import os
import requests
import pandas as pd

headers = {"X-AIO-Key": os.environ["ADAFRUIT_IO_KEY"]}
base = "https://io.adafruit.com/api/v2/Jesica_April/feeds"

data = {}
for feed in ["tds", "temperature", "humidity"]:
    r = requests.get(f"{base}/{feed}/data", headers=headers,
                     params={"limit": 100}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df[feed] = pd.to_numeric(df["value"], errors="coerce")
    data[feed] = df[["created_at", feed]].sort_values("created_at")

merged = pd.merge_asof(
    data["tds"], data["temperature"],
    on="created_at", direction="nearest",
    tolerance=pd.Timedelta(seconds=2)
)
merged = pd.merge_asof(
    merged.sort_values("created_at"), data["humidity"],
    on="created_at", direction="nearest",
    tolerance=pd.Timedelta(seconds=2)
)

print("TDS records:", len(merged))
print("Complete rows within 2 seconds:",
      merged[["tds", "temperature", "humidity"]].notna().all(axis=1).sum())
print("Missing after alignment:")
print(merged[["tds", "temperature", "humidity"]].isna().sum())
