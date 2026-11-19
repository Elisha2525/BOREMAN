import os
import requests
from datetime import datetime

headers = {"X-AIO-Key": os.environ["ADAFRUIT_IO_KEY"]}
base = "https://io.adafruit.com/api/v2/Jesica_April/feeds"

for feed in ["tds", "temperature", "humidity"]:
    r = requests.get(
        f"{base}/{feed}/data",
        headers=headers,
        params={"limit": 100},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    print("\n", feed.upper(), "records:", len(data))
    if data:
        print("Oldest:", data[-1]["created_at"])
        print("Newest:", data[0]["created_at"])
