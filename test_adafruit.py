import os
import requests

headers = {"X-AIO-Key": os.environ["ADAFRUIT_IO_KEY"]}
base = "https://io.adafruit.com/api/v2/Jesica_April/feeds"

for feed in ["tds", "temperature", "humidity"]:
    response = requests.get(
        f"{base}/{feed}/data",
        headers=headers,
        params={"limit": 5},
        timeout=20
    )
    print("\n", feed.upper(), "HTTP:", response.status_code)
    if response.ok:
        for item in response.json():
            print(item.get("value"), item.get("created_at"))
    else:
        print(response.text[:300])
