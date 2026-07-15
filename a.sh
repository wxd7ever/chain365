cd /mnt/data/robocasa-main

URL=$(python - <<'PY'
import json
import os
import robocasa

key = "target/atomic/PickPlaceCounterToCabinet/20250811/lerobot.tar"

json_path = os.path.join(
    robocasa.__path__[0],
    "models",
    "assets",
    "box_links",
    "box_links_ds.json",
)

with open(json_path, "r") as f:
    shared_url = json.load(f)[key]

shared_id = shared_url.rstrip("/").split("/")[-1]
base = shared_url.split("/s/")[0]
print(f"{base}/shared/static/{shared_id}.tar")
PY
)

echo "$URL"
