DIR=/mnt/data/robocasa-main/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811

mkdir -p "$DIR"

aria2c \
  -c \
  -x 8 \
  -s 8 \
  -k 1M \
  --file-allocation=none \
  --connect-timeout=30 \
  --timeout=60 \
  --retry-wait=5 \
  --max-tries=0 \
  -d "$DIR" \
  -o lerobot.tar \
  "$URL"
