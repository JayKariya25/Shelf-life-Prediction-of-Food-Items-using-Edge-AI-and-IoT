#!/usr/bin/env bash
# End-to-end reproduction. Run from the project root:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/scripts"

echo "== 1. dataset, target and splits =="
python3 prepare_data.py

echo "== 2. cache frozen backbone features =="
python3 extract_image_features.py

for SPLIT in blocked random; do
  echo "== 3. sensor branch  ($SPLIT) =="
  python3 train_sensor.py --split "$SPLIT"
  echo "== 4. image branch   ($SPLIT) =="
  python3 train_image.py  --split "$SPLIT"
  echo "== 5. fusion head    ($SPLIT) =="
  python3 train_fusion.py --split "$SPLIT"
  echo "== 6. evaluate       ($SPLIT) =="
  python3 evaluate.py     --split "$SPLIT"
done

echo "== 7. TFLite export for Raspberry Pi =="
python3 export_tflite.py --split blocked

echo "== 8. verify on-device feature reconstruction =="
python3 verify_pi_features.py

echo "== 8b. verify the serving image transform reproduces training =="
python3 verify_serving_transform.py

echo "== 9. ablations, duplicate probe, leave-one-day-out =="
python3 ablations.py

echo "== 10. report =="
python3 make_report.py

echo "done."
