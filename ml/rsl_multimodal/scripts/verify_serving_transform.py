"""Gate: the serving image path must reproduce the training image path.

Two checks, both on the blocked held-out split:

1. Rebuild each stored 224x224 training image from its full-resolution original with
   `image_io.load_image_bytes` and require the decoded tensor to be *bit-identical* to
   decoding the stored training JPEG.
2. Score the float16 TFLite model on the test set with the serving loader and require
   the MAE to match the figure `export_tflite.py` recorded.

Exits non-zero on failure so it can sit in run_all.sh.

    python scripts/verify_serving_transform.py
    python scripts/verify_serving_transform.py --originals ../Original_Images --limit 60
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from image_io import decoder, load_image, load_image_bytes

MAE_TOLERANCE = 5e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", type=Path, default=C.ROOT.parent / "Original_Images")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--model", type=Path, default=C.MODELS / "fusion_blocked_full_float16.tflite")
    args = ap.parse_args()

    name, exact, _ = decoder()
    print(f"decoder: {name} ({'exact' if exact else 'APPROXIMATE'})")
    if not exact:
        print("FAIL - install simplejpeg or tensorflow; PIL cannot reproduce training pixels")
        return 1

    df = C.load_master()
    test = df[df["split_blocked"] == "test"].reset_index(drop=True)
    failures = 0

    # --- 1. full-resolution originals -> training images, bit for bit -------------------
    checked = identical = 0
    if args.originals.is_dir():
        available = {p.name for p in args.originals.iterdir()}
        for f in test.loc[test["image_file"].isin(available), "image_file"].head(args.limit):
            rebuilt = load_image_bytes((args.originals / f).read_bytes()).pixels
            stored = load_image(C.IMAGE_DIR / f).pixels
            checked += 1
            identical += int(np.array_equal(rebuilt, stored))
        print(f"rebuild from originals: {identical}/{checked} bit-identical")
        if checked and identical != checked:
            failures += 1
    else:
        print(f"rebuild from originals: skipped ({args.originals} not found)")

    # --- 2. the served model reproduces the recorded test MAE -------------------------
    report_path = C.RESULTS / "blocked_full" / "tflite_report.json"
    recorded = None
    if report_path.exists():
        variants = json.loads(report_path.read_text()).get("variants", {})
        # fusion_blocked_full_float16.tflite -> "float16"
        entry = variants.get(args.model.stem.rsplit("_", 1)[-1], {})
        recorded = (entry.get("metrics") or {}).get("MAE")

    from predict_pi import load_interpreter

    with open(C.SENSOR_SCALER_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    feats = bundle["features"]["full"]
    X = bundle["scalers"]["blocked__full"].transform(test[feats].to_numpy("float64")).astype("float32")

    interp = load_interpreter(args.model)
    by_rank = {len(d["shape"]): d for d in interp.get_input_details()}
    out = interp.get_output_details()[0]
    preds = np.empty(len(test), "float64")
    for i, f in enumerate(test["image_file"]):
        interp.set_tensor(by_rank[4]["index"], load_image(C.IMAGE_DIR / f).pixels)
        interp.set_tensor(by_rank[2]["index"], X[i:i + 1])
        interp.invoke()
        preds[i] = float(interp.get_tensor(out["index"]).ravel()[0])
    mae = float(np.abs(preds - test[C.TARGET].to_numpy()).mean())

    if recorded is None:
        print(f"served test MAE: {mae:.4f} (no recorded figure to compare against)")
    else:
        ok = abs(mae - float(recorded)) <= MAE_TOLERANCE
        print(f"served test MAE: {mae:.4f}  recorded {float(recorded):.4f}  "
              f"{'match' if ok else 'MISMATCH'}")
        failures += int(not ok)

    C.save_json(C.RESULTS / "serving_transform_verification.json", {
        "decoder": name,
        "originals_checked": checked,
        "originals_bit_identical": identical,
        "served_test_mae": mae,
        "recorded_test_mae": recorded,
        "passed": failures == 0,
    })
    print("PASS" if failures == 0 else f"FAIL - {failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
