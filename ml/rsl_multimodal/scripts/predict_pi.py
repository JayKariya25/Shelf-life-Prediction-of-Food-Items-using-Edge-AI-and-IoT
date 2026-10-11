"""Raspberry Pi inference: a photograph -> Remaining Shelf Life.

Runs on `tflite_runtime` (the light Pi package) when available, LiteRT or full TensorFlow
otherwise, so the same file works on the Pi and on a dev machine.

    # Pi
    pip install tflite-runtime pillow numpy
    python scripts/predict_pi.py --image capture.jpg

Both exports are supported, and the model file decides which. The recommended
`image_ensemble_*` export takes one input - the photograph - and needs no readings, no
scaler and no session state. The older `fusion_*` export additionally takes the
27-feature sensor vector, so with it the three readings are required:

    pip install tflite-runtime pillow numpy scikit-learn
    python scripts/predict_pi.py --model models/fusion_blocked_full_float16.tflite \
        --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185

The engineered sensor vector is rebuilt by `pi_features.SensorFeatureBuilder`, whose
definitions are checked against the training CSV by `scripts/verify_pi_features.py`.
Images go through `image_io.load_image`, which reproduces the training transform exactly
(checked by `scripts/verify_serving_transform.py`).

Session state (the rolling window, the session's first smoothed reading, the last in-range
value per channel) is kept in `runtime/pi_state.json` so a restart mid-session does not
reset the history. Use --reset-session to start a new monitoring session, or --state to
keep one file per batch. (Earlier versions wrote this into artifacts/, mixing runtime
state with the trained artifacts.)

Ship to the Pi: the .tflite model, artifacts/preprocessing_config.json, and
scripts/predict_pi.py and image_io.py - plus artifacts/sensor_scaler.pkl and
pi_features.py if you are running the fusion export. Install simplejpeg there so images
decode exactly as in training.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from image_io import load_image
from pi_features import CHANNELS, SensorFeatureBuilder  # noqa: F401  (re-exported for callers)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
STATE_PATH = ROOT / "runtime" / "pi_state.json"


def load_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter      # Raspberry Pi
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # LiteRT
        except ImportError:
            import tensorflow as tf                             # dev machine
            Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=str(model_path))
    for d in interp.get_input_details():
        shape = d["shape"].copy()
        shape[0] = 1
        interp.resize_tensor_input(d["index"], shape, strict=False)
    interp.allocate_tensors()
    return interp


def quantize_in(detail, value: np.ndarray) -> np.ndarray:
    if detail["dtype"] in (np.int8, np.uint8):
        scale, zero = detail["quantization"]
        info = np.iinfo(detail["dtype"])
        value = np.round(value / scale + zero).clip(info.min, info.max)
    return value.astype(detail["dtype"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--temperature", type=float, help="required by the fusion export only")
    ap.add_argument("--humidity", type=float, help="required by the fusion export only")
    ap.add_argument("--gas", type=float, help="required by the fusion export only")
    ap.add_argument("--minutes-since-previous", type=float, default=2.0,
                    help="sampling interval; used for the *_Rate features")
    ap.add_argument("--model", type=Path,
                    default=MODELS / "image_ensemble_blocked_full_float16.tflite")
    ap.add_argument("--split", default="blocked")
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--reset-session", action="store_true",
                    help="forget the rolling history and start a new monitoring session")
    ap.add_argument("--state", type=Path, default=STATE_PATH,
                    help="session state file; use one per food batch")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    config = json.loads((ARTIFACTS / "preprocessing_config.json").read_text())
    bounds = config["outlier_bounds"]
    envelope = config.get("input_envelope", {})
    max_days = float(config["target"]["definition"].split("-")[0].strip())

    interp = load_interpreter(args.model)
    by_rank = {len(d["shape"]): d for d in interp.get_input_details()}
    # The graph's own contract decides what has to be prepared, so an export swap cannot
    # silently leave an input unfed or a reading quietly unused.
    image_only = 2 not in by_rank

    builder = None
    readings: dict[str, float] = {}
    if not image_only:
        missing = [name for name in ("temperature", "humidity", "gas")
                   if getattr(args, name) is None]
        if missing:
            ap.error(f"{args.model.name} takes a sensor vector, so --{' --'.join(missing)} "
                     f"{'is' if len(missing) == 1 else 'are'} required")
        with open(ARTIFACTS / "sensor_scaler.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        feature_names = bundle["features"][args.feature_set]
        scaler = bundle["scalers"][f"{args.split}__{args.feature_set}"]

        state = None
        if args.state.exists() and not args.reset_session:
            state = json.loads(args.state.read_text())
        builder = SensorFeatureBuilder.from_state(state, bounds)

        readings = {"Temperature": args.temperature, "Humidity": args.humidity,
                    "Gas": args.gas}
        raw = builder.vector(readings, args.minutes_since_previous / 60.0, feature_names)
        sensor = scaler.transform(raw.astype("float64")).astype("float32")

    loaded = load_image(args.image)
    image = loaded.pixels

    interp.set_tensor(by_rank[4]["index"], quantize_in(by_rank[4], image))
    if not image_only:
        interp.set_tensor(by_rank[2]["index"], quantize_in(by_rank[2], sensor))
    interp.invoke()

    out_d = interp.get_output_details()[0]
    raw_out = interp.get_tensor(out_d["index"])
    if out_d["dtype"] in (np.int8, np.uint8):
        scale, zero = out_d["quantization"]
        raw_out = (raw_out.astype("float32") - zero) * scale
    raw_rsl = float(np.asarray(raw_out).ravel()[0])
    rsl = float(np.clip(raw_rsl, 0.0, max_days))

    if builder is not None:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(builder.state_dict(), indent=2))

    notes = list(loaded.notes)
    # An image-only export reads none of the channels, so neither the envelope check nor
    # the outlier report below can apply to it: `readings` is empty and both loops no-op.
    for channel, value in readings.items():
        env = envelope.get(channel)
        if env and not (env["min"] <= value <= env["max"]):
            notes.append(
                f"UNRELIABLE: {channel} {value} is outside every reading the model was trained "
                f"on ({env['min']:g} to {env['max']:g}). The raw value still reaches the network "
                f"through the instantaneous feature, so this is extrapolation."
            )
    adjustments = builder.last_adjustments if builder is not None else {}
    for channel, change in adjustments.items():
        verb = "clipped to the outlier bound" if change["how"] == "clipped" else \
            "replaced by the last in-range reading"
        notes.append(f"{channel} {change['raw']} is an outlier; its smoothed features used "
                     f"{change['used']:.2f} ({verb})")
    if rsl != raw_rsl:
        notes.append(f"raw model output {raw_rsl:.2f} clamped to the 0-{max_days:g} day range")

    if args.json:
        print(json.dumps({"remaining_shelf_life_days": round(rsl, 2),
                          "raw_model_output": round(raw_rsl, 4),
                          "image": str(args.image), "model": args.model.name,
                          "decoder": loaded.decoder, "readings": readings,
                          "notes": notes}))
    else:
        print(f"Remaining shelf life: {rsl:.2f} days")
        for note in notes:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
