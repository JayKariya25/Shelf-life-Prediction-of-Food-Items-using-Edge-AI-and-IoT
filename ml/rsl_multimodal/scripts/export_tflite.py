"""Export the fused multimodal model to TFLite for the Raspberry Pi, and measure the cost.

Four conversions are produced and every one of them is re-evaluated on the held-out test
set, so the accuracy/size trade-off is measured rather than assumed:

  fusion_<tag>_float32.tflite   baseline, no quantization
  fusion_<tag>_float16.tflite   float16 weights            <- recommended for a Pi 4/5
  fusion_<tag>_dynamic.tflite   int8 weights, float activations
  fusion_<tag>_int8.tflite      full-integer weights+activations

Quantization is calibrated on TRAIN rows only - calibrating on val/test would leak.

Note on int8: MobileNetV3's hard-swish and squeeze-excite blocks are known to lose a lot
of accuracy under post-training full-integer quantization, and this task needs sub-day
resolution over a narrow output range, so the loss shows up plainly in the numbers below.
The float16 build is lossless here and is still XNNPACK-accelerated on a Pi. If int8 is a
hard requirement (Coral Edge TPU, very tight RAM), the route is quantization-aware
training or a quantization-friendlier backbone such as MobileNetV2 - not a different
converter flag.

    python scripts/export_tflite.py --split blocked
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import common as C
import datasets as D

CALIB_SAMPLES = 500
RECOMMENDED = "float16"


def load_images(files: list[str]) -> np.ndarray:
    out = np.empty((len(files), *C.IMAGE_SIZE, 3), dtype="float32")
    for i, f in enumerate(files):
        img = tf.io.decode_jpeg(tf.io.read_file(str(C.IMAGE_DIR / f)), channels=3)
        out[i] = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear").numpy()
    return out


def representative_dataset(images: np.ndarray, sensors: np.ndarray):
    """Keyed by input name so the converter cannot mis-assign the two inputs."""
    def gen():
        for i in range(len(images)):
            yield {"image": images[i:i + 1].astype("float32"),
                   "sensor": sensors[i:i + 1].astype("float32")}
    return gen


def make_interpreter(path: Path):
    """Interpreter with batch pinned to 1; XNNPACK when it will prepare the graph.

    Full-integer graphs are rejected by the XNNPACK delegate here, so they fall back to the
    reference kernels - slower, but correct. Which path was taken is reported with the
    latency, because the same choice happens on the Pi.
    """
    def _build(resolver=None):
        kwargs = {"model_path": str(path)}
        if resolver is not None:
            kwargs["experimental_op_resolver_type"] = resolver
        interp = tf.lite.Interpreter(**kwargs)
        for d in interp.get_input_details():
            shape = d["shape"].copy()
            shape[0] = 1
            interp.resize_tensor_input(d["index"], shape, strict=False)
        interp.allocate_tensors()
        return interp

    try:
        return _build(), "xnnpack"
    except RuntimeError:
        return _build(tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES), \
            "reference_kernels"


def run_tflite(path: Path, images: np.ndarray, sensors: np.ndarray):
    interp, backend = make_interpreter(path)
    by_rank = {len(d["shape"]): d for d in interp.get_input_details()}
    img_d, sen_d = by_rank[4], by_rank[2]
    out_d = interp.get_output_details()[0]

    preds = np.empty(len(images), dtype="float32")
    t0 = time.perf_counter()
    for i in range(len(images)):
        for detail, value in ((img_d, images[i:i + 1]), (sen_d, sensors[i:i + 1])):
            v = value.astype("float32")
            if detail["dtype"] in (np.int8, np.uint8):
                scale, zero = detail["quantization"]
                info = np.iinfo(detail["dtype"])
                v = np.round(v / scale + zero).clip(info.min, info.max)
            interp.set_tensor(detail["index"], v.astype(detail["dtype"]))
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])
        if out_d["dtype"] in (np.int8, np.uint8):
            scale, zero = out_d["quantization"]
            raw = (raw.astype("float32") - zero) * scale
        preds[i] = float(np.asarray(raw).ravel()[0])
    ms = (time.perf_counter() - t0) / len(images) * 1000.0
    return preds, ms, backend


def converters(model, rep) -> dict:
    def base():
        return tf.lite.TFLiteConverter.from_keras_model(model)

    def float32():
        return base()

    def float16():
        c = base()
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
        return c

    def dynamic():
        c = base()
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        return c

    def int8():
        c = base()
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return c

    return {"float32": float32, "float16": float16, "dynamic": dynamic, "int8": int8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    args = ap.parse_args()

    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    df, X, _F, y, (tr, _va, te) = D.load_all(args.split, args.feature_set)
    model = tf.keras.models.load_model(C.MODELS / f"fusion_model_{tag}.keras", compile=False)

    rng = np.random.default_rng(C.SEED)
    calib_idx = rng.choice(np.flatnonzero(tr), size=min(CALIB_SAMPLES, int(tr.sum())), replace=False)
    rep = representative_dataset(load_images(df.loc[calib_idx, "image_file"].tolist()), X[calib_idx])

    test_idx = np.flatnonzero(te)
    test_imgs = load_images(df.loc[test_idx, "image_file"].tolist())
    test_sen, y_te = X[test_idx], y[test_idx]
    keras_pred = model.predict([test_imgs, test_sen], verbose=0).ravel()

    report = {
        "split": args.split,
        "feature_set": args.feature_set,
        "recommended_variant": RECOMMENDED,
        "keras_model_bytes": (C.MODELS / f"fusion_model_{tag}.keras").stat().st_size,
        "calibration": {"samples": int(len(calib_idx)), "source": "train split only"},
        "keras_test_metrics": C.regression_metrics(y_te, keras_pred),
        "variants": {},
    }

    for name, make in converters(model, rep).items():
        path = C.MODELS / f"fusion_{tag}_{name}.tflite"
        try:
            path.write_bytes(make().convert())
        except Exception as exc:
            print(f"[warn] {name} conversion failed: {exc}")
            continue
        preds, ms, backend = run_tflite(path, test_imgs, test_sen)
        metrics = C.regression_metrics(y_te, preds)
        report["variants"][name] = {
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 3),
            "latency_ms_per_sample_x86_cpu": round(ms, 2),
            "cpu_backend": backend,
            "metrics": metrics,
            "max_abs_diff_vs_keras": float(np.max(np.abs(preds - keras_pred))),
        }
        print(f"{name:9s} {report['variants'][name]['size_mb']:>6.3f} MB   "
              f"MAE {metrics['MAE']:.4f}   R2 {metrics['R2']:.4f}   "
              f"{ms:5.1f} ms/sample  [{backend}]")

    C.save_json(outdir / "tflite_report.json", report)
    print(f"\nrecommended for Raspberry Pi: fusion_{tag}_{RECOMMENDED}.tflite")
    print(f"wrote {outdir / 'tflite_report.json'}")


if __name__ == "__main__":
    main()
