"""Train and export the deployed model: an ensemble of image heads over the frozen backbone.

Why this and not the fusion model
---------------------------------
scripts/experiments.py scores every candidate by leave-one-day-out (LODO), the only
protocol this dataset supports honestly - the blocked test split is largely duplicate
retrieval (1-NN on *untrained* features scores 0.081 MAE on it). Measured over six
training seeds:

    sensor branch   2.392 +/- 0.106
    fusion head     1.113 +/- 0.179     <- what was being served
    image head      0.920 +/- 0.086
    image head, 6-seed ensemble  0.856  <- what this script builds

Two findings drive the change. The image branch beats fusion on an unseen day (it wins on
4 of 6 seeds, mean margin 0.193 days): the sensor branch learned the ambient conditions of
the one 11-day run it saw, and the fusion head leans on it. And averaging several heads
beats any single head, because head training is the largest source of run-to-run variance
here (+/-0.086 is comparable to the differences between whole architectures).

Dropping the sensor input also removes three preconditions from serving - a gas reading, a
reading history long enough for the rolling features, and readings inside the trained
envelope. A recent photograph is now the only requirement.

    python scripts/train_image_ensemble.py --lodo      # measure (writes the honest figure)
    python scripts/train_image_ensemble.py             # train + export for serving

Results: results/image_ensemble_lodo.json, results/blocked_full/image_ensemble_*.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import common as C
import datasets as D
import models as M
from extract_image_features import build_backbone

# The six seeds experiments.py measured the 0.856 ensemble figure on. Kept fixed so the
# deployed ensemble is the one that was scored, not a same-sized stand-in.
SEEDS = (42, 7, 2026, 13, 99, 404)
EPOCHS = 200
PATIENCE = 12
BATCH = 64
MIN_SAMPLES_PER_DAY = 100
EXPORT_VARIANTS = ("float32", "float16")


# --------------------------------------------------------------------------------------
# the ensemble
# --------------------------------------------------------------------------------------
def train_heads(F_tr, y_tr, F_va, y_va, seeds=SEEDS, *, verbose: int = 0) -> list:
    """One image head per seed. Only the seed differs, so no head is favoured."""
    heads = []
    for i, seed in enumerate(seeds):
        C.set_seeds(seed)
        head = M.compile_regressor(M.build_image_head(F_tr.shape[1], name=f"image_head_{i}"))
        head.fit(F_tr, y_tr, validation_data=(F_va, y_va), epochs=EPOCHS,
                 batch_size=BATCH, callbacks=M.default_callbacks(PATIENCE), verbose=verbose)
        heads.append(head)
    return heads


def ensemble_predict(heads: list, F) -> np.ndarray:
    """Mean of the heads - the same arithmetic the exported graph performs."""
    return np.mean([h.predict(F, verbose=0).ravel() for h in heads], axis=0)


def assemble(heads: list, backbone) -> tf.keras.Model:
    """photo -> frozen backbone -> K heads -> mean -> RSL in days.

    One input and one scalar output, so the serving contract has no sensor vector in it.
    """
    backbone.trainable = False
    for head in heads:
        head.trainable = False
    inp = layers.Input(shape=(*C.IMAGE_SIZE, 3), name="image")
    pooled = backbone(inp, training=False)
    per_head = [head(pooled, training=False) for head in heads]
    out = layers.Average(name="remaining_shelf_life")(per_head)
    return tf.keras.Model(inp, out, name="image_ensemble_rsl")


# --------------------------------------------------------------------------------------
# honest measurement
# --------------------------------------------------------------------------------------
def run_lodo(seeds=SEEDS) -> dict:
    """Hold out one whole day for test and a different whole day for validation.

    Every day in the run is a batch the model never saw, which is what a user's item is.
    """
    df = C.load_master()
    F = np.load(C.ARTIFACTS / "backbone_features.npy")
    y = df[C.TARGET].to_numpy("float32")
    days = sorted(int(d) for d, n in df["Day"].value_counts().items() if n >= MIN_SAMPLES_PER_DAY)

    folds, started = {}, time.perf_counter()
    for i, day in enumerate(days):
        val_day = days[(i + 1) % len(days)]
        te = (df["Day"] == day).to_numpy()
        va = (df["Day"] == val_day).to_numpy()
        tr = ~(te | va)

        heads = train_heads(F[tr], y[tr], F[va], y[va], seeds)
        singles = [h.predict(F[te], verbose=0).ravel().astype(float) for h in heads]
        mean_pred = np.mean(singles, axis=0)

        folds[str(day)] = {
            "held_out_samples": int(te.sum()),
            "validation_day": val_day,
            "true_RSL": float(y[te][0]),
            "metrics": {"image_ensemble": C.regression_metrics(y[te], mean_pred)},
            "single_head_MAE": [round(float(np.abs(p - y[te]).mean()), 4) for p in singles],
        }
        single_mean = float(np.mean(folds[str(day)]["single_head_MAE"]))
        print(f"  day {day:>2}  true {y[te][0]:>4.1f}  "
              f"ensemble {folds[str(day)]['metrics']['image_ensemble']['MAE']:.3f}  "
              f"(mean single head {single_mean:.3f})")

    ensemble_mae = float(np.mean([f["metrics"]["image_ensemble"]["MAE"] for f in folds.values()]))
    single_mae = float(np.mean([np.mean(f["single_head_MAE"]) for f in folds.values()]))
    payload = {
        "protocol": "leave-one-day-out; one day held out for test, the next for validation",
        "heads": len(seeds),
        "seeds": list(seeds),
        "folds": folds,
        "mean_MAE": {"image_ensemble": round(ensemble_mae, 4),
                     "single_head_average": round(single_mae, 4)},
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    print(f"\n  LODO mean MAE: ensemble {ensemble_mae:.3f} d  "
          f"vs {single_mae:.3f} d for an average single head  "
          f"({payload['runtime_seconds']:.0f}s)")
    return payload


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------
def load_test_images(files: list[str]) -> np.ndarray:
    out = np.empty((len(files), *C.IMAGE_SIZE, 3), dtype="float32")
    for i, name in enumerate(files):
        img = tf.io.decode_jpeg(tf.io.read_file(str(C.IMAGE_DIR / name)), channels=3)
        out[i] = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear").numpy()
    return out


def make_interpreter(path: Path):
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


def export_tflite(model: tf.keras.Model, tag: str, images: np.ndarray, y_true: np.ndarray) -> dict:
    """Convert, then re-score every conversion on the held-out rows it must serve."""
    report = {}
    for variant in EXPORT_VARIANTS:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT] if variant == "float16" else []
        if variant == "float16":
            converter.target_spec.supported_types = [tf.float16]
        path = C.MODELS / f"image_ensemble_{tag}_{variant}.tflite"
        path.write_bytes(converter.convert())

        interp, delegate = make_interpreter(path)
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        preds, started = np.empty(len(images), dtype="float64"), time.perf_counter()
        for i in range(len(images)):
            interp.set_tensor(inp["index"], images[i:i + 1].astype(inp["dtype"]))
            interp.invoke()
            preds[i] = float(np.asarray(interp.get_tensor(out["index"])).ravel()[0])
        latency = (time.perf_counter() - started) / len(images) * 1000.0

        report[variant] = {
            "path": path.name,
            "size_mb": round(path.stat().st_size / 1e6, 3),
            "metrics": C.regression_metrics(y_true, preds),
            "latency_ms_per_sample_x86_cpu": round(latency, 2),
            "delegate": delegate,
        }
        print(f"  {variant:<9} {report[variant]['size_mb']:>6.2f} MB  "
              f"MAE {report[variant]['metrics']['MAE']:.4f}  "
              f"R2 {report[variant]['metrics']['R2']:.4f}  {latency:.1f} ms  ({delegate})")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lodo", action="store_true",
                    help="measure the ensemble leave-one-day-out instead of exporting")
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    ap.add_argument("--heads", type=int, default=len(SEEDS),
                    help=f"how many of the measured seeds to use (max {len(SEEDS)})")
    args = ap.parse_args()
    seeds = SEEDS[: max(1, min(args.heads, len(SEEDS)))]

    if args.lodo:
        print(f"image ensemble | leave-one-day-out | {len(seeds)} heads {list(seeds)}")
        C.save_json(C.RESULTS / "image_ensemble_lodo.json", run_lodo(seeds))
        print(f"  wrote {C.RESULTS / 'image_ensemble_lodo.json'}")
        return

    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    df, _X, F, y, (tr, va, te) = D.load_all(args.split, args.feature_set)
    print(f"image ensemble | split={args.split} heads={len(seeds)} backbone_dim={F.shape[1]} "
          f"train={tr.sum()} val={va.sum()} test={te.sum()}")

    heads = train_heads(F[tr], y[tr], F[va], y[va], seeds, verbose=0)

    single = {part: [C.regression_metrics(y[m], h.predict(F[m], verbose=0)) for h in heads]
              for part, m in (("val", va), ("test", te))}
    metrics = {part: C.regression_metrics(y[m], ensemble_predict(heads, F[m]))
               for part, m in (("train", tr), ("val", va), ("test", te))}
    print(f"  keras ensemble  test MAE {metrics['test']['MAE']:.4f}  "
          f"R2 {metrics['test']['R2']:.4f}   "
          f"(single heads {np.mean([m['MAE'] for m in single['test']]):.4f} "
          f"+/- {np.std([m['MAE'] for m in single['test']]):.4f})")

    for i, head in enumerate(heads):
        head.save(C.MODELS / f"image_ensemble_head{i}_{tag}.keras")
    model = assemble(heads, build_backbone())
    model.save(C.MODELS / f"image_ensemble_model_{tag}.keras")

    images = load_test_images(df.loc[te, "image_file"].tolist())
    keras_end_to_end = C.regression_metrics(y[te], model.predict(images, verbose=0))
    print(f"  end-to-end graph  test MAE {keras_end_to_end['MAE']:.4f} "
          f"(cache-based {metrics['test']['MAE']:.4f})")
    tflite = export_tflite(model, tag, images, y[te])

    D.save_predictions(df, te, y[te], ensemble_predict(heads, F[te]),
                       outdir / "image_ensemble_test_predictions.csv")
    C.save_json(outdir / "image_ensemble_metrics.json", {
        "model": "image_ensemble", "split": args.split, "heads": len(seeds), "seeds": list(seeds),
        "backbone": "MobileNetV3Small (frozen, ImageNet)", "backbone_dim": int(F.shape[1]),
        "metrics": metrics,
        "single_head_metrics": single,
        "end_to_end_graph_test": keras_end_to_end,
        "note": ("Within-session figures. The blocked split is largely duplicate retrieval, "
                 "so quote results/image_ensemble_lodo.json instead."),
    })
    C.save_json(outdir / "image_ensemble_tflite_report.json", {
        "model": "image_ensemble", "split": args.split, "heads": len(seeds),
        "recommended": "float16", "variants": tflite,
    })
    print(f"  wrote {outdir / 'image_ensemble_metrics.json'} and the TFLite report")


if __name__ == "__main__":
    main()
