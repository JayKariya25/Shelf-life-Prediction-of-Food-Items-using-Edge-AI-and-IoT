"""Evaluate all three models, verify the assembled graph, and draw the report figures.

    python scripts/evaluate.py --split blocked
"""

from __future__ import annotations

import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import common as C
import datasets as D

BATCH = 64


def image_tensor_dataset(files: list[str]) -> tf.data.Dataset:
    paths = [str(C.IMAGE_DIR / f) for f in files]

    def _load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear")
        return tf.cast(img, tf.float32)

    return tf.data.Dataset.from_tensor_slices(paths).map(
        _load, num_parallel_calls=tf.data.AUTOTUNE)


def plot_scatter(y_true, preds: dict[str, np.ndarray], path, title: str) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.4), sharey=True)
    lo, hi = float(min(y_true)) - 0.5, float(max(y_true)) + 0.5
    for ax, (name, yp) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y_true, yp, s=9, alpha=0.35, edgecolor="none", color="#2f6f9f")
        ax.plot([lo, hi], [lo, hi], lw=1.2, color="#c0392b", ls="--")
        m = C.regression_metrics(y_true, yp)
        ax.set_title(f"{name}\nMAE {m['MAE']:.3f} d   R² {m['R2']:.4f}", fontsize=10)
        ax.set_xlabel("Actual RSL (days)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25, lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel("Predicted RSL (days)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_hist(y_true, preds: dict[str, np.ndarray], path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, yp in preds.items():
        ax.hist(np.asarray(yp).ravel() - y_true, bins=45, alpha=0.5, label=name)
    ax.axvline(0, color="#c0392b", lw=1.1, ls="--")
    ax.set_xlabel("Prediction error (days)"); ax.set_ylabel("Test samples")
    ax.legend(); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    args = ap.parse_args()

    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    df, X, F, y, (tr, va, te) = D.load_all(args.split, args.feature_set)

    sensor_branch = tf.keras.models.load_model(C.MODELS / f"sensor_model_{tag}.keras", compile=False)
    image_head = tf.keras.models.load_model(C.MODELS / f"image_head_{tag}.keras", compile=False)
    fusion_model = tf.keras.models.load_model(C.MODELS / f"fusion_model_{tag}.keras", compile=False)
    fusion_head = tf.keras.models.load_model(C.MODELS / f"fusion_head_{tag}.keras", compile=False)

    y_te = y[te]
    p_sensor = sensor_branch.predict(X[te], verbose=0).ravel()
    p_image = image_head.predict(F[te], verbose=0).ravel()

    img_emb = tf.keras.Model(image_head.input, image_head.get_layer("image_embedding").output)
    sen_emb = tf.keras.Model(sensor_branch.input, sensor_branch.get_layer("sensor_embedding").output)
    E_te = np.concatenate([img_emb.predict(F[te], verbose=0),
                           sen_emb.predict(X[te], verbose=0),
                           p_image[:, None], p_sensor[:, None]], axis=1)
    p_fusion_staged = fusion_head.predict(E_te, verbose=0).ravel()

    # End-to-end graph: raw pixels + scaled sensors -> RSL --------------------------------
    files = df.loc[te, "image_file"].tolist()
    imgs = image_tensor_dataset(files).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    img_batches = np.concatenate([b.numpy() for b in imgs], axis=0)
    p_fusion_e2e = fusion_model.predict([img_batches, X[te]], verbose=0).ravel()

    max_drift = float(np.max(np.abs(p_fusion_e2e - p_fusion_staged)))
    print(f"staged vs assembled-graph max |diff| = {max_drift:.3e}")

    preds = {"Sensor only": p_sensor, "Image only": p_image, "Fusion": p_fusion_e2e}
    summary = {
        "split": args.split, "feature_set": args.feature_set,
        "n_test": int(te.sum()),
        "graph_consistency_max_abs_diff": max_drift,
        "test_metrics": {k: C.regression_metrics(y_te, v) for k, v in preds.items()},
    }
    C.save_json(outdir / "comparison_metrics.json", summary)
    print(json.dumps(summary["test_metrics"], indent=2))

    plot_scatter(y_te, preds, outdir / "actual_vs_predicted.png",
                 f"Held-out test set - {args.split} split (n={int(te.sum())})")
    plot_error_hist(y_te, preds, outdir / "error_distribution.png")

    pd.DataFrame({
        "image_file": files,
        "Day": df.loc[te, "Day"].to_numpy(),
        "actual_RSL": y_te,
        "pred_sensor": p_sensor,
        "pred_image": p_image,
        "pred_fusion": p_fusion_e2e,
    }).to_csv(outdir / "all_model_test_predictions.csv", index=False)
    print(f"figures + tables written to {outdir}")


if __name__ == "__main__":
    main()
