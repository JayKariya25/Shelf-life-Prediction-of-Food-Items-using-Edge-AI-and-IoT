"""Stage 2 - freeze both branches and train the fusion head.

Neither branch is updated here. The image branch contributes its 128-d `image_embedding`,
the sensor branch its 64-d `sensor_embedding`; the concatenated 192-d vector is the only
input the fusion head ever sees. Because both branches are frozen, their embeddings are
computed once and reused, and the saved multimodal graph is numerically identical to this
staged pipeline.

    python scripts/train_fusion.py --split blocked
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

import common as C
import datasets as D
import models as M
from extract_image_features import build_backbone


def embed(model: tf.keras.Model, layer: str, X: np.ndarray) -> np.ndarray:
    extractor = tf.keras.Model(model.input, model.get_layer(layer).output)
    return extractor.predict(X, verbose=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    C.set_seeds()
    df, X, F, y, (tr, va, te) = D.load_all(args.split, args.feature_set)
    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    sensor_branch = tf.keras.models.load_model(C.MODELS / f"sensor_model_{tag}.keras", compile=False)
    image_head = tf.keras.models.load_model(C.MODELS / f"image_head_{tag}.keras", compile=False)
    sensor_branch.trainable = False
    image_head.trainable = False

    E_img = embed(image_head, "image_embedding", F)
    E_sen = embed(sensor_branch, "sensor_embedding", X)
    P_img = image_head.predict(F, verbose=0)
    P_sen = sensor_branch.predict(X, verbose=0)
    E = np.concatenate([E_img, E_sen, P_img, P_sen], axis=1).astype("float32")
    print(f"fusion | split={args.split} input={E.shape[1]}d "
          f"(image {E_img.shape[1]} + sensor {E_sen.shape[1]} + 2 branch predictions)")

    fusion_head = M.compile_regressor(M.build_fusion_head(), lr=1e-3)
    hist = fusion_head.fit(
        E[tr], y[tr],
        validation_data=(E[va], y[va]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=M.default_callbacks(),
        verbose=2,
    )

    metrics = {
        part: C.regression_metrics(y[m], fusion_head.predict(E[m], verbose=0))
        for part, m in (("train", tr), ("val", va), ("test", te))
    }
    print(json.dumps(metrics["test"], indent=2))

    # Assemble and verify the deployable two-input graph -------------------------------
    backbone = build_backbone()
    multimodal = M.assemble_multimodal(backbone, image_head, sensor_branch,
                                       fusion_head, n_sensor_features=X.shape[1])
    multimodal.save(C.MODELS / f"fusion_model_{tag}.keras")
    fusion_head.save(C.MODELS / f"fusion_head_{tag}.keras")

    pd.DataFrame(hist.history).to_csv(outdir / "fusion_training_history.csv", index=False)
    D.save_predictions(df, te, y[te], fusion_head.predict(E[te], verbose=0),
                       outdir / "fusion_test_predictions.csv")
    C.save_json(outdir / "fusion_metrics.json", {
        "model": "fusion_head_on_frozen_branches", "split": args.split,
        "embedding_dims": {"image": int(E_img.shape[1]), "sensor": int(E_sen.shape[1])},
        "trainable_params": int(sum(np.prod(w.shape) for w in fusion_head.trainable_weights)),
        "epochs_run": len(hist.history["loss"]),
        "metrics": metrics,
    })
    print(f"saved {C.MODELS / f'fusion_model_{tag}.keras'}")


if __name__ == "__main__":
    main()
