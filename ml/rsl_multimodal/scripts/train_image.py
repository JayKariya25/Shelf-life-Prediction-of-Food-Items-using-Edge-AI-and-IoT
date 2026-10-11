"""Stage 1b - train the image-only RSL model.

The MobileNetV3Small backbone stays frozen, so its 576-d pooled output is constant and is
read from the cache built by extract_image_features.py. Only the head is trained here; the
frozen backbone and the trained head are then saved as one end-to-end image model.

    python scripts/train_image.py --split blocked
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    C.set_seeds()
    df, _X, F, y, (tr, va, te) = D.load_all(args.split, args.feature_set)
    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"image branch | split={args.split} backbone_dim={F.shape[1]} "
          f"train={tr.sum()} val={va.sum()} test={te.sum()}")

    head = M.compile_regressor(M.build_image_head(F.shape[1]), lr=1e-3)
    hist = head.fit(
        F[tr], y[tr],
        validation_data=(F[va], y[va]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=M.default_callbacks(),
        verbose=2,
    )

    metrics = {
        part: C.regression_metrics(y[m], head.predict(F[m], verbose=0))
        for part, m in (("train", tr), ("val", va), ("test", te))
    }
    print(json.dumps(metrics["test"], indent=2))

    head.save(C.MODELS / f"image_head_{tag}.keras")
    backbone = build_backbone()
    image_model = M.assemble_image_branch(backbone, head)
    image_model.save(C.MODELS / f"image_model_{tag}.keras")

    pd.DataFrame(hist.history).to_csv(outdir / "image_training_history.csv", index=False)
    D.save_predictions(df, te, y[te], head.predict(F[te], verbose=0),
                       outdir / "image_test_predictions.csv")
    C.save_json(outdir / "image_metrics.json", {
        "model": "image_branch", "split": args.split,
        "backbone": "MobileNetV3Small (frozen, ImageNet)",
        "backbone_dim": int(F.shape[1]), "epochs_run": len(hist.history["loss"]),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
