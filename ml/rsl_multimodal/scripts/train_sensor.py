"""Stage 1a - train the sensor-only RSL model.

    python scripts/train_sensor.py --split blocked --feature-set full
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import tensorflow as tf

import common as C
import datasets as D
import models as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="blocked", choices=C.SPLIT_STRATEGIES)
    ap.add_argument("--feature-set", default="full", choices=["full", "minimal"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    C.set_seeds()
    df, X, _F, y, (tr, va, te) = D.load_all(args.split, args.feature_set)
    tag = f"{args.split}_{args.feature_set}"
    outdir = C.RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"sensor branch | split={args.split} features={X.shape[1]} "
          f"train={tr.sum()} val={va.sum()} test={te.sum()}")

    model = M.compile_regressor(M.build_sensor_branch(X.shape[1]), lr=1e-3)
    hist = model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=M.default_callbacks(),
        verbose=2,
    )

    metrics = {
        part: C.regression_metrics(y[m], model.predict(X[m], verbose=0))
        for part, m in (("train", tr), ("val", va), ("test", te))
    }
    print(json.dumps(metrics["test"], indent=2))

    model.save(C.MODELS / f"sensor_model_{tag}.keras")
    pd.DataFrame(hist.history).to_csv(outdir / "sensor_training_history.csv", index=False)
    D.save_predictions(df, te, y[te], model.predict(X[te], verbose=0),
                       outdir / "sensor_test_predictions.csv")
    C.save_json(outdir / "sensor_metrics.json", {
        "model": "sensor_branch", "split": args.split, "feature_set": args.feature_set,
        "n_features": int(X.shape[1]), "epochs_run": len(hist.history["loss"]),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
