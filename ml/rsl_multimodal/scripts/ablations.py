"""Three checks that keep the headline numbers honest.

1. Nearest-neighbour duplicate probe. Frames arrive every ~2 minutes, so a split that lets
   a test frame sit beside its near-twin in train measures lookup, not learning. This probe
   scores 1-NN on *untrained* ImageNet features: if it rivals the trained model, the split
   is leaking. Run on both splits so the claim is measured rather than asserted.
2. Level-feature ablation - what the excluded, unreproducible *_Level_Ord columns were worth.
3. Leave-one-day-out - every frame of one day removed from training, and the validation set
   drawn from a *different* held-out day so early stopping is also selected across days.
   This is the hardest question the dataset supports and the number to quote for a new batch.

    python scripts/ablations.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import common as C
import datasets as D
import models as M


# --------------------------------------------------------------------------------------
def nearest_neighbour_probe() -> dict:
    """1-NN from test rows to train rows, using features that never saw the target."""
    out = {}
    for split in C.SPLIT_STRATEGIES:
        df, X, F, y, (tr, _va, te) = D.load_all(split, "full")
        res = {}
        for name, A in (("frozen_imagenet_features", F), ("scaled_sensor_vector", X)):
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            sims = An[te] @ An[tr].T
            nn = sims.argmax(axis=1)
            res[name] = {
                "metrics": C.regression_metrics(y[te], y[tr][nn]),
                "median_similarity_to_nearest_train": float(np.median(sims.max(axis=1))),
            }
        gaps = nearest_time_gap_minutes(df, tr, te)
        res["minutes_to_nearest_train_row"] = {
            "median": float(np.median(gaps)),
            "p05": float(np.percentile(gaps, 5)),
            "fraction_within_5_min": float((gaps <= 5).mean()),
        }
        out[split] = res
        print(f"  {split:8s} 1-NN image MAE {res['frozen_imagenet_features']['metrics']['MAE']:.3f}  "
              f"1-NN sensor MAE {res['scaled_sensor_vector']['metrics']['MAE']:.3f}  "
              f"median gap {res['minutes_to_nearest_train_row']['median']:.0f} min")
    return out


def nearest_time_gap_minutes(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    ts = df["Timestamp"].to_numpy("datetime64[s]").astype("int64")
    train_ts = np.sort(ts[tr])
    pos = np.searchsorted(train_ts, ts[te])
    left = np.where(pos > 0, ts[te] - train_ts[np.clip(pos - 1, 0, None)], np.inf)
    right = np.where(pos < len(train_ts),
                     train_ts[np.clip(pos, None, len(train_ts) - 1)] - ts[te], np.inf)
    return np.minimum(left, right) / 60.0


# --------------------------------------------------------------------------------------
def level_feature_ablation() -> dict:
    df = C.load_master()
    raw = C.add_level_ordinals(pd.read_csv(C.SENSOR_FEATURES_CSV))
    raw["image_file"] = raw["Image_Path"].astype(str).map(
        lambda p: p.replace("\\", "/").split("/")[-1])
    extra = raw.set_index("image_file").loc[df["image_file"], C.UNREPRODUCIBLE_AT_INFERENCE]
    extra = extra.to_numpy("float32")

    _df, X, _F, y, (tr, va, te) = D.load_all("blocked", "full")
    results = {}
    for name, Xi in (("deployable_features", X),
                     ("plus_excluded_columns", np.concatenate([X, extra], axis=1))):
        C.set_seeds()
        model = M.compile_regressor(M.build_sensor_branch(Xi.shape[1]))
        model.fit(Xi[tr], y[tr], validation_data=(Xi[va], y[va]), epochs=200,
                  batch_size=64, callbacks=M.default_callbacks(), verbose=0)
        results[name] = {"n_features": int(Xi.shape[1]),
                         "test": C.regression_metrics(y[te], model.predict(Xi[te], verbose=0))}
        print(f"  {name:22s} n={Xi.shape[1]:>2}  MAE {results[name]['test']['MAE']:.4f}")
    return results


# --------------------------------------------------------------------------------------
def embed(model, layer, A):
    return tf.keras.Model(model.input, model.get_layer(layer).output).predict(A, verbose=0)


def leave_one_day_out(min_samples: int = 100) -> dict:
    """Hold out one whole day for test and a different whole day for validation."""
    df = C.load_master()
    feats = C.feature_columns("full")
    raw_X = df[feats].to_numpy("float64")
    F = np.load(C.ARTIFACTS / "backbone_features.npy")
    y = df[C.TARGET].to_numpy("float32")

    days = sorted(int(d) for d, n in df["Day"].value_counts().items() if n >= min_samples)
    out = {}
    for i, day in enumerate(days):
        val_day = days[(i + 1) % len(days)]
        te = (df["Day"] == day).to_numpy()
        va = (df["Day"] == val_day).to_numpy()
        tr = ~(te | va)

        # Scaler fitted on this fold's train rows only - the held-out day must not even
        # contribute its mean and variance.
        scaler = StandardScaler().fit(raw_X[tr])
        X = scaler.transform(raw_X).astype("float32")

        C.set_seeds()
        sensor = M.compile_regressor(M.build_sensor_branch(X.shape[1]))
        sensor.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=120, batch_size=64,
                   callbacks=M.default_callbacks(8), verbose=0)
        image = M.compile_regressor(M.build_image_head(F.shape[1]))
        image.fit(F[tr], y[tr], validation_data=(F[va], y[va]), epochs=120, batch_size=64,
                  callbacks=M.default_callbacks(8), verbose=0)

        E = np.concatenate([embed(image, "image_embedding", F),
                            embed(sensor, "sensor_embedding", X),
                            image.predict(F, verbose=0),
                            sensor.predict(X, verbose=0)], axis=1)
        C.set_seeds()
        fusion = M.compile_regressor(M.build_fusion_head())
        fusion.fit(E[tr], y[tr], validation_data=(E[va], y[va]), epochs=120, batch_size=64,
                   callbacks=M.default_callbacks(8), verbose=0)

        preds = {
            "sensor": sensor.predict(X[te], verbose=0).ravel(),
            "image": image.predict(F[te], verbose=0).ravel(),
            "fusion": fusion.predict(E[te], verbose=0).ravel(),
        }
        out[str(day)] = {
            "held_out_samples": int(te.sum()),
            "validation_day": val_day,
            "true_RSL": float(y[te][0]),
            "mean_predicted_RSL": float(preds["fusion"].mean()),
            "metrics": {k: C.regression_metrics(y[te], v) for k, v in preds.items()},
        }
        print(f"  day {day:>2} (val day {val_day:>2})  true {y[te][0]:>4.1f}  "
              f"pred {preds['fusion'].mean():>5.2f}  fusion MAE "
              f"{out[str(day)]['metrics']['fusion']['MAE']:.3f}  (n={int(te.sum())})")
    return out


def main() -> None:
    print("Nearest-neighbour duplicate probe (no training involved):")
    probe = nearest_neighbour_probe()

    print("\nExcluded-column ablation (blocked split, sensor branch):")
    ablation = level_feature_ablation()

    print("\nLeave-one-day-out:")
    lodo = leave_one_day_out()

    per_model = {
        m: float(np.mean([v["metrics"][m]["MAE"] for v in lodo.values()]))
        for m in ("sensor", "image", "fusion")
    }
    payload = {
        "nearest_neighbour_probe": probe,
        "excluded_column_ablation": ablation,
        "leave_one_day_out": lodo,
        "leave_one_day_out_summary": {
            "days_evaluated": len(lodo),
            "mean_MAE": per_model,
            "worst_fusion_MAE": float(max(v["metrics"]["fusion"]["MAE"] for v in lodo.values())),
            "note": "No frame from the held-out day is seen in training, and validation "
                    "uses a different held-out day so early stopping is also selected "
                    "across days. This is the number to quote for generalisation to a new "
                    "monitoring session.",
        },
    }
    C.save_json(C.RESULTS / "ablations.json", payload)
    print(f"\nmean leave-one-day-out MAE: " +
          "  ".join(f"{k} {v:.3f}" for k, v in per_model.items()))
    print(f"wrote {C.RESULTS / 'ablations.json'}")


if __name__ == "__main__":
    main()
