"""Measure changes against the only honest yardstick: a day the model never saw.

The blocked test split cannot answer "is this model better?" - a 1-nearest-neighbour
lookup on *untrained* ImageNet features scores 0.081 MAE on it, so most of that number
is duplicate retrieval. Leave-one-day-out is the hardest question this dataset supports,
so every variant here is scored by it.

Each fold holds out one whole day for test and a *different* whole day for validation,
and fits the sensor scaler on that fold's training rows only. Per-fold predictions are
saved, so post-hoc combinations (blends, gates) can be scored without retraining.

    python scripts/experiments.py                 # every variant
    python scripts/experiments.py --only baseline augmented
    python scripts/experiments.py --list

Results: results/experiments.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import common as C
import models as M

MIN_SAMPLES_PER_DAY = 100
EPOCHS = 120
PATIENCE = 8
BATCH = 64

AUGMENTED_CACHE = C.ARTIFACTS / "backbone_features_augmented.npy"
# Colour is the signal for ripeness - a banana browns as it ages - so the augmentation
# deliberately leaves hue alone and keeps brightness/contrast mild. Geometry and framing
# are nuisance variables and get the real jitter.
AUGMENT_COPIES = 6


# --------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------
def embed(model, layer, A):
    return tf.keras.Model(model.input, model.get_layer(layer).output).predict(A, verbose=0)


def fold_days(df: pd.DataFrame) -> list[int]:
    return sorted(int(d) for d, n in df["Day"].value_counts().items() if n >= MIN_SAMPLES_PER_DAY)


def load_inputs():
    df = C.load_master()
    raw_X = df[C.feature_columns("full")].to_numpy("float64")
    F = np.load(C.ARTIFACTS / "backbone_features.npy")
    y = df[C.TARGET].to_numpy("float32")
    return df, raw_X, F, y


def train_branches(X, F, y, tr, va, *, sensor_dropout: float = 0.0,
                   image_features: np.ndarray | None = None,
                   augment_index: np.ndarray | None = None, seed: int = C.SEED):
    """Fit the sensor branch, image head and fusion head for one fold.

    ``image_features``/``augment_index`` let the image head train on an expanded,
    augmented copy of the training rows while everything else stays row-aligned.
    """
    C.set_seeds(seed)
    sensor = M.compile_regressor(M.build_sensor_branch(X.shape[1]))
    sensor.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=EPOCHS,
               batch_size=BATCH, callbacks=M.default_callbacks(PATIENCE), verbose=0)

    image = M.compile_regressor(M.build_image_head(F.shape[1]))
    if image_features is None:
        image.fit(F[tr], y[tr], validation_data=(F[va], y[va]), epochs=EPOCHS,
                  batch_size=BATCH, callbacks=M.default_callbacks(PATIENCE), verbose=0)
    else:
        # augment_index[i] is the original row each augmented row came from
        keep = tr[augment_index]
        image.fit(image_features[keep], y[augment_index][keep],
                  validation_data=(F[va], y[va]), epochs=EPOCHS, batch_size=BATCH,
                  callbacks=M.default_callbacks(PATIENCE), verbose=0)

    E = np.concatenate([embed(image, "image_embedding", F),
                        embed(sensor, "sensor_embedding", X),
                        image.predict(F, verbose=0),
                        sensor.predict(X, verbose=0)], axis=1)

    C.set_seeds(seed)
    fusion = M.compile_regressor(build_fusion_head(sensor_dropout))
    fusion.fit(E[tr], y[tr], validation_data=(E[va], y[va]), epochs=EPOCHS,
               batch_size=BATCH, callbacks=M.default_callbacks(PATIENCE), verbose=0)
    return sensor, image, fusion, E


def build_fusion_head(sensor_dropout: float):
    """The stock fusion head, optionally with the sensor half dropped during training.

    The sensor branch learns the ambient conditions of the one run it saw, and the fusion
    head then leans on it - which is why fusion is worse than the image branch alone on an
    unseen day. Dropping the sensor slice at train time forces the head to stay useful
    without it.
    """
    if sensor_dropout <= 0:
        return M.build_fusion_head()

    from tensorflow.keras import layers

    inp = layers.Input(shape=(M.FUSION_INPUT_DIM,), name="fused_embedding")
    image_part = layers.Lambda(
        lambda t: t[:, : C.IMAGE_EMBED_DIM], name="image_slice",
        output_shape=(C.IMAGE_EMBED_DIM,))(inp)
    sensor_part = layers.Lambda(
        lambda t: t[:, C.IMAGE_EMBED_DIM:], name="sensor_slice",
        output_shape=(M.FUSION_INPUT_DIM - C.IMAGE_EMBED_DIM,))(inp)
    # Drops the whole sensor contribution on a fraction of training steps.
    sensor_part = layers.Dropout(sensor_dropout, noise_shape=(None, 1),
                                 name="sensor_branch_dropout")(sensor_part)
    x = layers.Concatenate(name="fused_again")([image_part, sensor_part])
    x = layers.Dense(64, activation="relu", name="fusion_dense_1")(x)
    x = layers.BatchNormalization(name="fusion_bn_1")(x)
    x = layers.Dropout(0.20, name="fusion_drop_1")(x)
    x = layers.Dense(32, activation="relu", name="fusion_dense_2")(x)
    out = layers.Dense(1, name="remaining_shelf_life")(x)
    return tf.keras.Model(inp, out, name="fusion_head_sensor_dropout")


# --------------------------------------------------------------------------------------
# augmented feature cache
# --------------------------------------------------------------------------------------
def build_augmented_cache(copies: int = AUGMENT_COPIES) -> tuple[np.ndarray, np.ndarray]:
    """Backbone features for `copies` augmented versions of every image.

    The backbone stays frozen, so augmentation only has to pass forward once per copy and
    the result can be cached - keeping head training as fast as the plain cache while
    giving it many more views of each frame.
    """
    from extract_image_features import build_backbone

    df = C.load_master()
    files = [str(C.IMAGE_DIR / f) for f in df["image_file"]]
    index_path = AUGMENTED_CACHE.with_suffix(".index.npy")
    if AUGMENTED_CACHE.exists() and index_path.exists():
        feats, index = np.load(AUGMENTED_CACHE), np.load(index_path)
        if len(index) == len(files) * copies:
            print(f"  reusing {AUGMENTED_CACHE.name}  shape={feats.shape}")
            return feats, index

    backbone = build_backbone()
    backbone.trainable = False

    def augment(img, seed):
        img = tf.image.stateless_random_flip_left_right(img, seed)
        img = tf.image.stateless_random_flip_up_down(img, seed + 1)
        # framing / distance
        scale = tf.random.stateless_uniform([], seed + 2, 0.82, 1.0)
        size = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * scale, tf.int32)
        img = tf.image.stateless_random_crop(img, [size, size, 3], seed + 3)
        img = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear")
        # lighting only: no hue shift, because colour carries the ripeness signal
        img = tf.image.stateless_random_brightness(img, 12.0, seed + 4)
        img = tf.image.stateless_random_contrast(img, 0.92, 1.08, seed + 5)
        return tf.clip_by_value(img, 0.0, 255.0)

    all_feats, index = [], []
    for copy in range(copies):
        def _load(path, i, copy=copy):
            raw = tf.io.read_file(path)
            img = tf.cast(tf.io.decode_jpeg(raw, channels=3), tf.float32)
            img = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear")
            if copy > 0:  # copy 0 is the untouched frame
                img = augment(img, tf.stack([tf.cast(i, tf.int32) + copy * 100003, copy + 7]))
            return img

        ds = (tf.data.Dataset.from_tensor_slices((files, np.arange(len(files))))
              .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(64).prefetch(tf.data.AUTOTUNE))
        feats = backbone.predict(ds, verbose=0)
        all_feats.append(np.asarray(feats, dtype="float32"))
        index.append(np.arange(len(files)))
        print(f"  copy {copy + 1}/{copies} done")

    feats = np.concatenate(all_feats)
    index = np.concatenate(index)
    np.save(AUGMENTED_CACHE, feats)
    np.save(index_path, index)
    print(f"  wrote {AUGMENTED_CACHE.name}  shape={feats.shape}")
    return feats, index


# --------------------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------------------
def run_lodo(name: str, *, sensor_dropout: float = 0.0, augmented: bool = False,
             feature_subset: list[str] | None = None, seed: int = C.SEED) -> dict:
    df, raw_X, F, y = load_inputs()
    if feature_subset is not None:
        columns = C.feature_columns("full")
        keep = [columns.index(c) for c in feature_subset]
        raw_X = raw_X[:, keep]

    aug_feats = aug_index = None
    if augmented:
        aug_feats, aug_index = build_augmented_cache()

    days = fold_days(df)
    folds, started = {}, time.perf_counter()
    for i, day in enumerate(days):
        val_day = days[(i + 1) % len(days)]
        te = (df["Day"] == day).to_numpy()
        va = (df["Day"] == val_day).to_numpy()
        tr = ~(te | va)

        scaler = StandardScaler().fit(raw_X[tr])
        X = scaler.transform(raw_X).astype("float32")

        sensor, image, fusion, E = train_branches(
            X, F, y, tr, va, sensor_dropout=sensor_dropout,
            image_features=aug_feats, augment_index=aug_index, seed=seed)

        preds = {
            "sensor": sensor.predict(X[te], verbose=0).ravel().astype(float),
            "image": image.predict(F[te], verbose=0).ravel().astype(float),
            "fusion": fusion.predict(E[te], verbose=0).ravel().astype(float),
        }
        folds[str(day)] = {
            "held_out_samples": int(te.sum()),
            "validation_day": val_day,
            "true_RSL": float(y[te][0]),
            "metrics": {k: C.regression_metrics(y[te], v) for k, v in preds.items()},
            "predictions": {k: [round(p, 4) for p in v] for k, v in preds.items()},
        }
        print(f"  day {day:>2}  true {y[te][0]:>4.1f}  "
              + "  ".join(f"{k} {folds[str(day)]['metrics'][k]['MAE']:.3f}" for k in preds))

    summary = {
        m: float(np.mean([f["metrics"][m]["MAE"] for f in folds.values()]))
        for m in ("sensor", "image", "fusion")
    }
    print(f"  {name}: mean LODO MAE  " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items())
          + f"   ({time.perf_counter() - started:.0f}s)")
    return {"folds": folds, "mean_MAE": summary}


def blend_sweep(result: dict) -> dict:
    """Score w*image + (1-w)*fusion post-hoc, per fold, without retraining."""
    best, table = None, {}
    for w in [round(x / 10, 1) for x in range(11)]:
        maes = []
        for fold in result["folds"].values():
            image = np.array(fold["predictions"]["image"])
            fusion = np.array(fold["predictions"]["fusion"])
            blended = w * image + (1 - w) * fusion
            maes.append(float(np.abs(blended - fold["true_RSL"]).mean()))
        mean = float(np.mean(maes))
        table[str(w)] = round(mean, 4)
        if best is None or mean < best[1]:
            best = (w, mean)
    return {"table": table, "best_weight": best[0], "best_mean_MAE": round(best[1], 4)}


DYNAMICS_ONLY = [
    "Temperature_Change", "Humidity_Change", "Gas_Change",
    "Temperature_Rate", "Humidity_Rate", "Gas_Rate",
    "Temperature_Rolling_STD", "Humidity_Rolling_STD", "Gas_Rolling_STD",
    "Temperature_From_Initial", "Humidity_From_Initial", "Gas_From_Initial",
    "Gas_Increasing", "Temperature_Increasing", "Humidity_Increasing",
]

VARIANTS = {
    "baseline": dict(),
    "sensor_dropout": dict(sensor_dropout=0.5),
    "augmented": dict(augmented=True),
    "augmented_sensor_dropout": dict(augmented=True, sensor_dropout=0.5),
    "dynamics_only_sensors": dict(feature_subset=DYNAMICS_ONLY),
}


def run_variant(name: str, kwargs: dict, seeds: list[int]) -> dict:
    """Run a variant once per seed. Head training is stochastic and the differences
    between variants are small, so a single run cannot tell a change from noise."""
    runs = []
    for seed in seeds:
        print(f"  -- seed {seed}")
        runs.append(run_lodo(name, seed=seed, **kwargs))
    summary = {}
    for model in ("sensor", "image", "fusion"):
        values = [r["mean_MAE"][model] for r in runs]
        summary[model] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "runs": [round(v, 4) for v in values],
        }
    blends = [blend_sweep(r) for r in runs]
    best_w = [b["best_weight"] for b in blends]

    # Averaging the seeds' predictions per fold: the cheapest variance reduction there
    # is, and deployable as one graph that averages K heads.
    ensemble = {}
    for model in ("sensor", "image", "fusion"):
        maes = []
        for day in runs[0]["folds"]:
            stacked = np.array([r["folds"][day]["predictions"][model] for r in runs])
            truth = runs[0]["folds"][day]["true_RSL"]
            maes.append(float(np.abs(stacked.mean(axis=0) - truth).mean()))
        ensemble[model] = round(float(np.mean(maes)), 4)

    return {
        "seeds": seeds,
        "mean_MAE": summary,
        "ensemble_of_seeds_MAE": ensemble,
        "blend_best_weights": best_w,
        "blend_best_MAE": round(float(np.mean([b["best_mean_MAE"] for b in blends])), 4),
        "last_run": runs[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="variant names to run")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 2026])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, kwargs in VARIANTS.items():
            print(f"  {name:<26} {kwargs or '(stock pipeline)'}")
        return

    out_path = C.RESULTS / "experiments.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in (args.only or list(VARIANTS)):
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; try --list")
        print(f"\n== {name} ({len(args.seeds)} seeds) ==")
        results[name] = run_variant(name, VARIANTS[name], args.seeds)
        C.save_json(out_path, results)

    print(f"\n{'variant':<26}{'sensor':>16}{'image':>16}{'fusion':>16}{'ens.image':>11}")
    for name, r in results.items():
        if not isinstance(r.get("mean_MAE", {}).get("image"), dict):
            continue
        cells = "".join(f"{r['mean_MAE'][m]['mean']:>10.3f} ±{r['mean_MAE'][m]['std']:<5.3f}"
                        for m in ("sensor", "image", "fusion"))
        ens = r.get("ensemble_of_seeds_MAE", {}).get("image")
        tail = f"{ens:>11.3f}" if ens is not None else f"{'n/a':>11}"
        print(f"{name:<26}{cells}{tail}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
