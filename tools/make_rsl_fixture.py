"""Build a small stand-in for the RSL model bundle, for tests.

The weights are random, so its predictions mean nothing. What it reproduces exactly
is the *contract* of ``ml/rsl_multimodal``:

    models/image_ensemble_blocked_full_float16.tflite   image (1,224,224,3) -> (1,1)
    models/fusion_blocked_full_float16.tflite   image (1,224,224,3) + sensor (1,27) -> (1,1)
    artifacts/sensor_scaler.pkl                 {"scalers": {"blocked__full": ...}, "features": ...}
    artifacts/preprocessing_config.json         the real outlier bounds and input envelope
    artifacts/master_dataset.csv                a few held-out rows with the 27 features
    scripts/pi_features.py, image_io.py         copied from the real bundle
    results/...                                 published metrics with known values
    data_images/                                224x224 JPEGs

Both export shapes are built: the image-only ensemble the website serves by default, and
the image+sensor fusion graph, which the loader still supports and which tests point at
explicitly so that path keeps its coverage.

Copying the real feature builder and image loader means tests exercise the code that
actually ships with the model.

    .venv-model/bin/python tools/make_rsl_fixture.py <output-dir>
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REAL_BUNDLE = Path(__file__).resolve().parent.parent / "ml" / "rsl_multimodal"

FEATURES = [
    "Temperature", "Humidity", "Gas",
    "Temperature_Smoothed", "Humidity_Smoothed", "Gas_Smoothed",
    "Temperature_Change", "Humidity_Change", "Gas_Change",
    "Temperature_Rate", "Humidity_Rate", "Gas_Rate",
    "Temperature_Rolling_Mean", "Humidity_Rolling_Mean", "Gas_Rolling_Mean",
    "Temperature_Rolling_STD", "Humidity_Rolling_STD", "Gas_Rolling_STD",
    "Temperature_Humidity_Interaction", "Temperature_Gas_Interaction",
    "Humidity_Gas_Interaction",
    "Temperature_From_Initial", "Humidity_From_Initial", "Gas_From_Initial",
    "Gas_Increasing", "Temperature_Increasing", "Humidity_Increasing",
]

# Known published figures, so tests can assert on them.
PUBLISHED_MAE = 0.2636
NEW_BATCH_MAE = {"1": 2.0, "2": 0.5}      # fusion MAE per held-out day -> mean 1.25
ENSEMBLE_MAE = {"1": 1.2, "2": 0.4}       # ensemble MAE per held-out day -> mean 0.80
ENSEMBLE_HEADS = 3
ENSEMBLE_PUBLISHED_MAE = 0.1468
# 1-NN-on-untrained-features scores, the model card's leakage evidence.
PROBE_BLOCKED_MAE = 0.0812
PROBE_RANDOM_MAE = 0.0196

BOUNDS = {
    "Temperature": {"lower": 27.455, "upper": 29.335},
    "Humidity": {"lower": 52.55, "upper": 62.63},
    "Gas": {"lower": 171.5, "upper": 191.5},
}
ENVELOPE = {
    "Temperature": {"min": 25.83, "max": 29.31, "source": "raw readings, blocked train split"},
    "Humidity": {"min": 53.07, "max": 67.01, "source": "raw readings, blocked train split"},
    "Gas": {"min": 152.0, "max": 193.0, "source": "raw readings, blocked train split"},
}


def _rsl_output(tf, z, name: str = "rsl"):
    """Small weights around a bias of 5 days keep outputs inside 0-11, so tests reach
    the unclamped path as well as the clamped one."""
    return tf.keras.layers.Dense(
        1, name=name,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=7),
        bias_initializer=tf.keras.initializers.Constant(5.0),
    )(z)


def _convert(model, path: Path) -> None:
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        path.write_bytes(converter.convert())


def build_fusion_tflite(path: Path) -> None:
    """The two-input contract: image (rank 4) + sensor vector (rank 2)."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(7)
    image = tf.keras.Input(shape=(224, 224, 3), name="image")
    sensor = tf.keras.Input(shape=(len(FEATURES),), name="sensor")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(image)
    x = tf.keras.layers.Conv2D(4, 3, strides=8, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    y = tf.keras.layers.Dense(8, activation="relu")(sensor)
    z = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(8, activation="relu")(z)
    model = tf.keras.Model(inputs={"image": image, "sensor": sensor},
                           outputs=_rsl_output(tf, z))
    _convert(model, path)


def build_image_ensemble_tflite(path: Path, heads: int = ENSEMBLE_HEADS) -> None:
    """The served contract: one image input, and a mean over several heads inside the graph.

    Averaging in the graph is the part worth reproducing - it is what makes the export a
    single file with one scalar output despite being an ensemble.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(11)
    image = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(image)
    x = tf.keras.layers.Conv2D(4, 3, strides=8, activation="relu")(x)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
    per_head = [
        _rsl_output(tf, tf.keras.layers.Dense(8, activation="relu", name=f"head_{i}")(pooled),
                    name=f"rsl_{i}")
        for i in range(heads)
    ]
    out = tf.keras.layers.Average(name="remaining_shelf_life")(per_head)
    model = tf.keras.Model(inputs={"image": image}, outputs=out)
    _convert(model, path)


def main(out: Path) -> Path:
    from PIL import Image
    from sklearn.preprocessing import StandardScaler

    for sub in ("models", "artifacts", "scripts", "results/blocked_full", "data_images"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for name in ("pi_features.py", "image_io.py"):
        shutil.copyfile(REAL_BUNDLE / "scripts" / name, out / "scripts" / name)

    build_fusion_tflite(out / "models" / "fusion_blocked_full_float16.tflite")
    build_image_ensemble_tflite(out / "models" / "image_ensemble_blocked_full_float16.tflite")

    rng = np.random.default_rng(20260917)
    real_scaler = REAL_BUNDLE / "artifacts" / "sensor_scaler.pkl"
    if real_scaler.is_file():
        # The real scaler gives realistic z-scores for realistic readings.
        shutil.copyfile(real_scaler, out / "artifacts" / "sensor_scaler.pkl")
    else:
        centre = np.array([28.4, 58.0, 183.0] * 1 + [28.4, 58.0, 183.0] + [0.0] * 6
                          + [28.4, 58.0, 183.0] + [0.05, 0.3, 1.0]
                          + [1647.0, 5197.0, 10614.0] + [0.0] * 3 + [0.5] * 3)
        train = centre + rng.normal(size=(200, len(FEATURES))) * np.maximum(np.abs(centre) * 0.01, 0.1)
        scaler = StandardScaler().fit(train)
        with open(out / "artifacts" / "sensor_scaler.pkl", "wb") as fh:
            pickle.dump({"scalers": {"blocked__full": scaler},
                         "features": {"full": FEATURES, "minimal": FEATURES[:3]}}, fh)

    (out / "artifacts" / "preprocessing_config.json").write_text(json.dumps({
        "image_size": [224, 224],
        "target": {"column": "RemainingShelfLife", "definition": "11 - Day", "unit": "days"},
        "sensor_features": {"full": FEATURES, "minimal": FEATURES[:3]},
        "outlier_bounds": BOUNDS,
        "input_envelope": ENVELOPE,
        "smoothing_window": 5,
    }, indent=2))

    # Realistic feature rows: a synthetic in-envelope stream replayed through the real
    # feature builder, exactly as the training CSV was produced.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fixture_pi_features", out / "scripts" / "pi_features.py")
    pi_features = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pi_features
    spec.loader.exec_module(pi_features)
    builder = pi_features.SensorFeatureBuilder(bounds=BOUNDS)

    rows = []
    for index in range(8):
        name = f"fixture_{index:03d}.jpg"
        colour = tuple(int(v) for v in rng.integers(30, 225, 3))
        Image.new("RGB", (224, 224), colour).save(out / "data_images" / name, "JPEG", quality=95)
        reading = {"Temperature": 28.2 + 0.1 * index, "Humidity": 57.0 + index,
                   "Gas": 180.0 + index}
        values = builder.update(reading, 0.0 if index == 0 else 2 / 60)
        day = 1 + index
        rows.append({
            "image_file": name,
            "Timestamp": f"2026-06-{6 + index:02d} 12:00:00",
            "Day": day,
            "RemainingShelfLife": float(11 - day),
            "split_blocked": "test" if index % 2 == 0 else "train",
            "split_random": "test",
            **values,
        })
    with open(out / "artifacts" / "master_dataset.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {"MAE": PUBLISHED_MAE, "RMSE": 0.6, "R2": 0.87, "n": 271}
    (out / "results" / "blocked_full" / "fusion_metrics.json").write_text(
        json.dumps({"metrics": {"test": {**metrics, "MAE": 0.2662}}}))
    (out / "results" / "blocked_full" / "tflite_report.json").write_text(json.dumps({
        "variants": {"float16": {"file": "fusion_blocked_full_float16.tflite",
                                 "size_mb": 2.34, "latency_ms_per_sample_x86_cpu": 1.9,
                                 "metrics": metrics}}}))
    (out / "results" / "ablations.json").write_text(json.dumps({
        "leave_one_day_out": {
            day: {"true_RSL": 11 - int(day), "held_out_samples": 10,
                  "metrics": {"fusion": {"MAE": mae}, "image": {"MAE": mae / 2},
                              "sensor": {"MAE": mae * 2}}}
            for day, mae in NEW_BATCH_MAE.items()
        },
        # The leakage evidence the model card shows: a 1-NN lookup on untrained
        # features. The real bundle's blocked figure is 0.081; these are stand-ins
        # with the same shape.
        "nearest_neighbour_probe": {
            "blocked": {
                "frozen_imagenet_features": {"metrics": {"MAE": PROBE_BLOCKED_MAE}},
                "scaled_sensor_vector": {"metrics": {"MAE": 0.5166}},
                "minutes_to_nearest_train_row": {"median": 96.0},
            },
            "random": {
                "frozen_imagenet_features": {"metrics": {"MAE": PROBE_RANDOM_MAE}},
                "scaled_sensor_vector": {"metrics": {"MAE": 0.1155}},
                "minutes_to_nearest_train_row": {"median": 2.0},
            },
        }}))

    # The multi-seed variant study behind the choice of served model.
    (out / "results" / "experiments.json").write_text(json.dumps({
        "baseline": {
            "seeds": [42, 7, 2026],
            "mean_MAE": {
                "sensor": {"mean": 2.3921, "std": 0.1064, "runs": [2.46, 2.41, 2.20]},
                "image": {"mean": 0.9199, "std": 0.086, "runs": [0.81, 1.04, 0.88]},
                "fusion": {"mean": 1.1128, "std": 0.179, "runs": [1.24, 0.94, 0.95]},
            },
            "ensemble_of_seeds_MAE": {"image": 0.8558},
        },
        "sensor_dropout": {
            "seeds": [42, 7, 2026],
            "mean_MAE": {
                "sensor": {"mean": 2.3582, "std": 0.1133, "runs": [2.46, 2.41, 2.20]},
                "image": {"mean": 0.9102, "std": 0.0981, "runs": [0.81, 1.04, 0.88]},
                "fusion": {"mean": 1.7053, "std": 0.1855, "runs": [1.71, 1.93, 1.47]},
            },
            "ensemble_of_seeds_MAE": {},
        },
    }))

    # Branch-by-branch within-session scores, which the model card tabulates.
    (out / "results" / "blocked_full" / "comparison_metrics.json").write_text(json.dumps({
        "split": "blocked", "feature_set": "full", "n_test": 271,
        "graph_consistency_max_abs_diff": 0.0,
        "test_metrics": {
            "Sensor only": {"MAE": 0.7535, "RMSE": 1.3876, "R2": 0.4431,
                            "WithinHalfDay": 0.4502, "WithinOneDay": 0.8339, "n": 271},
            "Image only": {"MAE": 0.2424, "RMSE": 0.4371, "R2": 0.9448,
                           "WithinHalfDay": 0.8708, "WithinOneDay": 0.9631, "n": 271},
            "Fusion": {"MAE": 0.2662, "RMSE": 0.6694, "R2": 0.8704,
                       "WithinHalfDay": 0.9230, "WithinOneDay": 0.9700, "n": 271},
        },
    }))

    ensemble_metrics = {"MAE": ENSEMBLE_PUBLISHED_MAE, "RMSE": 0.39, "R2": 0.955,
                        "WithinHalfDay": 0.963, "WithinOneDay": 0.974, "n": 271}
    (out / "results" / "blocked_full" / "image_ensemble_metrics.json").write_text(json.dumps({
        "heads": ENSEMBLE_HEADS, "metrics": {"test": ensemble_metrics}}))
    (out / "results" / "blocked_full" / "image_ensemble_tflite_report.json").write_text(
        json.dumps({"variants": {
            "float16": {"file": "image_ensemble_blocked_full_float16.tflite",
                        "size_mb": 4.08, "latency_ms_per_sample_x86_cpu": 1.8,
                        "metrics": ensemble_metrics}}}))
    (out / "results" / "image_ensemble_lodo.json").write_text(json.dumps({
        "heads": ENSEMBLE_HEADS,
        "mean_MAE": {"image_ensemble": sum(ENSEMBLE_MAE.values()) / len(ENSEMBLE_MAE),
                     "single_head_average": 1.009},
        "folds": {
            day: {"true_RSL": 11 - int(day), "held_out_samples": 10,
                  "metrics": {"image_ensemble": {"MAE": mae}}}
            for day, mae in ENSEMBLE_MAE.items()
        }}))
    return out


if __name__ == "__main__":
    target = main(Path(sys.argv[1] if len(sys.argv) > 1 else "rsl_fixture"))
    print(f"fixture bundle written to {target}")
