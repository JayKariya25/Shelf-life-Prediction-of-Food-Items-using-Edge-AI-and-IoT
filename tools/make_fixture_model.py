"""Build a tiny stand-in model with the *same contract* as the trained one.

This is a test fixture, not a shelf-life model: its weights are random and its
predictions are meaningless. Its only job is to let the web application's
loading, validation and inference glue be exercised end to end without shipping
the real 20 MB artifact into the repository.

Contract reproduced exactly:
    inputs  : image  (None, 224, 224, 3)   named "image"
              sensor (None, 3)             named "sensor"
    output  : (None, 1) - remaining shelf life in days
    scaler  : StandardScaler fitted on [Temperature, Humidity, Gas]
    config  : {"sensor_columns": [...], "image_size": [224, 224], ...}

Usage:
    .venv-model/bin/python tools/make_fixture_model.py <output-dir>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler

SENSOR_COLUMNS = ["Temperature", "Humidity", "Gas"]


def build_model() -> tf.keras.Model:
    image_in = tf.keras.Input(shape=(224, 224, 3), name="image")
    sensor_in = tf.keras.Input(shape=(3,), name="sensor")

    x = tf.keras.layers.Conv2D(4, 3, strides=4, activation="relu")(image_in)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)

    y = tf.keras.layers.Dense(8, activation="relu")(sensor_in)

    merged = tf.keras.layers.Concatenate()([x, y])
    merged = tf.keras.layers.Dense(8, activation="relu")(merged)
    out = tf.keras.layers.Dense(1, name="remaining_shelf_life")(merged)

    return tf.keras.Model(inputs={"image": image_in, "sensor": sensor_in}, outputs=out)


def main(out_dir: Path) -> None:
    models = out_dir / "models"
    artifacts = out_dir / "artifacts"
    images = out_dir / "data" / "Images"
    for directory in (models, artifacts, images):
        directory.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.save(models / "best_multimodal_model.keras")

    rng = np.random.default_rng(20260826)
    sensor_train = np.column_stack([
        rng.uniform(12, 36, 400),    # Temperature
        rng.uniform(35, 95, 400),    # Humidity
        rng.uniform(100, 320, 400),  # Gas
    ])
    scaler = StandardScaler().fit(sensor_train)
    joblib.dump(scaler, artifacts / "sensor_scaler.pkl")

    (artifacts / "config.json").write_text(
        json.dumps(
            {
                "name": "Fixture multimodal model (random weights)",
                "version": "0.0.0-fixture",
                "sensor_columns": SENSOR_COLUMNS,
                "image_size": [224, 224],
                "metrics": {"test_mae": 0.82, "test_rmse": 1.14, "test_r2": 0.71},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    for index in range(12):
        name = f"fixture_{index:03d}.jpg"
        colour = tuple(int(v) for v in rng.integers(40, 220, 3))
        Image.new("RGB", (320, 240), colour).save(images / name, quality=88)
        rows.append(
            {
                "split": "test" if index % 3 == 0 else "train",
                "Imagepath": f"Images/{name}",
                "resolved_image_path": str((images / name).resolve()),
                "Temperature": round(float(rng.uniform(18, 34)), 2),
                "Humidity": round(float(rng.uniform(45, 90)), 2),
                "Gas": round(float(rng.uniform(120, 300)), 2),
                "RemainingShelfLife": round(float(rng.uniform(0, 9)), 2),
            }
        )

    with (artifacts / "dataset_splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixture artifacts written under {out_dir}")
    print(f"  test rows: {sum(1 for r in rows if r['split'] == 'test')}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
