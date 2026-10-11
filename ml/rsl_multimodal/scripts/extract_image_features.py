"""Run the frozen MobileNetV3Small backbone once over every image and cache the result.

The image branch keeps the backbone frozen, so its output never changes during training.
Caching it turns image-model training from hours of CPU convolutions into seconds, and the
cached matrix is reused unchanged by the fusion stage.

Output: artifacts/backbone_features.npy, row-aligned with artifacts/master_dataset.csv.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import common as C

BATCH = 64
CACHE = C.ARTIFACTS / "backbone_features.npy"


LOCAL_BACKBONE = C.ARTIFACTS / "mobilenetv3s_backbone.weights.h5"


def build_backbone() -> tf.keras.Model:
    """MobileNetV3Small, ImageNet weights, global-average-pooled -> 576-d.

    `include_preprocessing=True` keeps the [0,255] -> network scaling inside the graph, so
    the exported TFLite model takes raw camera pixels on the Pi and no preprocessing step
    can drift out of sync between training and deployment.

    Weights are downloaded from the Keras CDN when it is reachable; the offline fallback
    is a local copy (see scripts/recover_backbone_weights.py).
    """
    kwargs = dict(
        input_shape=(*C.IMAGE_SIZE, 3),
        include_top=False,
        pooling="avg",
        include_preprocessing=True,
    )
    try:
        return tf.keras.applications.MobileNetV3Small(weights="imagenet", **kwargs)
    except Exception as exc:  # no network access to the Keras weight CDN
        if not LOCAL_BACKBONE.exists():
            raise RuntimeError(
                "ImageNet weights unavailable and no local copy at "
                f"{LOCAL_BACKBONE} - run scripts/recover_backbone_weights.py"
            ) from exc
        print(f"[backbone] CDN unreachable, loading local weights: {LOCAL_BACKBONE}")
        model = tf.keras.applications.MobileNetV3Small(weights=None, **kwargs)
        model.load_weights(LOCAL_BACKBONE)
        return model


def image_dataset(files: list[str]) -> tf.data.Dataset:
    paths = [str(C.IMAGE_DIR / f) for f in files]

    def _load(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, C.IMAGE_SIZE, method="bilinear")
        return tf.cast(img, tf.float32)

    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )


def main() -> None:
    C.set_seeds()
    df = C.load_master()
    files = df["image_file"].tolist()

    backbone = build_backbone()
    backbone.trainable = False
    print(f"backbone output dim: {backbone.output_shape[-1]}  images: {len(files)}")

    feats = backbone.predict(image_dataset(files), verbose=1)
    feats = np.asarray(feats, dtype="float32")
    np.save(CACHE, feats)
    print(f"wrote {CACHE}  shape={feats.shape}")


if __name__ == "__main__":
    main()
