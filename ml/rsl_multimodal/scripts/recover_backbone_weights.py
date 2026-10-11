"""Recover MobileNetV3Small ImageNet weights from the existing DEMO_MODEL checkpoint.

Only needed on machines that cannot reach the Keras weights CDN
(storage.googleapis.com). The demo model embeds the full MobileNetV3Small stack inline
with its ImageNet weights intact, so every weighted backbone layer can be copied out by
name. On a machine with normal internet access this script is unnecessary -
`weights="imagenet"` just works.

Usage:  python scripts/recover_backbone_weights.py path/to/best_multimodal_model.keras
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

import common as C

OUT = C.ARTIFACTS / "mobilenetv3s_backbone.weights.h5"


def main() -> None:
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else C.ARTIFACTS / "best_multimodal_model.keras"
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    donor = tf.keras.models.load_model(src_path, compile=False)
    donor_layers = {l.name: l for l in donor.layers if l.weights}

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(*C.IMAGE_SIZE, 3),
        include_top=False,
        weights=None,
        pooling="avg",
        include_preprocessing=True,
    )

    copied, skipped = 0, []
    for layer in backbone.layers:
        if not layer.weights:
            continue
        donor_layer = donor_layers.get(layer.name)
        if donor_layer is None:
            skipped.append(layer.name)
            continue
        sw, tw = donor_layer.get_weights(), layer.get_weights()
        if len(sw) != len(tw) or any(a.shape != b.shape for a, b in zip(sw, tw)):
            skipped.append(layer.name)
            continue
        layer.set_weights(sw)
        copied += 1

    if skipped:
        raise RuntimeError(f"could not recover {len(skipped)} layers: {skipped[:10]}")

    backbone.save_weights(OUT)
    print(f"recovered {copied} weighted layers -> {OUT}")


if __name__ == "__main__":
    main()
