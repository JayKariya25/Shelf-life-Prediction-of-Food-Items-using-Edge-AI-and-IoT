"""Model definitions for the three stages: sensor branch, image branch, fusion head."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

import common as C


# --------------------------------------------------------------------------------------
# Stage 1a - sensor branch
# --------------------------------------------------------------------------------------
def build_sensor_branch(n_features: int, name: str = "sensor_branch") -> tf.keras.Model:
    """MLP over the scaled sensor features. Penultimate layer is the fusion embedding."""
    inp = layers.Input(shape=(n_features,), name="sensor")
    x = layers.Dense(128, activation="relu", name="sensor_dense_1")(inp)
    x = layers.BatchNormalization(name="sensor_bn_1")(x)
    x = layers.Dropout(0.20, name="sensor_drop_1")(x)
    x = layers.Dense(96, activation="relu", name="sensor_dense_2")(x)
    x = layers.Dropout(0.15, name="sensor_drop_2")(x)
    emb = layers.Dense(C.SENSOR_EMBED_DIM, activation="relu", name="sensor_embedding")(x)
    out = layers.Dense(1, name="sensor_rsl")(emb)
    return tf.keras.Model(inp, out, name=name)


# --------------------------------------------------------------------------------------
# Stage 1b - image branch
# --------------------------------------------------------------------------------------
def build_image_head(backbone_dim: int, name: str = "image_head") -> tf.keras.Model:
    """Trainable head that sits on the frozen backbone's pooled 576-d output."""
    inp = layers.Input(shape=(backbone_dim,), name="backbone_features")
    x = layers.Dense(256, activation="relu", name="image_dense_1")(inp)
    x = layers.BatchNormalization(name="image_bn_1")(x)
    x = layers.Dropout(0.30, name="image_drop_1")(x)
    emb = layers.Dense(C.IMAGE_EMBED_DIM, activation="relu", name="image_embedding")(x)
    out = layers.Dense(1, name="image_rsl")(emb)
    return tf.keras.Model(inp, out, name=name)


def assemble_image_branch(backbone: tf.keras.Model, head: tf.keras.Model,
                          name: str = "image_branch") -> tf.keras.Model:
    """Glue the frozen backbone and the trained head into one end-to-end image model."""
    backbone.trainable = False
    inp = layers.Input(shape=(*C.IMAGE_SIZE, 3), name="image")
    out = head(backbone(inp, training=False))
    return tf.keras.Model(inp, out, name=name)


# --------------------------------------------------------------------------------------
# Stage 2 - fusion
# --------------------------------------------------------------------------------------
FUSION_INPUT_DIM = C.IMAGE_EMBED_DIM + C.SENSOR_EMBED_DIM + 2


def build_fusion_head(name: str = "fusion_head") -> tf.keras.Model:
    """Trained on [image 128 | sensor 64 | image RSL | sensor RSL].

    Appending each frozen branch's own scalar prediction matters: it gives the head a
    ready-made baseline to lean on, so fusing can only refine the better branch rather
    than having to relearn it from embeddings and risk landing below it.
    """
    inp = layers.Input(shape=(FUSION_INPUT_DIM,), name="fused_embedding")
    x = layers.Dense(64, activation="relu", name="fusion_dense_1")(inp)
    x = layers.BatchNormalization(name="fusion_bn_1")(x)
    x = layers.Dropout(0.20, name="fusion_drop_1")(x)
    x = layers.Dense(32, activation="relu", name="fusion_dense_2")(x)
    out = layers.Dense(1, name="remaining_shelf_life")(x)
    return tf.keras.Model(inp, out, name=name)


def assemble_multimodal(backbone: tf.keras.Model,
                        image_head: tf.keras.Model,
                        sensor_branch: tf.keras.Model,
                        fusion_head: tf.keras.Model,
                        n_sensor_features: int,
                        name: str = "multimodal_rsl") -> tf.keras.Model:
    """Full deployable graph: raw image + scaled sensor vector -> RSL in days.

    Both branches are frozen; only `fusion_head` was trained on top of them. Taking the
    embeddings from named intermediate layers keeps this graph numerically identical to
    the staged training pipeline.
    """
    backbone.trainable = False
    image_head.trainable = False
    sensor_branch.trainable = False

    img_in = layers.Input(shape=(*C.IMAGE_SIZE, 3), name="image")
    sen_in = layers.Input(shape=(n_sensor_features,), name="sensor")

    image_embedder = tf.keras.Model(
        image_head.input,
        [image_head.get_layer("image_embedding").output, image_head.output],
        name="image_embedder",
    )
    sensor_embedder = tf.keras.Model(
        sensor_branch.input,
        [sensor_branch.get_layer("sensor_embedding").output, sensor_branch.output],
        name="sensor_embedder",
    )

    img_emb, img_pred = image_embedder(backbone(img_in, training=False), training=False)
    sen_emb, sen_pred = sensor_embedder(sen_in, training=False)
    fused = layers.Concatenate(name="concat_embeddings")(
        [img_emb, sen_emb, img_pred, sen_pred]
    )
    out = fusion_head(fused, training=False)
    return tf.keras.Model([img_in, sen_in], out, name=name)


def compile_regressor(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def default_callbacks(patience: int = 12) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=patience, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=5, min_lr=1e-5, mode="min"
        ),
    ]
