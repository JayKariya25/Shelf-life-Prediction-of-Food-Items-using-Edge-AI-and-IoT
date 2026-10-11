"""Turn any uploaded photo into exactly the tensor the model was trained on.

Why this module exists
----------------------
Training images went through two steps, and serving has to repeat both:

1. **Preparation** (`prepare_images.py` / the image preprocessing notebook):
   RGB -> LANCZOS resize to 224x224 -> JPEG quality 95.
2. **Decoding** (`extract_image_features.py`, `export_tflite.py`):
   `tf.io.decode_jpeg` with default arguments, which uses libjpeg's *fast integer*
   IDCT with fancy upsampling.

The first version of `serve.py` and `predict_pi.py` decoded with PIL (accurate IDCT)
and resized full-resolution frames with BILINEAR. Both are silent train/serve skews,
measured on the held-out blocked split:

    decoder only, 224x224 test images   MAE 0.264 (TF)   -> 0.278 (PIL)
    resize, 2592x1944 camera originals  MAE 0.152        -> 0.207 (BILINEAR, PIL)

`load_image_bytes` rebuilds a stored training image from its full-resolution original
bit for bit, and `verify_serving_transform.py` checks that as a pipeline gate.

Decoder selection
-----------------
    tensorflow   exact - used on a development machine or server
    simplejpeg   exact - fastdct=True, fastupsample=False; light enough for a Pi
    PIL          fallback only - ~1.2 grey levels off; the result says so

Orientation: training never applied EXIF rotation, so neither does this.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 95


@dataclass
class LoadedImage:
    pixels: np.ndarray                 # (1, 224, 224, 3) float32, raw 0-255
    decoder: str                       # "tensorflow" | "simplejpeg" | "pil"
    exact: bool                        # False only for the PIL fallback
    source_size: tuple[int, int]
    resized: bool
    notes: list[str] = field(default_factory=list)


def _decoder():
    """Pick the most faithful JPEG decoder that is importable."""
    try:
        import tensorflow as tf  # noqa: F401

        def decode(data: bytes) -> np.ndarray:
            return tf.io.decode_jpeg(data, channels=3).numpy()

        return "tensorflow", True, decode
    except ImportError:
        pass
    try:
        import simplejpeg

        def decode(data: bytes) -> np.ndarray:
            return simplejpeg.decode_jpeg(
                data, colorspace="RGB", fastdct=True, fastupsample=False
            )

        return "simplejpeg", True, decode
    except ImportError:
        pass

    from PIL import Image

    def decode(data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"))

    return "pil", False, decode


_DECODER = None


def decoder():
    global _DECODER
    if _DECODER is None:
        _DECODER = _decoder()
    return _DECODER


def load_image_bytes(data: bytes) -> LoadedImage:
    """Decode an uploaded image the way the training pipeline did."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ValueError("empty image")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, mode, size = probe.format, probe.mode, probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"could not read that image: {exc}") from exc

    name, exact, decode = decoder()
    notes: list[str] = []

    # A 224x224 RGB JPEG is already in training form: decode it directly. Re-encoding
    # it would change the pixels.
    if fmt == "JPEG" and mode == "RGB" and size == IMAGE_SIZE:
        payload, resized = data, False
    else:
        with Image.open(io.BytesIO(data)) as im:
            rgb = im.convert("RGB")
            if rgb.size != IMAGE_SIZE:
                rgb = rgb.resize(IMAGE_SIZE, Image.LANCZOS)
            buffer = io.BytesIO()
            rgb.save(buffer, "JPEG", quality=JPEG_QUALITY)
        payload, resized = buffer.getvalue(), size != IMAGE_SIZE
        if fmt != "JPEG":
            notes.append(f"{fmt} input was re-encoded as JPEG q{JPEG_QUALITY}, as training images were")

    pixels = decode(payload).astype("float32")
    if pixels.shape != (*IMAGE_SIZE, 3):
        raise ValueError(f"decoded image has shape {pixels.shape}, expected {(*IMAGE_SIZE, 3)}")
    if not exact:
        notes.append(
            "decoded with PIL: install simplejpeg (or tensorflow) for pixels that match "
            "training exactly - PIL costs about 0.014 days of MAE"
        )
    return LoadedImage(pixels[None, ...], name, exact, size, resized, notes)


def load_image(path: str | Path) -> LoadedImage:
    return load_image_bytes(Path(path).read_bytes())
