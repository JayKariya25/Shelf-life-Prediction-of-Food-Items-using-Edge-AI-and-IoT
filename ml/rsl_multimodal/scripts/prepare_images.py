"""Build `data_images/` - the single 224x224 image folder the pipeline trains on.

`Processed_Images/` already holds 224x224 RGB JPEGs, but it covers only 2058 of the 3316
sensor rows. Another 1009 rows have a frame in `Original_Images/` that was never
preprocessed. Those are put through the identical transform declared in
`image_preprocessing_config.json` (RGB -> LANCZOS 224x224 -> JPEG q95) instead of being
discarded, which raises usable multimodal samples from 2058 to 3067.

    python scripts/prepare_images.py --processed ../Processed_Images --original ../Original_Images
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

import common as C

EXTS = {".jpg", ".jpeg", ".png"}


def listing(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.suffix.lower() in EXTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.ROOT.parent / "Processed_Images")
    ap.add_argument("--original", type=Path, default=C.ROOT.parent / "Original_Images")
    ap.add_argument("--out", type=Path, default=C.IMAGE_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    processed, original = listing(args.processed), listing(args.original)
    done = listing(args.out)

    df = pd.read_csv(C.SENSOR_FEATURES_CSV)
    names = df["Image_Path"].astype(str).map(lambda p: p.replace("\\", "/").split("/")[-1])

    copied = resized = missing = 0
    for name in names:
        if name in done:
            continue
        if name in processed:
            shutil.copyfile(args.processed / name, args.out / name)
            copied += 1
        elif name in original:
            img = Image.open(args.original / name).convert("RGB")
            img.resize(C.IMAGE_SIZE, Image.LANCZOS).save(args.out / name, "JPEG", quality=95)
            resized += 1
        else:
            missing += 1

    print(f"copied from Processed_Images : {copied}")
    print(f"resized from Original_Images : {resized}")
    print(f"sensor rows with no image    : {missing}")
    print(f"total images in {args.out}   : {len(listing(args.out))}")


if __name__ == "__main__":
    main()
