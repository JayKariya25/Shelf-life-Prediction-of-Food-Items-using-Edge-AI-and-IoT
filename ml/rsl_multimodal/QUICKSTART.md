# Quickstart

Everything below runs from this folder:

```bash
cd ~/Downloads/"CAPSTONE DATASET"/RSL_Multimodal_Model
```

---

## 0. Just look at the results (no setup)

Nothing to install. Open these:

| File | What it is |
|---|---|
| `REPORT.md` | The write-up: data, splits, results, deployment, limitations |
| `results/summary.png` | The two headline charts |
| `results/blocked_full/actual_vs_predicted.png` | Predicted vs actual, all three models |
| `results/ablations.json` | Leave-one-day-out and the nearest-neighbour probe |
| `results/blocked_full/tflite_report.json` | Size / accuracy / latency of each TFLite export |

The trained models are already in `models/` — you do **not** have to retrain to demo this.

---

## 1. Set up Python (once, ~3 minutes)

You need Python 3.10–3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check it worked:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

> On an Apple Silicon Mac the plain `tensorflow` package is correct — `requirements.txt`
> already picks it for you. Don't install `tensorflow-cpu` there; it has no arm64 build.

---

## 2. Run one prediction (~20 seconds)

The fastest way to see it work end to end. Pick any image and give it three sensor readings:

```bash
python scripts/predict_pi.py \
  --image data_images/20260606_200106.jpg \
  --temperature 28.74 --humidity 58.04 --gas 185 \
  --reset-session
```

```
Remaining shelf life: 9.59 days
```

This is the exact code path a Raspberry Pi runs: it rebuilds the 27 engineered sensor
features from the live reading, decodes the image exactly as training did
(`scripts/image_io.py`), scales the features with the training scaler, and runs the
float16 TFLite model. Add `--json` for machine-readable output. The true value for this
frame is 10.0 days.

Readings outside what the model was trained on (25.8-29.3 C, 53-67 % RH, gas 152-193)
print an `UNRELIABLE` note - the model is extrapolating there - and the output is clamped
to 0-11 days.

`--reset-session` clears the rolling history. Leave it off for the 2nd reading onward so
the rolling/rate features build up properly — that is how a real deployment behaves.
History is kept in `runtime/pi_state.json`; pass `--state path.json` to keep one file per
food batch.

To check a prediction against ground truth:

```bash
grep 20260606_200106.jpg artifacts/master_dataset.csv | cut -d, -f1-4
# image_file, Timestamp, Day, RemainingShelfLife
```

---

## 3. Verify the pipeline is sound (~1 minute)

This is the check worth running before you trust anything:

```bash
python scripts/verify_pi_features.py
```

It replays all 3316 real sensor readings one at a time through the on-device feature code
and compares against the training CSV. Expect `PASS`. If it ever says `FAIL`, the Pi is
being fed different features than the model was trained on.

Then check the image side the same way:

```bash
python scripts/verify_serving_transform.py
```

It rebuilds training images from the full-resolution originals and re-scores the test set
through the serving loader. Expect `60/60 bit-identical`, a test MAE of `0.2636`, and `PASS`.

---

## 4. Open the notebooks (~2 minutes)

```bash
python -m pip install jupyter
jupyter notebook
```

Then open, in order:

1. `notebooks/01_sensor_model.ipynb` — sensor branch, and which columns are excluded and why
2. `notebooks/02_image_model.ipynb` — image branch, sample frames, worst errors
3. `notebooks/03_fusion_and_deployment.ipynb` — fusion, TFLite, generalisation

They already contain all their outputs, so you can read them without running anything.
"Restart & Run All" re-executes them (~3 minutes total).

---

## 5. Retrain everything from scratch (~40 minutes)

```bash
bash run_all.sh
```

Stages, in order:

| # | Stage | Time |
|---|---|---|
| 1 | `prepare_data.py` — target, splits, scalers | 5 s |
| 2 | `extract_image_features.py` — frozen backbone cache | 30 s |
| 3–6 | train sensor / image / fusion + evaluate, per split | 2 min |
| 7 | `export_tflite.py` — 4 conversions, each re-scored | 8 min |
| 8 | `verify_pi_features.py` | 1 min |
| 8b | `verify_serving_transform.py` | 20 s |
| 9 | `ablations.py` — NN probe + leave-one-day-out (8 folds) | 25 min |
| 10 | `make_report.py` — regenerates `REPORT.md` | 5 s |

To run a single stage instead:

```bash
python scripts/train_sensor.py --split blocked
python scripts/train_image.py  --split blocked
python scripts/train_fusion.py --split blocked
python scripts/evaluate.py     --split blocked
```

`--split random` gives the optimistic comparison numbers. `--split blocked` is the honest one.

---

## 6. Put it on the Raspberry Pi

Copy these five things to the Pi:

```
models/image_ensemble_blocked_full_float16.tflite
artifacts/preprocessing_config.json
scripts/predict_pi.py
scripts/image_io.py
```

That is the whole list for the recommended export: it takes the photograph and nothing
else, so `python scripts/predict_pi.py --image capture.jpg` is a complete command. Add
`artifacts/sensor_scaler.pkl` and `scripts/pi_features.py` only if you intend to run the
older fusion export, which also needs `--temperature --humidity --gas`.

On the Pi:

```bash
pip install tflite-runtime simplejpeg pillow numpy scikit-learn==1.8.0
python predict_pi.py --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185
```

`predict_pi.py` picks up `tflite_runtime` automatically when TensorFlow isn't installed.
Install `simplejpeg` on the Pi: without it the image falls back to PIL decoding, which is
about 1.2 grey levels off and costs ~0.014 days of MAE (the output says so).
Model is 2.3 MB; inference is a few milliseconds plus camera capture.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'common'`** — the scripts import each other by
plain module name, so run each one as a file (`python scripts/<name>.py` from the project
root, or `python <name>.py` from inside `scripts/` as `run_all.sh` does). Importing them as a
package (`python -m scripts.<name>`) does not work.

**`InconsistentVersionWarning` when loading `sensor_scaler.pkl`** — the scaler was pickled
with scikit-learn 1.8.0. Install exactly that: `pip install scikit-learn==1.8.0`.

**`URL fetch failure ... storage.googleapis.com`** — no internet for the ImageNet weights.
The offline copy is already in `artifacts/`, and `extract_image_features.py` falls back to
it automatically. Nothing to do.

**Images missing** — rebuild `data_images/` from the original folders:

```bash
python scripts/prepare_images.py \
  --processed ../Processed_Images --original ../Original_Images
```

**`FileNotFoundError: master_dataset.csv`** — run `python scripts/prepare_data.py` first.
