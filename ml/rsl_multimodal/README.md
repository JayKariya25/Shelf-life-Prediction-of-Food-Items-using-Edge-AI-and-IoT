# Multimodal Remaining Shelf Life (RSL) — edge-AI capstone

Predicts the remaining shelf life of food, in days, from a camera frame plus live
temperature / humidity / gas readings. Two models are trained **separately**, frozen, and
joined by a small fusion head — so each modality stays independently retrainable and the
fused graph exports cleanly to TFLite for a Raspberry Pi.

**New here? Start with [QUICKSTART.md](QUICKSTART.md)** — how to install, run one
prediction, and reproduce everything.

Read **[REPORT.md](REPORT.md)** for results and the deployment recommendation, and
**[INTEGRATION.md](INTEGRATION.md)** to connect the model to a website.

## Layout

```
notebooks/
  01_sensor_model.ipynb           sensor branch, walkthrough with plots
  02_image_model.ipynb            image branch, walkthrough with plots
  03_fusion_and_deployment.ipynb  fusion, TFLite export, generalisation checks
scripts/
  common.py                       paths, feature sets, splits, metrics
  datasets.py                     loading helpers
  models.py                       the three architectures
  prepare_data.py                 master dataset, RSL target, splits, scalers
  extract_image_features.py       frozen-backbone feature cache
  train_sensor.py                 stage 1a
  train_image.py                  stage 1b
  train_fusion.py                 stage 2 (branches frozen)
  evaluate.py                     comparison metrics, figures, graph consistency check
  export_tflite.py                float32 / float16 / dynamic / int8 + re-scoring
  ablations.py                    nearest-neighbour probe, ablation, leave-one-day-out
  pi_features.py                  on-device sensor feature reconstruction
  verify_pi_features.py           gate: replays the real stream through pi_features.py
  image_io.py                     training-identical image decoding for serving
  verify_serving_transform.py     gate: serving images and test MAE match training
  prepare_images.py               builds data_images/ from Processed_ + Original_Images
  predict_pi.py                   Raspberry Pi inference
  recover_backbone_weights.py     offline fallback for the ImageNet weights
  build_notebooks.py              regenerates the notebooks above
serve.py    HTTP API + built-in test page, for the website to call
models/     trained .keras models and .tflite exports
artifacts/  master dataset, scalers, preprocessing config, audits
results/    metrics, predictions, figures, per-split
```

## Requirements

```bash
pip install -r requirements.txt
```

Input data expected next to this README:

- `sensor_features.csv` — output of `02_sensor_feature_engineering.ipynb`
- `data_images/` — the 224×224 RGB JPEGs (`Processed_Images/`, plus any `Original_Images/`
  put through the same transform)

## Reproduce everything

```bash
bash run_all.sh
```

Roughly 6 minutes on a laptop CPU. The frozen backbone is cached after the first run, so
re-training the heads afterwards takes seconds.

## Predict on a Raspberry Pi

```bash
pip install tflite-runtime simplejpeg pillow numpy scikit-learn==1.8.0
python scripts/predict_pi.py --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185
```

Ship these to the Pi: `models/image_ensemble_blocked_full_float16.tflite`,
`artifacts/sensor_scaler.pkl`, `artifacts/preprocessing_config.json`, and
`scripts/predict_pi.py`, `scripts/pi_features.py` and `scripts/image_io.py`.

## Four things to know before the viva

1. **Several columns in `sensor_features.csv` are clocks, not sensors.** `Day_Of_Month`,
   `Elapsed_Time_*`, `Cumulative_*` and `Sl_No` correlate 0.97–1.00 with `Day`, and the
   target is `11 − Day`. Any model given them scores near-perfectly by doing arithmetic.
   `prepare_data.py` refuses to build a dataset if any model input correlates ≥ 0.90 with
   the target, so this is checked from the data rather than trusted to a list.
2. **Test error depends entirely on what "unseen" means.** The served image ensemble is at
   ~0.15 days MAE within a monitoring session already being observed, and ~0.85 days on a
   day it has never seen. Both numbers are in the report; the second is the honest one to
   quote for a new batch of food. The fusion model that was served first scored ~0.27 and
   ~1.17 on the same two protocols, which is why it was replaced — see REPORT section 5b.
3. **A nearest-neighbour probe is part of the pipeline.** Copying labels from the nearest
   train frame using untrained ImageNet features scores 0.02 days MAE on a random split.
   That is the number to bring up if anyone asks why a random split was not used — and it
   is why the report leads with leave-one-day-out.
4. **The model only knows one room.** Every raw reading it was trained on sits in
   25.8–29.3 °C, 53–67 % RH and gas 152–193. The instantaneous inputs are the raw readings,
   so anything outside reaches the network and the output is extrapolation — a 45 °C cold
   start returns −7.4 days before clamping. The serving code flags these as `UNRELIABLE`, and
   the monitoring website falls back to its kinetic baseline for them.

## Serving fixes (September 2026)

An audit of the serving path found problems that raised no errors but changed predictions.
All are fixed, and the two that affect accuracy now have gates in `run_all.sh`.

| Problem | Effect | Fix |
|---|---|---|
| `serve.py` / `predict_pi.py` decoded JPEGs with PIL; training used `tf.io.decode_jpeg` | test MAE 0.264 → 0.278 | `image_io.py` uses a TF-identical decoder (TensorFlow or `simplejpeg`) |
| Camera frames resized with BILINEAR; training used LANCZOS + JPEG q95 | MAE on 2592×1944 frames 0.152 → 0.207 | `image_io.py` repeats the training transform; 60/60 images rebuilt bit-for-bit |
| An out-of-range **first** reading was passed through unclamped, while the API said it was clamped | out-of-distribution input, misleading warning | clipped to the outlier bound; warnings report exactly what happened |
| No warning when a raw reading was outside anything seen in training | silent extrapolation (−7.4 d) | `input_envelope` in the config; `UNRELIABLE` warnings; output clamped in `predict_pi.py` too |
| Requests without `session_id` silently shared one `default` session | batches mixed their rolling history | still accepted, but now warned |
| A failed request still advanced the session's rolling window | corrupted history | the session is rolled back on failure |
| `scikit-learn` unpinned; the scaler was pickled with 1.8.0 | `InconsistentVersionWarning` on a fresh install | pinned to 1.8.0 |
| `predict_pi.py` wrote session state into `artifacts/` | runtime state mixed with trained artifacts | moved to `runtime/`, with `--state` |
| `serve.py` defaulted to port 8000 | collided with the monitoring website | defaults to 8001 |
| Docs claimed features matched "to ~1e-13" and preprocessing "cannot drift" | overstated | corrected in `REPORT.md` with measured figures |

The feature-verification gate reports identical numbers before and after these changes.

