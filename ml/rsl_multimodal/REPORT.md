# Multimodal Remaining Shelf Life (RSL) prediction

Edge-AI + IoT capstone. Two models are trained independently - one on the preprocessed
images, one on the engineered sensor features - then frozen and joined by a small fusion
head that predicts remaining shelf life in days.

```
  224x224 RGB image ---> MobileNetV3Small (frozen) ---> image head (frozen)
                                                          |-- image_embedding (128) --.
                                                          '-- image RSL         (1) --|
                                                                                      |--> concat (194) --> fusion head --> RSL in days
  27 sensor features ---> sensor MLP (frozen)                                          |
                                                          .-- sensor_embedding (64) --'
                                                          '-- sensor RSL        (1) --'
```

Stage 1 trains each branch separately with its own RSL head. Stage 2 freezes both and
trains only the fusion head on their concatenated embeddings plus their two scalar
predictions. Target: `RemainingShelfLife = 11 - Day`, in days; `Day` is never an input.

---

## Headline

| Question | Best model | MAE |
|---|---|---|
| How much shelf life is left, in a session already being monitored? | **Fusion** | **0.27 days** |
| How much is left, for a batch the model has never seen? | **Image only** | **0.87 days** |

Those are different questions with different answers, and the gap between them is the most
important thing in this report. Section 4 explains why.

---

## 1. Data

| | |
|---|---|
| Rows in `sensor_features.csv` | 3316 |
| Rows with a usable 224x224 image | 3067 |
| Rows dropped (no image found) | 249 |
| Distinct monitoring days | 10 |
| RSL range | 0 - 10 days |

`Processed_Images/` covers only 2058 of the sensor rows. The 1009 rows whose frame existed
only in `Original_Images/` were put through the identical transform recorded in
`image_preprocessing_config.json` (RGB -> LANCZOS 224x224 -> JPEG q95) rather than being
discarded, which is what lifts usable samples to 3067.

### Columns excluded from the model, and why

**Clocks, not sensors.** `Day_Of_Month`, `Elapsed_Time_*`, `Cumulative_*` and `Sl_No`
correlate 0.97-1.00 with `Day`, and the target is `11 - Day`. Any model given them
reproduces the target arithmetically and reports a near-perfect R² while learning nothing
about spoilage - and a Pi handed an arbitrary sample cannot supply them. `prepare_data.py`
now measures this from the data rather than trusting a hand-written list: it refuses to
build a dataset if any model input correlates ≥ 0.90 with `Day`. Highest among the
27 inputs actually used: `Temperature_Gas_Interaction` at
0.50.

**Not reproducible on the device.** The three `*_Level` bands are not a function of the
reading they describe - `Gas_Level` "Low" is exactly 179 while "Very_High" spans 152-195,
each of the four gas bands holds exactly 3316/4 = 829 rows (cut on rank, not value), and
`Temperature_Level` is the single value "Moderate" for all 3316 rows. The three `*_Trend`
columns are excluded for a blunter reason: their definition could not be recovered. Every
rolling least-squares slope over the raw, cleaned or smoothed series (windows 3-30, all
`min_periods`) fails to reproduce them, so a Pi cannot compute them.

| Sensor branch input | features | test MAE (days) |
|---|---|---|
| deployable features | 27 | 0.753 |
| + the excluded columns | 33 | 0.510 |

They are worth ~0.24 days of MAE. It is not a gain the deployed system can realise, so the
model does without them.

## 2. Splits, and why the obvious one does not work

Frames are captured roughly every 2 minutes. Consecutive frames of the same tray under the
same light are near-duplicates, so a random split puts a frame in train and its twin in
test. To measure that rather than argue about it, `ablations.py` runs a **1-nearest-neighbour
probe**: it looks up each test row's closest train row using *untrained* ImageNet features
and copies its label. No learning at all.

| Split | median minutes to nearest train row | 1-NN on frozen ImageNet features | 1-NN on sensor vector | trained fusion |
|---|---|---|---|---|
| blocked | 96 | 0.081 | 0.517 | 0.266 |
| random | 2 | 0.020 | 0.115 | 0.153 |

On the random split, nearest-neighbour lookup scores MAE 0.02 days - the test set is
answerable by memory alone, and any model score on it is meaningless. The **blocked** split
used for section 3 assigns whole 180-minute blocks and then
**embargoes** every val/test row within 60 minutes of any
train row (364 test and 282 val rows dropped),
pushing the median gap to 96 minutes.

That helps, but be honest about what it does not fix: 1-NN still scores
0.081 days on the blocked split, better than the
trained model. Within one continuous 11-day experiment, frames of the same day look nearly
identical no matter how far apart in time they are. **No within-day split can measure
generalisation on this dataset.** That is what section 4 is for.

| Split | train | val | test | embargoed |
|---|---|---|---|---|
| blocked | 1882 | 268 | 271 | 646 |
| random | 2148 | 460 | 459 | - |

## 3. Within-session performance (blocked split, n=271)

| Model | MAE (days) | RMSE | R² | within 0.5 d | within 1 d |
|---|---|---|---|---|---|
| Sensor only | 0.753 | 1.388 | 0.4431 | 45.0% | 83.4% |
| Image only | 0.242 | 0.437 | 0.9448 | 87.1% | 96.3% |
| Fusion | 0.266 | 0.669 | 0.8704 | 92.3% | 97.0% |
| **Image ensemble (6 heads)** | **0.145** | **0.393** | **0.9553** | **96.3%** | 97.4% |

Fusion beats the single image head on the within-half-day rate (92.3% against 87.1%), but
not on MAE, RMSE or R²; an earlier draft of this report claimed it beat both branches
outright, which the table does not support. The 6-head image ensemble added in section 5b
beats every row here on every column. These are the realistic numbers for a Pi watching
one batch continuously, having already seen earlier frames of it. They are **not**
generalisation numbers, for the reason in section 2.

For reference, the earlier single-stage DEMO_MODEL reported MAE 0.281 / R² 0.966 on a plain
random split — the same protocol under which 1-NN lookup here scores 0.02.

## 4. A day the model has never seen (leave-one-day-out)

Every frame of one day is removed from training, and validation uses a *different* held-out
day so early stopping is also selected across days.

| Held-out day | true RSL | fusion mean prediction | sensor MAE | image MAE | fusion MAE | n |
|---|---|---|---|---|---|---|
| 1 | 10 | 7.63 | 5.303 | 1.016 | 2.371 | 111 |
| 2 | 9 | 9.03 | 1.866 | 0.518 | 0.453 | 647 |
| 3 | 8 | 7.93 | 0.731 | 0.303 | 0.307 | 628 |
| 4 | 7 | 6.95 | 0.261 | 1.034 | 2.005 | 327 |
| 5 | 6 | 4.93 | 1.416 | 0.517 | 1.117 | 162 |
| 6 | 5 | 5.03 | 1.267 | 0.696 | 0.172 | 653 |
| 7 | 4 | 3.86 | 2.775 | 0.700 | 0.370 | 360 |
| 9 | 2 | 4.60 | 6.161 | 2.188 | 2.597 | 170 |

| Mean across held-out days | sensor | image | fusion | image ensemble |
|---|---|---|---|---|
| MAE (days) | 2.473 | 0.871 | 1.174 | **0.848** |

The ensemble column is measured by `scripts/train_image_ensemble.py --lodo`, which runs
the same protocol over the exact graph that is served; per-day figures are in
`results/image_ensemble_lodo.json`.

**The ordering flips.** Within a session, fusion is best. Across sessions, the image branch
alone is best and fusion is worse, because the sensor branch generalises poorly to a new day
(2.47 days MAE) and the fusion head has learned to trust it. Temperature,
humidity and a single gas channel describe the *ambient conditions of that particular run*
more than they describe spoilage; the camera sees the food itself.

The largest errors are at the ends of the range - day 1 and day 9 - which is expected:
holding out an end day asks the model to extrapolate beyond every target value it has seen.

**What to do with this:**

- For continuous monitoring of a batch already under observation, deploy the fusion model.
- For a cold read on an unfamiliar batch, the image branch alone is the more reliable
  predictor, and roughly 1 day of error is the honest expectation.
- The fix is not a bigger model. It is more monitoring runs: with a single 11-day
  experiment, "a new day" and "a new batch" are the same axis, and the sensor branch has no
  way to learn what generalises across runs.

## 5b. Why the served model is a photograph-only ensemble

Section 4 shows the image branch beating fusion on an unseen day. Before acting on that,
`scripts/experiments.py` re-ran leave-one-day-out over six training seeds (42, 7, 2026,
13, 99, 404), because head training is stochastic and the differences being chased are
small:

| | LODO MAE, mean ± sd over 6 seeds |
|---|---|
| sensor branch | 2.392 ± 0.106 |
| fusion head | 1.113 ± 0.179 |
| image head | 0.920 ± 0.086 |
| image head, 6 predictions averaged | **0.856** |

The image branch wins on 4 of the 6 seeds, with a mean margin of 0.193 days, so the
ordering in section 4 is not a single-run artifact. The seed-to-seed spread (±0.086) is as
large as the gap between whole architectures, which is the second finding: averaging the
heads removes that variance and lands below every individual head.

Four other variants were tried and every one was worse — sensor dropout in the fusion head
(1.705), augmented image training (1.003 image / 1.108 fusion), both together (2.095), and
a dynamics-only sensor feature set (1.188). A post-hoc blend of image and fusion
predictions never beat the image branch alone. The full table is in
`results/experiments.json`.

Rebuilt as its own export, `scripts/train_image_ensemble.py` measures 0.848 days
leave-one-day-out — confirming the 0.856 estimate through the serving graph itself — and
0.145 within-session, better than fusion on both. Dropping the sensor input also removes
three serving preconditions: a gas reading, a reading history long enough for the rolling
features, and readings inside the trained envelope.

## 5. Raspberry Pi deployment

| Variant | size (MB) | MAE (days) | R² | ms/sample (x86) | CPU backend |
|---|---|---|---|---|---|
Fusion export (image + 27 sensor features):

| Variant | size (MB) | MAE (days) | R² | ms/sample (x86) | CPU backend |
|---|---|---|---|---|---|
| float32 | 4.60 | 0.266 | 0.8704 | 1.8 | xnnpack |
| float16 | 2.34 | 0.264 | 0.8705 | 1.9 | xnnpack |
| dynamic | 1.34 | 0.590 | 0.8222 | 3.0 | xnnpack |
| int8 | 1.46 | 3.179 | -2.7200 | 10.2 | reference_kernels |

Image-ensemble export (one image input, 6 heads averaged inside the graph):

| Variant | size (MB) | MAE (days) | R² | ms/sample (x86) | CPU backend |
|---|---|---|---|---|---|
| float32 | 8.08 | 0.145 | 0.9553 | 2.2 | xnnpack |
| float16 | 4.08 | 0.147 | 0.9549 | 1.8 | xnnpack |

**Recommended artifact: `models/image_ensemble_blocked_full_float16.tflite`** — half the
size of its float32 build, accuracy unchanged, XNNPACK-accelerated on a Pi 4/5, and the
best of everything measured here on an unseen day. It is larger than the fusion float16
build (4.1 MB against 2.3 MB) because it carries six heads over one shared backbone; the
backbone dominates, so each extra head costs about 0.36 MB.

The fusion float16 build stays in the bundle and stays loadable, for comparison and
because it is the artifact the earlier results were measured on.

Full-integer int8 costs far more accuracy than usual here (MAE 3.18 vs
0.27 days). MobileNetV3's hard-swish and squeeze-excite blocks are hostile
to post-training quantization, and this task needs sub-day resolution over a narrow output
range, so quantization noise is large relative to the signal. The XNNPACK delegate also
refuses the fully-quantized graph and falls back to reference kernels, making int8 ~5x
*slower* than float16 as well as less accurate. If int8 becomes a hard requirement (Coral
Edge TPU, tight memory), the route is quantization-aware training or a
quantization-friendlier backbone such as MobileNetV2 — not a different converter flag.

### On-device feature reconstruction is verified, not assumed

The sensor branch expects 27 engineered features, most of them defined over a rolling
window of *cleaned* readings. Getting any of them wrong on the Pi is silent: the model
still returns a number, it is just fed a different distribution. `scripts/pi_features.py`
implements the definitions recovered from the training CSV, and
`scripts/verify_pi_features.py` replays all 3316 real readings through it one at a time
and compares against the CSV in scaler-z units.

Result: **PASS** on the 3010 rows whose window contains no IQR outlier.
Continuous features agree to about 1e-6 in raw units (under 1e-5 in scaler-z units). The two
binary direction flags disagree on a handful of rows - `Temperature_Increasing` on 0.56% of rows, `Humidity_Increasing` on 0.13% of rows - because training
computed `Change > 0` with no tolerance and recorded a float-noise change of +3.6e-15 as
"increasing"; the device treats that as zero. The other, larger difference: training
replaced outliers by *linear interpolation*, which needs the next reading; a live device
forward-fills the last in-range value instead, and clips an out-of-range *first* reading to
the nearest bound. The frozen IQR bounds are shipped in `preprocessing_config.json` so the
device applies the same thresholds causally.

### The serving image path is verified too

The first serving code decoded JPEGs with PIL and resized camera frames with BILINEAR.
Training decoded with `tf.io.decode_jpeg` (fast integer IDCT) and resized with LANCZOS
followed by a JPEG q95 round trip. Neither mismatch raised an error; both moved predictions:

| Serving path | 224x224 test images | 2592x1944 camera originals |
|---|---|---|
| PIL decode, BILINEAR resize (original `serve.py`) | 0.278 | 0.207 |
| `image_io.py` (LANCZOS + JPEG q95 + TF-identical IDCT) | 0.264 | 0.152 |

`scripts/verify_serving_transform.py` rebuilds stored training images from their full-resolution originals through `scripts/image_io.py`: **60/60 bit-identical**. Scoring the float16 model on the test set through that loader gives MAE **0.2636**, matching the 0.2636 recorded at export (decoder: `tensorflow`). On a Pi without TensorFlow, `simplejpeg` with `fastdct=True,
fastupsample=False` reproduces TensorFlow's decoder bit for bit.

### Operating envelope

The instantaneous `Temperature`, `Humidity` and `Gas` inputs are the **raw** readings, exactly
as in training - only the smoothed and rolling features are cleaned. So a reading far outside
anything the model saw still reaches the network. Outside this envelope the output is an
extrapolation (a 45 C cold start returns -7.4 days before clamping), and the serving code
flags it as unreliable:

| Channel | trained on, min | trained on, max | outlier bound, lower | outlier bound, upper |
|---|---|---|---|---|
| Temperature | 25.83 | 29.31 | 27.46 | 29.33 |
| Humidity | 53.07 | 67.01 | 52.55 | 62.63 |
| Gas | 152 | 193 | 171.50 | 191.50 |

```bash
pip install tflite-runtime simplejpeg pillow numpy scikit-learn==1.8.0
python scripts/predict_pi.py --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185
```

## 6. Scaling this

- **Each branch is independently retrainable.** Retrain the sensor MLP on new gas chemistry,
  or swap the backbone, and only the fusion head needs refitting — seconds of CPU.
- **The frozen backbone is cached** (`artifacts/backbone_features.npy`), so a full retrain
  of both heads takes under a minute for 3067 samples.
- **More food types**: add a food-type input to the sensor vector, or train one fusion head
  per class over the shared backbone.
- **The highest-value next step is more monitoring runs**, not more model. Section 4's gap
  is a data limitation, and no architecture closes it.

## 7. Reproducing

```bash
bash run_all.sh
```

Checks that run as part of the pipeline, not as afterthoughts:

- model inputs are refused if any correlates ≥ 0.90 with the target (`prepare_data.py`);
- the assembled two-input graph is compared against the staged pipeline on every test
  sample (max abs difference 0.0e+00);
- scalers and quantization calibration use train rows only;
- every TFLite variant is re-scored on the test set rather than assumed equivalent;
- the on-device feature builder is replayed against the real stream and gated;
- a nearest-neighbour probe reports how much of each split is answerable by lookup alone.
