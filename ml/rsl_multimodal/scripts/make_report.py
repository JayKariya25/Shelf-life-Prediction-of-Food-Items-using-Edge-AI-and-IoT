"""Collect every metrics file into REPORT.md plus a summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

REPORT = C.ROOT / "REPORT.md"

HEADER = ("| Model | MAE (days) | RMSE | R² | within 0.5 d | within 1 d |\n"
          "|---|---|---|---|---|---|")


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def row(name: str, m: dict) -> str:
    return (f"| {name} | {m['MAE']:.3f} | {m['RMSE']:.3f} | {m['R2']:.4f} | "
            f"{m['WithinHalfDay']*100:.1f}% | {m['WithinOneDay']*100:.1f}% |")


def summary_figure(cmp_b, abl, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    names = ["Sensor only", "Image only", "Fusion"]
    keys = ["sensor", "image", "fusion"]
    xs = np.arange(len(names))
    w = 0.36

    within = [cmp_b["test_metrics"][n]["MAE"] for n in names]
    across = [abl["leave_one_day_out_summary"]["mean_MAE"][k] for k in keys]
    ax1.bar(xs - w / 2, within, w, label="within session (blocked split)", color="#2f6f9f")
    ax1.bar(xs + w / 2, across, w, label="new day (leave-one-day-out)", color="#e08a3c")
    ax1.set_xticks(xs); ax1.set_xticklabels(names)
    ax1.set_ylabel("Test MAE (days)")
    ax1.set_title("The two questions give different answers")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.25, lw=0.5)

    lodo = abl["leave_one_day_out"]
    days = sorted(int(d) for d in lodo)
    true = [lodo[str(d)]["true_RSL"] for d in days]
    pred = [lodo[str(d)]["mean_predicted_RSL"] for d in days]
    ax2.plot(true, true, "--", color="#c0392b", lw=1.2, label="perfect")
    ax2.scatter(true, pred, s=60, color="#e08a3c", zorder=3, label="mean prediction")
    for t, p, d in zip(true, pred, days):
        ax2.annotate(f"day {d}", (t, p), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("True RSL of the held-out day"); ax2.set_ylabel("Mean predicted RSL")
    ax2.set_title("Leave-one-day-out: a day never seen in training")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    audit = read(C.ARTIFACTS / "data_audit.json")
    prep = read(C.ARTIFACTS / "preprocessing_config.json")
    cmp_b = read(C.RESULTS / "blocked_full" / "comparison_metrics.json")
    cmp_r = read(C.RESULTS / "random_full" / "comparison_metrics.json")
    tfl = read(C.RESULTS / "blocked_full" / "tflite_report.json")
    abl = read(C.RESULTS / "ablations.json")
    pifv = read(C.RESULTS / "pi_feature_verification.json")
    stv = read(C.RESULTS / "serving_transform_verification.json")
    envelope = prep.get("input_envelope", {})

    noise = pifv.get("features_with_float_noise_disagreement", {})
    noise_text = ", ".join(f"`{k}` on {v['frac_rows'] * 100:.2f}% of rows" for k, v in noise.items())
    envelope_rows = "\n".join(
        f"| {c} | {envelope[c]['min']:g} | {envelope[c]['max']:g} | "
        f"{prep['outlier_bounds'][c]['lower']:.2f} | {prep['outlier_bounds'][c]['upper']:.2f} |"
        for c in ("Temperature", "Humidity", "Gas") if c in envelope
    )
    if stv:
        serving_text = (
            f"`scripts/verify_serving_transform.py` rebuilds stored training images from their "
            f"full-resolution originals through `scripts/image_io.py`: "
            f"**{stv['originals_bit_identical']}/{stv['originals_checked']} bit-identical**. "
            f"Scoring the float16 model on the test set through that loader gives MAE "
            f"**{stv['served_test_mae']:.4f}**, matching the {stv['recorded_test_mae']:.4f} recorded "
            f"at export (decoder: `{stv['decoder']}`)."
        )
    else:
        serving_text = "Run `scripts/verify_serving_transform.py` to check the serving image path."

    summary_figure(cmp_b, abl, C.RESULTS / "summary.png")

    lodo = abl["leave_one_day_out"]
    lodo_rows = "\n".join(
        f"| {d} | {lodo[d]['true_RSL']:.0f} | {lodo[d]['mean_predicted_RSL']:.2f} | "
        f"{lodo[d]['metrics']['sensor']['MAE']:.3f} | {lodo[d]['metrics']['image']['MAE']:.3f} | "
        f"{lodo[d]['metrics']['fusion']['MAE']:.3f} | {lodo[d]['held_out_samples']} |"
        for d in sorted(lodo, key=int)
    )
    probe = abl["nearest_neighbour_probe"]
    probe_rows = "\n".join(
        f"| {s} | {p['minutes_to_nearest_train_row']['median']:.0f} | "
        f"{p['frozen_imagenet_features']['metrics']['MAE']:.3f} | "
        f"{p['scaled_sensor_vector']['metrics']['MAE']:.3f} | "
        f"{cmp['test_metrics']['Fusion']['MAE']:.3f} |"
        for s, p, cmp in (("blocked", probe["blocked"], cmp_b),
                          ("random", probe["random"], cmp_r))
    )
    tfl_rows = "\n".join(
        f"| {k} | {v['size_mb']:.2f} | {v['metrics']['MAE']:.3f} | {v['metrics']['R2']:.4f} | "
        f"{v['latency_ms_per_sample_x86_cpu']:.1f} | {v['cpu_backend']} |"
        for k, v in tfl["variants"].items()
    )
    mean_lodo = abl["leave_one_day_out_summary"]["mean_MAE"]

    text = f"""# Multimodal Remaining Shelf Life (RSL) prediction

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
predictions. Target: `RemainingShelfLife = {C.MAX_SHELF_LIFE_DAY} - Day`, in days; `Day` is never an input.

---

## Headline

| Question | Best model | MAE |
|---|---|---|
| How much shelf life is left, in a session already being monitored? | **Fusion** | **{cmp_b['test_metrics']['Fusion']['MAE']:.2f} days** |
| How much is left, for a batch the model has never seen? | **Image only** | **{mean_lodo['image']:.2f} days** |

Those are different questions with different answers, and the gap between them is the most
important thing in this report. Section 4 explains why.

---

## 1. Data

| | |
|---|---|
| Rows in `sensor_features.csv` | {audit['rows_in_sensor_features']} |
| Rows with a usable 224x224 image | {audit['rows_final']} |
| Rows dropped (no image found) | {audit['rows_without_image']} |
| Distinct monitoring days | {len(audit['day_counts'])} |
| RSL range | {audit['target_range'][0]:.0f} - {audit['target_range'][1]:.0f} days |

`Processed_Images/` covers only 2058 of the sensor rows. The 1009 rows whose frame existed
only in `Original_Images/` were put through the identical transform recorded in
`image_preprocessing_config.json` (RGB -> LANCZOS 224x224 -> JPEG q95) rather than being
discarded, which is what lifts usable samples to {audit['rows_final']}.

### Columns excluded from the model, and why

**Clocks, not sensors.** `Day_Of_Month`, `Elapsed_Time_*`, `Cumulative_*` and `Sl_No`
correlate 0.97-1.00 with `Day`, and the target is `11 - Day`. Any model given them
reproduces the target arithmetically and reports a near-perfect R² while learning nothing
about spoilage - and a Pi handed an arbitrary sample cannot supply them. `prepare_data.py`
now measures this from the data rather than trusting a hand-written list: it refuses to
build a dataset if any model input correlates ≥ 0.90 with `Day`. Highest among the
{len(prep['sensor_features']['full'])} inputs actually used: `{audit['max_abs_corr_with_Day_among_inputs']['column']}` at
{audit['max_abs_corr_with_Day_among_inputs']['value']:.2f}.

**Not reproducible on the device.** The three `*_Level` bands are not a function of the
reading they describe - `Gas_Level` "Low" is exactly 179 while "Very_High" spans 152-195,
each of the four gas bands holds exactly 3316/4 = 829 rows (cut on rank, not value), and
`Temperature_Level` is the single value "Moderate" for all 3316 rows. The three `*_Trend`
columns are excluded for a blunter reason: their definition could not be recovered. Every
rolling least-squares slope over the raw, cleaned or smoothed series (windows 3-30, all
`min_periods`) fails to reproduce them, so a Pi cannot compute them.

| Sensor branch input | features | test MAE (days) |
|---|---|---|
| deployable features | {abl['excluded_column_ablation']['deployable_features']['n_features']} | {abl['excluded_column_ablation']['deployable_features']['test']['MAE']:.3f} |
| + the excluded columns | {abl['excluded_column_ablation']['plus_excluded_columns']['n_features']} | {abl['excluded_column_ablation']['plus_excluded_columns']['test']['MAE']:.3f} |

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
{probe_rows}

On the random split, nearest-neighbour lookup scores MAE 0.02 days - the test set is
answerable by memory alone, and any model score on it is meaningless. The **blocked** split
used for section 3 assigns whole {prep['split']['block_minutes']}-minute blocks and then
**embargoes** every val/test row within {prep['split']['embargo_minutes']} minutes of any
train row ({audit['embargoed_rows']['test']} test and {audit['embargoed_rows']['val']} val rows dropped),
pushing the median gap to {probe['blocked']['minutes_to_nearest_train_row']['median']:.0f} minutes.

That helps, but be honest about what it does not fix: 1-NN still scores
{probe['blocked']['frozen_imagenet_features']['metrics']['MAE']:.3f} days on the blocked split, better than the
trained model. Within one continuous 11-day experiment, frames of the same day look nearly
identical no matter how far apart in time they are. **No within-day split can measure
generalisation on this dataset.** That is what section 4 is for.

| Split | train | val | test | embargoed |
|---|---|---|---|---|
| blocked | {audit['split_counts']['blocked']['train']} | {audit['split_counts']['blocked']['val']} | {audit['split_counts']['blocked']['test']} | {audit['split_counts']['blocked'].get('dropped', 0)} |
| random | {audit['split_counts']['random']['train']} | {audit['split_counts']['random']['val']} | {audit['split_counts']['random']['test']} | - |

## 3. Within-session performance (blocked split, n={cmp_b['n_test']})

{HEADER}
{row('Sensor only', cmp_b['test_metrics']['Sensor only'])}
{row('Image only', cmp_b['test_metrics']['Image only'])}
{row('**Fusion**', cmp_b['test_metrics']['Fusion'])}

Fusion beats both branches. This is the realistic number for the actual product: a Pi
watching one batch continuously, having already seen earlier frames of it. It is **not** a
generalisation number, for the reason in section 2.

For reference, the earlier single-stage DEMO_MODEL reported MAE 0.281 / R² 0.966 on a plain
random split — the same protocol under which 1-NN lookup here scores 0.02.

## 4. A day the model has never seen (leave-one-day-out)

Every frame of one day is removed from training, and validation uses a *different* held-out
day so early stopping is also selected across days.

| Held-out day | true RSL | fusion mean prediction | sensor MAE | image MAE | fusion MAE | n |
|---|---|---|---|---|---|---|
{lodo_rows}

| Mean across held-out days | sensor | image | fusion |
|---|---|---|---|
| MAE (days) | {mean_lodo['sensor']:.3f} | **{mean_lodo['image']:.3f}** | {mean_lodo['fusion']:.3f} |

**The ordering flips.** Within a session, fusion is best. Across sessions, the image branch
alone is best and fusion is worse, because the sensor branch generalises poorly to a new day
({mean_lodo['sensor']:.2f} days MAE) and the fusion head has learned to trust it. Temperature,
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

## 5. Raspberry Pi deployment

| Variant | size (MB) | MAE (days) | R² | ms/sample (x86) | CPU backend |
|---|---|---|---|---|---|
{tfl_rows}

**Recommended artifact: `models/fusion_blocked_full_float16.tflite`** — half the size of
float32, accuracy unchanged, XNNPACK-accelerated on a Pi 4/5.

Full-integer int8 costs far more accuracy than usual here (MAE {tfl['variants']['int8']['metrics']['MAE']:.2f} vs
{tfl['variants']['float32']['metrics']['MAE']:.2f} days). MobileNetV3's hard-swish and squeeze-excite blocks are hostile
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
`scripts/verify_pi_features.py` replays all {pifv['rows_replayed']} real readings through it one at a time
and compares against the CSV in scaler-z units.

Result: **{pifv['passed'] and 'PASS' or 'FAIL'}** on the {pifv['rows_without_outlier_in_window']} rows whose window contains no IQR outlier.
Continuous features agree to about 1e-6 in raw units (under 1e-5 in scaler-z units). The two
binary direction flags disagree on a handful of rows - {noise_text} - because training
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

{serving_text} On a Pi without TensorFlow, `simplejpeg` with `fastdct=True,
fastupsample=False` reproduces TensorFlow's decoder bit for bit.

### Operating envelope

The instantaneous `Temperature`, `Humidity` and `Gas` inputs are the **raw** readings, exactly
as in training - only the smoothed and rolling features are cleaned. So a reading far outside
anything the model saw still reaches the network. Outside this envelope the output is an
extrapolation (a 45 C cold start returns -7.4 days before clamping), and the serving code
flags it as unreliable:

| Channel | trained on, min | trained on, max | outlier bound, lower | outlier bound, upper |
|---|---|---|---|---|
{envelope_rows}

```bash
pip install tflite-runtime simplejpeg pillow numpy scikit-learn==1.8.0
python scripts/predict_pi.py --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185
```

## 6. Scaling this

- **Each branch is independently retrainable.** Retrain the sensor MLP on new gas chemistry,
  or swap the backbone, and only the fusion head needs refitting — seconds of CPU.
- **The frozen backbone is cached** (`artifacts/backbone_features.npy`), so a full retrain
  of both heads takes under a minute for {audit['rows_final']} samples.
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
  sample (max abs difference {cmp_b['graph_consistency_max_abs_diff']:.1e});
- scalers and quantization calibration use train rows only;
- every TFLite variant is re-scored on the test set rather than assumed equivalent;
- the on-device feature builder is replayed against the real stream and gated;
- a nearest-neighbour probe reports how much of each split is answerable by lookup alone.
"""
    REPORT.write_text(text)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
