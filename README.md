# Shelf-Life Monitor — Edge-AI Web Application

Web dashboard for the B.Tech capstone **“Shelf-Life Prediction of Food Items
using Edge-AI”** (PES University, Bengaluru).

It tracks individual fruit and vegetable items, records the temperature and
humidity they are actually stored in, estimates how much shelf life each one has
left, and raises alerts before produce is wasted. Inference runs on the same
machine that serves the page, so a Raspberry Pi 5 can run the whole system with
no internet connection.

---

## Two interfaces, one login

The role on the account decides what appears after signing in. There is no
separate admin login and no separate URL to remember.

| | `user` | `admin` |
|---|---|---|
| Dashboard, Items, Alerts, Device, Profile | yes | yes |
| Developer console, Model card | no | yes |
| Inference playground | no | yes |
| Held-out evaluation | no | yes |
| Accounts and roles | no | yes |
| System and maintenance | no | yes |

A `user` account that navigates to any `/admin/...` URL gets a **403**, not a
redirect — it is authenticated, simply not authorised — and the developer
section is absent from its navigation entirely. An `admin` keeps full access to
the ordinary monitoring interface as well, so there is no need to switch
accounts to see what a user sees.

Admin is an operator role, **not** a data-access role: an admin still cannot
read another account's items, alerts or devices through the normal interface.
That is covered by tests.

New accounts are always created as `user`. Promote one from the command line:

```bash
flask --app app set-role --email you@example.com --role admin
flask --app app list-users
```

The last remaining administrator cannot be demoted or deactivated, so the
console can never be locked away behind a database edit.

---

## What this application does and does not claim

This is the part that matters most for the report, so it is stated plainly.

Two estimators are available, and every prediction records which one produced it
(`model_kind`: `trained-model` or `heuristic-baseline`). That tag drives the banners on
the dashboard and item pages; nothing in the UI implies the trained model ran when it
did not.

| | Status |
|---|---|
| Remaining shelf life from a photograph | **Working** — the project's trained image ensemble, `ml/rsl_multimodal` |
| Temperature/humidity kinetic estimate | **Working** — Q10 baseline, always available |
| Reproduces the model's published test error inside the website | **Verified** — MAE 0.1468 on all 271 held-out frames |
| Calibrated class confidence | **Not available** — the model is a single-output regressor |
| Uncertainty band | ±0.85 days — the model's error on days it never saw |

### When the trained model is used, and when it is not

The served model reads the photograph and nothing else, so it has two preconditions:

- **The produce it was trained on** — `RSL_TRAINED_FOOD_KEYS`, which is `banana`. The
  training data is a single 11-day banana run and has no food column at all, so the
  bundle cannot state this and the application does. A tomato photograph scored by a
  banana-only model would be a confident number about a food it has never seen, so
  tomato items use the kinetic baseline, which *is* food-aware through `FOOD_TYPE_SEED`.
- **A recent photo** — under `RSL_MAX_PHOTO_AGE_HOURS` (6 h) from the newest reading.
  The model estimates from what the food looks like *now*; a day-old photo would
  report a day-old answer.

Otherwise the item falls back to the kinetic baseline, and the item page says exactly
why:

> Estimated with the kinetic baseline because the trained model was fitted on banana
> only, and this item is tomato (its photograph would be scored by a model that has
> never seen this food).

No gas reading, no reading history and no temperature inside a trained envelope are
required any more. That is a consequence of the accuracy work below, not a relaxation:
the export that needed those three things is the one that scored worse.

**The honest error figure is ±0.85 days, not 0.15.** The 0.147-day MAE is measured on
frames from the same 11-day run the model trained on, and that split is largely
duplicate retrieval — a 1-nearest-neighbour lookup on *untrained* ImageNet features
scores 0.081 MAE on it. Every item a user tracks is a batch the model has never seen,
so the website shows the leave-one-day-out error instead. The model was trained on one
food, one camera and one room; treat its output elsewhere with that in mind.

### Why a photograph-only model, and why an ensemble

`ml/rsl_multimodal/scripts/experiments.py` scores candidates by leave-one-day-out, the
only protocol this dataset supports honestly. Over six training seeds:

| | LODO MAE (days) | Within-session MAE |
|---|---|---|
| sensor branch | 2.392 ± 0.106 | 0.720 |
| fusion head (served until now) | 1.113 ± 0.179 | 0.266 |
| image head, one seed | 0.920 ± 0.086 | 0.220 |
| **image head, 6-seed ensemble** | **0.848** | **0.147** |

Two findings. The image branch beats fusion on an unseen day — it wins on 4 of 6 seeds,
mean margin 0.193 days — because the sensor branch learned the ambient conditions of the
one 11-day run it saw, and the fusion head leans on it. And averaging several heads beats
any single head, because head training is the largest source of run-to-run variance here
(±0.086 is as large as the gaps between whole architectures). Every other variant tried —
sensor dropout, image augmentation, a dynamics-only feature set — made things worse; the
numbers are in `results/experiments.json`.

Rebuild and re-measure with:

```bash
.venv-model/bin/python scripts/train_image_ensemble.py --lodo   # the honest figure
.venv-model/bin/python scripts/train_image_ensemble.py          # train + export
```

The freshness class is *derived* from the predicted days (days ÷ 11), not predicted
separately.

### The baseline estimator (fallback)

Shelf life at a given temperature uses the standard Q10 formulation:

```
life(T) = reference_life × Q10 ^ ((T_ref − T) / 10)
```

then multiplied by three correction factors:

* **Humidity** — deviation outside the produce’s ideal band. Too damp is
  penalised harder than too dry, because microbial growth dominates spoilage for
  the produce this project targets.
* **Chilling injury** — for cold-sensitive produce (tomato, banana, mango,
  cucumber, capsicum), cooling below the chilling threshold buys *no* further
  slowdown and causes damage. The rate model is held at the threshold and an
  injury penalty is applied, so the app never claims a tomato lasts a month in
  a fridge that is actively harming it.
* **Spoilage gas** — applied only when an optional MQ-135 style reading is
  present. The reference build (BMP280 + SHT31) has no gas sensor, so this is
  normally neutral.

Degradation is then **integrated across the stored environment history** rather
than assuming the current reading held the whole time. An item left warm for two
days and then refrigerated is charged for the warm days:

| Scenario (tomato, 72 h stored, now at 12 °C) | Life consumed | Remaining |
|---|---|---|
| 48 h at 32 °C, then 24 h at 12 °C | 89% | ~1.6 days |
| 72 h at 12 °C throughout | 21% | ~11 days |

### Produce scope

The project covers **banana and tomato only**, matching the trained model's
training set. `SUPPORTED_FOOD_KEYS` in [`shelflife/db.py`](shelflife/db.py) is
the single source of truth; the add-item form, the API and the seed data all
follow it.

On startup, produce profiles outside that set are pruned — but **only if no item
references them**, so narrowing the scope can never silently delete somebody's
tracked item. Anything still in use is left alone and reported.

### Sensors used

Temperature, humidity and gas. The BMP280's **pressure** reading is still accepted
and stored when hardware sends it, but it is not displayed anywhere and not sent to
the browser, because neither estimator uses it.

Gas is used by **neither** estimator as it now stands. The project's gas sensor reports
raw, uncalibrated values (about 150–195 across the whole training run), so the kinetic
baseline ignores it rather than guess a ppm scale, and the served image ensemble takes no
sensor input at all. Devices still send it as `gas_ppm`, in the same units the training
rig recorded: it is stored, charted, and read by the fusion export if you switch to it.

### Where the temperature on the page comes from

Three sources, in preference order. The page always says which one it is showing, and a
real measurement is never labelled as simulated:

| Mode | What it is | Shown as |
|---|---|---|
| `live` | a Raspberry Pi on the shelf reported within `DEVICE_ONLINE_SECONDS` | **Live sensor** |
| `observed` | the real current conditions for `WEATHER_PLACE`, from Open-Meteo | **Local weather · _place_** |
| `simulated` | `sensors.synthesise` — a deterministic model of a Bengaluru kitchen | **Simulated data** |

`SENSOR_SOURCE=auto` (the default) walks down that list. `observed` exists because the
simulator is a *model*, not a measurement: on a hot afternoon it sat several degrees from
the real thing, so with no hardware attached the pages were showing a plausible number
rather than the actual environment. It needs no API key, uses only the standard library,
caches one fetch per `WEATHER_TTL_SECONDS` across all request threads, and times out in
`WEATHER_TIMEOUT_SECONDS` — a failure falls back to the simulator, labelled, and is shown
as *Unreachable* on the developer console's system page. Set `WEATHER_ENABLED=false` to
keep the machine fully offline.

**What `observed` is not.** It is outdoor air for a whole locality, measured nowhere near
the food. It is a reasonable stand-in for fruit on a counter in an unconditioned room and
the wrong number for a closed fridge — so **enclosed storage keeps its nominal profile**
(`services.STORAGE_PROFILES`: fridge 4 °C, freezer −18 °C) and never uses it. A weather
station also has no gas channel, so `gas_ppm` stays null rather than being invented. The
only way to get the real temperature *beside the food* is to link a device to the item.

Reference shelf-life and Q10 values come from standard postharvest storage
guidance and live in `FOOD_TYPE_SEED` in [`shelflife/db.py`](shelflife/db.py).
They are order-of-magnitude reference values, tunable per produce type, and they
are the *baseline to beat* once the multimodal model is trained.

---

## Quick start

### Monitoring interface only (no trained model)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app.py                  # http://127.0.0.1:5000
```

Runs on any Python 3.10+. Serves the Q10 kinetic baseline and says so.

### With the trained shelf-life model

> **TensorFlow publishes no wheel for Python 3.14.** Serve the model from a Python
> 3.12 environment; on 3.14 the Model card page reports that the runtime is missing
> and the site keeps working on the baseline.

```bash
python3.12 -m venv .venv-model
.venv-model/bin/pip install -r requirements-model.txt
sh tools/serve.sh                    # http://127.0.0.1:8000 and your LAN address
```

The model is already in the repository, in `ml/rsl_multimodal/` — see
[The model folder](#the-model-folder). Confirm what loaded:

```bash
.venv-model/bin/python -m flask --app app model-check
```

The database is created and migrated automatically on first start. To create a
local account without going through the sign-up form:

```bash
python tools/dev_login.py you@example.com yourpassword123
flask --app app seed-demo --email you@example.com    # optional sample items
```

### Running the tests

```bash
python -m unittest discover -s tests -t .                  # fast, baseline only
.venv-model/bin/python -m unittest discover -s tests -t .  # everything
```

Standard library `unittest` only. Without TensorFlow, the model tests skip.

With it, they cover three layers:

* a **fixture** TFLite model with the real model's exact input contract
  (`tools/make_rsl_fixture.py`), bundled with the real `pi_features.py` and
  `image_io.py`, driving the whole website path — device readings, photo capture,
  fallback rules, playground sessions, evaluation;
* the **real bundle**, checking the published envelope, error figures and gates;
* the **real accuracy**: the website scores all 271 held-out frames and must match the
  published test MAE (skipped if `data_images/` has not been copied in).

---

## Architecture

```
                  ┌──────────────────────────────┐
   SHT31   ───┐   │  Raspberry Pi 5              │
   Gas     ───┼──▶│  tools/pi_agent.py           │
   Camera  ───┘   │  readings every 2 min, photos │
                  └──────────────┬───────────────┘
                                 │ HTTP + bearer token
                                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Flask application (shelflife/)                          │
   │                                                          │
   │   blueprints/device.py  ── ingest readings & captures     │
   │   blueprints/api.py     ── browser JSON API               │
   │   blueprints/auth.py    ── sessions, CSRF, throttling     │
   │   blueprints/pages.py   ── user interface (any account)   │
   │   blueprints/admin.py   ── developer console (admin only) │
   │                                                          │
   │   services.py   ── items, alerts, snapshots               │
   │   inference.py  ── registry: trained model │ Q10 baseline │
   │   rsl_model.py  ── TFLite RSL model, served in-process     │
   │   dataset.py    ── held-out split + regression metrics    │
   │   sensors.py    ── validation + labelled simulator        │
   │   db.py         ── SQLite schema, migrations, seed data   │
   └───────────────┬───────────────────────────┬─────────────┘
                   ▼                           ▼
           SQLite (database.db)      ml/rsl_multimodal/
                                     model, scaler, pi_features.py,
                                     image_io.py, results
```

### Files

| Path | Purpose |
|---|---|
| `app.py` | Entry point |
| `shelflife/config.py` | Environment-driven config; refuses to start prod without `SECRET_KEY` |
| `shelflife/db.py` | Schema, forward-only migrations, produce reference data |
| `shelflife/inference.py` | Registry that picks the estimator that can serve each request |
| `shelflife/rsl_model.py` | Loads and runs the trained model from its bundle |
| `ml/rsl_multimodal/` | The trained model, its training pipeline, notebooks and report |
| `shelflife/dataset.py` | Held-out split reader and MAE/RMSE/R² |
| `shelflife/blueprints/admin.py` | Developer console, admin-gated |
| `shelflife/sensors.py` | Reading validation and the deterministic simulator |
| `shelflife/services.py` | Items, predictions, alerting, dashboard snapshot |
| `shelflife/security.py` | CSRF, sessions, device tokens, rate limiting |
| `templates/`, `static/` | UI — no build step, no CDN |
| `tools/pi_agent.py` | Raspberry Pi sensor agent |
| `tools/make_rsl_fixture.py` | Builds a contract-identical stand-in bundle for tests |
| `tools/serve.sh` | Serves the site on your network with waitress |
| `tests/` | 264 unit and integration tests |

---

## Hosting it on your network

For a lab demo or viva, serve it to every device on the same Wi-Fi:

```bash
sh tools/serve.sh
```

That reads `.env`, starts **waitress** (a real WSGI server, not Flask's dev
server) on `0.0.0.0:8000`, and prints both the loopback and LAN URLs. Use
`VENV=.venv sh tools/serve.sh` to run the baseline-only environment instead.

`.env` holds the generated `SECRET_KEY` and is gitignored — never commit it.
Sessions survive a restart because the key is persisted.

Because the site is served over plain **HTTP** on the LAN, `.env` sets
`SESSION_COOKIE_SECURE=false` and `FORCE_HTTPS=false`; otherwise the browser
would refuse to send the session cookie and every request would be redirected to
a `https://` URL that nothing is listening on. Session cookies therefore travel
unencrypted on that network. Set both back to `true` the moment the site sits
behind TLS.

If another device cannot reach it:

* macOS firewall — allow incoming connections for the Python interpreter, or
  turn the firewall off for the demo.
* Campus and guest Wi-Fi frequently enable **client isolation**, which blocks
  device-to-device traffic entirely. A phone hotspot with both devices joined is
  the usual way around it.
* Confirm the address is current: `ipconfig getifaddr en0`. DHCP leases change.

## Raspberry Pi deployment

### 1. Serve the app

```bash
export APP_ENV=raspi-prod
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export SENSOR_SOURCE=auto

pip install waitress
waitress-serve --host 0.0.0.0 --port 8000 --call "shelflife:create_app"
```

`raspi-prod` will not start without an explicit `SECRET_KEY`, and it enables
secure cookies and HSTS. Put it behind nginx or Caddy for TLS.

As a systemd unit:

```ini
[Unit]
Description=Shelf-Life Monitor
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/capstone_project
Environment=APP_ENV=raspi-prod
EnvironmentFile=/home/pi/capstone_project/.env
ExecStart=/home/pi/capstone_project/.venv/bin/waitress-serve \
          --host 0.0.0.0 --port 8000 --call "shelflife:create_app"
Restart=always

[Install]
WantedBy=multi-user.target
```

### 2. Register the device

Open **Device & Model → Register device**. The token is displayed exactly once —
only its SHA-256 hash is stored.

### 3. Run the sensor agent

```bash
pip install adafruit-circuitpython-bmp280 adafruit-circuitpython-sht31d
export SHELFLIFE_URL=http://<server-address>:8000
export SHELFLIFE_TOKEN=slp_...

# for the trained model:
export SHELFLIFE_GAS_COMMAND="python3 /home/pi/read_gas.py"   # prints one number
export SHELFLIFE_ITEM_ID=7                                      # item to photograph
python3 tools/pi_agent.py
```

The agent does not guess how to read your gas sensor. `SHELFLIFE_GAS_COMMAND` must
print the value **in the same units your training rig recorded** — about 150–195 in
the training run. A differently-scaled value lands every reading outside the model's
envelope, and the website will (correctly) fall back to the baseline.

Photos are taken every 30 minutes with `rpicam-still` or `libcamera-still` (override
with `SHELFLIFE_CAPTURE_COMMAND`, using `{path}` for the output file).

The agent reads the SHT31 for temperature and humidity and the BMP280 for
pressure, buffers to disk when the network drops, and flushes the buffer as a
batch when it returns. **If no sensor is detected it exits rather than inventing
readings.**

Housekeeping:

```bash
flask --app app prune-readings --days 30   # trim stored readings
flask --app app model-check                # print the loaded model contract
flask --app app list-users                 # accounts and their roles
flask --app app set-role --email x --role admin
```

---

## Data sources and honest labelling

The dashboard always says where its numbers came from:

* **Live sensor** — a registered device reported within `DEVICE_ONLINE_SECONDS`.
* **Simulated data** — no hardware connected, so the built-in simulator is used.
* **Sensor offline** — `SENSOR_SOURCE=live` and the hardware has gone quiet.

Simulated readings are a **deterministic function of their timestamp**, so the
same time window always renders the same trace instead of re-randomising on
every poll. They are **never written to the `readings` table**, which therefore
only ever counts genuine hardware samples — “Live readings” on the profile page
cannot be inflated by a demo.

Predictions never mix real and synthetic environment history within one storage
window. If real readings cover part of it, only those are used, and the
uncovered time is charged at the current condition rather than treated as free.

---

## HTTP API

All browser endpoints return `{"ok": true, "data": {...}}` or
`{"ok": false, "error": {"code": "...", "message": "..."}}`.

### Browser API (session cookie + `X-CSRF-Token`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/dashboard` | Full dashboard snapshot (`?refresh=1` forces re-inference) |
| GET | `/api/v1/readings/latest` | Newest environment reading |
| GET | `/api/v1/readings/history` | `?hours=&points=` down-sampled trend |
| GET/POST | `/api/v1/items` | List / create tracked items |
| GET/PATCH/DELETE | `/api/v1/items/<id>` | Read, update, remove |
| POST | `/api/v1/items/<id>/predict` | Run inference now |
| POST | `/api/v1/uploads` | Upload an item photo |
| GET | `/api/v1/alerts` | Alert feed (`?open=1` for unread) |
| POST | `/api/v1/alerts/<id>/acknowledge` | Mark read |
| GET/PUT | `/api/v1/settings` | Notification preferences |
| GET/POST/DELETE | `/api/v1/devices` | Manage edge devices |
| GET | `/api/v1/system` | Model and runtime status |

### Admin API (session cookie + `X-CSRF-Token`, admin role required)

Every one of these returns **403** for a `user` account.

| Method | Path | Purpose |
|---|---|---|
| POST | `/admin/api/predict` | One playground reading: `image`, `temperature`, `humidity`, `gas`, `session`, `minutes_since_previous`, `reset` |
| POST | `/admin/api/playground/reset` | Clear a playground session |
| GET | `/admin/api/test-samples` | List the held-out rows |
| GET | `/admin/api/test-samples/<i>/image` | Serve a held-out image, addressed by index |
| POST | `/admin/api/test-samples/<i>/predict` | Score one held-out row |
| POST | `/admin/api/evaluate` | MAE / RMSE / R² over the split |
| POST | `/admin/api/reload-model` | Re-read the artifacts from disk |
| PATCH | `/admin/api/users/<id>/role` | Promote or demote an account |
| PATCH | `/admin/api/users/<id>/active` | Enable or disable an account |
| POST | `/admin/api/prune-readings` | Delete readings older than N days |

### Device API (`Authorization: Bearer <token>`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/device/v1/config` | Verify the token, learn the reporting interval |
| POST | `/api/device/v1/readings` | One reading |
| POST | `/api/device/v1/readings/batch` | Up to 500 buffered readings |
| POST | `/api/device/v1/captures` | Camera frame (multipart: `image`, `item_id`, optional `captured_at`) |

```bash
curl -X POST http://<host>:8000/api/device/v1/readings \
  -H "Authorization: Bearer slp_..." -H "Content-Type: application/json" \
  -d '{"temperature_c": 28.7, "humidity_pct": 58.0, "gas_ppm": 185}'
```

Readings are range-checked against the BMP280 and SHT31 datasheet limits.
Timestamps more than five minutes in the future are rejected, so a Pi with an
unsynced clock cannot poison the charts. In a batch, valid rows are stored even
when some rows are rejected, so one bad sample does not cost the device its whole
buffer.

---

## The model folder

`ml/rsl_multimodal/` is the complete model project — training scripts, notebooks,
report, results and the exported model — copied in from
`~/Downloads/CAPSTONE DATASET/RSL_Multimodal_Model`. The website reads from it
directly, so the code that must agree with the model always ships with it:

```
ml/rsl_multimodal/
├── models/image_ensemble_blocked_full_float16.tflite   the served model (4.1 MB)
├── models/fusion_blocked_full_float16.tflite   the older two-input export (2.3 MB)
├── artifacts/sensor_scaler.pkl                 scaler + the 27 feature names
├── artifacts/preprocessing_config.json         outlier bounds, input envelope
├── artifacts/master_dataset.csv                held-out split for the admin console
├── scripts/pi_features.py                      builds the 27 sensor features
├── scripts/image_io.py                         decodes photos exactly as training did
├── results/                                    published metrics and gate results
├── data_images/                                training frames (not in git)
└── README.md, REPORT.md, QUICKSTART.md, INTEGRATION.md
```

**After retraining**, run the pipeline inside that folder (`bash run_all.sh`), then press
**Reload artifacts** in the developer console — no restart needed. `run_all.sh` ends
with two gates: `verify_pi_features.py` and `verify_serving_transform.py`. Don't serve a
model that fails either.

**What is in git.** The code, notebooks, docs, results, scaler, configs, dataset CSVs and
the float16 export — about 8 MB, enough to run the site and reproduce the report. The
3,067 training images, the `.keras` checkpoints, the other TFLite variants and the
backbone cache are ignored; copy `data_images/` in by hand to use the evaluation page.

### Fixes made to the model project

Serving the model through the website turned up problems that raised no errors but
changed predictions. They are fixed in `ml/rsl_multimodal`, and documented in its
README and REPORT:

| Problem | Effect | Fix |
|---|---|---|
| Serving decoded JPEGs with PIL; training used `tf.io.decode_jpeg` | test MAE 0.264 → 0.278 | `image_io.py`: TF-identical decoder |
| Camera frames resized with BILINEAR; training used LANCZOS + JPEG q95 | camera-frame MAE 0.152 → 0.207 | `image_io.py` repeats the training transform; 60/60 rebuilt bit-for-bit |
| An out-of-range first reading passed through while the API said it was clamped | out-of-distribution input | clipped to the outlier bound; warnings say what happened |
| No flag for readings outside anything seen in training | silent extrapolation (−7.4 d) | `input_envelope` in the config; `UNRELIABLE` warnings |
| `scikit-learn` unpinned; scaler pickled with 1.8.0 | version warning on a fresh install | pinned |
| Missing `session_id` silently shared one session; failed requests advanced history | batches mixed | warned; rolled back on failure |

### How the website runs it

| | |
|---|---|
| Model | `image_ensemble_blocked_full_float16.tflite`: image (1, 224, 224, 3) → days, 6 heads averaged inside the graph |
| Runtime | `tflite-runtime`, LiteRT or TensorFlow, whichever is installed |
| Photo | RGB → LANCZOS 224×224 → JPEG q95 → TF-identical decode (skipped if already 224×224 JPEG) |
| Sensors | not an input. They are still recorded, charted, and used by the baseline |
| Output | clamped to 0–11 days; raw value kept for the developer console |
| Band | ± leave-one-day-out MAE (0.85 days) |
| Loading | validated on start — the graph's own input contract decides whether the sensor half of the bundle is read at all; failure degrades to the baseline |

The two-input fusion export is still in the bundle and still supported: set
`RSL_MODEL_FILE=models/fusion_blocked_full_float16.tflite` and the loader reads the
scaler and feature builder again, re-imposes the gas and envelope preconditions, and
quotes that model's own 1.17-day band. Nothing else changes.

Devices are told to report every 120 s, the cadence the model was trained on. Photos
come from `POST /api/device/v1/captures`, which also links the item to that device.

### The developer console

| Page | What it does |
|---|---|
| **Console** | Model state, where predictions came from |
| **Model card** | Bundle files, input contract, operating envelope, leave-one-day-out table, TFLite variants, the 27 features, gate results |
| **Playground** | A photo plus one reading at a time, in named sessions that build history like a real batch |
| **Evaluation** | Scores the blocked test split from stored features; confirms the published MAE is reproduced |
| **Accounts** | Roles and account activation |
| **System** | Runtime, paths, thresholds, reading pruning |

Everything is inference-only. Held-out images are served **by index**, never by path.

## Security

* Passwords hashed with Werkzeug PBKDF2; a minimum length, a common-password
  list and a digits-only check are enforced server-side.
* Session cookies are HttpOnly, SameSite=Lax, secure under `raspi-prod`, and the
  session is rotated on sign-in to defeat fixation.
* CSRF tokens on every state-changing request. The device blueprint is exempt by
  prefix — it is bearer-authenticated and cookie-free — so a new device endpoint
  cannot silently break the Pi.
* Sign-in is throttled per identifier and client address; sign-out is POST-only.
* Every query is scoped by `user_id`, so one account cannot read or modify
  another's items, alerts or devices. There are dedicated isolation tests.
* Uploads are extension-checked *and* content-sniffed, and stored under a
  generated filename, so a renamed script cannot be saved as an image.
* Content-Security-Policy is strict (`script-src 'self'`, `frame-ancestors
  'none'`) because no asset comes from a CDN.
* `?next=` redirects are restricted to same-site relative paths.
* The developer console is gated on the account role at every route, not hidden
  by navigation alone; `/admin/api/...` returns JSON 403 rather than an HTML
  page. Held-out dataset images are addressed by index, never by path.
* The last active administrator cannot be demoted or deactivated.

Known limitation: rate limiting is in-process, which suits the single-process
edge deployment this project targets. A multi-worker deployment would need a
shared store.

---

## Development notes

* **No build step and no CDN.** The stylesheet is hand-written and the charts are
  a small SVG renderer in `static/js/charts.js`, because the Pi may have no
  internet and a broken CDN link would take the dashboard down mid-demo.
* Pages are server-rendered first, then enhanced by JavaScript, so the dashboard
  shows real content before any script runs.
* Polling backs off after failures and pauses while the tab is hidden.
* Timestamps are stored as ISO-8601 UTC strings and rendered in the viewer's
  local timezone.

```bash
sh tools/check_js.sh     # syntax-check every ES module
```

---

## Project context

Capstone project, PES University, Bengaluru — *Shelf-Life Prediction of Food
Items using Edge-AI*. This repository is the user-facing monitoring and alerting
layer of that system; the model-training work lives alongside it.
