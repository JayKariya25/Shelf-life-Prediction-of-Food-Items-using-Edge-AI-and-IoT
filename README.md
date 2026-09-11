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

The app runs **whichever estimator can actually serve the request**, and always
says which one it used.

| | Status |
|---|---|
| Multimodal remaining-shelf-life regression (image + T/H/Gas) | **Working** — when the trained model is installed |
| Temperature / humidity driven estimation | **Working** — Q10 kinetic baseline, always available |
| Degradation integrated over the conditions actually experienced | **Working** (baseline path) |
| Calibrated class confidence | **Not available** — the trained model is a single-output regressor |
| Uncertainty interval | Only when `config.json` publishes a `test_mae` |

Every prediction is tagged `model_kind` — `trained-model` or
`heuristic-baseline` — and the tag drives the banners on the dashboard and item
pages. Nothing in the UI implies a trained model is running when it is not.

### When the trained model is used, and when it is not

The trained model requires **an image and all three sensor values**
(Temperature, Humidity, Gas). If either is missing for a given item, the app
falls back to the baseline and records *why*, which is shown on the item page:

> Estimated with the kinetic baseline because this item has no photograph on file
> and no gas reading is available.

This matters for the reference hardware: **the BMP280 + SHT31 build has no gas
sensor**, so live monitoring falls back to the baseline until an MQ-135 (or
equivalent) reports. The admin playground and evaluation pages supply all three
inputs directly, so the trained model runs there regardless.

The freshness class shown in the user interface is *derived* from the predicted
remaining days, not predicted separately — the rationale on each item says so.

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

Temperature and humidity only. The BMP280's **pressure** reading is still
accepted and stored when hardware sends it, but it is not displayed anywhere and
not sent to the browser, because nothing in the model or the baseline uses it.
Gas is optional and only needed by the trained model.

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

### With the trained multimodal model

> **TensorFlow publishes no wheel for Python 3.14.** If your main virtualenv is
> on 3.14 the Keras model cannot load there — the Model card page will tell you
> exactly that. Use a Python 3.12 environment instead.

```bash
python3.12 -m venv .venv-model
.venv-model/bin/pip install -r requirements-model.txt

# drop the artifacts in place (see models/README.md)
#   models/best_multimodal_model.keras
#   artifacts/sensor_scaler.pkl
#   artifacts/config.json
#   artifacts/dataset_splits.csv      (admin evaluation page)
#   data/Images/                      (images named by the splits file)

.venv-model/bin/python app.py
```

Confirm what loaded:

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
python -m unittest discover -s tests -t .
```

264 tests, standard library only — no test framework to install.

The 32 tests that exercise the real Keras path skip automatically when the model
runtime is absent. To run them, use the model environment:

```bash
.venv-model/bin/python -m unittest discover -s tests -t .
```

They build a throwaway Keras model with the *same input contract* as the trained
export and drive the whole path — load, signature validation, preprocessing,
inference, evaluation and fallback — so the integration is covered without
committing a large binary to the repository. See `tools/make_fixture_model.py`.

---

## Architecture

```
                  ┌──────────────────────────────┐
   BMP280  ───┐   │  Raspberry Pi 5              │
   SHT31   ───┼──▶│  tools/pi_agent.py           │
   Camera  ───┘   │  reads I2C, buffers, uploads │
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
   │   multimodal.py ── Keras adapter (image + T/H/Gas)        │
   │   dataset.py    ── held-out split + regression metrics    │
   │   sensors.py    ── validation + labelled simulator        │
   │   db.py         ── SQLite schema, migrations, seed data   │
   └─────────────────────────────┬────────────────────────────┘
                                 ▼
                         SQLite (database.db)
```

### Files

| Path | Purpose |
|---|---|
| `app.py` | Entry point |
| `shelflife/config.py` | Environment-driven config; refuses to start prod without `SECRET_KEY` |
| `shelflife/db.py` | Schema, forward-only migrations, produce reference data |
| `shelflife/inference.py` | Registry that picks the estimator that can serve each request |
| `shelflife/multimodal.py` | Trained Keras model adapter (loading, validation, inference) |
| `shelflife/dataset.py` | Held-out split reader and MAE/RMSE/R² |
| `shelflife/blueprints/admin.py` | Developer console, admin-gated |
| `shelflife/sensors.py` | Reading validation and the deterministic simulator |
| `shelflife/services.py` | Items, predictions, alerting, dashboard snapshot |
| `shelflife/security.py` | CSRF, sessions, device tokens, rate limiting |
| `templates/`, `static/` | UI — no build step, no CDN |
| `tools/pi_agent.py` | Raspberry Pi sensor agent |
| `tools/make_fixture_model.py` | Builds a contract-identical stand-in model for tests |
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
export SHELFLIFE_URL=http://<pi-address>:8000
export SHELFLIFE_TOKEN=slp_...
python3 tools/pi_agent.py
```

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
| POST | `/admin/api/predict` | Manual inference on an uploaded image (multipart) |
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
| POST | `/api/device/v1/captures` | Camera frame (multipart) |

```bash
curl -X POST http://<host>:8000/api/device/v1/readings \
  -H "Authorization: Bearer slp_..." -H "Content-Type: application/json" \
  -d '{"temperature_c": 24.8, "humidity_pct": 61.2, "pressure_hpa": 1008.1}'
```

Readings are range-checked against the BMP280 and SHT31 datasheet limits.
Timestamps more than five minutes in the future are rejected, so a Pi with an
unsynced clock cannot poison the charts. In a batch, valid rows are stored even
when some rows are rejected, so one bad sample does not cost the device its whole
buffer.

---

## The trained multimodal model

### Inference contract

The adapter reproduces the project's Streamlit demo exactly, and **validates it
on load** rather than trusting it:

```
inputs   image  (None, 224, 224, 3)   named "image"
         sensor (None, 3)             named "sensor"
output   (None, 1)                    remaining shelf life, in days

image    RGB -> resize(224, 224, BILINEAR) -> mobilenet_v3.preprocess_input
sensor   [Temperature, Humidity, Gas] -> fitted StandardScaler.transform -> float32
```

A model whose input names, shapes or output rank differ is **refused**, and the
reason is displayed on the Model card page. `config.json` is checked for the
sensor column order; a permuted order is rejected rather than silently producing
wrong numbers. Day is never an input.

Loading is lazy and failure-tolerant: a missing artifact, an absent TensorFlow,
or an exception mid-inference all degrade to the kinetic baseline with the cause
recorded, instead of taking the site down.

### The developer console

| Page | What it does |
|---|---|
| **Console** | Model state, artifact checklist, where predictions came from |
| **Model card** | Full input contract, published metrics, load errors |
| **Playground** | Upload any image, set T/H/Gas, run one forward pass |
| **Evaluation** | Score the held-out split: MAE, RMSE, R², bias, worst error, predicted-vs-actual scatter |
| **Accounts** | Roles and account activation |
| **System** | Runtime paths, storage counts, reading pruning |

The evaluation page reports its own measured metrics **next to** the ones
published in `config.json`, with the delta, so drift between the training run
and this environment is visible rather than assumed away.

Everything is inference-only. The console never retrains, refits the scaler, or
writes to the split metadata. Held-out images are served **by index** — the
client never supplies a path — so the endpoint cannot become an arbitrary file
read.

`POST /admin/api/reload-model` re-reads the artifacts from disk, so a new export
can be swapped in without restarting the server.

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
