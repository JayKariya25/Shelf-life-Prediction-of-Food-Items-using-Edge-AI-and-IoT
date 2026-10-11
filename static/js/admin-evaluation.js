/* Held-out evaluation: browse test samples, run one, or score the whole split. */
import { api, escapeHtml, formatNumber, readJsonScript, toast } from "./core.js";

const config = readJsonScript("evalConfig", { available: false, modelReady: false, count: 0 });
const strip = document.getElementById("sampleStrip");
const detail = document.getElementById("sampleDetail");
const results = document.getElementById("evalResults");

let samples = [];
let selected = null;

/* ------------------------------------------------------------- browser --- */
async function loadSamples() {
    if (!config.available || !strip) return;
    try {
        const data = await api("/admin/api/test-samples");
        samples = data.samples;
        strip.innerHTML = samples
            .map(
                (sample) => `
                <button class="sample-thumb" data-index="${sample.index}"
                        title="${escapeHtml(sample.label)} - day ${sample.day}, ${sample.actual_days} days left">
                    ${sample.has_image
                        ? `<img src="/admin/api/test-samples/${sample.index}/image" alt="" loading="lazy">`
                        : '<span class="tiny muted">no image</span>'}
                    <span class="sample-thumb__label">${sample.actual_days}d</span>
                </button>`
            )
            .join("");
        if (samples.length) select(0);
    } catch (error) {
        strip.innerHTML = `<p class="small text-crit">${escapeHtml(error.message)}</p>`;
    }
}

function select(index) {
    selected = index;
    document.querySelectorAll(".sample-thumb").forEach((node) => {
        node.classList.toggle("is-selected", Number(node.dataset.index) === index);
    });
    renderSample();
}

function renderSample() {
    const sample = samples[selected];
    if (!sample || !detail) return;
    detail.innerHTML = `
        ${sample.has_image
            ? `<img src="/admin/api/test-samples/${sample.index}/image" alt="${escapeHtml(sample.label)}"
                    style="max-height:220px;border-radius:var(--radius);margin-bottom:0.9rem">`
            : '<p class="small text-crit">Image file is missing for this row.</p>'}
        <dl class="kv">
            <dt>File</dt><dd class="mono tiny">${escapeHtml(sample.label)}</dd>
            <dt>Captured</dt><dd class="tiny">${escapeHtml(sample.timestamp || "")} &middot; day ${sample.day}</dd>
            <dt>Temperature</dt><dd>${formatNumber(sample.temperature, 2)}</dd>
            <dt>Humidity</dt><dd>${formatNumber(sample.humidity, 2)}</dd>
            <dt>Gas</dt><dd>${formatNumber(sample.gas, 2)}</dd>
            <dt>Actual RSL</dt><dd>${formatNumber(sample.actual_days, 2)} days</dd>
        </dl>
        <button class="btn btn--primary btn--block" id="runSampleBtn" style="margin-top:0.9rem"
                ${config.modelReady && sample.has_image ? "" : "disabled"}>
            Run the model on this sample
        </button>
        <div id="sampleResult" style="margin-top:0.9rem"></div>`;

    document.getElementById("runSampleBtn")?.addEventListener("click", runSelected);
}

async function runSelected() {
    const button = document.getElementById("runSampleBtn");
    const target = document.getElementById("sampleResult");
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Running';
    try {
        const data = await api(`/admin/api/test-samples/${selected}/predict`, { method: "POST" });
        target.innerHTML = `
            <div class="grid grid--stats" style="gap:0.6rem">
                <div class="stat"><div class="stat__label">Actual</div>
                    <div class="stat__value" style="font-size:1.4rem">${formatNumber(data.actual_days, 2)}<small>d</small></div></div>
                <div class="stat"><div class="stat__label">Predicted</div>
                    <div class="stat__value" style="font-size:1.4rem">${formatNumber(data.predicted_days, 2)}<small>d</small></div></div>
                <div class="stat"><div class="stat__label">Abs. error</div>
                    <div class="stat__value" style="font-size:1.4rem">${formatNumber(data.absolute_error, 2)}<small>d</small></div></div>
            </div>
            <p class="tiny muted" style="margin-top:0.6rem">
                ${formatNumber(data.inference_ms, 1)} ms &middot; raw output
                ${formatNumber(data.predicted_days_raw, 4)} days
            </p>`;
    } catch (error) {
        target.innerHTML = `<div class="banner banner--crit"><div class="banner__body small">
            ${escapeHtml(error.message)}</div></div>`;
    } finally {
        button.disabled = false;
        button.textContent = "Run the model on this sample";
    }
}

strip?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-index]");
    if (button) select(Number(button.dataset.index));
});

document.getElementById("randomSampleBtn")?.addEventListener("click", () => {
    if (samples.length) select(Math.floor(Math.random() * samples.length));
});

/* ---------------------------------------------------------- aggregate --- */
function scatter(points) {
    if (!points.length) return "";
    const values = points.flatMap((p) => [p.actual, p.predicted]);
    const max = Math.max(...values, 1);
    const min = Math.min(...values, 0);
    const span = max - min || 1;
    const W = 320;
    const H = 320;
    const pad = 34;
    const scale = (v) => pad + ((v - min) / span) * (W - pad * 2);

    const dots = points
        .map(
            (p) =>
                `<circle class="scatter__point" cx="${scale(p.actual).toFixed(1)}" ` +
                `cy="${(H - scale(p.predicted)).toFixed(1)}" r="4" fill="var(--brand-500)"><title>` +
                `actual ${p.actual}d, predicted ${p.predicted}d</title></circle>`
        )
        .join("");

    return `
        <div class="scatter">
            <svg viewBox="0 0 ${W} ${H}" role="img"
                 aria-label="Predicted against actual remaining shelf life">
                <line class="scatter__ideal" x1="${pad}" y1="${H - pad}"
                      x2="${W - pad}" y2="${pad}"></line>
                <line class="chart__grid" x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}"></line>
                <line class="chart__grid" x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}"></line>
                <text class="chart__axis" x="${W / 2}" y="${H - 8}" text-anchor="middle">actual (days)</text>
                <text class="chart__axis" x="10" y="${H / 2}" text-anchor="middle"
                      transform="rotate(-90 10 ${H / 2})">predicted (days)</text>
                <text class="chart__axis" x="${pad}" y="${H - pad + 13}" text-anchor="middle">${min.toFixed(1)}</text>
                <text class="chart__axis" x="${W - pad}" y="${H - pad + 13}" text-anchor="middle">${max.toFixed(1)}</text>
                ${dots}
            </svg>
            <p class="tiny muted" style="text-align:center">
                The dashed line is a perfect prediction.
            </p>
        </div>`;
}

document.getElementById("runEvalBtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const label = button.innerHTML;
    const limit = Number(document.getElementById("evalLimit").value) || config.count;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Evaluating';
    results.innerHTML = '<p class="small muted">Running the model over the held-out split&hellip;</p>';

    try {
        const data = await api("/admin/api/evaluate", { method: "POST", body: { limit } });
        const m = data.metrics;
        const pub = data.published || {};
        const compare = (value, published) => {
            if (published === undefined || published === null || value === null) return "";
            const delta = value - published;
            return `<div class="stat__meta">published ${formatNumber(published, 4)} ` +
                   `(${delta >= 0 ? "+" : ""}${formatNumber(delta, 4)})</div>`;
        };
        const reproduced = data.complete && pub.mae != null && Math.abs(m.mae - pub.mae) < 5e-4;

        results.innerHTML = `
            ${data.complete
                ? (reproduced
                    ? `<div class="banner banner--ok" style="margin-bottom:0.9rem"><div class="banner__body small">
                         Reproduced the published ${escapeHtml(pub.variant || "")} test MAE on all ${m.n} rows.</div></div>`
                    : `<div class="banner banner--warn" style="margin-bottom:0.9rem"><div class="banner__body small">
                         The measured MAE differs from the published figure. Check that the served export
                         and the images match the ones the report was generated from.</div></div>`)
                : `<div class="banner banner--info" style="margin-bottom:0.9rem"><div class="banner__body small">
                     Partial run (${m.n} of ${config.count} rows), so the comparison with the published
                     figure is only indicative.</div></div>`}
            <div class="grid grid--stats">
                <div class="stat"><div class="stat__label">MAE</div>
                    <div class="stat__value">${formatNumber(m.mae, 4)}<small>days</small></div>
                    ${compare(m.mae, pub.mae)}</div>
                <div class="stat"><div class="stat__label">RMSE</div>
                    <div class="stat__value">${formatNumber(m.rmse, 4)}<small>days</small></div>
                    ${compare(m.rmse, pub.rmse)}</div>
                <div class="stat"><div class="stat__label">R&sup2;</div>
                    <div class="stat__value">${m.r2 === null ? "n/a" : formatNumber(m.r2, 4)}</div>
                    ${m.r2 === null ? '<div class="stat__meta">actuals have zero variance</div>' : compare(m.r2, pub.r2)}</div>
                <div class="stat"><div class="stat__label">Within 1 day</div>
                    <div class="stat__value">${formatNumber(m.within_one_day * 100, 1)}<small>%</small></div>
                    <div class="stat__meta">${formatNumber(m.within_half_day * 100, 1)}% within half a day</div></div>
            </div>
            <div class="grid grid--halves" style="margin-top:1rem">
                ${scatter(data.points)}
                <div>
                    <dl class="kv">
                        <dt>Rows scored</dt><dd>${m.n} (${data.skipped_count} skipped)</dd>
                        <dt>Mean bias</dt><dd>${formatNumber(m.bias, 3)} days</dd>
                        <dt>Worst error</dt><dd>${formatNumber(m.max_error, 3)} days</dd>
                        <dt>Mean inference</dt><dd>${formatNumber(data.mean_inference_ms, 2)} ms</dd>
                        <dt>New-batch error</dt><dd>${data.new_batch_mae_days != null ? formatNumber(data.new_batch_mae_days, 3) + " days" : "--"}</dd>
                    </dl>
                    ${data.skipped_count
                        ? `<div class="banner banner--warn" style="margin-top:0.9rem">
                             <div class="banner__body small">
                               ${data.skipped_count} row(s) skipped:
                               ${escapeHtml(data.skipped.map((s) => `#${s.index} ${s.reason}`).join("; "))}
                             </div></div>`
                        : ""}
                    <p class="tiny muted" style="margin-top:0.9rem">
                        A positive bias means the model over-predicts remaining shelf life,
                        which is the less safe direction for a spoilage warning.
                    </p>
                </div>
            </div>`;
    } catch (error) {
        results.innerHTML = `<div class="banner banner--crit"><div class="banner__body">
            ${escapeHtml(error.message)}</div></div>`;
        toast(error.message || "Evaluation failed.", "error");
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});

loadSamples();
