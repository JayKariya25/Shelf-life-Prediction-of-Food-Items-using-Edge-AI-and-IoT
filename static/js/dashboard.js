/* Dashboard: live environment tiles, trend chart and item cards. */
import {
    api, formatDuration, formatNumber, poll, readJsonScript, relativeTime, toast, escapeHtml,
} from "./core.js";
import { lineChart, sparkline, distributionBar, SERIES_COLORS } from "./charts.js";
import { wireAddItemForm } from "./item-form.js";

let chartRange = 6;
let historyCache = null;

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------- tiles ------ */
function renderReading(reading, mode) {
    const temp = $("tempValue");
    const humidity = $("humidityValue");

    if (temp) {
        temp.innerHTML =
            reading.temperature_c === null || reading.temperature_c === undefined
                ? "--"
                : `${formatNumber(reading.temperature_c, 1)}<small>&deg;C</small>`;
    }
    if (humidity) {
        humidity.innerHTML =
            reading.humidity_pct === null || reading.humidity_pct === undefined
                ? "--"
                : `${formatNumber(reading.humidity_pct, 0)}<small>%</small>`;
    }
    const meta = $("tempMeta");
    if (meta) {
        meta.textContent = reading.recorded_at
            ? `Updated ${relativeTime(reading.recorded_at)}`
            : "No reading";
    }

    const pill = $("sensorPill");
    if (pill) {
        const config = {
            live: ["ok", "Live sensor", true],
            offline: ["crit", "Sensor offline", false],
            simulated: ["warn", "Simulated data", false],
        }[mode] || ["muted", "Unknown source", false];
        pill.innerHTML =
            `<span class="pill pill--${config[0]}">` +
            `<span class="pill__dot${config[2] ? " pill__dot--pulse" : ""}"></span>${config[1]}</span>`;
    }

    const modeText = $("sensorModeText");
    if (modeText) modeText.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
}

/* ------------------------------------------------------------- charts ----- */
function renderCharts(history) {
    const points = history.points || [];
    const labels = points.map((point) =>
        new Date(point.recorded_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    );

    lineChart($("envChart"), {
        labels,
        height: 210,
        ariaLabel: "Temperature and relative humidity over time",
        emptyMessage:
            history.mode === "offline"
                ? "No device has reported yet. Connect the Raspberry Pi agent to see live data."
                : "No readings in this window yet.",
        series: [
            {
                key: "temperature", label: "Temperature", unit: " °C",
                values: points.map((p) => p.temperature_c), color: SERIES_COLORS.temperature,
            },
            {
                key: "humidity", label: "Humidity", unit: "%",
                values: points.map((p) => p.humidity_pct), color: SERIES_COLORS.humidity,
                fill: false, axis: "right",
            },
        ],
    });

    sparkline($("tempSpark"), points.map((p) => p.temperature_c), SERIES_COLORS.temperature);
    sparkline($("humiditySpark"), points.map((p) => p.humidity_pct), SERIES_COLORS.humidity);
}

async function loadHistory() {
    const points = chartRange > 48 ? 56 : 48;
    historyCache = await api(`/api/v1/readings/history?hours=${chartRange}&points=${points}`);
    renderCharts(historyCache);
}

/* -------------------------------------------------------------- items ---- */
function toneFor(level) {
    return { critical: "crit", warning: "warn", ok: "ok" }[level] || "muted";
}

function renderItemCards(items) {
    const grid = $("itemGrid");
    if (!grid) return;
    items.forEach((item) => {
        const card = grid.querySelector(`[data-item-id="${item.id}"]`);
        if (!card) return;
        const tone = toneFor(item.alert_level);
        card.className = `item-card item-card--${tone}`;

        const remaining = card.querySelector('[data-field="remaining"]');
        if (remaining) remaining.textContent = item.remaining_display;

        const meter = card.querySelector(".meter__fill");
        if (meter && item.remaining_hours !== null) {
            const total = item.remaining_hours + item.hours_stored;
            meter.style.width = `${total > 0 ? (item.remaining_hours / total) * 100 : 0}%`;
            meter.className = `meter__fill meter__fill--${tone}`;
        }

        const pill = card.querySelector(".item-card__status .pill");
        if (pill) {
            pill.className = `pill pill--${tone}`;
            pill.innerHTML =
                `<span class="pill__dot${item.alert_level === "critical" ? " pill__dot--pulse" : ""}"></span>` +
                escapeHtml(item.freshness_class);
        }
    });
}

function renderAtRisk(atRisk) {
    const list = $("atRiskList");
    const count = $("atRiskCount");
    if (count) count.textContent = atRisk.length;
    if (!list) return;

    if (!atRisk.length) {
        list.innerHTML = `
            <div class="card__body">
                <div class="row row--tight text-ok">
                    <span class="small">Everything is comfortably within its estimated shelf life.</span>
                </div>
            </div>`;
        return;
    }

    list.innerHTML = atRisk
        .map((item) => {
            const tone = item.alert_level === "critical" ? "crit" : "warning";
            return `
            <div class="alert-row">
                <div class="alert-row__icon alert-row__icon--${tone}">!</div>
                <div class="alert-row__body">
                    <div class="alert-row__title">${escapeHtml(item.food_emoji)} ${escapeHtml(item.label)}</div>
                    <div class="alert-row__msg">${escapeHtml(item.remaining_display)} &middot; ${escapeHtml(item.freshness_class)}</div>
                </div>
                <a class="btn btn--subtle btn--sm" href="/items/${item.id}">Open</a>
            </div>`;
        })
        .join("");
}

function renderFreshnessMix(counts) {
    const container = $("freshnessBar");
    if (!container) return;
    distributionBar(container, [
        { label: "Fresh", value: counts["Fresh"] || 0, tone: "ok" },
        { label: "Moderately fresh", value: counts["Moderately Fresh"] || 0, tone: "warn" },
        { label: "Spoiled", value: counts["Spoiled"] || 0, tone: "crit" },
    ]);
}

function renderConditions(items) {
    const container = $("conditionList");
    if (!container) return;
    const flagged = items.filter((item) => item.condition_flags && item.condition_flags.length);

    if (!flagged.length) {
        container.innerHTML =
            '<p class="small muted">Every tracked item is inside its ideal temperature and humidity band.</p>';
        return;
    }
    container.innerHTML = flagged
        .slice(0, 6)
        .map(
            (item) => `
            <div>
                <div class="small strong">${escapeHtml(item.food_emoji)} ${escapeHtml(item.label)}</div>
                ${item.condition_flags
                    .map((flag) => `<div class="tiny text-warn">${escapeHtml(flag)}</div>`)
                    .join("")}
            </div>`
        )
        .join("");
}

function renderCounts(snapshot) {
    const total = $("itemCount");
    if (total) total.textContent = snapshot.items.length;
    const meta = $("itemCountMeta");
    if (meta) {
        meta.innerHTML =
            `<span class="text-crit">${snapshot.counts.critical} urgent</span> &middot; ` +
            `<span class="text-warn">${snapshot.counts.warning} soon</span> &middot; ` +
            `<span class="text-ok">${snapshot.counts.ok} fine</span>`;
    }
    const badge = $("alertBadge");
    if (badge) {
        const open = snapshot.open_alerts ? snapshot.open_alerts.length : 0;
        badge.hidden = open === 0;
        badge.textContent = open > 9 ? "9+" : String(open);
    }
}

function render(snapshot) {
    renderReading(snapshot.reading, snapshot.sensor_mode);
    renderCounts(snapshot);
    renderItemCards(snapshot.items);
    renderAtRisk(snapshot.at_risk);
    renderFreshnessMix(snapshot.freshness_counts);
    renderConditions(snapshot.items);
}

/* ---------------------------------------------------------------- boot ---- */
const initial = readJsonScript("snapshotData");
if (initial) render(initial);

loadHistory().catch(() => renderCharts({ points: [], mode: "offline" }));

const dashboardPoll = poll(async () => {
    render(await api("/api/v1/dashboard"));
}, 15000, { immediate: false });

document.querySelectorAll("[data-range]").forEach((button) => {
    button.addEventListener("click", async () => {
        document.querySelectorAll("[data-range]").forEach((other) => other.classList.remove("is-active"));
        button.classList.add("is-active");
        chartRange = Number(button.dataset.range);
        try {
            await loadHistory();
        } catch {
            toast("Could not load the trend for that range.", "error");
        }
    });
});

$("refreshBtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Recomputing';
    try {
        render(await api("/api/v1/dashboard?refresh=1"));
        await loadHistory();
        toast("Estimates recomputed from the latest readings.");
    } catch (error) {
        toast(error.message || "Refresh failed.", "error");
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});

wireAddItemForm("addItemForm", () => window.location.reload());

// Refresh the trend every couple of minutes; the tiles poll faster.
setInterval(() => {
    if (!document.hidden) loadHistory().catch(() => {});
}, 120000);

window.addEventListener("beforeunload", () => dashboardPoll.stop());
