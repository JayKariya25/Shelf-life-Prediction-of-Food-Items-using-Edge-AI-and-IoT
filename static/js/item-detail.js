/* Item detail: gauge, prediction history, environment chart and item actions. */
import { api, formatDurationLong, readJsonScript, toast } from "./core.js";
import { lineChart, gauge, SERIES_COLORS } from "./charts.js";

const item = readJsonScript("itemData");
const history = readJsonScript("historyData", []) || [];
const environment = readJsonScript("envData", { points: [] }) || { points: [] };

const $ = (id) => document.getElementById(id);
const itemId = item ? item.id : null;

/* ------------------------------------------------------------- gauge ----- */
function fractionRemaining(record) {
    // Computed server-side so it matches the estimator that produced the value.
    return typeof record.life_fraction === "number" ? record.life_fraction : 0;
}

function renderGauge(record) {
    const container = $("lifeGauge");
    if (!container) return;
    const fraction = fractionRemaining(record);
    gauge(container, {
        fraction,
        tone: { critical: "crit", warning: "warn", ok: "ok" }[record.alert_level] || "muted",
        value: `${Math.round(fraction * 100)}%`,
        unit: "life left",
        size: 156,
    });
}

/* ------------------------------------------------------------ charts ----- */
function renderHistoryChart() {
    const container = $("historyChart");
    if (!container) return;
    if (history.length < 2) {
        lineChart(container, {
            labels: [], series: [],
            emptyMessage: "Run the estimate a few times to build a trend. It updates automatically every 15 minutes.",
        });
        return;
    }
    lineChart(container, {
        labels: history.map((row) =>
            new Date(row.created_at).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })
        ),
        height: 200,
        ariaLabel: "Estimated remaining shelf life over successive inference runs",
        series: [
            {
                key: "remaining",
                label: "Remaining",
                unit: " h",
                values: history.map((row) => row.remaining_hours),
                color: SERIES_COLORS.shelfLife,
            },
        ],
    });
}

function renderEnvChart() {
    const container = $("envChart");
    if (!container) return;
    const points = environment.points || [];
    lineChart(container, {
        labels: points.map((point) =>
            new Date(point.recorded_at).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit" })
        ),
        height: 190,
        ariaLabel: "Temperature and humidity during storage",
        emptyMessage: "No environment data covering this storage period.",
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
}

/* ------------------------------------------------------------ actions ---- */
$("predictBtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Running';
    try {
        const result = await api(`/api/v1/items/${itemId}/predict`, { method: "POST" });
        const updated = result.item;
        const which = result.prediction.model_kind === "trained-model" ? "trained model" : "kinetic baseline";
        toast(`Re-estimated with the ${which}: ${formatDurationLong(updated.remaining_hours)} remaining.`);
        // The model used, its notes and any fallback reason can all change, so redraw
        // the page from the server rather than patching pieces of it.
        setTimeout(() => window.location.reload(), 700);
    } catch (error) {
        toast(error.message || "Could not run the estimate.", "error");
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});

document.querySelectorAll("[data-close-item]").forEach((button) => {
    button.addEventListener("click", async () => {
        const status = button.dataset.closeItem;
        button.disabled = true;
        try {
            await api(`/api/v1/items/${itemId}`, { method: "PATCH", body: { status } });
            window.location.reload();
        } catch (error) {
            toast(error.message || "Could not update the item.", "error");
            button.disabled = false;
        }
    });
});

$("deleteItemBtn")?.addEventListener("click", async () => {
    if (!window.confirm("Delete this item along with its predictions and alerts? This cannot be undone.")) {
        return;
    }
    try {
        await api(`/api/v1/items/${itemId}`, { method: "DELETE" });
        window.location.href = "/items";
    } catch (error) {
        toast(error.message || "Could not delete the item.", "error");
    }
});

/* --------------------------------------------------------------- edit ---- */
const editForm = $("editItemForm");
if (editForm && item) {
    const storedAt = editForm.querySelector('input[name="stored_at"]');
    if (storedAt && item.stored_at) {
        const date = new Date(item.stored_at);
        date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
        storedAt.value = date.toISOString().slice(0, 16);
    }

    editForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const submit = editForm.querySelector('button[type="submit"]');
        const label = submit.innerHTML;
        submit.disabled = true;
        submit.innerHTML = '<span class="spinner"></span>Saving';
        try {
            const data = new FormData(editForm);
            const payload = {
                label: String(data.get("label") || "").trim(),
                food_type_id: Number(data.get("food_type_id")),
                quantity: String(data.get("quantity") || "").trim(),
                storage: String(data.get("storage") || ""),
                notes: String(data.get("notes") || "").trim(),
            };
            const rawStoredAt = String(data.get("stored_at") || "").trim();
            if (rawStoredAt) payload.stored_at = new Date(rawStoredAt).toISOString();

            const file = data.get("image");
            if (file && file.size > 0) {
                const upload = new FormData();
                upload.append("image", file);
                const result = await api("/api/v1/uploads", { method: "POST", formData: upload });
                payload.image_path = result.image_path;
            }

            await api(`/api/v1/items/${itemId}`, { method: "PATCH", body: payload });
            window.location.reload();
        } catch (error) {
            toast(error.message || "Could not save the changes.", "error");
            submit.disabled = false;
            submit.innerHTML = label;
        }
    });
}

/* ---------------------------------------------------------------- boot --- */
if (item) renderGauge(item);
renderHistoryChart();
renderEnvChart();
