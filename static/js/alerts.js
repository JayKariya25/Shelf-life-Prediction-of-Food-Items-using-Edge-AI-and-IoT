/* Alerts page: acknowledge alerts and persist notification preferences. */
import { api, toast } from "./core.js";

const feed = document.getElementById("alertFeed");

function setAcknowledged(row) {
    row.classList.add("is-acknowledged");
    row.dataset.open = "0";
    row.querySelector("[data-ack]")?.remove();
}

feed?.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-ack]");
    if (!button) return;
    const row = button.closest("[data-alert-id]");
    button.disabled = true;
    try {
        await api(`/api/v1/alerts/${button.dataset.ack}/acknowledge`, { method: "POST" });
        setAcknowledged(row);
        updateBadge();
    } catch (error) {
        toast(error.message || "Could not mark that alert as read.", "error");
        button.disabled = false;
    }
});

document.getElementById("ackAllBtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
        const result = await api("/api/v1/alerts/acknowledge-all", { method: "POST" });
        document.querySelectorAll('[data-alert-id][data-open="1"]').forEach(setAcknowledged);
        updateBadge();
        toast(result.acknowledged ? `Marked ${result.acknowledged} alert(s) as read.` : "Nothing left to mark.");
    } catch (error) {
        toast(error.message || "Could not update the alerts.", "error");
    } finally {
        button.disabled = false;
    }
});

document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll("[data-filter]").forEach((other) => other.classList.remove("is-active"));
        button.classList.add("is-active");
        const openOnly = button.dataset.filter === "open";
        document.querySelectorAll("[data-alert-id]").forEach((row) => {
            row.hidden = openOnly && row.dataset.open !== "1";
        });
    });
});

function updateBadge() {
    const open = document.querySelectorAll('[data-alert-id][data-open="1"]').length;
    const badge = document.getElementById("alertBadge");
    if (!badge) return;
    badge.hidden = open === 0;
    badge.textContent = open > 9 ? "9+" : String(open);
}

document.getElementById("saveSettings")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Saving';
    try {
        await api("/api/v1/settings", {
            method: "PUT",
            body: {
                alert_days_before: Number(document.getElementById("alertDays").value),
                email_notifications: document.getElementById("emailNotify").checked,
                sms_notifications: document.getElementById("smsNotify").checked,
            },
        });
        toast("Preferences saved.");
    } catch (error) {
        toast(error.message || "Could not save your preferences.", "error");
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});

updateBadge();
