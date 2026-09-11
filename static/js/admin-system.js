/* System maintenance: prune stored readings. */
import { api, escapeHtml, toast } from "./core.js";

document.getElementById("pruneBtn")?.addEventListener("click", async (event) => {
    const days = Number(document.getElementById("pruneDays").value);
    if (!Number.isFinite(days) || days < 1) {
        toast("Enter a retention window of at least one day.", "error");
        return;
    }
    if (!window.confirm(`Permanently delete stored readings older than ${days} days?`)) return;

    const button = event.currentTarget;
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Pruning';
    const output = document.getElementById("pruneResult");
    try {
        const data = await api("/admin/api/prune-readings", { method: "POST", body: { days } });
        output.innerHTML = `Removed <strong>${data.removed}</strong> reading(s) older than ${escapeHtml(String(data.days))} days.`;
        toast(`Removed ${data.removed} reading(s).`);
    } catch (error) {
        output.innerHTML = `<span class="text-crit">${escapeHtml(error.message)}</span>`;
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});
