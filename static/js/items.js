/* Items page: client-side search and sort over the rendered cards. */
import { readJsonScript } from "./core.js";
import { wireAddItemForm } from "./item-form.js";

const grid = document.getElementById("itemGrid");
const search = document.getElementById("itemSearch");
const sort = document.getElementById("itemSort");
const noMatches = document.getElementById("noMatches");
const items = readJsonScript("itemsData", []) || [];

const byId = new Map(items.map((item) => [String(item.id), item]));
const URGENCY = { critical: 0, warning: 1, ok: 2 };

function cards() {
    return grid ? Array.from(grid.querySelectorAll("[data-item-id]")) : [];
}

function apply() {
    if (!grid) return;
    const term = (search?.value || "").trim().toLowerCase();
    const mode = sort?.value || "urgency";

    let visible = 0;
    cards().forEach((card) => {
        const item = byId.get(card.dataset.itemId);
        const haystack = item
            ? `${item.label} ${item.food_name} ${item.storage} ${item.quantity || ""}`.toLowerCase()
            : card.textContent.toLowerCase();
        const matches = !term || haystack.includes(term);
        card.hidden = !matches;
        if (matches) visible += 1;
    });
    if (noMatches) noMatches.hidden = visible !== 0 || !term;

    const ordered = cards().sort((a, b) => {
        const left = byId.get(a.dataset.itemId);
        const right = byId.get(b.dataset.itemId);
        if (!left || !right) return 0;
        switch (mode) {
            case "name":
                return left.label.localeCompare(right.label);
            case "stored":
                return new Date(right.stored_at) - new Date(left.stored_at);
            case "remaining":
                return (right.remaining_hours ?? -1) - (left.remaining_hours ?? -1);
            default: {
                const rank = URGENCY[left.alert_level] - URGENCY[right.alert_level];
                if (rank !== 0) return rank;
                return (left.remaining_hours ?? 1e9) - (right.remaining_hours ?? 1e9);
            }
        }
    });
    ordered.forEach((card) => grid.appendChild(card));
}

search?.addEventListener("input", apply);
sort?.addEventListener("change", apply);
apply();

wireAddItemForm("addItemForm", () => window.location.reload());
