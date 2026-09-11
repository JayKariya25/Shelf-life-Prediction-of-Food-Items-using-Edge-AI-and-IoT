/* Shared frontend runtime: API client, theme, layout chrome, toasts, modals.
   No build step and no external libraries - the Pi serves these files as-is. */

/* ------------------------------------------------------------------ theme -- */
const THEME_KEY = "slp.theme";

export function applyTheme(theme) {
    const resolved = theme === "dark" || theme === "light" ? theme : null;
    if (resolved) {
        document.documentElement.setAttribute("data-theme", resolved);
    } else {
        document.documentElement.removeAttribute("data-theme");
    }
    document.querySelectorAll("[data-theme-state]").forEach((node) => {
        node.dataset.themeState = resolved || "system";
    });
}

export function storedTheme() {
    try {
        return localStorage.getItem(THEME_KEY);
    } catch {
        return null; // private browsing / storage disabled
    }
}

export function setTheme(theme) {
    try {
        if (theme) localStorage.setItem(THEME_KEY, theme);
        else localStorage.removeItem(THEME_KEY);
    } catch {
        /* not fatal - the theme still applies for this page view */
    }
    applyTheme(theme);
}

export function currentTheme() {
    const stored = storedTheme();
    if (stored) return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/* ------------------------------------------------------------- api client -- */
export class ApiError extends Error {
    constructor(message, code, status, payload) {
        super(message);
        this.name = "ApiError";
        this.code = code;
        this.status = status;
        this.payload = payload;
    }
}

function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : "";
}

/**
 * Call the JSON API. Resolves with the `data` envelope, throws ApiError otherwise.
 */
export async function api(path, { method = "GET", body, formData, signal } = {}) {
    const headers = {};
    const options = { method, headers, credentials: "same-origin", signal };

    if (method !== "GET" && method !== "HEAD") {
        headers["X-CSRF-Token"] = csrfToken();
    }
    if (formData) {
        options.body = formData; // browser sets the multipart boundary
    } else if (body !== undefined) {
        headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
    }

    let response;
    try {
        response = await fetch(path, options);
    } catch (error) {
        if (error.name === "AbortError") throw error;
        throw new ApiError("Cannot reach the server. Check the connection.", "network", 0, null);
    }

    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        payload = await response.json().catch(() => null);
    }

    if (!response.ok || !payload || payload.ok !== true) {
        const error = payload && payload.error ? payload.error : {};
        throw new ApiError(
            error.message || `Request failed (${response.status})`,
            error.code || "http_error",
            response.status,
            payload
        );
    }
    return payload.data;
}

/* ----------------------------------------------------------------- toasts -- */
const ICONS = {
    success: '<path d="M20 6 9 17l-5-5"/>',
    error: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
    warn: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
};

export function toast(message, kind = "success", timeout = 4200) {
    let stack = document.querySelector(".toast-stack");
    if (!stack) {
        stack = document.createElement("div");
        stack.className = "toast-stack";
        stack.setAttribute("role", "status");
        stack.setAttribute("aria-live", "polite");
        document.body.appendChild(stack);
    }
    const node = document.createElement("div");
    node.className = `toast${kind === "error" ? " toast--error" : kind === "warn" ? " toast--warn" : ""}`;
    node.innerHTML = `
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"
             style="flex:0 0 17px;width:17px;height:17px;margin-top:1px">${ICONS[kind] || ICONS.info}</svg>
        <div></div>`;
    node.lastElementChild.textContent = message;
    stack.appendChild(node);
    setTimeout(() => {
        node.style.opacity = "0";
        node.style.transition = "opacity .2s";
        setTimeout(() => node.remove(), 220);
    }, timeout);
}

/* ----------------------------------------------------------------- modals -- */
export function openModal(id) {
    const modal = document.getElementById(id);
    if (!modal) return null;
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    const focusable = modal.querySelector(
        "input:not([type=hidden]), select, textarea, button, [href]"
    );
    if (focusable) setTimeout(() => focusable.focus(), 40);
    return modal;
}

export function closeModal(modal) {
    const node = typeof modal === "string" ? document.getElementById(modal) : modal;
    if (!node) return;
    node.classList.remove("is-open");
    node.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
}

function wireModals() {
    document.addEventListener("click", (event) => {
        const opener = event.target.closest("[data-modal-open]");
        if (opener) {
            event.preventDefault();
            openModal(opener.dataset.modalOpen);
            return;
        }
        const closer = event.target.closest("[data-modal-close]");
        if (closer) {
            event.preventDefault();
            closeModal(closer.closest(".modal"));
            return;
        }
        if (event.target.classList.contains("modal")) closeModal(event.target);
    });
    document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        document.querySelectorAll(".modal.is-open").forEach((modal) => closeModal(modal));
    });
}

/* ------------------------------------------------------------- formatting -- */
export function formatDuration(hours) {
    if (hours === null || hours === undefined || Number.isNaN(hours)) return "--";
    if (hours <= 0) return "0h";
    if (hours < 1) return `${Math.round(hours * 60)}m`;
    if (hours < 48) return `${Math.floor(hours)}h`;
    return `${(hours / 24).toFixed(1)}d`;
}

export function formatDurationLong(hours) {
    if (hours === null || hours === undefined || Number.isNaN(hours)) return "Unknown";
    if (hours <= 0) return "Past its estimate";
    const days = Math.floor(hours / 24);
    const rest = Math.round(hours % 24);
    if (days === 0) return `${rest} hour${rest === 1 ? "" : "s"}`;
    if (rest === 0) return `${days} day${days === 1 ? "" : "s"}`;
    return `${days}d ${rest}h`;
}

export function formatNumber(value, digits = 1, fallback = "--") {
    if (value === null || value === undefined || Number.isNaN(value)) return fallback;
    return Number(value).toFixed(digits);
}

export function formatTime(iso) {
    if (!iso) return "--";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "--";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatDateTime(iso) {
    if (!iso) return "--";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "--";
    return date.toLocaleString([], {
        day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    });
}

export function relativeTime(iso) {
    if (!iso) return "never";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "never";
    const seconds = (Date.now() - then) / 1000;
    if (seconds < 0) return "just now";
    if (seconds < 60) return `${Math.floor(seconds)}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

export function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value === null || value === undefined ? "" : String(value);
    return div.innerHTML;
}

export function readJsonScript(id, fallback = null) {
    const node = document.getElementById(id);
    if (!node) return fallback;
    try {
        return JSON.parse(node.textContent);
    } catch {
        console.warn(`Malformed JSON payload in #${id}`);
        return fallback;
    }
}

/* --------------------------------------------------------------- polling --- */
/**
 * Repeat `task` on an interval, pausing while the tab is hidden and backing off
 * after failures. Prevents a disconnected Pi from being hammered by a tab that
 * has been left open for days.
 */
export function poll(task, intervalMs, { immediate = true } = {}) {
    let timer = null;
    let failures = 0;
    let stopped = false;

    const schedule = (delay) => {
        clearTimeout(timer);
        timer = setTimeout(run, delay);
    };

    async function run() {
        if (stopped) return;
        if (document.hidden) {
            schedule(intervalMs);
            return;
        }
        try {
            await task();
            failures = 0;
            schedule(intervalMs);
        } catch (error) {
            if (error.name === "AbortError") return;
            failures += 1;
            // Exponential backoff, capped at 8x the base interval.
            schedule(Math.min(intervalMs * 2 ** failures, intervalMs * 8));
        }
    }

    document.addEventListener("visibilitychange", () => {
        if (!document.hidden && !stopped) schedule(250);
    });

    if (immediate) run();
    else schedule(intervalMs);

    return {
        stop() { stopped = true; clearTimeout(timer); },
        refresh() { schedule(0); },
    };
}

/* ------------------------------------------------------------- app chrome -- */
const SIDEBAR_KEY = "slp.sidebar";

function wireChrome() {
    const sidebar = document.getElementById("sidebar");
    const scrim = document.getElementById("scrim");
    const isMobile = () => window.matchMedia("(max-width: 820px)").matches;

    if (sidebar) {
        try {
            if (localStorage.getItem(SIDEBAR_KEY) === "collapsed") {
                sidebar.classList.add("is-collapsed");
            }
        } catch { /* storage unavailable */ }
    }

    document.querySelectorAll("[data-sidebar-toggle]").forEach((button) => {
        button.addEventListener("click", () => {
            if (!sidebar) return;
            if (isMobile()) {
                const open = sidebar.classList.toggle("is-open");
                if (scrim) scrim.classList.toggle("is-visible", open);
                return;
            }
            const collapsed = sidebar.classList.toggle("is-collapsed");
            try {
                localStorage.setItem(SIDEBAR_KEY, collapsed ? "collapsed" : "expanded");
            } catch { /* storage unavailable */ }
        });
    });

    if (scrim) {
        scrim.addEventListener("click", () => {
            sidebar?.classList.remove("is-open");
            scrim.classList.remove("is-visible");
        });
    }

    wireThemePicker();

    document.querySelectorAll(".flash__close").forEach((button) => {
        button.addEventListener("click", () => button.closest(".flash")?.remove());
    });

    // Keep relative timestamps honest without a full page refresh.
    const refreshRelatives = () => {
        document.querySelectorAll("[data-relative]").forEach((node) => {
            node.textContent = relativeTime(node.dataset.relative);
        });
    };
    refreshRelatives();
    setInterval(refreshRelatives, 30000);
}

const SUN_PATH =
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4' +
    'M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>';
const MOON_PATH = '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>';
const SYSTEM_PATH =
    '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>';

/** Keep the icon beside the picker in step with the *resolved* appearance. */
function paintThemeIcon(choice) {
    const holder = document.getElementById("themeIcon");
    if (!holder) return;
    const svg = holder.querySelector("svg");
    if (!svg) return;
    const paths =
        choice === "system" ? SYSTEM_PATH : choice === "dark" ? MOON_PATH : SUN_PATH;
    svg.innerHTML = paths;
}

function wireThemePicker() {
    const select = document.getElementById("themeSelect");
    if (!select) return;
    // storedTheme() is null when the user has not overridden the OS setting.
    const choice = storedTheme() || "system";
    select.value = choice;
    paintThemeIcon(choice);

    select.addEventListener("change", () => {
        const next = select.value;
        setTheme(next === "system" ? null : next);
        paintThemeIcon(next);
    });
}

/* Applied before first paint by the inline-free bootstrap in base.html. */
applyTheme(storedTheme());

document.addEventListener("DOMContentLoaded", () => {
    applyTheme(storedTheme());
    wireChrome();
    wireModals();
});

// Follow the OS live while the user is on "System setting".
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!storedTheme()) applyTheme(null);
});
