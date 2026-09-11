/* Dependency-free SVG charts.

   Chart.js was pulled from a CDN by the prototype, which breaks the moment the
   Raspberry Pi has no internet. These renderers cover what the dashboard needs
   (line, sparkline, gauge, donut) in a few hundred lines, self-hosted, and they
   inherit the CSS theme tokens so dark mode works for free. */

const SVG_NS = "http://www.w3.org/2000/svg";

export const SERIES_COLORS = {
    temperature: "#ef6820",
    humidity: "#0ba5ec",
    gas: "#dd2590",
    shelfLife: "#12b76a",
};

function el(name, attrs = {}) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) {
        if (value !== null && value !== undefined) node.setAttribute(key, String(value));
    }
    return node;
}

/** Round an axis range outwards to human-friendly numbers. */
function niceScale(min, max, ticks = 4) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, step: 0.5 };
    if (min === max) {
        const pad = Math.abs(min) > 1 ? Math.abs(min) * 0.1 : 1;
        min -= pad;
        max += pad;
    }
    const rawStep = (max - min) / Math.max(1, ticks);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalised = rawStep / magnitude;
    const step = (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
    return {
        min: Math.floor(min / step) * step,
        max: Math.ceil(max / step) * step,
        step,
    };
}

function emptyState(container, message) {
    container.innerHTML = "";
    const div = document.createElement("div");
    div.className = "chart__empty";
    div.textContent = message;
    container.appendChild(div);
}

/**
 * Multi-series line chart.
 *
 * @param {HTMLElement} container
 * @param {{labels: string[], series: Array<{key,label,values,color,unit}>,
 *          height?: number, emptyMessage?: string, yUnit?: string}} config
 */
export function lineChart(container, config) {
    const { labels = [], series = [], height = 210, emptyMessage = "No data yet." } = config;
    const active = series.filter((s) => s.values.some((v) => v !== null && v !== undefined));
    if (!labels.length || !active.length) {
        emptyState(container, emptyMessage);
        return;
    }

    // Series carrying different units (°C vs %) must not share one scale, or the
    // smaller-range series is squashed into a flat line at the bottom. Series
    // marked axis:"right" get their own scale and their own labelled axis.
    const leftSeries = active.filter((s) => s.axis !== "right");
    const rightSeries = active.filter((s) => s.axis === "right");
    const dualAxis = leftSeries.length > 0 && rightSeries.length > 0;

    const W = 640;
    const H = height;
    const pad = { top: 14, right: dualAxis ? 40 : 12, bottom: 24, left: 42 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const scaleFor = (group) => {
        const values = group.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
        return values.length ? niceScale(Math.min(...values), Math.max(...values)) : niceScale(0, 1);
    };
    const leftScale = scaleFor(leftSeries.length ? leftSeries : active);
    const rightScale = dualAxis ? scaleFor(rightSeries) : leftScale;
    const count = labels.length;

    const xAt = (index) => (count === 1 ? pad.left + plotW / 2 : pad.left + (index / (count - 1)) * plotW);
    const project = (value, scale) =>
        pad.top + plotH * (1 - (value - scale.min) / (scale.max - scale.min || 1));
    const scaleOf = (entry) => (dualAxis && entry.axis === "right" ? rightScale : leftScale);
    const yAt = (value, entry) => project(value, scaleOf(entry || {}));

    container.innerHTML = "";
    container.style.position = "relative";
    const svg = el("svg", {
        viewBox: `0 0 ${W} ${H}`,
        role: "img",
        "aria-label": config.ariaLabel || `Line chart of ${active.map((s) => s.label).join(", ")}`,
        // Uniform scaling: "none" would squash the axis text horizontally on a
        // phone. The CSS gives the element width:100%;height:auto, so the chart
        // simply gets shorter on narrow screens instead of distorting.
        preserveAspectRatio: "xMidYMid meet",
    });

    const tickLabel = (value) =>
        Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1).replace(/\.0$/, "");

    // horizontal grid, driven by the left scale
    for (let value = leftScale.min; value <= leftScale.max + 1e-9; value += leftScale.step) {
        const y = project(value, leftScale);
        svg.appendChild(el("line", { class: "chart__grid", x1: pad.left, y1: y, x2: W - pad.right, y2: y }));
        const text = el("text", { class: "chart__axis", x: pad.left - 7, y: y + 3.5, "text-anchor": "end" });
        text.textContent = tickLabel(value);
        if (dualAxis) text.setAttribute("fill", leftSeries[0].color || "currentColor");
        svg.appendChild(text);
    }

    // right-hand axis labels, aligned to the same grid lines
    if (dualAxis) {
        const steps = Math.round((leftScale.max - leftScale.min) / leftScale.step);
        for (let i = 0; i <= steps; i += 1) {
            const y = project(leftScale.min + i * leftScale.step, leftScale);
            const value = rightScale.min + (i / steps) * (rightScale.max - rightScale.min);
            const text = el("text", {
                class: "chart__axis", x: W - pad.right + 7, y: y + 3.5, "text-anchor": "start",
                fill: rightSeries[0].color || "currentColor",
            });
            text.textContent = tickLabel(value);
            svg.appendChild(text);
        }
    }

    // x labels: show at most 6 so they never collide
    const labelStride = Math.max(1, Math.ceil(count / 6));
    labels.forEach((label, index) => {
        if (index % labelStride !== 0 && index !== count - 1) return;
        const text = el("text", {
            class: "chart__axis",
            x: xAt(index),
            y: H - 6,
            "text-anchor": index === 0 ? "start" : index === count - 1 ? "end" : "middle",
        });
        text.textContent = label;
        svg.appendChild(text);
    });

    // series paths (gaps in the data break the line rather than inventing a value)
    active.forEach((entry) => {
        const color = entry.color || SERIES_COLORS.temperature;
        const segments = [];
        let current = [];
        entry.values.forEach((value, index) => {
            if (value === null || value === undefined) {
                if (current.length) segments.push(current);
                current = [];
                return;
            }
            current.push([xAt(index), yAt(value, entry)]);
        });
        if (current.length) segments.push(current);

        segments.forEach((points) => {
            if (points.length > 1 && entry.fill !== false) {
                const area =
                    `M ${points[0][0]} ${pad.top + plotH} ` +
                    points.map(([x, y]) => `L ${x} ${y}`).join(" ") +
                    ` L ${points[points.length - 1][0]} ${pad.top + plotH} Z`;
                svg.appendChild(el("path", { class: "chart__area", d: area, fill: color }));
            }
            const d = points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
            svg.appendChild(el("path", { class: "chart__line", d, stroke: color }));
            if (points.length === 1) {
                svg.appendChild(el("circle", { cx: points[0][0], cy: points[0][1], r: 3, fill: color }));
            }
        });
    });

    // hover layer
    const cursor = el("line", {
        class: "chart__cursor", x1: 0, y1: pad.top, x2: 0, y2: pad.top + plotH, opacity: 0,
    });
    svg.appendChild(cursor);
    const markers = active.map((entry) =>
        el("circle", { class: "chart__dot", r: 4, fill: entry.color || SERIES_COLORS.temperature, opacity: 0 })
    );
    markers.forEach((marker) => svg.appendChild(marker));

    const overlay = el("rect", {
        x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "transparent",
    });
    overlay.style.cursor = "crosshair";
    svg.appendChild(overlay);
    container.appendChild(svg);

    const tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    container.appendChild(tooltip);

    const hide = () => {
        cursor.setAttribute("opacity", "0");
        markers.forEach((marker) => marker.setAttribute("opacity", "0"));
        tooltip.classList.remove("is-visible");
    };

    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("pointermove", (event) => {
        const box = svg.getBoundingClientRect();
        // Map client pixels back into viewBox units (preserveAspectRatio="none").
        const vbX = ((event.clientX - box.left) / box.width) * W;
        const ratio = count === 1 ? 0 : (vbX - pad.left) / plotW;
        const index = Math.max(0, Math.min(count - 1, Math.round(ratio * (count - 1))));
        const x = xAt(index);

        cursor.setAttribute("x1", x);
        cursor.setAttribute("x2", x);
        cursor.setAttribute("opacity", "1");

        const lines = [`<strong>${labels[index]}</strong>`];
        active.forEach((entry, seriesIndex) => {
            const value = entry.values[index];
            const marker = markers[seriesIndex];
            if (value === null || value === undefined) {
                marker.setAttribute("opacity", "0");
                return;
            }
            marker.setAttribute("cx", x);
            marker.setAttribute("cy", yAt(value, entry));
            marker.setAttribute("opacity", "1");
            lines.push(`${entry.label}: ${Number(value).toFixed(1)}${entry.unit || ""}`);
        });
        tooltip.innerHTML = lines.join("<br>");
        tooltip.style.left = `${(x / W) * 100}%`;
        tooltip.style.top = `${(pad.top / H) * 100}%`;
        tooltip.classList.add("is-visible");
    });
}

/** Compact trend line with no axes, for stat tiles. */
export function sparkline(container, values, color = SERIES_COLORS.temperature) {
    const clean = values.filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
    if (clean.length < 2) {
        container.innerHTML = "";
        return;
    }
    const W = 120;
    const H = 34;
    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const span = max - min || 1;
    const step = W / (values.length - 1);

    const points = [];
    values.forEach((value, index) => {
        if (value === null || value === undefined) return;
        points.push([index * step, H - 3 - ((value - min) / span) * (H - 6)]);
    });

    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
    svg.style.height = `${H}px`;
    const area =
        `M ${points[0][0]} ${H} ` + points.map(([x, y]) => `L ${x} ${y}`).join(" ") +
        ` L ${points[points.length - 1][0]} ${H} Z`;
    svg.appendChild(el("path", { d: area, fill: color, class: "chart__area" }));
    svg.appendChild(
        el("path", {
            d: points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" "),
            class: "chart__line", stroke: color, "stroke-width": 1.8,
        })
    );
    const [lastX, lastY] = points[points.length - 1];
    svg.appendChild(el("circle", { cx: lastX, cy: lastY, r: 2.6, fill: color }));
    container.appendChild(svg);
}

const TONE_COLORS = {
    ok: "var(--ok-line)",
    warn: "var(--warn-line)",
    crit: "var(--crit-line)",
    muted: "var(--border-strong)",
};

/**
 * Radial gauge, used for "fraction of shelf life remaining".
 * @param {{fraction: number, tone: string, value: string, unit: string, size?: number}} config
 */
export function gauge(container, config) {
    const { fraction = 0, tone = "ok", value = "--", unit = "", size = 150 } = config;
    const clamped = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
    const radius = size / 2 - 12;
    const circumference = 2 * Math.PI * radius;
    // Leave a 90-degree gap at the bottom so it reads as a gauge, not a pie.
    const arc = circumference * 0.75;

    container.innerHTML = "";
    container.classList.add("gauge");
    container.style.width = `${size}px`;
    container.style.height = `${size}px`;

    const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, "aria-hidden": "true" });
    const common = {
        cx: size / 2, cy: size / 2, r: radius, fill: "none",
        "stroke-width": 11, transform: `rotate(135 ${size / 2} ${size / 2})`,
    };
    svg.appendChild(
        el("circle", { ...common, class: "gauge__track", "stroke-dasharray": `${arc} ${circumference}` })
    );
    svg.appendChild(
        el("circle", {
            ...common,
            class: "gauge__value",
            stroke: TONE_COLORS[tone] || TONE_COLORS.ok,
            "stroke-linecap": "round",
            "stroke-dasharray": `${arc} ${circumference}`,
            "stroke-dashoffset": arc * (1 - clamped),
        })
    );
    container.appendChild(svg);

    const center = document.createElement("div");
    center.className = "gauge__center";
    center.innerHTML = `<div class="gauge__number"></div><div class="gauge__unit"></div>`;
    center.firstElementChild.textContent = value;
    center.lastElementChild.textContent = unit;
    container.appendChild(center);
}

/**
 * Horizontal stacked bar used for the freshness distribution.
 * @param {Array<{label: string, value: number, tone: string}>} segments
 */
export function distributionBar(container, segments) {
    const total = segments.reduce((sum, segment) => sum + segment.value, 0);
    container.innerHTML = "";
    if (!total) {
        emptyState(container, "Nothing tracked yet.");
        return;
    }
    const bar = document.createElement("div");
    bar.className = "meter";
    bar.style.height = "10px";
    bar.style.display = "flex";
    segments.forEach((segment) => {
        if (!segment.value) return;
        const part = document.createElement("div");
        part.style.width = `${(segment.value / total) * 100}%`;
        part.style.background = TONE_COLORS[segment.tone] || TONE_COLORS.muted;
        part.title = `${segment.label}: ${segment.value}`;
        bar.appendChild(part);
    });
    container.appendChild(bar);

    const legend = document.createElement("div");
    legend.className = "chart-legend";
    legend.style.marginTop = "0.6rem";
    segments.forEach((segment) => {
        const key = document.createElement("span");
        key.className = "chart-legend__key";
        const swatch = document.createElement("span");
        swatch.className = "chart-legend__swatch";
        swatch.style.background = TONE_COLORS[segment.tone] || TONE_COLORS.muted;
        swatch.style.height = "8px";
        swatch.style.width = "8px";
        swatch.style.borderRadius = "50%";
        key.appendChild(swatch);
        key.appendChild(document.createTextNode(`${segment.label} (${segment.value})`));
        legend.appendChild(key);
    });
    container.appendChild(legend);
}
