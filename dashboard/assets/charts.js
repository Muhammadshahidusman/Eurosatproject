/* Minimal SVG chart helpers (no dependencies).
 * Each renderer draws into a container at its current width and returns nothing;
 * callers re-invoke on resize / theme change. Colours are CSS custom properties. */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function svgRoot(container, w, h) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img" });
    container.appendChild(svg);
    return svg;
  }

  function tooltip(container) {
    let tip = container.querySelector(":scope > .tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "tip";
      tip.hidden = true;
      container.appendChild(tip);
    }
    return {
      show(x, y, html) { tip.innerHTML = html; tip.hidden = false; tip.style.left = x + "px"; tip.style.top = y + "px"; },
      hide() { tip.hidden = true; },
    };
  }

  // Rounded-end bar (4px radius on the data end only) for horizontal bars.
  function hbarPath(x0, y, len, h, r) {
    r = Math.min(r, len, h / 2);
    if (len <= 0) return "";
    return `M${x0},${y}H${x0 + len - r}Q${x0 + len},${y} ${x0 + len},${y + r}` +
      `V${y + h - r}Q${x0 + len},${y + h} ${x0 + len - r},${y + h}H${x0}Z`;
  }

  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map(h => parseInt(h, 16));
    const pb = b.match(/\w\w/g).map(h => parseInt(h, 16));
    return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("");
  }

  function luminance(hex) {
    const [r, g, b] = hex.match(/\w\w/g).map(h => parseInt(h, 16) / 255)
      .map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  /* -------------------------------------------------- grouped horizontal bars */
  function groupedHBar(container, { categories, series, min = 0, max = 1, fmt = v => v.toFixed(3) }) {
    const W = Math.max(container.clientWidth, 280);
    const labelW = Math.min(150, W * 0.36);
    const barH = 7, gap = 2, groupPad = 10;
    const groupH = series.length * barH + (series.length - 1) * gap;
    const top = 6, bottom = 22;
    const H = top + categories.length * (groupH + groupPad) + bottom;
    const plotW = W - labelW - 12;
    const x = v => ((v - min) / (max - min)) * plotW;
    const svg = svgRoot(container, W, H);
    svg.setAttribute("aria-label", "Grouped bar chart");

    const grid = el("g", { class: "grid" }, svg);
    const ticks = 5;
    for (let i = 0; i <= ticks; i++) {
      const v = min + (i / ticks) * (max - min);
      const gx = labelW + x(v);
      el("line", { x1: gx, x2: gx, y1: top, y2: H - bottom }, grid);
      const t = el("text", { x: gx, y: H - 6, "text-anchor": "middle" }, svg);
      t.textContent = v.toFixed(2);
    }
    const tip = tooltip(container);
    categories.forEach((cat, ci) => {
      const gy = top + ci * (groupH + groupPad) + groupPad / 2;
      const lbl = el("text", { x: labelW - 8, y: gy + groupH / 2 + 4, "text-anchor": "end", class: "axis-label" }, svg);
      lbl.textContent = cat;
      series.forEach((s, si) => {
        const v = s.values[ci];
        const y = gy + si * (barH + gap);
        el("path", { d: hbarPath(labelW, y, Math.max(x(v), 1), barH, 3.5), fill: s.color }, svg);
      });
      const hit = el("rect", { x: 0, y: gy - groupPad / 2, width: W, height: groupH + groupPad, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", e => {
        const r = container.getBoundingClientRect();
        tip.show(e.clientX - r.left, gy, `<div class="t">${cat}</div>` + series.map(s =>
          `<div class="r"><i style="background:${s.color}"></i>${s.name}<b>${fmt(s.values[ci])}</b></div>`).join(""));
      });
      hit.addEventListener("mouseleave", tip.hide);
    });
    el("line", { class: "baseline", x1: labelW, x2: labelW, y1: top, y2: H - bottom }, svg);
  }

  /* -------------------------------------------------- confusion heatmap */
  function heatmap(container, { labels, matrix, short }) {
    const W = Math.max(container.clientWidth, 280);
    const n = labels.length;
    const left = Math.min(110, W * 0.27), top = 8, bottom = 64;
    const cell = Math.floor((W - left - 4) / n);
    const H = top + cell * n + bottom;
    const svg = svgRoot(container, W, H);
    svg.setAttribute("aria-label", "Confusion matrix heatmap");
    const c0 = css("--seq-0"), c1 = css("--seq-1"), surface = css("--surface");
    const tip = tooltip(container);
    const rowSums = matrix.map(r => r.reduce((a, b) => a + b, 0));
    const fontSize = cell < 30 ? 8.5 : 10;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = rowSums[i] ? matrix[i][j] / rowSums[i] : 0;
        const fill = mix(c0, c1, Math.sqrt(v));
        const x = left + j * cell, y = top + i * cell;
        el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, fill }, svg);
        if (v >= 0.05) {
          const ink = luminance(fill) > 0.35 ? "#0f0f0e" : "#ffffff";
          const t = el("text", { x: x + cell / 2, y: y + cell / 2 + 3.5, "text-anchor": "middle",
            style: `fill:${ink};font-size:${fontSize}px;font-weight:${i === j ? 600 : 400}` }, svg);
          t.textContent = v >= 0.995 ? "1.0" : v.toFixed(2).replace(/^0/, "");
        }
        const hit = el("rect", { x, y, width: cell, height: cell, fill: "transparent" }, svg);
        hit.addEventListener("mousemove", () => {
          tip.show(x + cell / 2, y, `<div class="t">${labels[i]} → ${labels[j]}</div>` +
            `<div class="r">Share of true ${labels[i]}<b>${(v * 100).toFixed(1)}%</b></div>` +
            `<div class="r">Images<b>${matrix[i][j]} / ${rowSums[i]}</b></div>`);
        });
        hit.addEventListener("mouseleave", tip.hide);
      }
      const yl = el("text", { x: left - 6, y: top + i * cell + cell / 2 + 4, "text-anchor": "end", class: "axis-label" }, svg);
      yl.textContent = short[i];
      const xl = el("text", { x: 0, y: 0, "text-anchor": "end", class: "axis-label",
        transform: `translate(${left + i * cell + cell / 2 + 3},${top + n * cell + 8}) rotate(-45)` }, svg);
      xl.textContent = short[i];
    }
    const ax = el("text", { x: left + (n * cell) / 2, y: H - 2, "text-anchor": "middle" }, svg);
    ax.textContent = "Predicted class →";
    void surface;
  }

  /* -------------------------------------------------- line chart w/ crosshair */
  function lineChart(container, { xs, series, yMin, yMax, fmt, yTicks = 5, xLabel = "Epoch" }) {
    const W = Math.max(container.clientWidth, 280);
    const H = Math.round(Math.min(300, Math.max(220, W * 0.55)));
    const left = 40, right = 12, top = 10, bottom = 34;
    const pw = W - left - right, ph = H - top - bottom;
    const xMax = Math.max(...xs);
    const X = v => left + ((v - 1) / (xMax - 1)) * pw;
    const Y = v => top + (1 - (v - yMin) / (yMax - yMin)) * ph;
    const svg = svgRoot(container, W, H);
    svg.setAttribute("aria-label", "Line chart");
    const grid = el("g", { class: "grid" }, svg);
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (i / yTicks) * (yMax - yMin);
      el("line", { x1: left, x2: W - right, y1: Y(v), y2: Y(v) }, grid);
      const t = el("text", { x: left - 6, y: Y(v) + 4, "text-anchor": "end" }, svg);
      t.textContent = fmt(v, true);
    }
    xs.forEach(v => {
      const t = el("text", { x: X(v), y: H - bottom + 16, "text-anchor": "middle" }, svg);
      t.textContent = v;
    });
    const xl = el("text", { x: left + pw / 2, y: H - 2, "text-anchor": "middle" }, svg);
    xl.textContent = xLabel;
    const surface = css("--surface");

    series.forEach(s => {
      const pts = s.values.map((v, i) => [X(i + 1), Y(v)]);
      el("path", {
        d: "M" + pts.map(p => p.join(",")).join("L"), fill: "none", stroke: s.color,
        "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-opacity": s.faint ? 0.42 : 1,
      }, svg);
      pts.forEach(p => el("circle", { cx: p[0], cy: p[1], r: s.faint ? 2.5 : 3.5, fill: s.faint ? surface : s.color,
        stroke: s.faint ? s.color : surface, "stroke-width": s.faint ? 1.5 : 2, "stroke-opacity": s.faint ? 0.6 : 1 }, svg));
    });

    const cross = el("line", { y1: top, y2: top + ph, stroke: css("--muted"), "stroke-width": 1, opacity: 0 }, svg);
    const tip = tooltip(container);
    const hit = el("rect", { x: left, y: top, width: pw, height: ph, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => {
      const r = svg.getBoundingClientRect();
      const mx = (e.clientX - r.left) * (W / r.width);
      const ep = Math.round(1 + ((mx - left) / pw) * (xMax - 1));
      const rows = series.filter(s => ep <= s.values.length);
      if (!rows.length) { tip.hide(); cross.setAttribute("opacity", 0); return; }
      cross.setAttribute("x1", X(ep)); cross.setAttribute("x2", X(ep)); cross.setAttribute("opacity", 0.6);
      tip.show(X(ep) * (r.width / W), top, `<div class="t">${xLabel} ${ep}</div>` + rows.map(s =>
        `<div class="r"><i style="background:${s.color};opacity:${s.faint ? 0.5 : 1}"></i>${s.name}<b>${fmt(s.values[ep - 1])}</b></div>`).join(""));
    });
    hit.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("opacity", 0); });
  }

  window.Charts = { groupedHBar, heatmap, lineChart, css };
})();
