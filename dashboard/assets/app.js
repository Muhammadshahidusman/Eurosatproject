/* EuroSAT Land Cover Explorer - dashboard logic. */
(async function () {
  "use strict";

  const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const MODEL_META = {
    resnet50: { name: "ResNet50", color: () => Charts.css("--series-1"), size: 224 },
    cnn: { name: "Custom CNN", color: () => Charts.css("--series-2"), size: 64 },
  };
  const ORDER = ["resnet50", "cnn"];
  const SHORT = {
    AnnualCrop: "AnnualCrop", Forest: "Forest", HerbaceousVegetation: "HerbVeg", Highway: "Highway",
    Industrial: "Industrial", Pasture: "Pasture", PermanentCrop: "PermCrop", Residential: "Residential",
    River: "River", SeaLake: "SeaLake",
  };
  const $ = s => document.querySelector(s);
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";

  if (window.self !== window.top) document.documentElement.classList.add("embedded");

  /* ------------------------------------------------------------ theme */
  const storage = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const savedTheme = storage.get("theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage.set("theme", document.documentElement.dataset.theme);
    redrawAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => redrawAll());

  /* ------------------------------------------------------------ data */
  const [M, L] = await Promise.all([
    fetch("data/metrics.json").then(r => r.json()),
    fetch("data/lahore.json").then(r => r.json()),
  ]);
  const CLASSES = M.classes;
  const MEAN = M.normalisation.mean, STD = M.normalisation.std;

  /* ------------------------------------------------------------ KPIs */
  function renderKpis() {
    const rn = M.models.resnet50, cnn = M.models.cnn;
    const nTiles = [...L.grids.resnet50].filter(c => c !== ".").length;
    const km2 = nTiles * (L.tile_m / 1000) ** 2;
    const tiles = [
      { v: pct(rn.accuracy, 2), k: "ResNet50 test accuracy", s: `Macro F1 ${rn.macro_f1.toFixed(3)}`, c: MODEL_META.resnet50.color() },
      { v: pct(cnn.accuracy, 2), k: "Custom CNN test accuracy", s: `${(cnn.params / 1e6).toFixed(2)}M params · 64 px`, c: MODEL_META.cnn.color() },
      { v: "27,000", k: "EuroSAT tiles · 10 classes", s: `${M.split.test.toLocaleString()} held out for test` },
      { v: nTiles.toLocaleString(), k: "Lahore tiles classified", s: `≈ ${Math.round(km2).toLocaleString()} km² at 640 m / tile` },
    ];
    $("#kpis").innerHTML = tiles.map(t => `<div class="kpi">
      <div class="v">${t.v}</div>
      <div class="k">${t.c ? `<span class="dot" style="background:${t.c}"></span>` : ""}${t.k}</div>
      <div class="s">${t.s}</div></div>`).join("");
  }

  /* ------------------------------------------------------------ PIL-exact bilinear resize
   * Mirrors Pillow's ImagingResample (BILINEAR, 22-bit fixed point, horizontal then
   * vertical pass) so browser inputs match torchvision's Resize on PIL images. */
  const PREC = 22;
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const fscale = Math.max(scale, 1);
    const support = fscale;
    const out = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      const k = [];
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        let t = Math.abs((x + xmin - center + 0.5) / fscale);
        const w = t < 1 ? 1 - t : 0;
        k.push(w); ww += w;
      }
      out.push({ xmin, k: k.map(w => { const v = (ww ? w / ww : 0) * (1 << PREC); return v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5); }) });
    }
    return out;
  }
  const clip8 = v => { v = Math.floor(v / (1 << PREC)); return v < 0 ? 0 : v > 255 ? 255 : v; };
  function resizeRGB(src, w, h, nw, nh) {
    let cur = src, cw = w;
    if (nw !== w) {
      const cs = coeffs(w, nw), dst = new Uint8ClampedArray(nw * h * 3);
      for (let y = 0; y < h; y++) for (let x = 0; x < nw; x++) {
        const { xmin, k } = cs[x];
        for (let c = 0; c < 3; c++) {
          let ss = 1 << (PREC - 1);
          for (let i = 0; i < k.length; i++) ss += cur[(y * cw + xmin + i) * 3 + c] * k[i];
          dst[(y * nw + x) * 3 + c] = clip8(ss);
        }
      }
      cur = dst; cw = nw;
    }
    if (nh !== h) {
      const cs = coeffs(h, nh), dst = new Uint8ClampedArray(cw * nh * 3);
      for (let y = 0; y < nh; y++) {
        const { xmin, k } = cs[y];
        for (let x = 0; x < cw; x++) for (let c = 0; c < 3; c++) {
          let ss = 1 << (PREC - 1);
          for (let i = 0; i < k.length; i++) ss += cur[((xmin + i) * cw + x) * 3 + c] * k[i];
          dst[(y * cw + x) * 3 + c] = clip8(ss);
        }
      }
      cur = dst;
    }
    return cur;
  }
  function toTensor(rgb, size) {
    const n = size * size, t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    return new ort.Tensor("float32", t, [1, 3, size, size]);
  }

  /* Image -> 64x64 RGB (center square crop, PIL-style downsample when needed). */
  function imageTo64(img) {
    const w = img.naturalWidth, h = img.naturalHeight, s = Math.min(w, h);
    const cv = document.createElement("canvas");
    cv.width = s; cv.height = s;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, (w - s) / 2, (h - s) / 2, s, s, 0, 0, s, s);
    const d = ctx.getImageData(0, 0, s, s).data;
    let rgb = new Uint8ClampedArray(s * s * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
    if (s !== 64) rgb = resizeRGB(rgb, s, s, 64, 64);
    return rgb;
  }

  /* ------------------------------------------------------------ ONNX sessions */
  const status = $("#status");
  let sessions = {};
  function loadSession(key) {
    if (!sessions[key]) {
      sessions[key] = ort.InferenceSession.create(`models/${key}.onnx`, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    return sessions[key];
  }
  async function predict(key, rgb64) {
    const sess = await loadSession(key);
    const size = MODEL_META[key].size;
    const rgb = size === 64 ? rgb64 : resizeRGB(rgb64, 64, 64, size, size);
    const t0 = performance.now();
    const out = await sess.run({ [sess.inputNames[0]]: toTensor(rgb, size) });
    const ms = performance.now() - t0;
    const logits = Array.from(out[sess.outputNames[0]].data);
    const mx = Math.max(...logits);
    const ex = logits.map(v => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    return { probs: ex.map(v => v / sum), ms };
  }
  if (typeof ort !== "undefined") {
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = 1;
  }

  /* ------------------------------------------------------------ classifier UI */
  let modelChoice = "both";
  let current = null; // { rgb64, truth, sample }
  const preview = $("#preview");

  function drawPreview(rgb64) {
    const cv = document.createElement("canvas"); cv.width = 64; cv.height = 64;
    const ctx = cv.getContext("2d"), id = ctx.createImageData(64, 64);
    for (let i = 0, j = 0; j < rgb64.length; i += 4, j += 3) { id.data[i] = rgb64[j]; id.data[i + 1] = rgb64[j + 1]; id.data[i + 2] = rgb64[j + 2]; id.data[i + 3] = 255; }
    ctx.putImageData(id, 0, 0);
    const pctx = preview.getContext("2d");
    pctx.imageSmoothingEnabled = false;
    pctx.clearRect(0, 0, preview.width, preview.height);
    pctx.drawImage(cv, 0, 0, preview.width, preview.height);
    return cv.toDataURL();
  }

  function renderSamples() {
    const box = $("#samples");
    box.innerHTML = M.gallery.map((g, i) => `<button type="button" role="option" data-i="${i}"
      class="${g.resnet50.pred !== g.true ? "miss" : ""}" title="${g.true}${g.resnet50.pred !== g.true ? " (ResNet50 error)" : ""}">
      <img src="img/samples/${g.file}" alt="${g.true}" loading="lazy" width="64" height="64"></button>`).join("");
    box.addEventListener("click", e => {
      const b = e.target.closest("button"); if (b) selectSample(+b.dataset.i);
    });
  }

  function loadImg(src) {
    return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
  }

  async function selectSample(i) {
    document.querySelectorAll("#samples button").forEach(b => b.classList.toggle("on", +b.dataset.i === i));
    const g = M.gallery[i];
    const img = await loadImg(`img/samples/${g.file}`);
    current = { rgb64: imageTo64(img), truth: g.true, sample: g };
    $("#truth").textContent = g.true;
    $("#drop-hint").hidden = false;
    drawPreview(current.rgb64);
    $("#cam-row").hidden = false;
    $("#cam-orig").src = `img/samples/${g.file}`;
    $("#cam-orig").style.imageRendering = "pixelated";
    $("#cam-img").src = `img/gradcam/${g.file}`;
    runCurrent();
  }

  async function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImg(url);
      current = { rgb64: imageTo64(img), truth: null, sample: null };
    } finally { URL.revokeObjectURL(url); }
    document.querySelectorAll("#samples button").forEach(b => b.classList.remove("on"));
    $("#truth").textContent = "Unknown (your upload)";
    $("#cam-row").hidden = true;
    drawPreview(current.rgb64);
    runCurrent();
  }

  let runToken = 0;
  async function runCurrent() {
    if (!current) return;
    const token = ++runToken;
    const keys = modelChoice === "both" ? ORDER : [modelChoice];
    const box = $("#predictions");
    box.className = "results" + (keys.length > 1 ? " two" : "");
    box.innerHTML = keys.map(k => `<div class="model-result" id="res-${k}">
      <h4><i style="background:${MODEL_META[k].color()}"></i>${MODEL_META[k].name}</h4>
      <div class="muted small">Running…</div></div>`).join("");
    status.className = "status muted";
    for (const k of keys) {
      try {
        if (!(await isLoaded(k))) status.textContent = `Downloading ${MODEL_META[k].name} (${M.models[k].onnx_int8.size_mb.toFixed(1)} MB, cached after first load)…`;
        const r = await predict(k, current.rgb64);
        if (token !== runToken) return;
        renderResult(k, r);
      } catch (err) {
        console.error(err);
        status.className = "status err";
        status.textContent = `Could not run ${MODEL_META[k].name}: ${err.message || err}`;
        return;
      }
    }
    status.textContent = `Ran locally in your browser on int8 ONNX exports of the trained models (test accuracy after export: ResNet50 ${pct(M.models.resnet50.onnx_int8.accuracy, 2)}, CNN ${pct(M.models.cnn.onnx_int8.accuracy, 2)}).`;
  }
  const loaded = {};
  async function isLoaded(k) { return loaded[k]; }

  function renderResult(k, { probs, ms }) {
    loaded[k] = true;
    const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const top = CLASSES[idx[0][1]];
    const color = MODEL_META[k].color();
    const verdict = current.truth == null ? "" :
      top === current.truth ? `<span class="badge ok">✓ correct</span>` : `<span class="badge no">✕ incorrect</span>`;
    $(`#res-${k}`).innerHTML = `<h4><i style="background:${color}"></i>${MODEL_META[k].name}</h4>
      <div class="top1">${top} ${verdict}</div>
      ${idx.slice(0, 5).map(([p, i]) => `<div class="prob-row">
        <span class="n" title="${CLASSES[i]}">${CLASSES[i]}</span>
        <span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%;background:${color}"></span></span>
        <span class="p">${(p * 100).toFixed(1)}%</span></div>`).join("")}
      <div class="latency">${ms.toFixed(1)} ms · ${MODEL_META[k].size}×${MODEL_META[k].size} input</div>`;
  }

  document.querySelectorAll(".result-card .seg button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".result-card .seg button").forEach(x => x.classList.toggle("on", x === b));
    modelChoice = b.dataset.model;
    runCurrent();
  }));
  const drop = $("#drop"), file = $("#file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => handleFile(file.files[0]));
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handleFile(e.dataTransfer.files[0]));
  $("#random-sample").addEventListener("click", () => selectSample(Math.floor(Math.random() * M.gallery.length)));

  /* ------------------------------------------------------------ metrics table */
  function renderMetricsTable() {
    const eff = key => M.efficiency.find(r => (key === "cnn") === /cnn/i.test(String(r.model))) || {};
    const rows = ORDER.map(k => {
      const m = M.models[k], e = eff(k);
      return {
        k, name: MODEL_META[k].name, acc: m.accuracy, f1: m.macro_f1, wf1: m.weighted_f1,
        params: m.params / 1e6, gmacs: e.macs_giga, lat: e.latency_ms_per_image, size: m.onnx_int8.size_mb, q: m.onnx_int8.accuracy,
      };
    });
    const best = (f, low) => { const v = rows.map(r => r[f]); return low ? Math.min(...v) : Math.max(...v); };
    const cell = (r, f, txt, low) => `<td class="${r[f] === best(f, low) ? "best" : ""}">${txt}</td>`;
    $("#metrics-table").innerHTML = `<thead><tr><th>Model</th><th>Test accuracy</th><th>Macro F1</th><th>Weighted F1</th>
      <th>Params</th><th>GMACs</th><th>Latency¹</th><th>Epochs</th><th>Input</th></tr></thead><tbody>` +
      rows.map(r => `<tr><td><span class="sw" style="background:${MODEL_META[r.k].color()}"></span>${r.name}</td>
        ${cell(r, "acc", pct(r.acc, 2))}${cell(r, "f1", r.f1.toFixed(3))}${cell(r, "wf1", r.wf1.toFixed(3))}
        ${cell(r, "params", r.params.toFixed(2) + "M", true)}${cell(r, "gmacs", r.gmacs != null ? r.gmacs.toFixed(2) : "-", true)}
        ${cell(r, "lat", r.lat != null ? r.lat.toFixed(1) + " ms" : "-", true)}
        <td>${M.models[r.k].history.length}</td><td>${MODEL_META[r.k].size} px</td></tr>`).join("") + "</tbody>";
    $("#metrics-foot").innerHTML = `Bold marks the better model per column. ¹ Batch size 1 on an NVIDIA Quadro P2000 (4 GB), median of timed runs after warm-up.`;
    renderEfficiencyTable();

    $("#perclass-table").innerHTML = `<thead><tr><th>Class</th>${ORDER.map(k =>
      `<th>${MODEL_META[k].name} P</th><th>R</th><th>F1</th>`).join("")}<th>Support</th></tr></thead><tbody>` +
      CLASSES.map((c, i) => `<tr><td>${c}</td>${ORDER.map(k => { const p = M.models[k].per_class[i];
        return `<td>${p.precision.toFixed(3)}</td><td>${p.recall.toFixed(3)}</td><td>${p.f1.toFixed(3)}</td>`; }).join("")}
        <td>${M.models.resnet50.per_class[i].support}</td></tr>`).join("") + "</tbody>";
  }

  function renderEfficiencyTable() {
    const e = k => M.efficiency.find(r => (k === "cnn") === /cnn/i.test(String(r.model))) || {};
    const c = e("cnn"), r = e("resnet50");
    const row = (label, f, fmt) =>
      `<tr><td>${label}</td><td>${fmt(c[f])}</td><td>${fmt(r[f])}</td><td class="ratio">${(r[f] / c[f]).toFixed(1)}×</td></tr>`;
    $("#eff-table").innerHTML = `<thead><tr><th>Metric</th>
      <th><span class="sw" style="background:${MODEL_META.cnn.color()}"></span>CNN</th>
      <th><span class="sw" style="background:${MODEL_META.resnet50.color()}"></span>ResNet50</th><th>Ratio</th></tr></thead><tbody>` +
      row("Parameters", "params_millions", v => v.toFixed(2) + " M") +
      row("GMACs / image", "macs_giga", v => v.toFixed(2)) +
      row("GFLOPs / image", "gflops", v => v.toFixed(2)) +
      row("Latency (bs = 1)", "latency_ms_per_image", v => v.toFixed(2) + " ms") +
      row("Peak GPU memory", "peak_gpu_memory_mb", v => v.toFixed(0) + " MB") +
      row("Weights on disk", "size_mb", v => v.toFixed(1) + " MB") + "</tbody>";
  }

  /* ------------------------------------------------------------ tabs */
  document.querySelectorAll("[data-tabs]").forEach(group => {
    const name = group.dataset.tabs;
    group.addEventListener("click", e => {
      const b = e.target.closest("button[data-tab]"); if (!b) return;
      group.querySelectorAll("button").forEach(x => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
      document.querySelectorAll(`[data-panel^="${name}:"]`).forEach(p => { p.hidden = p.dataset.panel !== `${name}:${b.dataset.tab}`; });
      renderCharts();
      if (name === "lahore") drawMap();
    });
  });

  /* ------------------------------------------------------------ lightbox */
  const lb = $("#lightbox");
  const figs = [];
  document.querySelectorAll("[data-lightbox]").forEach(btn => {
    const id = btn.dataset.lightbox;
    const capEl = (btn.closest("figure") && btn.closest("figure").querySelector("figcaption")) || btn.querySelector(".hero-figure-cap");
    const cap = capEl ? capEl.textContent.trim().replace(/\s+/g, " ") : btn.getAttribute("aria-label");
    let idx = figs.findIndex(f => f.id === id);
    if (idx < 0) { const im = btn.querySelector("img"); figs.push({ id, cap, alt: im ? im.alt : "" }); idx = figs.length - 1; }
    btn.addEventListener("click", () => openFig(idx));
  });
  let lbIdx = 0;
  function openFig(i) {
    lbIdx = (i + figs.length) % figs.length;
    const f = figs[lbIdx];
    $("#lb-img").src = `img/figures/${f.id}.webp`;
    $("#lb-img").alt = f.alt;
    $("#lb-cap").textContent = f.cap;
    if (!lb.open) lb.showModal();
  }
  $("#lb-close").addEventListener("click", () => lb.close());
  $("#lb-prev").addEventListener("click", () => openFig(lbIdx - 1));
  $("#lb-next").addEventListener("click", () => openFig(lbIdx + 1));
  lb.addEventListener("click", e => { if (e.target === lb) lb.close(); });
  lb.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft") openFig(lbIdx - 1);
    if (e.key === "ArrowRight") openFig(lbIdx + 1);
  });

  /* ------------------------------------------------------------ charts */
  let cmModel = "resnet50";
  function legendHTML(items) {
    return items.map(i => `<span><i class="${i.line ? "line" : ""}" style="background:${i.color};opacity:${i.faint ? 0.5 : 1}"></i>${i.name}</span>`).join("");
  }
  const visible = el => el && el.offsetParent !== null;
  function renderCharts() {
    if (visible($("#f1-chart"))) renderPerClassCharts();
    if (visible($("#acc-chart"))) renderCurves();
  }
  function renderPerClassCharts() {
    const series = ORDER.map(k => ({ name: MODEL_META[k].name, color: MODEL_META[k].color(), values: M.models[k].per_class.map(p => p.f1) }));
    const minF1 = Math.floor(Math.min(...series.flatMap(s => s.values)) * 10) / 10;
    $("#f1-legend").innerHTML = legendHTML(series);
    Charts.groupedHBar($("#f1-chart"), { categories: CLASSES.map(c => SHORT[c]), series, min: Math.min(minF1, 0.5), max: 1 });

    Charts.heatmap($("#cm-chart"), { labels: CLASSES, short: CLASSES.map(c => SHORT[c]), matrix: M.models[cmModel].confusion });
  }
  function renderCurves() {
    const curves = [];
    ORDER.forEach(k => {
      const h = M.models[k].history, c = MODEL_META[k].color();
      curves.push({ k, name: `${MODEL_META[k].name} · val`, color: c, acc: h.map(e => e.Val_Acc), loss: h.map(e => e.Val_Loss) });
      curves.push({ k, name: `${MODEL_META[k].name} · train`, color: c, faint: true, acc: h.map(e => e.Train_Acc), loss: h.map(e => e.Train_Loss) });
    });
    const xs = Array.from({ length: Math.max(...ORDER.map(k => M.models[k].history.length)) }, (_, i) => i + 1);
    $("#curve-legend").innerHTML = legendHTML(curves.map(c => ({ ...c, line: true })));
    Charts.lineChart($("#acc-chart"), { xs, yMin: 0.3, yMax: 1, yTicks: 7, fmt: (v, axis) => axis ? Math.round(v * 100) + "%" : pct(v),
      series: curves.map(c => ({ name: c.name, color: c.color, faint: c.faint, values: c.acc })) });
    const maxLoss = Math.ceil(Math.max(...curves.flatMap(c => c.loss)) / 0.4) * 0.4;
    Charts.lineChart($("#loss-chart"), { xs, yMin: 0, yMax: maxLoss, yTicks: Math.round(maxLoss / 0.4), fmt: (v, axis) => axis ? v.toFixed(1) : v.toFixed(3),
      series: curves.map(c => ({ name: c.name, color: c.color, faint: c.faint, values: c.loss })) });
  }
  $("#cm-toggle").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    cmModel = b.dataset.model;
    document.querySelectorAll("#cm-toggle button").forEach(x => x.classList.toggle("on", x === b));
    Charts.heatmap($("#cm-chart"), { labels: CLASSES, short: CLASSES.map(c => SHORT[c]), matrix: M.models[cmModel].confusion });
  });

  /* ------------------------------------------------------------ Grad-CAM gallery */
  let camFilter = "all";
  function renderCamFilters() {
    const nMiss = M.gallery.filter(g => g.resnet50.pred !== g.true).length;
    const chips = [["all", `All (${M.gallery.length})`], ["miss", `Misclassified (${nMiss})`], ...CLASSES.map(c => [c, SHORT[c]])];
    $("#cam-filters").innerHTML = chips.map(([v, t]) => `<button type="button" class="chip ${v === camFilter ? "on" : ""}" data-v="${v}">${t}</button>`).join("");
  }
  let camExpanded = false;
  const CAM_PREVIEW = 8;
  function renderCamGrid() {
    let items = M.gallery.map((g, i) => ({ ...g, i })).filter(g =>
      camFilter === "all" ? true : camFilter === "miss" ? g.resnet50.pred !== g.true : g.true === camFilter);
    if (camFilter === "all") {
      // Interleave classes (and one error) so the collapsed preview is varied.
      const correct = items.filter(g => g.resnet50.pred === g.true), wrong = items.filter(g => g.resnet50.pred !== g.true);
      const byClass = CLASSES.map(c => correct.filter(g => g.true === c));
      const mixed = [];
      for (let r = 0; mixed.length < correct.length; r++) byClass.forEach(l => { if (l[r]) mixed.push(l[r]); });
      items = [...mixed.slice(0, CAM_PREVIEW - 2), ...wrong.slice(0, 2), ...mixed.slice(CAM_PREVIEW - 2), ...wrong.slice(2)];
    }
    const total = items.length;
    if (!camExpanded && total > CAM_PREVIEW + 2) items = items.slice(0, CAM_PREVIEW);
    $("#cam-more").hidden = items.length === total && !camExpanded;
    $("#cam-more").textContent = camExpanded ? "Show fewer" : `Show all ${total}`;
    $("#cam-grid").innerHTML = items.map(g => {
      const miss = g.resnet50.pred !== g.true;
      return `<div class="cam-card ${miss ? "miss" : ""}">
        <div class="pair"><img class="px" src="img/samples/${g.file}" alt="${g.true} test tile" loading="lazy">
        <img src="img/gradcam/${g.file}" alt="Grad-CAM for predicted ${g.resnet50.pred}" loading="lazy"></div>
        <div class="meta">
          <div class="row"><span class="k">True</span><span class="val">${SHORT[g.true]}</span></div>
          <div class="row"><span class="k">ResNet50</span><span class="val pred">${SHORT[g.resnet50.pred]} · ${pct(g.resnet50.conf, 0)}</span></div>
        </div></div>`;
    }).join("") || `<p class="muted">No samples for this filter.</p>`;
  }
  $("#cam-filters").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    camFilter = b.dataset.v; camExpanded = false; renderCamFilters(); renderCamGrid();
  });
  $("#cam-more").addEventListener("click", () => {
    camExpanded = !camExpanded; renderCamGrid();
    if (!camExpanded) $("#explain").scrollIntoView();
  });

  /* ------------------------------------------------------------ Lahore map */
  let layer = "resnet50", highlight = null;
  const canvas = $("#map-canvas"), mapTip = $("#map-tip");
  canvas.width = L.cols; canvas.height = L.rows;
  const hexRGB = h => h.match(/\w\w/g).map(x => parseInt(x, 16));
  const classRGB = CLASSES.map(c => hexRGB(M.colors[c]));

  function drawMap() {
    const ctx = canvas.getContext("2d");
    const id = ctx.createImageData(L.cols, L.rows);
    const op = +$("#opacity").value / 100;
    const a = L.grids.resnet50, b = L.grids.cnn;
    const disagree = hexRGB(Charts.css("--bad").replace("#", "").length === 6 ? Charts.css("--bad") : "#e34948");
    for (let i = 0; i < a.length; i++) {
      const o = i * 4;
      if (layer === "rgb" || a[i] === "." ) { id.data[o + 3] = 0; continue; }
      let rgb, alpha = op;
      if (layer === "disagree") {
        if (a[i] !== b[i]) rgb = disagree; else { rgb = [255, 255, 255]; alpha = op * 0.25; }
      } else {
        const c = +(layer === "cnn" ? b : a)[i];
        rgb = classRGB[c];
        if (highlight != null && c !== highlight) alpha = op * 0.08;
        else if (highlight != null) alpha = Math.max(op, 0.9);
      }
      id.data[o] = rgb[0]; id.data[o + 1] = rgb[1]; id.data[o + 2] = rgb[2]; id.data[o + 3] = Math.round(alpha * 255);
    }
    ctx.putImageData(id, 0, 0);
  }

  function counts(grid) {
    const c = new Array(CLASSES.length).fill(0); let n = 0;
    for (const ch of grid) if (ch !== ".") { c[+ch]++; n++; }
    return c.map(v => v / n);
  }
  function renderLahoreBars() {
    const s = { resnet50: counts(L.grids.resnet50), cnn: counts(L.grids.cnn) };
    const order = CLASSES.map((c, i) => i).sort((x, y) => s.resnet50[y] - s.resnet50[x] || s.cnn[y] - s.cnn[x]);
    const max = Math.max(...s.resnet50, ...s.cnn);
    $("#lahore-legend").innerHTML = legendHTML(ORDER.map(k => ({ name: MODEL_META[k].name, color: MODEL_META[k].color() })));
    $("#lahore-bars").innerHTML = `<div class="lb-head"><span></span><span>Class</span><span>ResNet</span><span>CNN</span></div>` +
      order.map(i => `<button type="button" class="lb-row ${highlight === i ? "on" : highlight != null ? "dim" : ""}" data-i="${i}"
        aria-pressed="${highlight === i}" title="Click to isolate ${CLASSES[i]} on the map">
        <span class="sw" style="background:${M.colors[CLASSES[i]]}"></span>
        <span class="name">${CLASSES[i]}</span>
        <span class="pc">${pct(s.resnet50[i])}</span><span class="pc">${pct(s.cnn[i])}</span>
        <span></span><span class="bars">${ORDER.map(k => `<span class="bar" style="width:${s[k][i] ? Math.max(s[k][i] / max * 100, 0.8) : 0}%;background:${MODEL_META[k].color()}"></span>`).join("")}</span>
      </button>`).join("");
    $("#agree-pct").textContent = pct(L.agreement, 0);
    $("#map-foot").textContent = `${L.rows} × ${L.cols} tile grid · ${L.acquisition}. Tiles that are more than half outside the district are left blank.`;
  }
  $("#lahore-bars").addEventListener("click", e => {
    const b = e.target.closest(".lb-row"); if (!b) return;
    const i = +b.dataset.i;
    highlight = highlight === i ? null : i;
    if (layer === "disagree" || layer === "rgb") setLayer("resnet50");
    renderLahoreBars(); drawMap();
  });
  function setLayer(v) {
    layer = v;
    document.querySelectorAll("#map-toggle button").forEach(x => x.classList.toggle("on", x.dataset.layer === v));
    drawMap();
  }
  $("#map-toggle").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setLayer(b.dataset.layer); });
  $("#opacity").addEventListener("input", drawMap);
  canvas.addEventListener("mousemove", e => {
    const r = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - r.left) / r.width * L.cols), rr = Math.floor((e.clientY - r.top) / r.height * L.rows);
    const i = rr * L.cols + c;
    if (c < 0 || rr < 0 || c >= L.cols || rr >= L.rows || L.grids.resnet50[i] === ".") { mapTip.hidden = true; return; }
    const a = CLASSES[+L.grids.resnet50[i]], b = CLASSES[+L.grids.cnn[i]];
    mapTip.innerHTML = `<div class="t">Tile ${rr}, ${c}${a === b ? "" : " · models disagree"}</div>
      <div class="r"><i style="background:${M.colors[a]}"></i>ResNet50<b>${a}</b></div>
      <div class="r"><i style="background:${M.colors[b]}"></i>Custom CNN<b>${b}</b></div>`;
    mapTip.hidden = false;
    const wrap = $("#map-wrap").getBoundingClientRect();
    let x = e.clientX - wrap.left, y = e.clientY - wrap.top;
    if (x > wrap.width - 230) x -= 250;
    if (y > wrap.height - 90) y -= 100;
    mapTip.style.left = x + "px"; mapTip.style.top = y + "px";
  });
  canvas.addEventListener("mouseleave", () => { mapTip.hidden = true; });

  /* ------------------------------------------------------------ render all */
  function redrawAll() {
    renderKpis(); renderMetricsTable(); renderCharts(); renderLahoreBars(); drawMap();
    if (current) runCurrent();
  }
  renderSamples(); renderCamFilters(); renderCamGrid();
  redrawAll();
  let rt; new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(renderCharts, 120); }).observe(document.body);

  if (typeof ort === "undefined") {
    status.className = "status err";
    status.textContent = "ONNX Runtime Web failed to load from the CDN. Check your connection or content blocker.";
    return;
  }
  const selftest = new URLSearchParams(location.search).has("selftest");
  // The live classifier is an extra: only fetch the ONNX models once it scrolls into view.
  if (selftest) await selectSample(0);
  else new IntersectionObserver((entries, obs) => {
    if (entries.some(e => e.isIntersecting)) { obs.disconnect(); selectSample(0); }
  }, { rootMargin: "200px" }).observe($("#try"));

  /* ------------------------------------------------------------ self-test (?selftest) */
  if (selftest) {
    const report = { n: M.gallery.length, match: { cnn: 0, resnet50: 0 }, maxConfDiff: { cnn: 0, resnet50: 0 }, mismatches: [] };
    for (const g of M.gallery) {
      const rgb = imageTo64(await loadImg(`img/samples/${g.file}`));
      for (const k of ORDER) {
        const { probs } = await predict(k, rgb);
        const top = probs.indexOf(Math.max(...probs));
        if (CLASSES[top] === g[k].pred) report.match[k]++;
        else report.mismatches.push({ file: g.file, model: k, browser: CLASSES[top], python: g[k].pred });
        report.maxConfDiff[k] = Math.max(report.maxConfDiff[k], Math.abs(Math.max(...probs) - g[k].conf));
      }
    }
    const pre = document.createElement("pre");
    pre.id = "selftest"; pre.textContent = JSON.stringify(report);
    document.body.appendChild(pre);
    document.title = "SELFTEST DONE";
  }
})();
