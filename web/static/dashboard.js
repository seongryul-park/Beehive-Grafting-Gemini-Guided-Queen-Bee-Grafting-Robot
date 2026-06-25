// =============================================================================
//  dashboard.js — two views over the same image:
//    • Gemini Analysis  -> POST /analyze       (detect+classify+rank+select)
//    • OpenCV Detection -> POST /detect_opencv  (detection only, recall check)
// =============================================================================

const TYPE_COLORS = {
  young_larva: "#22c55e", egg: "#38bdf8", old_larva: "#f59e0b",
  empty: "#64748b", unknown: "#a855f7",
};
const TYPE_LABEL = {
  young_larva: "young larva", egg: "egg", old_larva: "old larva",
  empty: "empty", unknown: "unknown",
};
const DETECT_COLOR = "#22d3ee"; // neutral cyan for raw OpenCV boxes

const $ = (id) => document.getElementById(id);
const fileInput = $("file-input"), analyzeBtn = $("analyze-btn"), statusEl = $("status");
const img = $("hive-image"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const candList = $("candidate-list"), cellList = $("cell-list"), selectedTag = $("selected-tag");

let mode = "gemini";          // "gemini" | "opencv"
let selectedFile = null;
let lastResult = null;

// --- mode toggle ------------------------------------------------------------
$("mode-gemini").addEventListener("click", () => setMode("gemini"));
$("mode-opencv").addEventListener("click", () => setMode("opencv"));

function setMode(m) {
  mode = m;
  $("mode-gemini").classList.toggle("active", m === "gemini");
  $("mode-opencv").classList.toggle("active", m === "opencv");
  $("gemini-panels").hidden = m !== "gemini";
  $("opencv-panel").hidden = m !== "opencv";
  $("gemini-legend").hidden = m !== "gemini";
  $("opencv-legend").hidden = m !== "opencv";
  $("left-title").textContent = m === "opencv" ? "OpenCV detected cells" : "Detected cells";
  analyzeBtn.textContent = m === "opencv" ? "Detect" : "Analyze";
  lastResult = null;
  clearOverlay();
  setStatus("", "");
}

// --- file selection ---------------------------------------------------------
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  analyzeBtn.disabled = !selectedFile;
  lastResult = null;
  clearOverlay();
  if (selectedFile) {
    img.onload = () => sizeCanvas();
    img.src = URL.createObjectURL(selectedFile);
  }
});

window.addEventListener("resize", () => {
  if (lastResult) { sizeCanvas(); mode === "opencv" ? drawDetect(lastResult) : draw(lastResult); }
});

// --- run --------------------------------------------------------------------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const endpoint = mode === "opencv" ? "/detect_opencv" : "/analyze";
  setStatus(mode === "opencv" ? "Detecting (OpenCV)…" : "Analyzing with Gemini…", "busy");
  analyzeBtn.disabled = true;

  const form = new FormData();
  form.append("image", selectedFile);
  try {
    const resp = await fetch(endpoint, { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    lastResult = data;
    sizeCanvas();
    if (mode === "opencv") { drawDetect(data); renderDetect(data); }
    else { draw(data); renderLists(data); }
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  } finally {
    analyzeBtn.disabled = false;
  }
});

// --- canvas helpers ---------------------------------------------------------
function sizeCanvas() {
  canvas.width = img.clientWidth; canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px"; canvas.style.height = img.clientHeight + "px";
}
function clearOverlay() { ctx.clearRect(0, 0, canvas.width, canvas.height); }
function scalers(data) {
  const sw = (data.image && data.image.width) || data.width || img.naturalWidth || canvas.width;
  const sh = (data.image && data.image.height) || data.height || img.naturalHeight || canvas.height;
  return { sx: canvas.width / sw, sy: canvas.height / sh };
}

// --- OpenCV detection view --------------------------------------------------
function drawDetect(data) {
  clearOverlay();
  const { sx, sy } = scalers(data);
  ctx.lineWidth = 2; ctx.strokeStyle = DETECT_COLOR;
  ctx.font = "600 11px system-ui, sans-serif";
  data.cells.forEach((c) => {
    const [x1, y1, x2, y2] = c.bbox;
    const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    ctx.strokeStyle = DETECT_COLOR;
    ctx.strokeRect(x, y, w, h);
    const label = String(c.cell_id);
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(x, Math.max(0, y - 14), tw + 6, 14);
    ctx.fillStyle = DETECT_COLOR; ctx.fillText(label, x + 3, Math.max(10, y - 3));
  });
}

function renderDetect(data) {
  const d = data.debug || {};
  // detector-agnostic count: geometry -> detected_centers, color -> kept_components
  const n = d.detected_centers ?? d.kept_components ?? data.cells.length;
  $("detect-count").textContent = `${n} cells detected`;
  setStatus(`Done — ${n} cells (OpenCV detection, no Gemini).`, "ok");

  // Render whatever debug keys the active detector returned, plus image size.
  $("debug-list").innerHTML = "";
  const rows = Object.entries(d).concat([["image", `${data.width}×${data.height}`]]);
  rows.forEach(([k, v]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="k">${k}</span><span class="v">${v ?? "—"}</span>`;
    $("debug-list").appendChild(li);
  });

  const ul = $("detect-cells"); ul.innerHTML = "";
  data.cells.forEach((c) => {
    const li = document.createElement("li");
    li.textContent = `#${c.cell_id}  [${c.bbox.join(", ")}]`;
    ul.appendChild(li);
  });
}

// --- Gemini analysis view (unchanged) ---------------------------------------
function draw(data) {
  clearOverlay();
  const { sx, sy } = scalers(data);
  data.cells.forEach((c) => {
    const [x1, y1, x2, y2] = c.bbox;
    const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    const isTarget = c.cell_id === data.selected;
    const color = TYPE_COLORS[c.classification] || TYPE_COLORS.unknown;
    ctx.fillStyle = hexA(color, c.classification === "empty" ? 0.06 : 0.16);
    ctx.fillRect(x, y, w, h);
    ctx.lineWidth = isTarget ? 4 : 2;
    ctx.strokeStyle = isTarget ? "#ffffff" : color;
    ctx.strokeRect(x, y, w, h);
    if (isTarget) {
      ctx.lineWidth = 2; ctx.strokeStyle = TYPE_COLORS.young_larva;
      ctx.strokeRect(x + 3, y + 3, w - 6, h - 6);
    }
    const label = `#${c.cell_id} ${Math.round(c.confidence * 100)}%`;
    ctx.font = "600 12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.65)"; ctx.fillRect(x, Math.max(0, y - 16), tw + 8, 16);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x + 4, Math.max(11, y - 4));
  });
}

function renderLists(data) {
  const calls = data.gemini_calls === 0 ? "mock, 0 API calls" : `${data.gemini_calls} API call`;
  setStatus(`Done — ${data.cells.length} cells (${data.detector} detector, ${calls}).`, "ok");
  selectedTag.textContent = data.selected ? `selected: cell ${data.selected}` : "no graft candidate found";

  candList.innerHTML = "";
  if (!data.candidates.length) candList.innerHTML = '<li class="muted">No young larvae detected.</li>';
  data.candidates.forEach((c, i) => {
    const full = data.cells.find((x) => x.cell_id === c.cell_id) || c;
    candList.appendChild(cellItem(full, c.cell_id === data.selected, i + 1));
  });

  cellList.innerHTML = "";
  [...data.cells].sort((a, b) => b.confidence - a.confidence)
    .forEach((c) => cellList.appendChild(cellItem(c, c.cell_id === data.selected, null)));
}

function cellItem(c, isTarget, rank) {
  const li = document.createElement("li");
  li.className = "cell-item" + (isTarget ? " target" : "");
  const color = TYPE_COLORS[c.classification] || TYPE_COLORS.unknown;
  li.innerHTML = `
    <div class="row1">
      ${rank ? `<span class="rank">${rank}</span>` : ""}
      <span class="dot" style="background:${color}"></span>
      <strong>Cell ${c.cell_id}</strong>
      <span class="type">${TYPE_LABEL[c.classification] || c.classification}</span>
      ${isTarget ? '<span class="badge">GRAFT</span>' : ""}
      <span class="conf">${Math.round((c.confidence || 0) * 100)}%</span>
    </div>
    <div class="bar"><i style="width:${Math.round((c.confidence || 0) * 100)}%;background:${color}"></i></div>
    <div class="reason">${escapeHtml(c.reason || "")}</div>`;
  return li;
}

// --- helpers ----------------------------------------------------------------
function setStatus(t, k) { statusEl.textContent = t; statusEl.className = "status " + (k || ""); }
function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])); }
