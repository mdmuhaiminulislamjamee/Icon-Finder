// MediaControl Studio Dashboard Client

let allRecords = [];
let activeRecord = null;
let currentAnnotations = [];

// Streaming state
let isStreaming = false;
let streamFpsTimer = null;
let isFillMode = false;

// Auto-Capture Loop state
let isAutoCapturing = false;
let autoCaptureTimer = null;
let autoSessionCount = 0;

// Class Color Palette
const CLASS_COLORS = {
  skip_ad: "#ec4899",
  progress_bar: "#f97316",
  playhead: "#eab308",
  play: "#10b981",
  pause: "#f59e0b",
  volume_on: "#06b6d4",
  volume_muted: "#ef4444",
  fullscreen: "#a855f7",
  exit_fullscreen: "#9333ea",
  like: "#22c55e",
  dislike: "#f43f5e",
  comments: "#3b82f6",
  share: "#14b8a6",
  remix: "#6366f1",
  sound: "#8b5cf6",
  save: "#d946ef",
  search: "#64748b",
  more_options: "#94a3b8",
};

// Canvas drawing state
let canvas, ctx;
let canvasImage = null;
let isDrawing = false;
let drawStartX = 0;
let drawStartY = 0;

document.addEventListener("DOMContentLoaded", () => {
  initDevicePolling();
  loadRecords();
  setupEventListeners();
  initCanvas();
});

// -------------------------------------------------------------
// 1. Device Polling & Status
// -------------------------------------------------------------
function initDevicePolling() {
  checkDevice();
  setInterval(checkDevice, 5000);
}

async function checkDevice() {
  try {
    const res = await fetch("/api/device");
    const data = await res.json();
    const pill = document.getElementById("device-pill");
    const text = document.getElementById("device-text");

    if (data.connected) {
      pill.className = "device-pill connected";
      text.textContent = `Connected: ${data.name} (${data.model})`;
    } else {
      pill.className = "device-pill disconnected";
      text.textContent = "No iOS Device Connected";
      if (isAutoCapturing) {
        stopAutoCapture();
      }
    }
  } catch (err) {
    console.error("Device check failed:", err);
  }
}

// -------------------------------------------------------------
// 2. High-Performance Live Device Screen Streaming
// -------------------------------------------------------------
function toggleStream(forceStart = false) {
  if (isStreaming && !forceStart) {
    stopStream();
  } else {
    startStream();
  }
}

async function startStream() {
  isStreaming = true;
  const toggleBtn = document.getElementById("btn-toggle-stream");
  toggleBtn.className = "btn btn-sm btn-danger";
  document.getElementById("stream-icon").textContent = "⏹";
  document.getElementById("stream-btn-text").textContent = "Stop Live Stream";

  document.getElementById("monitor-placeholder").classList.add("hidden");
  document.getElementById("live-screen-img").classList.remove("hidden");
  document.getElementById("monitor-overlay-bar").classList.remove("hidden");

  // Notify backend to begin fast persistent frame grabber
  try {
    await fetch("/api/device/stream/start", { method: "POST" });
  } catch (e) {
    console.warn("Stream start notice:", e);
  }

  // Connect native MJPEG stream for zero-lag continuous display
  const img = document.getElementById("live-screen-img");
  img.src = `/api/device/stream?t=${Date.now()}`;

  pollStreamStatus();
}

async function stopStream() {
  isStreaming = false;
  clearTimeout(streamFpsTimer);

  const toggleBtn = document.getElementById("btn-toggle-stream");
  toggleBtn.className = "btn btn-sm btn-primary";
  document.getElementById("stream-icon").textContent = "▶";
  document.getElementById("stream-btn-text").textContent = "Start Live Stream";

  const img = document.getElementById("live-screen-img");
  img.src = "";
  img.classList.add("hidden");
  document.getElementById("monitor-overlay-bar").classList.add("hidden");
  document.getElementById("monitor-placeholder").classList.remove("hidden");

  try {
    await fetch("/api/device/stream/stop", { method: "POST" });
  } catch (e) {
    console.warn("Stream stop notice:", e);
  }
}

async function pollStreamStatus() {
  if (!isStreaming) return;

  try {
    const res = await fetch("/api/device/stream/status");
    if (res.ok) {
      const data = await res.json();
      if (data.fps > 0) {
        document.getElementById("stream-fps").textContent = `${data.fps} fps`;
      }
    }
  } catch (err) {
    // ignore polling glitches
  }

  if (isStreaming) {
    streamFpsTimer = setTimeout(pollStreamStatus, 1000);
  }
}

function toggleViewMode() {
  isFillMode = !isFillMode;
  const img = document.getElementById("live-screen-img");
  const btn = document.getElementById("btn-view-mode");

  if (isFillMode) {
    img.classList.add("fill-mode");
    btn.textContent = "🖼️ Fill";
    btn.title = "Current: Full Width Fill. Click to switch to Fitted Ratio.";
  } else {
    img.classList.remove("fill-mode");
    btn.textContent = "🔍 Fit";
    btn.title = "Current: Fitted Aspect Ratio. Click to switch to Full Fill.";
  }
}

function toggleFullscreen() {
  const vp = document.getElementById("monitor-viewport");
  if (!document.fullscreenElement) {
    if (vp.requestFullscreen) {
      vp.requestFullscreen();
    } else if (vp.webkitRequestFullscreen) {
      vp.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
}

async function snapLiveFrame() {
  try {
    const res = await fetch(`/api/device/live_frame?t=${Date.now()}`);
    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    } else {
      alert("Failed to grab snapshot from device.");
    }
  } catch (err) {
    alert("Snapshot error: " + err.message);
  }
}

// -------------------------------------------------------------
// 3. Capture Station (Single & Continuous Auto-Capture)
// -------------------------------------------------------------
function setCaptureMode(mode) {
  const singleBtn = document.getElementById("tab-mode-single");
  const autoBtn = document.getElementById("tab-mode-auto");
  const singleSection = document.getElementById("section-single-capture");
  const autoSection = document.getElementById("section-auto-capture");

  if (mode === "auto") {
    singleBtn.classList.remove("active");
    autoBtn.classList.add("active");
    singleSection.classList.add("hidden");
    autoSection.classList.remove("hidden");
  } else {
    autoBtn.classList.remove("active");
    singleBtn.classList.add("active");
    autoSection.classList.add("hidden");
    singleSection.classList.remove("hidden");
    if (isAutoCapturing) stopAutoCapture();
  }
}

async function triggerSingleCapture() {
  const platform = document.getElementById("cap-platform").value;
  const device = document.getElementById("cap-device").value;
  const content_format = document.getElementById("cap-format").value;
  const capture_state = document.getElementById("cap-state").value;

  const btn = document.getElementById("btn-trigger-capture");
  const overlay = document.getElementById("countdown-overlay");
  const bar = document.getElementById("countdown-bar");
  const statusBox = document.getElementById("capture-status-box");
  const statusTitle = document.getElementById("capture-status-title");
  const logPre = document.getElementById("capture-log");

  btn.disabled = true;
  overlay.classList.remove("hidden");
  bar.style.width = "100%";
  setTimeout(() => (bar.style.width = "0%"), 50);

  statusBox.classList.remove("hidden");
  statusTitle.textContent = "Running capture runner on device...";

  try {
    const res = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ platform, device, content_format, capture_state }),
    });
    const data = await res.json();

    overlay.classList.add("hidden");
    btn.disabled = false;

    if (data.success) {
      statusTitle.textContent = "✅ " + data.message;
      logPre.textContent = data.log || "No log returned";
      setTimeout(loadRecords, 800);
    } else {
      statusTitle.textContent = "❌ " + data.message;
      logPre.textContent = data.log || data.message;
      logPre.classList.remove("hidden");
    }
  } catch (err) {
    overlay.classList.add("hidden");
    btn.disabled = false;
    statusTitle.textContent = "❌ Capture error: " + err.message;
  }
}

// Continuous Auto-Capture Loop
function startAutoCapture() {
  isAutoCapturing = true;
  autoSessionCount = 0;

  document.getElementById("btn-start-auto").classList.add("hidden");
  document.getElementById("btn-stop-auto").classList.remove("hidden");
  updateAutoBadge();

  const statusBox = document.getElementById("capture-status-box");
  const statusTitle = document.getElementById("capture-status-title");
  statusBox.classList.remove("hidden");
  statusTitle.textContent = "🔄 Auto-capture loop running... Swipe and browse on your iPad!";

  runAutoCaptureIteration();
}

function stopAutoCapture() {
  isAutoCapturing = false;
  clearTimeout(autoCaptureTimer);

  document.getElementById("btn-stop-auto").classList.add("hidden");
  document.getElementById("btn-start-auto").classList.remove("hidden");

  const statusTitle = document.getElementById("capture-status-title");
  statusTitle.textContent = `⏹ Auto-capture stopped. Total captured in session: ${autoSessionCount} frames.`;
  loadRecords();
}

async function runAutoCaptureIteration() {
  if (!isAutoCapturing) return;

  const platform = document.getElementById("cap-platform").value;
  const device = document.getElementById("cap-device").value;
  const content_format = document.getElementById("cap-format").value;
  const capture_state = document.getElementById("cap-state").value;
  const intervalSeconds = parseInt(document.getElementById("auto-interval-select").value, 10) || 5;

  try {
    const res = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ platform, device, content_format, capture_state }),
    });
    const data = await res.json();

    if (data.success) {
      autoSessionCount++;
      updateAutoBadge();
      loadRecords();
    }
  } catch (err) {
    console.error("Auto-capture iteration failed:", err);
  }

  if (isAutoCapturing) {
    autoCaptureTimer = setTimeout(runAutoCaptureIteration, intervalSeconds * 1000);
  }
}

function updateAutoBadge() {
  document.getElementById("auto-counter-badge").textContent = `Session: ${autoSessionCount} frames`;
}

function toggleLog() {
  const log = document.getElementById("capture-log");
  if (!log.textContent.trim()) {
    log.textContent = "No capture log recorded in this session yet.\nClick 'Trigger Single Capture (4s)' or 'Start Auto-Capture' above to run the capture runner on the iPad.";
  }
  log.classList.toggle("hidden");
}

async function syncFromDevice() {
  const btn = document.getElementById("btn-sync-device");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="icon">⏳</span> Syncing...`;

  try {
    const res = await fetch("/api/device/sync", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      btn.innerHTML = `<span class="icon">✅</span> Synced (${data.synced_images} new)`;
      setTimeout(() => {
        btn.innerHTML = origText;
        btn.disabled = false;
      }, 2000);
      loadRecords();
    } else {
      alert("Sync failed: " + (data.detail || "Unknown error"));
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  } catch (err) {
    alert("Sync error: " + err.message);
    btn.innerHTML = origText;
    btn.disabled = false;
  }
}

// -------------------------------------------------------------
// 4. Dataset Records Loading & Filtering
// -------------------------------------------------------------
async function loadRecords() {
  try {
    const res = await fetch("/api/records");
    allRecords = await res.json();
    renderGallery();
  } catch (err) {
    console.error("Failed to load records:", err);
  }
}

function renderGallery() {
  const grid = document.getElementById("gallery-grid");
  const emptyState = document.getElementById("empty-state");
  const countBadge = document.getElementById("record-count-badge");

  const filterPlatform = document.getElementById("filter-platform").value;
  const filterFormat = document.getElementById("filter-format").value;
  const filterOrientation = document.getElementById("filter-orientation").value;
  const filterClass = document.getElementById("filter-class").value;
  const filterStatus = document.getElementById("filter-status").value;
  const filterSearch = document.getElementById("filter-search").value.toLowerCase().trim();

  const filtered = allRecords.filter((r) => {
    if (filterPlatform !== "all" && r.platform !== filterPlatform) return false;
    if (filterFormat !== "all" && r.content_format !== filterFormat) return false;
    if (filterOrientation !== "all" && r.orientation !== filterOrientation) return false;
    if (filterStatus === "curated" && !r.is_curated) return false;
    if (filterStatus === "raw" && r.is_curated) return false;
    if (filterSearch && !r.record_id.toLowerCase().includes(filterSearch)) return false;

    if (filterClass !== "all") {
      const hasClass = r.effective_annotations.some((a) => a.class === filterClass);
      if (!hasClass) return false;
    }
    return true;
  });

  countBadge.textContent = `${filtered.length} frames (${allRecords.length} total)`;

  if (filtered.length === 0) {
    grid.innerHTML = "";
    emptyState.classList.remove("hidden");
    return;
  }
  emptyState.classList.add("hidden");

  grid.innerHTML = filtered
    .map((r) => {
      const statusBadge = r.is_curated
        ? `<span class="thumb-curated-badge">✓ Curated</span>`
        : `<span class="thumb-raw-badge">Raw</span>`;

      const boxCount = r.effective_annotations.length;
      const hasSkipAd = r.effective_annotations.some((a) => a.class === "skip_ad");
      const orientationLabel = r.orientation === "landscape" ? "Landscape 🔄" : "Portrait 📱";

      return `
      <div class="frame-card">
        <div class="frame-thumb-box" onclick="openEditor('${r.record_id}')">
          <img src="${r.image_url}" class="frame-thumb" loading="lazy" alt="${r.record_id}" />
          ${statusBadge}
          <span class="thumb-box-count">${boxCount} boxes</span>
        </div>
        <div class="frame-info">
          <div class="frame-id" title="${r.record_id}">${r.record_id}</div>
          <div class="frame-meta-tags">
            <span class="badge badge-platform">${r.platform}</span>
            <span class="badge">${r.device}</span>
            <span class="badge">${r.content_format === "short_video" ? "Shorts" : "Full Video"}</span>
            <span class="badge">${orientationLabel}</span>
            ${hasSkipAd ? '<span class="badge badge-skip-ad">Skip Ad</span>' : ""}
          </div>
          <div class="frame-actions">
            <button class="btn btn-xs btn-primary btn-block" onclick="openEditor('${r.record_id}')">
              ✏️ Inspect & Curate
            </button>
            <button class="btn btn-xs btn-ghost" onclick="deleteFrame('${r.platform}', '${r.device}', '${r.content_format}', '${r.record_id}')" title="Delete frame">
              🗑️
            </button>
          </div>
        </div>
      </div>
    `;
    })
    .join("");
}

// -------------------------------------------------------------
// 5. Interactive Annotation Canvas Editor
// -------------------------------------------------------------
function initCanvas() {
  canvas = document.getElementById("annotation-canvas");
  ctx = canvas.getContext("2d");

  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
}

function openEditor(recordId) {
  const rec = allRecords.find((r) => r.record_id === recordId);
  if (!rec) return;

  activeRecord = rec;
  currentAnnotations = JSON.parse(JSON.stringify(rec.effective_annotations));

  document.getElementById("modal-record-title").textContent = rec.record_id;
  document.getElementById("modal-meta-tag").textContent = `${rec.platform.toUpperCase()} &bull; ${rec.device.toUpperCase()} &bull; ${rec.content_format} &bull; ${rec.orientation.toUpperCase()}`;

  document.getElementById("curation-status-select").value = rec.curation_status || "approved";
  document.getElementById("curation-notes-input").value = rec.curation_notes || "";

  document.getElementById("editor-modal").classList.remove("hidden");

  // Load image onto canvas
  canvasImage = new Image();
  canvasImage.src = rec.image_url;
  canvasImage.onload = () => {
    canvas.width = canvasImage.naturalWidth;
    canvas.height = canvasImage.naturalHeight;
    redrawCanvas();
    renderAnnotationList();
  };
}

function closeEditor() {
  document.getElementById("editor-modal").classList.add("hidden");
  activeRecord = null;
}

function redrawCanvas() {
  if (!canvasImage) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(canvasImage, 0, 0);

  // Draw existing boxes
  currentAnnotations.forEach((ann) => {
    const [x1, y1, x2, y2] = ann.xyxy;
    const color = CLASS_COLORS[ann.class] || "#ffffff";

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(4, canvas.width * 0.003);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label tag
    ctx.fillStyle = color;
    const fontSize = Math.max(22, canvas.width * 0.015);
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const text = `${ann.class}`;
    const textWidth = ctx.measureText(text).width;

    ctx.fillRect(x1, Math.max(0, y1 - fontSize - 6), textWidth + 12, fontSize + 6);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x1 + 6, Math.max(fontSize, y1 - 4));
  });
}

function getCanvasCoordinates(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
}

function onMouseDown(e) {
  const [x, y] = getCanvasCoordinates(e);
  isDrawing = true;
  drawStartX = x;
  drawStartY = y;
}

function onMouseMove(e) {
  if (!isDrawing) return;
  const [currX, currY] = getCanvasCoordinates(e);

  redrawCanvas();

  const color = CLASS_COLORS[document.getElementById("draw-class-select").value] || "#3b82f6";
  ctx.strokeStyle = color;
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = Math.max(4, canvas.width * 0.003);
  ctx.strokeRect(
    Math.min(drawStartX, currX),
    Math.min(drawStartY, currY),
    Math.abs(currX - drawStartX),
    Math.abs(currY - drawStartY)
  );
  ctx.setLineDash([]);
}

function onMouseUp(e) {
  if (!isDrawing) return;
  isDrawing = false;
  const [currX, currY] = getCanvasCoordinates(e);

  const x1 = Math.round(Math.min(drawStartX, currX));
  const y1 = Math.round(Math.min(drawStartY, currY));
  const x2 = Math.round(Math.max(drawStartX, currX));
  const y2 = Math.round(Math.max(drawStartY, currY));

  // Minimum size check (must be at least 10px wide and tall)
  if (x2 - x1 > 10 && y2 - y1 > 10) {
    const selectedClass = document.getElementById("draw-class-select").value;
    currentAnnotations.push({
      class: selectedClass,
      xyxy: [x1, y1, x2, y2],
      source_label: selectedClass,
      descriptor: "Manual box curation",
      source: "studio_manual_curation",
      resource_id: "",
      visible_evidence: "",
    });
    redrawCanvas();
    renderAnnotationList();
  }
}

function renderAnnotationList() {
  const list = document.getElementById("annotation-list");
  document.getElementById("box-count").textContent = currentAnnotations.length;

  list.innerHTML = currentAnnotations
    .map((ann, idx) => {
      const color = CLASS_COLORS[ann.class] || "#ffffff";
      const [x1, y1, x2, y2] = ann.xyxy.map(Math.round);
      return `
      <div class="ann-item" style="border-left-color: ${color}">
        <div>
          <span class="ann-item-cls">${ann.class}</span>
          <div class="ann-item-coords">[${x1}, ${y1}, ${x2}, ${y2}]</div>
        </div>
        <button class="ann-delete-btn" onclick="deleteAnnotation(${idx})" title="Remove box">✕</button>
      </div>
    `;
    })
    .join("");
}

function deleteAnnotation(idx) {
  currentAnnotations.splice(idx, 1);
  redrawCanvas();
  renderAnnotationList();
}

async function saveCuration() {
  if (!activeRecord) return;
  const status = document.getElementById("curation-status-select").value;
  const notes = document.getElementById("curation-notes-input").value;

  const btn = document.getElementById("btn-save-curation");
  btn.disabled = true;
  btn.textContent = "Saving...";

  try {
    const res = await fetch(
      `/api/curate/${activeRecord.platform}/${activeRecord.device}/${activeRecord.content_format}/${activeRecord.record_id}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations: currentAnnotations, status, notes }),
      }
    );

    if (res.ok) {
      btn.textContent = "Saved!";
      setTimeout(() => {
        btn.textContent = "💾 Save Curation";
        btn.disabled = false;
        closeEditor();
        loadRecords();
      }, 700);
    } else {
      alert("Failed to save curation.");
      btn.disabled = false;
      btn.textContent = "💾 Save Curation";
    }
  } catch (err) {
    alert("Error: " + err.message);
    btn.disabled = false;
    btn.textContent = "💾 Save Curation";
  }
}

async function deleteFrame(platform, device, content_format, recordId) {
  if (
    !confirm(
      `Are you sure you want to delete ${recordId}? This removes the image and annotation records.`
    )
  )
    return;

  try {
    const res = await fetch(
      `/api/record/${platform}/${device}/${content_format}/${recordId}`,
      { method: "DELETE" }
    );
    if (res.ok) {
      if (activeRecord && activeRecord.record_id === recordId) {
        closeEditor();
      }
      loadRecords();
    }
  } catch (err) {
    alert("Failed to delete record: " + err.message);
  }
}

// -------------------------------------------------------------
// 6. Handover Package Modal
// -------------------------------------------------------------
function openPackageModal() {
  document.getElementById("package-modal").classList.remove("hidden");
  document.getElementById("pkg-result").classList.add("hidden");
}

function closePackageModal() {
  document.getElementById("package-modal").classList.add("hidden");
}

async function buildPackage() {
  const platform = document.getElementById("pkg-platform").value;
  const device = document.getElementById("pkg-device").value;
  const content_format = document.getElementById("pkg-format").value;

  const btn = document.getElementById("btn-build-package");
  btn.disabled = true;
  btn.textContent = "Compiling & Hashing...";

  try {
    const res = await fetch("/api/package", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ platform, device, content_format }),
    });
    const data = await res.json();

    btn.disabled = false;
    btn.textContent = "Build Package Now";

    if (data.success) {
      document.getElementById("pkg-result").classList.remove("hidden");
      document.getElementById("pkg-filename").textContent = data.zip_name;
      const mb = (data.file_size / (1024 * 1024)).toFixed(2);
      document.getElementById("pkg-size").textContent = `${mb} MB`;
      document.getElementById("pkg-download-link").href = data.download_url;
    } else {
      alert("Error: " + (data.detail || "Failed to package"));
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Build Package Now";
    alert("Packaging error: " + err.message);
  }
}

// -------------------------------------------------------------
// Setup Event Listeners
// -------------------------------------------------------------
function setupEventListeners() {
  document.getElementById("btn-refresh").addEventListener("click", () => {
    checkDevice();
    loadRecords();
  });

  const syncBtn = document.getElementById("btn-sync-device");
  if (syncBtn) {
    syncBtn.addEventListener("click", syncFromDevice);
  }

  document.getElementById("btn-toggle-stream").addEventListener("click", () => toggleStream());
  document.getElementById("btn-snap-frame").addEventListener("click", snapLiveFrame);

  document.getElementById("btn-trigger-capture").addEventListener("click", triggerSingleCapture);

  document.getElementById("btn-start-auto").addEventListener("click", startAutoCapture);
  document.getElementById("btn-stop-auto").addEventListener("click", stopAutoCapture);

  document.getElementById("btn-save-curation").addEventListener("click", saveCuration);

  document.getElementById("btn-delete-frame").addEventListener("click", () => {
    if (activeRecord) {
      deleteFrame(
        activeRecord.platform,
        activeRecord.device,
        activeRecord.content_format,
        activeRecord.record_id
      );
    }
  });

  document.getElementById("btn-open-package-modal").addEventListener("click", openPackageModal);

  document.getElementById("btn-build-package").addEventListener("click", buildPackage);

  // Filters
  ["filter-platform", "filter-format", "filter-orientation", "filter-class", "filter-status"].forEach((id) => {
    document.getElementById(id).addEventListener("change", renderGallery);
  });
  document.getElementById("filter-search").addEventListener("input", renderGallery);
}
