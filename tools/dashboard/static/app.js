// MediaControl Studio Dashboard Client

let allRecords = [];
let activeRecord = null;
let currentAnnotations = [];
let currentProposals = [];
let proposalDecisions = [];
let selectedAnnotation = null;
let editorDirty = false;
let suggestionRequest = 0;
let annotationClasses = [];
let geometryPolicies = {subscribe: "visible_button", skip_ad: "text_and_icon", progress_bar: "visible_track", playhead: "visible_marker", sound: "artwork", channel: "artwork", autoplay: "visible_toggle", time_display: "text_readout", more_videos: "composite_control"};

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
  previous: "#14b8a6", next: "#2dd4bf", seek_backward: "#a3e635", seek_forward: "#84cc16",
  settings: "#94a3b8", captions: "#c084fc", autoplay: "#38bdf8", time_display: "#cbd5e1",
  more_videos: "#a78bfa", show_cards: "#818cf8", ask: "#f472b6",
  minimize_player: "#a78bfa", cast: "#38bdf8", hide_controls: "#facc15",
  subscribe: "#f472b6", channel: "#38bdf8", download: "#818cf8", report_problem: "#fb7185",
};

// Canvas drawing state
let canvas, ctx;
let canvasImage = null;
let inspectedBox = null;
let isDrawing = false;
let drawStartX = 0;
let drawStartY = 0;

document.addEventListener("DOMContentLoaded", () => {
  initDevicePolling();
  loadRecords();
  setupEventListeners();
  initCanvas();
  loadAnnotationClasses();
});

async function loadAnnotationClasses() {
  annotationClasses = Object.keys(CLASS_COLORS);
  try {
    const response = await fetch("/api/annotation-classes");
    if (response.ok) {
      const result = await response.json();
      annotationClasses = result.classes;
      geometryPolicies = result.geometry_policies || geometryPolicies;
    }
  } catch (error) {
    console.warn("Using built-in annotation classes", error);
  }
  const drawSelect = document.getElementById("draw-class-select");
  drawSelect.replaceChildren(...annotationClasses.map((cls) => new Option(cls, cls)));
  const filter = document.getElementById("filter-class");
  filter.replaceChildren(new Option("All Classes", "all"), ...annotationClasses.map((cls) => new Option(cls, cls)));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
}

function recordAnnotations(record) {
  return record.display_annotations || record.effective_annotations;
}

function visualPolicy(cls) {
  return geometryPolicies[cls] || "icon_only";
}

function matchingBox(a, b) {
  return a.class === b.class && a.xyxy.every((value, index) => Math.abs(value-b.xyxy[index]) <= .01);
}

function sameControl(a, b) {
  if (a.class !== b.class) return false;
  if (boxIoU(a.xyxy, b.xyxy) >= .5) return true;
  const x = a.xyxy, y = b.xyxy;
  const intersection = Math.max(0, Math.min(x[2], y[2])-Math.max(x[0], y[0])) * Math.max(0, Math.min(x[3], y[3])-Math.max(x[1], y[1]));
  const smaller = Math.min((x[2]-x[0])*(x[3]-x[1]), (y[2]-y[0])*(y[3]-y[1]));
  const tolerance = Math.max(2, .15*Math.min(Math.max(x[2]-x[0], x[3]-x[1]), Math.max(y[2]-y[0], y[3]-y[1])));
  return intersection >= .85*smaller && Math.abs(x[0]+x[2]-y[0]-y[2])/2 <= tolerance && Math.abs(x[1]+x[3]-y[1]-y[3])/2 <= tolerance;
}

function loadEditorAnnotations(record) {
  const effective = JSON.parse(JSON.stringify(recordAnnotations(record)));
  currentAnnotations = effective.filter((ann) => !ann.review_required);
  proposalDecisions = JSON.parse(JSON.stringify(record.proposal_review || []));
  currentProposals = proposalDecisions.filter((item) => item.decision === "pending" && typeof item.class === "string" && Array.isArray(item.xyxy) && item.xyxy.length === 4);
  for (const annotation of effective.filter((ann) => ann.review_required)) {
    const candidate = {...annotation, proposal_kind: "native_capture",
      proposal_id: annotation.proposal_id || `native:${record.record_id}:${annotation.class}:${annotation.xyxy.map((value) => Number(value).toFixed(3)).join(",")}`};
    const prior = [...proposalDecisions].reverse().find((item) => item.proposal_id === candidate.proposal_id || (Array.isArray(item.xyxy) && matchingBox(item, candidate)));
    if (prior?.decision === "accepted") {
      currentAnnotations.push({...annotation, review_required: false, geometry_reviewed: true, reviewed_by: "human"});
    } else if (!prior || prior.decision === "pending") {
      if (!prior) proposalDecisions.push({...candidate, decision: "pending"});
      const existing = currentProposals.find((item) => matchingBox(item, candidate));
      if (existing) existing.proposal_kind = "native_capture";
      else currentProposals.push(candidate);
    }
  }
}

function reviewedGeometry(annotation, box, cls) {
  if (cls === "progress_bar" && canvas?.height) {
    const center = (box[1]+box[3])/2;
    const halfHeight = Math.max((box[3]-box[1])/2, Math.round(canvas.height*0.0135)/2);
    const boundedHeight = Math.min(halfHeight, center, canvas.height-center);
    box = [box[0], center-boundedHeight, box[2], center+boundedHeight];
  }
  const result = {...annotation, class: cls, xyxy: box, visual_xyxy: box, source: "studio_manual_curation",
    geometry_policy: visualPolicy(cls), geometry_source: "human_drawn", geometry_reviewed: true, review_required: false};
  if (!result.interaction_xyxy && annotation.xyxy && ["ios_xctest_accessibility", "studio_accessibility_matcher", "xctest", "uiautomator", "android_accessibility"].includes(annotation.source)) result.interaction_xyxy = annotation.xyxy;
  delete result.canonical_xyxy;
  delete result.visible_content_xyxy;
  delete result.geometry_policy_version;
  return result;
}

function boxIoU(a, b) {
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection;
  return intersection / Math.max(area, 1e-9);
}

function markEditorChanged() {
  editorDirty = true;
  document.getElementById("curation-status-select").value = "needs_review";
}

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
      statusTitle.textContent = "❌ " + (data.message || data.detail || "Capture failed");
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
    } else {
      stopAutoCapture();
      document.getElementById("capture-status-title").textContent = data.message || data.detail || "Capture needs attention; loop stopped.";
      document.getElementById("capture-log").textContent = data.log || data.message || data.detail;
      document.getElementById("capture-log").classList.remove("hidden");
    }
  } catch (err) {
    stopAutoCapture();
    document.getElementById("capture-status-title").textContent = "Capture request failed; loop stopped: " + err.message;
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
    log.textContent = "No capture log recorded in this session yet.\nClick 'Capture One Frame' or 'Start Auto-Capture' above to run the capture runner on the iPad.";
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
    if (res.ok && data.success) {
      const imported = data.synced_records ?? data.synced_images ?? 0;
      btn.innerHTML = imported
        ? `<span class="icon">✅</span> Imported ${imported} new`
        : `<span class="icon">✅</span> No new captures`;
      setTimeout(() => {
        btn.innerHTML = origText;
        btn.disabled = false;
      }, 2000);
      loadRecords();
    } else {
      alert(data.message || data.detail || "Capture import failed.");
      if (data.synced_records) loadRecords();
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
    if (filterStatus === "needs_review" && !["raw", "needs_review"].includes(r.curation_status)) return false;
    if (filterStatus === "approved" && !["approved", "resolved"].includes(r.curation_status)) return false;
    if (filterSearch && !r.record_id.toLowerCase().includes(filterSearch)) return false;

    if (filterClass !== "all") {
      const hasClass = recordAnnotations(r).some((a) => a.class === filterClass);
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
      const ready = ["approved", "resolved"].includes(r.curation_status);
      const statusBadge = ready
        ? `<span class="thumb-curated-badge">Reviewed</span>`
        : `<span class="thumb-raw-badge thumb-review-badge">${r.is_curated ? "Needs Review" : "Raw / Review"}</span>`;

      const boxCount = recordAnnotations(r).length;
      const hasSkipAd = recordAnnotations(r).some((a) => a.class === "skip_ad");
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
  window.addEventListener("mouseup", onMouseUp);
}

function openEditor(recordId) {
  if (activeRecord && editorDirty && !confirm("Discard unsaved annotation changes?")) return;
  const rec = allRecords.find((r) => r.record_id === recordId);
  if (!rec) return;

  activeRecord = rec;
  loadEditorAnnotations(rec);
  editorDirty = false;
  suggestionRequest++;
  fitEditorImage();
  editAnnotation(null);
  renderSuggestions();
  document.getElementById("suggestion-message").textContent = currentProposals.length ? `${currentProposals.length} candidate(s) need review. Accept correct boxes, reject incorrect ones, and draw missing targets.` : "";
  const suggestButton = document.getElementById("btn-suggest-annotations");
  suggestButton.disabled = false;
  suggestButton.textContent = "Suggest Missing Labels";
  document.getElementById("btn-refine-geometry").disabled = false;
  document.getElementById("show-interaction-boxes").checked = false;
  const quality = rec.annotation_quality || {};
  document.getElementById("annotation-quality-message").textContent =
    `${quality.duplicates_removed || 0} duplicate box(es) hidden. ${quality.invalid_boxes || 0} invalid box(es) excluded.\n${(quality.unverified_icon_regions || 0)+(quality.unverified_text_regions || 0)} tap region(s) need visible bounds checked.\n` +
    (rec.has_accessibility_snapshot ? "Accessibility snapshot is available." : "This capture has no full accessibility snapshot.");
  document.getElementById("diagnostics-link").href = `/api/annotation-diagnostics/${rec.platform}/${rec.device}/${rec.content_format}/${rec.record_id}`;

  document.getElementById("modal-record-title").textContent = rec.record_id;
  document.getElementById("modal-meta-tag").textContent = `${rec.platform.toUpperCase()} · ${rec.device.toUpperCase()} · ${rec.content_format} · ${rec.orientation.toUpperCase()}`;

  document.getElementById("curation-status-select").value = ["approved", "resolved", "needs_review"].includes(rec.curation_status) ? rec.curation_status : "needs_review";
  document.getElementById("curation-notes-input").value = rec.curation_notes || "";

  document.getElementById("editor-modal").classList.remove("hidden");

  // Load image onto canvas
  const image = new Image();
  canvasImage = image;
  image.onload = () => {
    if (activeRecord !== rec || canvasImage !== image) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    redrawCanvas();
    renderAnnotationList();
  };
  image.onerror = () => {
    if (activeRecord === rec) document.getElementById("annotation-quality-message").textContent = "Could not load this capture image.";
  };
  image.src = rec.image_url;
  if (rec.has_accessibility_snapshot || recordAnnotations(rec).some((ann) => ann.review_required)) requestAnnotationSuggestions("detect");
}

function closeEditor(force = false) {
  if (!force && editorDirty && !confirm("Discard unsaved annotation changes?")) return;
  document.getElementById("editor-modal").classList.add("hidden");
  activeRecord = null;
  isDrawing = false;
  editorDirty = false;
  suggestionRequest++;
}

function redrawCanvas() {
  if (!canvasImage || !canvasImage.complete || !canvasImage.naturalWidth) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(canvasImage, 0, 0);

  // Draw existing boxes
  const drawBox = (ann, index, pending = false) => {
    if (inspectedBox && !matchingBox(inspectedBox, ann)) return;
    const [x1, y1, x2, y2] = ann.xyxy;
    const color = CLASS_COLORS[ann.class] || "#ffffff";

    if (!pending && ann.interaction_xyxy && document.getElementById("show-interaction-boxes").checked) {
      const [tx1, ty1, tx2, ty2] = ann.interaction_xyxy;
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = Math.max(2, canvas.width*.0015);
      ctx.setLineDash([6, 10]);
      ctx.strokeRect(tx1, ty1, tx2-tx1, ty2-ty1);
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = inspectedBox ? 0.75 : Math.max(1.5, canvas.width * (index === selectedAnnotation && !pending ? 0.0024 : 0.0016));
    ctx.setLineDash(pending ? [12, 8] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (inspectedBox) return;

    // Label tag
    ctx.fillStyle = color;
    const fontSize = Math.max(16, canvas.width * 0.013);
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const text = `${pending ? "? " : ""}${ann.class}`;
    const textWidth = ctx.measureText(text).width;

    const tx = Math.min(x1, Math.max(0, canvas.width-textWidth-12));
    const ty = y1 >= fontSize+6 ? y1-fontSize-6 : Math.min(y2+2, canvas.height-fontSize-6);
    ctx.fillRect(tx, ty, textWidth + 12, fontSize + 6);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, tx + 6, ty + fontSize);
  };
  currentAnnotations.forEach((ann, index) => drawBox(ann, index));
  currentProposals.forEach((ann) => drawBox(ann, -1, true));
}

function fitEditorImage() {
  inspectedBox = null;
  const viewport = document.getElementById("canvas-container");
  viewport.classList.remove("zoomed");
  canvas.style.width = "";
  canvas.style.height = "";
  viewport.scrollLeft = 0;
  viewport.scrollTop = 0;
  document.getElementById("canvas-zoom-tools").classList.add("hidden");
  redrawCanvas();
}

function zoomToBox(index, pending = false) {
  const annotation = (pending ? currentProposals : currentAnnotations)[index];
  if (!annotation || !canvasImage?.complete || !canvasImage.naturalWidth) return;
  const viewport = document.getElementById("canvas-container");
  inspectedBox = annotation;
  viewport.classList.add("zoomed");
  canvas.style.width = `${2*canvas.width}px`;
  canvas.style.height = `${2*canvas.height}px`;
  const [x1, y1, x2, y2] = annotation.xyxy;
  viewport.scrollLeft = x1+x2-viewport.clientWidth/2;
  viewport.scrollTop = y1+y2-viewport.clientHeight/2;
  document.getElementById("canvas-zoom-label").textContent = `${annotation.class} · 200%`;
  document.getElementById("canvas-zoom-tools").classList.remove("hidden");
  redrawCanvas();
}

function getCanvasCoordinates(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [Math.max(0, Math.min(canvas.width, (e.clientX - rect.left) * scaleX)), Math.max(0, Math.min(canvas.height, (e.clientY - rect.top) * scaleY))];
}

function onMouseDown(e) {
  if (e.button !== 0 || !activeRecord || !canvasImage?.complete) return;
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

  const selectedClass = document.getElementById("draw-class-select").value;
  // A thin hand-drawn track can be expanded around its measured center.
  if (x2-x1 > 3 && y2-y1 >= (selectedClass === "progress_bar" ? 1 : 4)) {
    let annotation = {
      class: selectedClass,
      xyxy: [x1, y1, x2, y2],
      source_label: selectedClass,
      descriptor: "Manual box curation",
      source: "studio_manual_curation",
      resource_id: "",
      visible_evidence: "",
    };
    annotation = reviewedGeometry(selectedAnnotation === null ? annotation : currentAnnotations[selectedAnnotation], annotation.xyxy, selectedClass);
    if (currentAnnotations.some((old, index) => index !== selectedAnnotation && old.class === selectedClass && boxIoU(old.xyxy, annotation.xyxy) >= .98)) {
      document.getElementById("annotation-quality-message").textContent = "That box already exists. Select Edit to adjust it.";
      redrawCanvas();
      return;
    }
    if (selectedAnnotation === null) currentAnnotations.push(annotation);
    else currentAnnotations[selectedAnnotation] = annotation;
    if (inspectedBox) inspectedBox = annotation;
    markEditorChanged();
    if (selectedAnnotation !== null) editAnnotation(selectedAnnotation);
    redrawCanvas();
    renderAnnotationList();
  }
  redrawCanvas();
}

function renderAnnotationList() {
  const list = document.getElementById("annotation-list");
  document.getElementById("box-count").textContent = currentAnnotations.length;

  list.innerHTML = currentAnnotations
    .map((ann, idx) => {
      const color = CLASS_COLORS[ann.class] || "#ffffff";
      const [x1, y1, x2, y2] = ann.xyxy.map(Math.round);
      return `
      <div class="ann-item ${selectedAnnotation === idx ? "selected" : ""}" style="border-left-color: ${color}">
        <div class="ann-details">
          <span class="ann-item-cls">${escapeHtml(ann.class)}</span>
          <div class="ann-item-coords">[${x1}, ${y1}, ${x2}, ${y2}]</div>
          <div class="ann-source">${escapeHtml(ann.source || "source unknown")}</div>
          <div class="ann-source">${escapeHtml(ann.geometry_policy || visualPolicy(ann.class))}${ann.interaction_xyxy ? " · tap region preserved" : ""}</div>
        </div>
        <div class="review-actions">
          <button class="btn btn-xs btn-ghost" onclick="zoomToBox(${idx})">Zoom</button>
          <button class="btn btn-xs btn-ghost" onclick="editAnnotation(${idx})">Edit</button>
          <button class="ann-delete-btn" onclick="deleteAnnotation(${idx})" title="Remove box">✕</button>
        </div>
      </div>
    `;
    })
    .join("");
}

function deleteAnnotation(idx) {
  currentAnnotations.splice(idx, 1);
  markEditorChanged();
  editAnnotation(null);
  redrawCanvas();
  renderAnnotationList();
}

function editAnnotation(index) {
  selectedAnnotation = index;
  const editing = index !== null && currentAnnotations[index];
  document.getElementById("box-editor-heading").textContent = editing ? `Edit Box ${index + 1}` : "Add New Box";
  document.getElementById("box-editor-help").textContent = editing ? "Change class or coordinates, then apply. Drag on the image to replace this box." : "Select class, then drag over a visible control to draw its box.";
  document.getElementById("box-geometry-editor").classList.toggle("hidden", !editing);
  if (editing) {
    const select = document.getElementById("draw-class-select");
    if (![...select.options].some((option) => option.value === editing.class)) select.add(new Option(editing.class, editing.class));
    select.value = editing.class;
    ["box-x1", "box-y1", "box-x2", "box-y2"].forEach((id, i) => document.getElementById(id).value = editing.xyxy[i]);
  }
  if (canvas) {
    redrawCanvas();
    renderAnnotationList();
  }
}

function applyBoxChanges() {
  if (selectedAnnotation === null || !activeRecord) return;
  const box = ["box-x1", "box-y1", "box-x2", "box-y2"].map((id) => Number(document.getElementById(id).value));
  if (box.some((value) => !Number.isFinite(value)) || box[0] < 0 || box[1] < 0 || box[2] > canvas.width || box[3] > canvas.height || box[2] <= box[0] || box[3] <= box[1]) {
    alert("Enter a valid box inside the image.");
    return;
  }
  const cls = document.getElementById("draw-class-select").value;
  if (currentAnnotations.some((ann, i) => i !== selectedAnnotation && ann.class === cls && boxIoU(ann.xyxy, box) >= .98)) {
    alert("That annotation already exists.");
    return;
  }
  currentAnnotations[selectedAnnotation] = reviewedGeometry(currentAnnotations[selectedAnnotation], box, cls);
  markEditorChanged();
  if (inspectedBox) zoomToBox(selectedAnnotation);
  editAnnotation(selectedAnnotation);
}

async function suggestAnnotations() {
  return requestAnnotationSuggestions("suggest");
}

async function refineIconBoxes() {
  return requestAnnotationSuggestions("refine-geometry");
}

async function requestAnnotationSuggestions(kind) {
  if (!activeRecord) return;
  const rec = activeRecord;
  const request = ++suggestionRequest;
  const automatic = kind === "detect";
  const retainPending = (item) => kind !== "suggest" || ["native_capture", "geometry_recovery"].includes(item.proposal_kind);
  const requestPending = currentProposals.filter(retainPending);
  const button = document.getElementById("btn-suggest-annotations");
  button.disabled = true;
  document.getElementById("btn-refine-geometry").disabled = true;
  button.textContent = automatic ? "Checking Saved Capture..." : "Checking Image...";
  document.getElementById("suggestion-message").textContent = automatic ? "Checking the saved capture for missed controls and box sizing..." : kind === "refine-geometry" ? "Checking visible bounds and padding..." : "Checking accessibility, visible geometry and reviewed icon examples...";
  try {
    const response = await fetch(`/api/${kind}/${rec.platform}/${rec.device}/${rec.content_format}/${rec.record_id}`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({annotations: [...currentAnnotations, ...requestPending]})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Suggestion request failed");
    if (activeRecord !== rec || request !== suggestionRequest) return;
    // Read the live state after the response: the user may have accepted a
    // native candidate while this request was running.
    let retained = currentProposals.filter(retainPending);
    const retainedIds = new Set(retained.map((item) => item.proposal_id));
    let superseded = 0;
    proposalDecisions.forEach((item) => { if (item.decision === "pending" && !retainedIds.has(item.proposal_id)) { item.decision = "superseded"; superseded++; } });
    const incoming = [];
    for (const suggestion of result.proposals) {
      if (proposalDecisions.some((decision) => decision.decision === "rejected" && Array.isArray(decision.xyxy) &&
        (decision.proposal_id === suggestion.proposal_id || sameControl(decision, suggestion)))) continue;
      const item = {...suggestion};
      if (item.action === "replace_geometry") {
        const pending = retained.find((old) => matchingBox(old, item.replaces));
        const reviewed = currentAnnotations.some((old) => matchingBox(old, item.replaces));
        if (automatic && pending) {
          retained = retained.filter((old) => old !== pending);
          const decision = proposalDecisions.find((old) => old.proposal_id === pending.proposal_id && old.decision === "pending");
          if (decision) decision.decision = "superseded";
          if (inspectedBox === pending) inspectedBox = item;
          delete item.action;
          delete item.replaces;
        } else if (!reviewed && !pending) continue;
      } else if ([...currentAnnotations, ...retained, ...incoming].some((old) => sameControl(old, item))) continue;
      if (automatic) item.proposal_kind = "geometry_recovery";
      incoming.push(item);
    }
    for (const diagnostic of result.geometry_diagnostics || []) {
      if (!diagnostic.manual_geometry) continue;
      const candidate = retained.find((ann) => matchingBox(ann, diagnostic.manual_geometry));
      if (candidate) {
        candidate.manual_geometry = diagnostic.reason;
        const decision = proposalDecisions.find((ann) => ann.proposal_id === candidate.proposal_id && ann.decision === "pending");
        if (decision) decision.manual_geometry = diagnostic.reason;
      }
    }
    currentProposals = [...retained, ...incoming];
    proposalDecisions.push(...incoming.map((ann) => ({...ann, decision: "pending"})));
    if (incoming.length || superseded) {
      if (automatic) document.getElementById("curation-status-select").value = "needs_review";
      else markEditorChanged();
    }
    const reasons = (result.geometry_diagnostics || []).filter((item) => !currentProposals.some((proposal) => proposal.class === item.class)).map((item) => `${item.class}: ${item.reason}`);
    document.getElementById("suggestion-message").textContent = `${automatic ? `Recovered ${incoming.length} additional candidate(s). ` : ""}${currentProposals.length} suggestion(s) to review.\n${result.warnings.join("\n")}\n${[...new Set(reasons)].join("\n")}`;
    renderSuggestions();
    redrawCanvas();
  } catch (error) {
    if (activeRecord === rec && request === suggestionRequest) document.getElementById("suggestion-message").textContent = error.message;
  } finally {
    if (activeRecord === rec && request === suggestionRequest) {
      button.disabled = false;
      button.textContent = "Suggest Missing Labels";
      document.getElementById("btn-refine-geometry").disabled = false;
    }
  }
}

function renderSuggestions() {
  document.getElementById("suggestion-list").innerHTML = currentProposals.map((ann, index) => {
    const score = ann.proposal_kind === "native_capture" ? "Native iPad candidate" : ann.confidence !== undefined ? `Model score ${Math.round(ann.confidence * 100)}%` : ann.similarity !== undefined ? `Template similarity ${Math.round(ann.similarity * 100)}%` : ann.proposal_kind === "geometry_recovery" || ann.source === "studio_icon_geometry" ? "Geometry checked" : "Accessibility match";
    return `<div class="ann-item suggestion-item" style="border-left-color:${CLASS_COLORS[ann.class] || "#94a3b8"}">
      <strong>${escapeHtml(ann.class)}</strong><div class="ann-source">${escapeHtml(score)} · ${escapeHtml(ann.source)}</div>
      <div class="ann-source">${ann.action === "replace_geometry" ? "Replace existing box with visible geometry" : escapeHtml(ann.manual_geometry || ann.geometry_policy || visualPolicy(ann.class))}</div>
      <div class="ann-item-coords">[${ann.xyxy.map(Math.round).join(", ")}]</div>
      <div class="review-actions"><button class="btn btn-xs btn-ghost" onclick="zoomToBox(${index}, true)">Zoom</button><button class="btn btn-xs btn-success" onclick="reviewSuggestion(${index}, true)" ${ann.manual_geometry ? "disabled" : ""}>Accept</button><button class="btn btn-xs btn-ghost" onclick="reviewSuggestion(${index}, false)">Reject</button></div></div>`;
  }).join("");
}

function reviewSuggestion(index, accept) {
  const proposal = currentProposals[index];
  if (!proposal || (accept && proposal.manual_geometry)) return;
  const accepted = {...proposal, review_required: false, geometry_reviewed: true, reviewed_by: "human", reviewed_at: new Date().toISOString()};
  let replacedPending = null;
  if (accept && proposal.action === "replace_geometry") {
    const target = currentAnnotations.findIndex((ann) => matchingBox(ann, proposal.replaces));
    if (target === -1) {
      replacedPending = currentProposals.find((item) => item !== proposal && matchingBox(item, proposal.replaces));
      if (!replacedPending) {
        alert("The original box changed. Run Apply Annotation Rules again.");
        return;
      }
      currentAnnotations.push({...replacedPending, ...accepted});
      const originalDecision = proposalDecisions.find((item) => item.proposal_id === replacedPending.proposal_id && item.decision === "pending");
      if (originalDecision) originalDecision.decision = "superseded";
    } else {
      currentAnnotations[target] = {...currentAnnotations[target], ...accepted};
    }
  } else if (accept && !currentAnnotations.some((ann) => sameControl(ann, proposal))) {
    currentAnnotations.push(accepted);
  }
  const decision = [...proposalDecisions].reverse().find((item) => item.proposal_id === proposal.proposal_id && item.decision === "pending");
  if (decision) decision.decision = accept ? "accepted" : "rejected";
  currentProposals = currentProposals.filter((item) => item !== proposal && item !== replacedPending);
  if (inspectedBox === proposal && !accept) fitEditorImage();
  if (replacedPending && inspectedBox === replacedPending) inspectedBox = accepted;
  markEditorChanged();
  renderSuggestions();
  renderAnnotationList();
  redrawCanvas();
}

async function saveCuration() {
  if (!activeRecord) return;
  const status = document.getElementById("curation-status-select").value;
  const notes = document.getElementById("curation-notes-input").value;
  if (["approved", "resolved"].includes(status) && currentProposals.length) {
    alert("Accept or reject remaining suggestions, or save with needs_review status.");
    return;
  }

  const btn = document.getElementById("btn-save-curation");
  btn.disabled = true;
  btn.textContent = "Saving...";

  try {
    const res = await fetch(
      `/api/curate/${activeRecord.platform}/${activeRecord.device}/${activeRecord.content_format}/${activeRecord.record_id}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations: currentAnnotations, status, notes, reviewed_proposals: proposalDecisions }),
      }
    );

    if (res.ok) {
      btn.textContent = "Saved!";
      setTimeout(() => {
        btn.textContent = "💾 Save Curation";
        btn.disabled = false;
        closeEditor(true);
        loadRecords();
      }, 700);
    } else {
      const error = await res.json();
      alert(typeof error.detail === "string" ? error.detail : "Failed to save curation.");
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
        closeEditor(true);
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
  document.getElementById("btn-suggest-annotations").addEventListener("click", suggestAnnotations);
  document.getElementById("btn-refine-geometry").addEventListener("click", refineIconBoxes);
  document.getElementById("btn-fit-image").addEventListener("click", fitEditorImage);
  document.getElementById("show-interaction-boxes").addEventListener("change", redrawCanvas);
  document.getElementById("btn-update-box").addEventListener("click", applyBoxChanges);
  document.getElementById("btn-add-box-mode").addEventListener("click", () => editAnnotation(null));
  document.getElementById("curation-status-select").addEventListener("change", () => {editorDirty = true;});
  document.getElementById("curation-notes-input").addEventListener("input", () => {editorDirty = true;});

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
