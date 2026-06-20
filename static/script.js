/* ── State ───────────────────────────────────────────────────────────────── */
let selectedFile = null;
let currentTab = "file";      // "file" | "youtube"
let currentResult = null;     // { summary, quiz }
let activeResultPane = "summary";

/* Step → progress-bar % mapping */
const STEP_PROGRESS = {
  download:   10,
  audio:      30,
  transcribe: 55,
  summarize:  80,
  quiz:       95,
};

/* ── Tab switching ────────────────────────────────────────────────────────── */
function switchTab(tab) {
  currentTab = tab;
  document.getElementById("tabFile").classList.toggle("active", tab === "file");
  document.getElementById("tabYT").classList.toggle("active", tab === "youtube");
  document.getElementById("zoneFile").classList.toggle("hidden", tab !== "file");
  document.getElementById("zoneYT").classList.toggle("hidden", tab !== "youtube");
  updateRunBtn();
}

/* ── File selection ───────────────────────────────────────────────────────── */
function onFileSelect(input) {
  const file = input.files[0];
  if (!file) return;

  if (file.size > 500 * 1024 * 1024) {
    showToast("File is too large (max 500 MB). Please compress or trim the video.");
    input.value = "";
    return;
  }

  selectedFile = file;
  document.getElementById("fileName").textContent = `✓ ${file.name}`;
  updateRunBtn();
}

function onYTInput(input) {
  updateRunBtn();
}

function updateRunBtn() {
  const hasFile = currentTab === "file" && selectedFile;
  const hasYT   = currentTab === "youtube" && document.getElementById("ytInput").value.trim().length > 10;
  document.getElementById("runBtn").disabled = !(hasFile || hasYT);
}

/* ── Drag & drop ──────────────────────────────────────────────────────────── */
const dropArea = document.getElementById("dropArea");

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragging");
});
dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragging");
});
dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) {
    // Simulate the file input selecting it
    const input = document.getElementById("fileInput");
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    onFileSelect(input);
  }
});

/* ── Start processing ─────────────────────────────────────────────────────── */
async function startProcessing() {
  const btn = document.getElementById("runBtn");
  btn.disabled = true;

  // Show progress card
  const progressCard = document.getElementById("progressCard");
  progressCard.classList.remove("hidden");
  setProgress(0, "Starting…");
  resetSteps();

  // Build form data
  const formData = new FormData();
  formData.append(
  "output_language",
  document.getElementById("outputLanguage").value
);
  if (currentTab === "file" && selectedFile) {
    formData.append("video", selectedFile);
  } else {
    const url = document.getElementById("ytInput").value.trim();
    formData.append("youtube", url);
    markStep("download", "active");
  }

  let jobId;
  try {
    const res = await fetch("/process", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      showError(data.error);
      btn.disabled = false;
      return;
    }
    jobId = data.job_id;
  } catch (e) {
    showError("Could not connect to the server. Is it running?");
    btn.disabled = false;
    return;
  }

  // Open SSE stream
  listenToJob(jobId);
}

/* ── SSE listener ─────────────────────────────────────────────────────────── */
function listenToJob(jobId) {
  const es = new EventSource(`/stream/${jobId}`);

  es.addEventListener("progress", (e) => {
    const { step, message } = JSON.parse(e.data);
    const pct = STEP_PROGRESS[step] ?? 20;
    setProgress(pct, message);
    markAllStepsBefore(step);
    markStep(step, "active");
  });

  es.addEventListener("done", (e) => {
    const { summary, quiz } = JSON.parse(e.data);
    es.close();

    // Mark all steps done
    Object.keys(STEP_PROGRESS).forEach((s) => markStep(s, "done"));
    setProgress(100, "Complete!");

    currentResult = { summary, quiz };

    setTimeout(() => {
      document.getElementById("progressCard").classList.add("hidden");
      renderResults(summary, quiz);
    }, 600);
  });

  es.addEventListener("error", (e) => {
    es.close();
    let msg = "An error occurred during processing.";
    try { msg = JSON.parse(e.data).message || msg; } catch (_) {}
    showError(msg);
    document.getElementById("runBtn").disabled = false;
  });

  // Native SSE error (connection lost)
  es.onerror = () => {
    es.close();
    showError("Lost connection to server. Please try again.");
    document.getElementById("runBtn").disabled = false;
  };
}

/* ── Progress helpers ─────────────────────────────────────────────────────── */
const STEP_ORDER = ["download", "audio", "transcribe", "summarize", "quiz"];

function resetSteps() {
  STEP_ORDER.forEach((s) => markStep(s, "pending"));
  // For file uploads, skip the download step visually
  if (currentTab === "file") {
    markStep("download", "done");
  }
}

function markStep(step, state) {
  const el = document.getElementById(`step-${step}`);
  if (!el) return;
  el.className = `step ${state}`;
}

function markAllStepsBefore(step) {
  const idx = STEP_ORDER.indexOf(step);
  for (let i = 0; i < idx; i++) {
    const s = document.getElementById(`step-${STEP_ORDER[i]}`);
    if (s && !s.classList.contains("done")) markStep(STEP_ORDER[i], "done");
  }
}

function setProgress(pct, msg) {
  document.getElementById("progressBar").style.width = pct + "%";
  document.getElementById("progressMsg").textContent = msg;
}

/* ── Render results ───────────────────────────────────────────────────────── */
function renderResults(summary, quiz) {
  document.getElementById("resultsCard").classList.remove("hidden");

  // Summary — render each bullet as a styled div
  const summaryPane = document.getElementById("paneSummary");
  summaryPane.innerHTML = "";
  const lines = summary.split("\n").filter((l) => l.trim());
  lines.forEach((line) => {
    const text = line.replace(/^•\s*/, "");
    const div = document.createElement("div");
    div.className = "bullet";
    div.innerHTML = `<span class="bullet-dot">◆</span><span>${escapeHtml(text)}</span>`;
    summaryPane.appendChild(div);
  });

  // Quiz — render each Q block
  const quizPane = document.getElementById("paneQuiz");
  quizPane.innerHTML = "";
  const blocks = quiz.split(/\n(?=Q\d+:)/);
  blocks.forEach((block) => {
    if (!block.trim()) return;
    const div = document.createElement("div");
    div.className = "q-block";
    const lines = block.split("\n").filter((l) => l.trim());
    lines.forEach((line) => {
      const p = document.createElement("div");
      if (/^Q\d+:/.test(line)) {
        p.className = "q-text";
        p.textContent = line;
      } else if (/^\s+[A-D]\)/.test(line)) {
        p.className = "q-option";
        p.textContent = line.trim();
      } else if (/✅/.test(line)) {
        p.className = "q-answer";
        p.textContent = line.trim();
      } else {
        p.className = "q-option";
        p.textContent = line.trim();
      }
      div.appendChild(p);
    });
    quizPane.appendChild(div);
  });

  showResult("summary");
}

function showResult(pane) {
  activeResultPane = pane;
  document.getElementById("rtabSummary").classList.toggle("active", pane === "summary");
  document.getElementById("rtabQuiz").classList.toggle("active", pane === "quiz");
  document.getElementById("paneSummary").classList.toggle("hidden", pane !== "summary");
  document.getElementById("paneQuiz").classList.toggle("hidden", pane !== "quiz");
}

/* ── Actions ──────────────────────────────────────────────────────────────── */
function copyResult() {
  if (!currentResult) return;
  const text =
    activeResultPane === "summary"
      ? currentResult.summary
      : currentResult.quiz;
  navigator.clipboard.writeText(text).then(() => {
    showToast("Copied to clipboard ✓", "info");
  });
}

function resetApp() {
  selectedFile = null;
  currentResult = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("fileName").textContent = "";
  document.getElementById("ytInput").value = "";
  document.getElementById("progressCard").classList.add("hidden");
  document.getElementById("resultsCard").classList.add("hidden");
  document.getElementById("runBtn").disabled = true;
  setProgress(0, "Initializing…");
  resetSteps();
  switchTab("file");
}

/* ── Error / toast ────────────────────────────────────────────────────────── */
function showError(msg) {
  document.getElementById("progressCard").classList.add("hidden");
  showToast(msg);
}

let toastTimer;
function showToast(msg, type = "error") {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.background = type === "info" ? "#0e1e16" : "#1e0e0e";
  toast.style.color = type === "info" ? "#63d3a6" : "#f07070";
  toast.style.borderColor = type === "info" ? "rgba(99,211,166,0.35)" : "rgba(240,112,112,0.35)";
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 4000);
}

/* ── Utils ────────────────────────────────────────────────────────────────── */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}