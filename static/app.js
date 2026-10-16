/**
 * Sentinel Gaze // Minimalist In-Browser Proctoring Controller
 * - Captures webcam and mic directly in the candidate's browser
 * - Streams frames & audio RMS to server via WebSocket
 * - Displays EXACT local OpenCV Proctoring HUD (details like mic, lips, gaze, score)
 * - Persistent audit vault & report generation
 */

// Application State
let ws = null;
let mediaStream = null;
let audioCtx = null;
let analyser = null;
let audioDataArray = null;
let frameInterval = null;
let localRenderInterval = null;
let timerInterval = null;
let isRunning = false;
let lastServerFrameTime = 0;

// Elements
const el = {
  testNameInput: document.getElementById("testNameInput"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  timerDisplay: document.getElementById("timerDisplay"),
  systemStatusBadge: document.getElementById("systemStatusBadge"),
  systemStatusText: document.getElementById("systemStatusText"),
  speechStateTag: document.getElementById("speechStateTag"),

  webcam: document.getElementById("webcam"),
  cameraCanvas: document.getElementById("cameraCanvas"),
  videoPlaceholder: document.getElementById("videoPlaceholder"),
  captureCanvas: document.getElementById("captureCanvas"),

  liveSpeechVal: document.getElementById("liveSpeechVal"),
  liveLipsVal: document.getElementById("liveLipsVal"),
  liveLinesVal: document.getElementById("liveLinesVal"),

  resultsSection: document.getElementById("resultsSection"),
  resultVerdictTag: document.getElementById("resultVerdictTag"),
  resultVerdictTitle: document.getElementById("resultVerdictTitle"),
  resultVerdictDesc: document.getElementById("resultVerdictDesc"),
  resultScore: document.getElementById("resultScore"),
  resultLines: document.getElementById("resultLines"),
  resultDuration: document.getElementById("resultDuration"),
  resultGraphImg: document.getElementById("resultGraphImg"),
  graphDownloadBtn: document.getElementById("graphDownloadBtn"),
  incidentsTableBody: document.getElementById("incidentsTableBody"),
  jsonDownloadBtn: document.getElementById("jsonDownloadBtn"),
  txtDownloadBtn: document.getElementById("txtDownloadBtn"),

  historyTableBody: document.getElementById("historyTableBody"),
  vaultCount: document.getElementById("vaultCount"),
};

// ─────────────────────────────────────────────────────────────────────────────
// Audio RMS Calculator
// ─────────────────────────────────────────────────────────────────────────────
function initAudioAnalyser(stream) {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    audioDataArray = new Float32Array(analyser.fftSize);
  } catch (err) {
    console.warn("Web Audio API not supported or mic unavailable:", err);
  }
}

function getAudioRMS() {
  if (!analyser || !audioDataArray) return 0.0;
  analyser.getFloatTimeDomainData(audioDataArray);
  let sum = 0.0;
  for (let i = 0; i < audioDataArray.length; i++) {
    sum += audioDataArray[i] * audioDataArray[i];
  }
  return Math.sqrt(sum / audioDataArray.length);
}

// ─────────────────────────────────────────────────────────────────────────────
// Start Assessment Session
// ─────────────────────────────────────────────────────────────────────────────
async function startSession() {
  const defaultName = `eval_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}_${new Date().toTimeString().slice(0, 5).replace(/:/g, "")}`;
  const testName = el.testNameInput.value.trim() || defaultName;
  el.testNameInput.value = testName;

  try {
    el.startBtn.disabled = true;
    el.startBtn.innerHTML = `<span>Activating Camera...</span>`;

    // 1. Capture Candidate's local camera & microphone directly in browser
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: true,
    });

    el.webcam.srcObject = mediaStream;
    initAudioAnalyser(mediaStream);

    try {
      await el.webcam.play();
    } catch (e) {
      console.log("Webcam play notice:", e);
    }

    // Show canvas immediately and hide placeholder
    el.cameraCanvas.style.display = "block";
    el.videoPlaceholder.style.display = "none";

    const displayCtx = el.cameraCanvas.getContext("2d");
    const captureCtx = el.captureCanvas.getContext("2d");

    // Local mirroring fallback loop: ensures canvas NEVER appears black or frozen
    localRenderInterval = setInterval(() => {
      if (!isRunning) return;
      // If server frame hasn't arrived within 300ms, keep drawing mirrored local webcam
      if (Date.now() - lastServerFrameTime > 300 && el.webcam.videoWidth > 0) {
        displayCtx.save();
        displayCtx.translate(640, 0);
        displayCtx.scale(-1, 1);
        displayCtx.drawImage(el.webcam, 0, 0, 640, 480);
        displayCtx.restore();
      }
    }, 40);

    // 2. Open WebSocket connection to Python backend
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws/session?test_name=${encodeURIComponent(testName)}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      isRunning = true;
      el.startBtn.style.display = "none";
      el.stopBtn.style.display = "inline-flex";
      el.stopBtn.disabled = false;
      el.testNameInput.disabled = true;

      el.systemStatusBadge.className = "status-badge";
      el.systemStatusText.textContent = "Proctoring Armed";
      el.speechStateTag.textContent = "HUD Live";

      // Live timer (ticks smoothly every second)
      const sessionStart = Date.now();
      timerInterval = setInterval(() => {
        if (!isRunning) return;
        const totalSec = Math.floor((Date.now() - sessionStart) / 1000);
        const mm = String(Math.floor(totalSec / 60)).padStart(2, "0");
        const ss = String(totalSec % 60).padStart(2, "0");
        el.timerDisplay.textContent = `${mm}:${ss}`;
      }, 500);

      // 3. Start sending frames at ~18 FPS (55ms interval)
      frameInterval = setInterval(() => {
        if (!isRunning || ws.readyState !== WebSocket.OPEN) return;
        if (el.webcam.videoWidth === 0) return;

        try {
          captureCtx.drawImage(el.webcam, 0, 0, 640, 480);
          const base64Data = el.captureCanvas.toDataURL("image/jpeg", 0.65);
          const rms = getAudioRMS();

          ws.send(JSON.stringify({ frame: base64Data, rms: rms }));
        } catch (e) {
          // Keep running smoothly even if a single frame drops
        }
      }, 55);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
          lastServerFrameTime = Date.now();
          if (data.frame) {
            const img = new Image();
            img.onload = () => {
              displayCtx.drawImage(img, 0, 0, 640, 480);
            };
            img.src = data.frame;
          }
          updateTelemetry(data);
        } else if (data.type === "report") {
          displayReport(data);
        }
      } catch (e) {
        console.error("Frame parse error:", e);
      }
    };

    ws.onerror = (err) => {
      console.warn("WebSocket status:", err);
    };

    ws.onclose = () => {
      console.log("WebSocket connection finalized");
      if (isRunning) {
        el.systemStatusText.textContent = "Connection Closed";
      }
    };

  } catch (err) {
    alert(`Could not start assessment: ${err.message}. Please check browser camera permissions.`);
    el.startBtn.disabled = false;
    el.startBtn.innerHTML = `<span>Start Assessment</span>`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop Assessment Session (Explicitly on user click)
// ─────────────────────────────────────────────────────────────────────────────
function stopSession() {
  if (!isRunning && (!ws || ws.readyState !== WebSocket.OPEN)) return;

  el.stopBtn.disabled = true;
  el.stopBtn.innerHTML = `<span>Finalizing...</span>`;

  // Stop frame capture loops & timer
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  if (frameInterval) {
    clearInterval(frameInterval);
    frameInterval = null;
  }
  if (localRenderInterval) {
    clearInterval(localRenderInterval);
    localRenderInterval = null;
  }

  // Request final report from server
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "stop" }));
  }

  // Stop hardware media tracks
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }

  isRunning = false;

  // Reset controls
  el.startBtn.style.display = "inline-flex";
  el.startBtn.disabled = false;
  el.startBtn.innerHTML = `
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
    <span>Start Assessment</span>`;
  el.stopBtn.style.display = "none";
  el.testNameInput.disabled = false;
  el.videoPlaceholder.style.display = "flex";
  el.cameraCanvas.style.display = "none";

  el.systemStatusText.textContent = "Completed";
  el.speechStateTag.textContent = "Camera Idle";
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-time Telemetry Updates
// ─────────────────────────────────────────────────────────────────────────────
function updateTelemetry(data) {
  el.timerDisplay.textContent = data.elapsed_formatted || "00:00";

  // Speech & Lips
  el.liveSpeechVal.textContent = data.is_speaking ? "Talking" : "Silent";
  el.liveSpeechVal.style.color = data.is_speaking ? "var(--accent-blue)" : "var(--text-muted)";
  el.liveLipsVal.textContent = data.lip_moving ? "Active" : "Still";
  el.liveLinesVal.textContent = data.n_cheat || 0;

  // Highlight active reading sentence in teleprompter
  highlightSentence(data.flow_streak);
}

function highlightSentence(streak) {
  const lineNum = Math.min(8, Math.max(1, (streak || 0) + 1));
  document.querySelectorAll(".sentence-row").forEach((row) => {
    const rowNum = parseInt(row.dataset.line, 10);
    row.classList.toggle("active", rowNum === lineNum && isRunning);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Display Assessment Report
// ─────────────────────────────────────────────────────────────────────────────
function displayReport(data) {
  el.resultsSection.style.display = "block";

  const meta = (data.report && data.report.test_metadata) || {};
  const score = meta.confidence_score || data.score || 0.0;
  const risk = (meta.risk_level || data.risk || "CLEAN").toUpperCase();
  const verdict = meta.final_verdict || data.verdict || "Assessment Finalized";

  el.resultsSection.className = "card results-card" + (risk === "CRITICAL" ? " critical" : "");
  el.resultVerdictTag.textContent = risk;
  el.resultVerdictTitle.textContent = verdict;

  if (score >= 0.70 || risk === "CRITICAL") {
    el.resultVerdictDesc.textContent =
      "High confidence script reading detected. Consecutive left-to-right eye saccades correlated with spoken reciting.";
  } else if (score >= 0.40) {
    el.resultVerdictDesc.textContent = "Suspicious staircase sweep activity observed during evaluation.";
  } else {
    el.resultVerdictDesc.textContent = "Candidate maintained natural conversational eye contact with webcam.";
  }

  el.resultScore.textContent = meta.confidence_percentage || `${(score * 100).toFixed(1)}%`;
  el.resultLines.textContent = meta.total_cheating_lines || 0;
  el.resultDuration.textContent = meta.duration_formatted || `${meta.duration_seconds || 0}s`;

  // Forensic Chart
  const chartUrl = data.graph_url || `/results/${meta.test_name}_graph.png?t=${Date.now()}`;
  el.resultGraphImg.src = chartUrl;
  el.graphDownloadBtn.href = chartUrl;

  // Downloads
  el.jsonDownloadBtn.href = data.report_url || `/results/${meta.test_name}_report.json`;
  el.txtDownloadBtn.href = data.summary_url || `/results/${meta.test_name}_summary.txt`;

  // Incidents Table
  const lines = (data.report && data.report.cheating_reading_lines) || [];
  el.incidentsTableBody.innerHTML = "";

  if (lines.length === 0) {
    el.incidentsTableBody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align:center; padding: 18px; color: var(--text-dim);">
          No cheating reading incidents recorded. Normal conversational behavior verified.
        </td>
      </tr>`;
  } else {
    lines.forEach((inc) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:600; color: #ffffff;">#${inc.incident_number}</td>
        <td>Line ${inc.line_number}</td>
        <td>${inc.timestamp_formatted}</td>
        <td>${inc.duration_seconds.toFixed(2)}s</td>
        <td>${inc.horizontal_span.toFixed(3)}</td>
        <td>${inc.vertical_step > 0 ? "+" : ""}${inc.vertical_step.toFixed(3)}</td>
        <td>${inc.candidate_speaking ? "Aloud" : "Silent"}</td>
      `;
      el.incidentsTableBody.appendChild(tr);
    });
  }

  // Scroll smoothly to results
  el.resultsSection.scrollIntoView({ behavior: "smooth" });

  // Refresh history list
  loadHistory();
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent History Vault
// ─────────────────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = await res.json();
    const sessions = data.sessions || [];

    el.vaultCount.textContent = sessions.length;
    el.historyTableBody.innerHTML = "";

    if (sessions.length === 0) {
      el.historyTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center; padding: 20px; color: var(--text-dim);">
            No saved sessions found in results/.
          </td>
        </tr>`;
      return;
    }

    sessions.forEach((sess) => {
      const tr = document.createElement("tr");
      const risk = (sess.risk_level || "CLEAN").toUpperCase();
      let tagClass = "tag-clean";
      if (risk === "CRITICAL") tagClass = "tag-critical";
      else if (risk === "HIGH" || risk === "LOW") tagClass = "tag-high";

      const dateStr = sess.timestamp ? new Date(sess.timestamp).toLocaleDateString() : "—";

      tr.innerHTML = `
        <td style="font-weight:600; color: #ffffff;">${sess.test_name}</td>
        <td>${dateStr}</td>
        <td>${sess.duration_formatted || `${sess.duration_seconds}s`}</td>
        <td><span class="tag-pill ${tagClass}">${risk}</span></td>
        <td style="font-family: var(--font-mono); font-weight:600;">${sess.confidence_percentage}</td>
        <td>${sess.total_cheating_lines} lines</td>
        <td>
          <button class="btn-inspect" data-test="${sess.test_name}">Inspect</button>
        </td>
      `;
      el.historyTableBody.appendChild(tr);
    });

    document.querySelectorAll(".btn-inspect").forEach((btn) => {
      btn.addEventListener("click", () => inspectPastTest(btn.dataset.test));
    });

  } catch (err) {
    console.error("History fetch error:", err);
  }
}

async function inspectPastTest(testName) {
  try {
    const res = await fetch(`/api/session/${testName}`);
    if (!res.ok) throw new Error("Could not load past test");
    const data = await res.json();
    displayReport(data);
  } catch (err) {
    alert(`Could not load '${testName}': ${err.message}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────
function init() {
  el.startBtn.addEventListener("click", startSession);
  el.stopBtn.addEventListener("click", stopSession);

  const todayStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const timeStr = new Date().toTimeString().slice(0, 5).replace(/:/g, "");
  el.testNameInput.placeholder = `manager_eval_${todayStr}_${timeStr}`;

  loadHistory();
}

document.addEventListener("DOMContentLoaded", init);
