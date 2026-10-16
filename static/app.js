/**
 * Sentinel Gaze // In-Browser MediaPipe 60 FPS Proctoring Controller
 * - Runs MediaPipe FaceMesh directly in candidate/manager browser via WebGL/WASM (0ms video lag!)
 * - Direct 60 FPS hardware-accelerated Canvas HUD (Mic levels, lip aperture, gaze crosshair, confidence bar)
 * - Streams lightweight 80-byte telemetry JSON to server over WebSocket
 * - Server runs ReadingPatternEngine with full mathematical accuracy and generates forensic charts
 */

// Application State
let ws = null;
let mediaStream = null;
let audioCtx = null;
let analyser = null;
let audioDataArray = null;
let faceMesh = null;
let cameraLoop = null;
let telemetryInterval = null;
let isRunning = false;
let sessionStartTime = 0;

// Telemetry & State Cache
let currentGaze = { rawX: 0.5, rawY: 0.5, screenX: 0.5, screenY: 0.5, pitch: 0.0, yaw: 0.0, detected: false };
let isSpeaking = false;
let lipMoving = false;
let currentScore = 0.0;
let currentStatus = "Calibrating posture...";
let currentLabel = "CALIBRATING";
let flowStreak = 0;
let driftX = 0.0;
let nCheat = 0;
let nClean = 0;

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
// In-Browser Visual Lip Motion Tracker (Visual VAD)
// ─────────────────────────────────────────────────────────────────────────────
class BrowserLipTracker {
  constructor() {
    this.rawBuf = [];
    this.historyInner = [];
    this.hangoverUntil = 0;
    this.isMoving = false;
    this.innerRatio = 0;
  }

  update(lm, t) {
    if (!lm) {
      this.isMoving = false;
      return false;
    }
    try {
      const mouthW = Math.max(Math.abs(lm[291].x - lm[61].x), 1e-4);
      const innerH = (Math.abs(lm[14].y - lm[13].y) * 0.50 +
                      Math.abs(lm[87].y - lm[82].y) * 0.25 +
                      Math.abs(lm[317].y - lm[312].y) * 0.25);
      const rawInner = innerH / mouthW;

      this.rawBuf.push(rawInner);
      if (this.rawBuf.length > 5) this.rawBuf.shift();

      const sorted = [...this.rawBuf].sort((a, b) => a - b);
      const smoothedInner = sorted[Math.floor(sorted.length / 2)];
      this.innerRatio = smoothedInner;

      this.historyInner.push(smoothedInner);
      if (this.historyInner.length > 24) this.historyInner.shift();

      if (this.historyInner.length >= 6) {
        const vals = this.historyInner;
        let sumDiff = 0;
        for (let i = vals.length - 5; i < vals.length; i++) {
          sumDiff += Math.abs(vals[i] - vals[i - 1]);
        }
        const marVel = sumDiff / 5;
        const recent = vals.slice(-10);
        const dynRange = Math.max(...recent) - Math.min(...recent);

        const speakingOpened = (smoothedInner >= 0.055 && (marVel >= 0.0030 || dynRange >= 0.020));
        const mouthWide = (smoothedInner >= 0.090);
        const syllableActive = (dynRange >= 0.045 && marVel >= 0.0040 && smoothedInner >= 0.035);

        const active = speakingOpened || mouthWide || syllableActive;
        if (active) {
          this.hangoverUntil = t + 0.35;
          this.isMoving = true;
        } else {
          this.isMoving = (t < this.hangoverUntil);
        }
      } else {
        this.isMoving = false;
      }
    } catch (e) {
      this.isMoving = false;
    }
    return this.isMoving;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// In-Browser Audio RMS & Multimodal VAD Fusion Tracker
// ─────────────────────────────────────────────────────────────────────────────
class BrowserAudioTracker {
  constructor() {
    this.ambient = 0.007;
    this.margin = 0.003;
    this.calSamples = [];
    this.noiseBuf = [];
    this.calibrated = false;
    this.isSpeaking = false;
    this.hangoverUntil = 0;
    this.currentRms = 0;
  }

  get sensitiveThreshold() {
    return Math.max(0.006, this.ambient + this.margin);
  }

  updateRms(rms) {
    this.currentRms = rms;
    this.noiseBuf.push(rms);
    if (this.noiseBuf.length > 30) this.noiseBuf.shift();

    if (!this.calibrated) {
      this.calSamples.push(rms);
      if (this.calSamples.length >= 15) {
        const sorted = [...this.calSamples].sort((a, b) => a - b);
        this.ambient = sorted[Math.floor(sorted.length * 0.20)];
        this.calibrated = true;
      }
      return;
    }

    if (this.noiseBuf.length >= 20) {
      const sorted = [...this.noiseBuf].sort((a, b) => a - b);
      const p15 = sorted[Math.floor(sorted.length * 0.15)];
      this.ambient = Math.min(0.025, Math.max(0.001, 0.98 * this.ambient + 0.02 * p15));
    }
  }

  evaluateSpeech(lipMoving, t) {
    if (!lipMoving) {
      if (t > this.hangoverUntil) this.isSpeaking = false;
      return this.isSpeaking;
    }
    if (this.currentRms >= this.sensitiveThreshold) {
      this.isSpeaking = true;
      this.hangoverUntil = t + 0.40;
    } else if (t > this.hangoverUntil) {
      this.isSpeaking = false;
    }
    return this.isSpeaking;
  }
}

const lipTracker = new BrowserLipTracker();
const audioTracker = new BrowserAudioTracker();

// ─────────────────────────────────────────────────────────────────────────────
// Eye Gaze Extraction (Exact Replica of OpenCV Local Detector with Blink Guard)
// ─────────────────────────────────────────────────────────────────────────────
const EYE_L = { inner: 133, outer: 33, top: 159, bot: 145 };
const EYE_R = { inner: 362, outer: 263, top: 386, bot: 374 };

function getEyeGaze(lm, irisIdx, eye) {
  const iris = lm[irisIdx];
  const inner = lm[eye.inner];
  const outer = lm[eye.outer];
  const top = lm[eye.top];
  const bot = lm[eye.bot];

  const leftX = Math.min(inner.x, outer.x);
  const rightX = Math.max(inner.x, outer.x);
  const topY = Math.min(top.y, bot.y);
  const botY = Math.max(top.y, bot.y);

  const w = rightX - leftX;
  const h = botY - topY;
  if (w < 1e-5 || h < 1e-5) return null;

  const hr = (iris.x - leftX) / w;   // 0..1 left-to-right in camera image
  const vr = (iris.y - topY) / h;    // 0..1 top-to-bottom
  const op = h / Math.max(w, 1e-5);  // Eyelid openness
  return { hr, vr, op };
}

function extractGaze(lm) {
  if (!lm || lm.length < 478) {
    return { detected: false, rawX: 0.5, rawY: 0.5, screenX: 0.5, screenY: 0.5, pitch: 0.0, yaw: 0.0 };
  }
  try {
    const le = getEyeGaze(lm, 468, EYE_L);
    const re = getEyeGaze(lm, 473, EYE_R);
    if (!le || !re) {
      return { detected: false, rawX: 0.5, rawY: 0.5, screenX: 0.5, screenY: 0.5, pitch: 0.0, yaw: 0.0 };
    }

    // Blink & eyelid occlusion guard (exact parity with anti_cheat_detector.py line 328)
    if (le.op < 0.18 || re.op < 0.18) {
      return { detected: false, rawX: 0.5, rawY: 0.5, screenX: 0.5, screenY: 0.5, pitch: 0.0, yaw: 0.0 };
    }

    const rawX = (le.hr + re.hr) / 2.0;
    const rawY = (le.vr + re.vr) / 2.0;

    // Head pose estimation in degrees (pitch, yaw)
    const midEyeX = (lm[33].x + lm[263].x) / 2.0;
    const midEyeY = (lm[33].y + lm[263].y) / 2.0;
    const faceW = Math.max(Math.abs(lm[263].x - lm[33].x), 1e-4);
    const faceH = Math.max(Math.abs(lm[152].y - midEyeY), 1e-4);

    const yaw = ((lm[1].x - midEyeX) / faceW) * 55.0;
    const pitch = ((lm[1].y - (midEyeY + faceH * 0.45)) / faceH) * 55.0;

    // Physical screen coordinates (1.0 - rawX so reading Left->Right across screen increases screenX)
    const screenX = Math.max(0.0, Math.min(1.0, 1.0 - rawX));
    const screenY = Math.max(0.0, Math.min(1.0, rawY));

    return {
      detected: true,
      rawX: rawX,
      rawY: rawY,
      screenX: screenX,
      screenY: screenY,
      pitch: pitch,
      yaw: yaw
    };
  } catch (e) {
    return { detected: false, rawX: 0.5, rawY: 0.5, screenX: 0.5, screenY: 0.5, pitch: 0.0, yaw: 0.0 };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Context Initialization
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
// 60 FPS Canvas HUD Renderer (Exact OpenCV Proctoring HUD Replica)
// ─────────────────────────────────────────────────────────────────────────────
function drawProctoringHUD(ctx, width, height, t, testName) {
  // 1. Draw Gaze Reticle at (gx, gy)
  if (currentGaze.detected) {
    const gx_px = currentGaze.screenX * width;
    const gy_px = currentGaze.screenY * height;

    // Glowing cyan/yellow reticle
    ctx.beginPath();
    ctx.arc(gx_px, gy_px, 12, 0, 2 * Math.PI);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#00dcf5";
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(gx_px, gy_px, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#00dcf5";
    ctx.fill();
  }

  // 2. Top HUD Bar Background
  ctx.fillStyle = "rgba(16, 16, 16, 0.94)";
  ctx.fillRect(0, 0, width, 115);

  ctx.strokeStyle = "rgba(70, 70, 70, 0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 115);
  ctx.lineTo(width, 115);
  ctx.stroke();

  // 3. Status Color Logic
  let scol = "#00dc00"; // Green
  if (currentScore >= 0.70) {
    scol = "#ff3333"; // Red
  } else if (currentScore >= 0.40) {
    scol = "#ffd700"; // Yellow
  } else if (nCheat >= 1 || currentLabel.includes("ALERT") || currentLabel.includes("PENDING")) {
    scol = "#ffff00";
  }

  // Row 1: Test Name & Live Clock [MM:SS.s]
  const sec = Math.max(0, t);
  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String((sec % 60).toFixed(1)).padStart(4, "0");
  const timeStr = `${mm}:${ss}`;

  ctx.fillStyle = "#aaaaaa";
  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.fillText(`TEST: ${testName}  [${timeStr}]`, 16, 24);

  // Row 2: Status
  ctx.fillStyle = scol;
  ctx.font = "bold 14px 'Inter', sans-serif";
  ctx.fillText(`STATUS: ${currentStatus}`, 16, 52);

  // Row 3: Telemetry Line
  ctx.fillStyle = "#cccccc";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(
    `Cheat-lines: ${nCheat}  |  Clean-speak: ${nClean}  |  Flow: ${flowStreak}L  |  Drift: +${driftX.toFixed(3)}  |  ${currentLabel}`,
    16,
    78
  );

  // Row 4: Speech & Lip VAD status
  const vadCol = isSpeaking ? "#00f0ff" : "#a0a0a0";
  const vadStr = isSpeaking ? "TALKING" : "SILENT";
  const lipStr = lipMoving ? "ACTIVE" : "STILL";
  const cRms = audioTracker.currentRms || 0.0;
  const sThr = audioTracker.sensitiveThreshold || 0.015;

  ctx.fillStyle = vadCol;
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(
    `SPEECH: ${vadStr} [LIPS: ${lipStr} | MIC: ${cRms.toFixed(3)}/${sThr.toFixed(3)}]`,
    16,
    102
  );

  // Audio RMS VU Meter Bar
  const bx = 325, by = 93, bw = 105, bh = 10;
  ctx.fillStyle = "#282828";
  ctx.fillRect(bx, by, bw, bh);

  const fr = Math.min(bw, Math.max(0, bw * (cRms / 0.050)));
  ctx.fillStyle = isSpeaking ? "#00f0ff" : "#888888";
  ctx.fillRect(bx, by, fr, bh);

  const tx = Math.min(bw, Math.max(0, bw * (sThr / 0.050)));
  ctx.strokeStyle = "#ff2222";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(bx + tx, by - 2);
  ctx.lineTo(bx + tx, by + bh + 2);
  ctx.stroke();

  // Confidence Bar (top-right)
  const cbw = 175;
  const cbx = width - cbw - 20;
  const cby = 16;

  ctx.fillStyle = "#282828";
  ctx.fillRect(cbx, cby, cbw, 16);

  const fc = Math.min(cbw, Math.max(0, cbw * currentScore));
  let mcol = "#00dc00";
  if (currentScore >= 0.70) mcol = "#ff3333";
  else if (currentScore >= 0.40) mcol = "#ffd700";

  ctx.fillStyle = mcol;
  ctx.fillRect(cbx, cby, fc, 16);

  ctx.fillStyle = "#dcdcdc";
  ctx.font = "12px 'Inter', sans-serif";
  const stateTag = !isSpeaking ? " [FROZEN]" : (currentLabel.includes("READING") ? " [ALERT]" : "");
  ctx.fillText(`Confidence: ${(currentScore * 100).toFixed(0)}%${stateTag}`, cbx, cby + 33);
}

// ─────────────────────────────────────────────────────────────────────────────
// Start Assessment Session (Direct In-Browser 60 FPS)
// ─────────────────────────────────────────────────────────────────────────────
async function startSession() {
  const defaultName = `eval_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}_${new Date().toTimeString().slice(0, 5).replace(/:/g, "")}`;
  const testName = el.testNameInput.value.trim() || defaultName;
  el.testNameInput.value = testName;

  try {
    el.startBtn.disabled = true;
    el.startBtn.innerHTML = `<span>Loading Camera...</span>`;

    // 1. Capture Candidate's local camera & microphone directly
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: true,
    });

    el.webcam.srcObject = mediaStream;
    initAudioAnalyser(mediaStream);

    try {
      await el.webcam.play();
    } catch (e) {}

    // Show canvas immediately and hide placeholder
    el.cameraCanvas.style.display = "block";
    el.videoPlaceholder.style.display = "none";

    const displayCtx = el.cameraCanvas.getContext("2d");

    // 2. Initialize in-browser FaceMesh engine
    el.startBtn.innerHTML = `<span>Initializing Vision...</span>`;
    if (!faceMesh) {
      faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });
      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults((results) => {
        if (!isRunning) return;

        const t = Math.max(0.001, (Date.now() - sessionStartTime) / 1000);
        const rms = getAudioRMS();
        audioTracker.updateRms(rms);

        // Mirror video frame directly to canvas at full 60 FPS
        displayCtx.save();
        displayCtx.translate(640, 0);
        displayCtx.scale(-1, 1);
        displayCtx.drawImage(results.image, 0, 0, 640, 480);
        displayCtx.restore();

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const lms = results.multiFaceLandmarks[0];
          currentGaze = extractGaze(lms);
          lipMoving = lipTracker.update(lms, t);
          isSpeaking = audioTracker.evaluateSpeech(lipMoving, t);
        } else {
          currentGaze.detected = false;
          lipMoving = false;
          isSpeaking = audioTracker.evaluateSpeech(false, t);
        }

        // Draw HUD overlay in real time
        drawProctoringHUD(displayCtx, 640, 480, t, testName);

        // Update HTML bottom metrics
        el.liveSpeechVal.textContent = isSpeaking ? "Talking" : "Silent";
        el.liveSpeechVal.style.color = isSpeaking ? "var(--accent-blue)" : "var(--text-muted)";
        el.liveLipsVal.textContent = lipMoving ? "Active" : "Still";
        el.liveLinesVal.textContent = nCheat;

        const sec = Math.max(0, t);
        const mm = String(Math.floor(sec / 60)).padStart(2, "0");
        const ss = String(Math.floor(sec % 60)).padStart(2, "0");
        el.timerDisplay.textContent = `${mm}:${ss}`;
      });
    }

    // 3. Open WebSocket to Python backend for ReadingPatternEngine
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws/session?test_name=${encodeURIComponent(testName)}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      isRunning = true;
      sessionStartTime = Date.now();

      el.startBtn.style.display = "none";
      el.stopBtn.style.display = "inline-flex";
      el.stopBtn.disabled = false;
      el.testNameInput.disabled = true;

      el.systemStatusBadge.className = "status-badge";
      el.systemStatusText.textContent = "Proctoring Armed";
      el.speechStateTag.textContent = "HUD Live";

      // 4. Send telemetry stream every 40ms (25 FPS) — only 80 bytes!
      telemetryInterval = setInterval(() => {
        if (!isRunning || ws.readyState !== WebSocket.OPEN) return;
        if (!currentGaze.detected) return; // Suppress during blinks!

        const t = Math.max(0.001, (Date.now() - sessionStartTime) / 1000);

        ws.send(JSON.stringify({
          action: "telemetry",
          t: t,
          raw_x: currentGaze.rawX,
          raw_y: currentGaze.rawY,
          gaze_x: currentGaze.screenX,
          gaze_y: currentGaze.screenY,
          pitch: currentGaze.pitch,
          yaw: currentGaze.yaw,
          is_speaking: isSpeaking,
          lip_moving: lipMoving,
          detected: true,
        }));
      }, 40);

      // 5. Run continuous camera detection loop
      async function processCamera() {
        if (!isRunning) return;
        if (el.webcam.readyState >= 2) {
          await faceMesh.send({ image: el.webcam });
        }
        requestAnimationFrame(processCamera);
      }
      requestAnimationFrame(processCamera);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "telemetry") {
          currentScore = data.score !== undefined ? data.score : currentScore;
          currentStatus = data.status || currentStatus;
          currentLabel = data.label || currentLabel;
          flowStreak = data.flow_streak !== undefined ? data.flow_streak : flowStreak;
          driftX = data.drift_x !== undefined ? data.drift_x : driftX;
          nCheat = data.n_cheat !== undefined ? data.n_cheat : nCheat;
          nClean = data.n_clean !== undefined ? data.n_clean : nClean;

          highlightSentence(flowStreak);
        } else if (data.type === "report") {
          displayReport(data);
        }
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onerror = (err) => {
      console.warn("WebSocket status:", err);
    };

    ws.onclose = () => {
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
// Stop Assessment Session
// ─────────────────────────────────────────────────────────────────────────────
function stopSession() {
  if (!isRunning && (!ws || ws.readyState !== WebSocket.OPEN)) return;

  el.stopBtn.disabled = true;
  el.stopBtn.innerHTML = `<span>Finalizing...</span>`;

  if (telemetryInterval) {
    clearInterval(telemetryInterval);
    telemetryInterval = null;
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
// Highlight Sentence in Reading Evaluation
// ─────────────────────────────────────────────────────────────────────────────
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
  const timePart = new Date().toTimeString().slice(0, 5).replace(/:/g, "");
  el.testNameInput.value = `manager_eval_${todayStr}_${timePart}`;

  loadHistory();
}

window.addEventListener("DOMContentLoaded", init);
