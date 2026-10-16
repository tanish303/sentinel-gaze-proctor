"""
FastAPI Server for Remote & Internet-Deployed Anti-Cheat Proctoring
Supports:
  - In-browser camera and microphone via navigator.mediaDevices.getUserMedia
  - Real-time WebSocket frame & audio streaming (/ws/session)
  - Processing candidate frames with MediaPipe Face Landmarker & ReadingPatternEngine
  - Instant session persistence to results/ (*_report.json, *_graph.png, *_summary.txt)
  - Simple, minimalist, executive web UI
"""

import sys
import os
import time
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from anti_cheat_detector import (
    ReadingPatternEngine,
    LipMotionTracker,
    extract_gaze,
    fmt_time,
    save_session_deliverables,
    setup_model,
    MODEL_PATH,
    RESULTS_DIR,
)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

app = FastAPI(title="Minimalist Gaze Proctor Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Remote Audio Tracker (processes RMS computed directly in candidate's browser)
# ─────────────────────────────────────────────────────────────────────────────
class RemoteAudioTracker:
    def __init__(self):
        self.current_rms = 0.0
        self.ambient = 0.007
        self.margin = 0.0020
        self.is_speaking = False
        self.lip_moving = False
        self._noise_buf = deque(maxlen=40)
        self.hangover_until = 0.0
        self.calibrated = False
        self.cal_samples = []

    @property
    def sensitive_threshold(self):
        return max(0.006, self.ambient + self.margin)

    def update_rms(self, rms: float):
        self.current_rms = float(rms)
        self._noise_buf.append(self.current_rms)
        if not self.calibrated:
            self.cal_samples.append(self.current_rms)
            if len(self.cal_samples) >= 12:
                self.ambient = float(np.percentile(self.cal_samples, 20))
                self.calibrated = True
            return
        if len(self._noise_buf) >= 20:
            quiet_p15 = float(np.percentile(self._noise_buf, 15))
            self.ambient = float(np.clip(0.98 * self.ambient + 0.02 * quiet_p15, 0.001, 0.025))

    def evaluate_speech(self, lip_moving: bool, t: float) -> bool:
        self.lip_moving = bool(lip_moving)
        if not self.lip_moving:
            if t > self.hangover_until:
                self.is_speaking = False
            return self.is_speaking

        if self.current_rms >= self.sensitive_threshold:
            self.is_speaking = True
            self.hangover_until = t + 0.400
        elif t > self.hangover_until:
            self.is_speaking = False
        return self.is_speaking

    def adjust(self, delta: float):
        self.margin = float(np.clip(self.margin + delta, -0.005, 0.050))


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time WebSocket Proctoring Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await websocket.accept()
    
    # Query parameters
    test_name_param = websocket.query_params.get("test_name", "")
    clean_name = test_name_param.strip().replace(" ", "_") if test_name_param else ""
    if not clean_name:
        clean_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    setup_model()
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
    )

    engine = ReadingPatternEngine(warmup=2.5)
    lip_tracker = LipMotionTracker()
    audio_tracker = RemoteAudioTracker()

    start_time = time.perf_counter()
    last_ts = -1

    try:
        with vision.FaceLandmarker.create_from_options(opts) as lmk:
            while True:
                msg_text = await websocket.receive_text()
                msg = json.loads(msg_text)
                action = msg.get("action")

                if action == "stop":
                    # Finalize session
                    duration = time.perf_counter() - start_time
                    rpt, json_path, txt_path, graph_path, verdict, risk = save_session_deliverables(
                        clean_name, duration, engine
                    )
                    await websocket.send_json({
                        "type": "report",
                        "test_name": clean_name,
                        "report": rpt,
                        "verdict": verdict,
                        "risk": risk,
                        "score": round(engine.score, 4),
                        "score_pct": f"{engine.score*100:.1f}%",
                        "graph_url": f"/results/{clean_name}_graph.png?t={int(time.time())}",
                        "summary_url": f"/results/{clean_name}_summary.txt",
                        "report_url": f"/results/{clean_name}_report.json",
                    })
                    break

                if "frame" in msg:
                    # Decode base64 frame from browser
                    base64_str = msg["frame"]
                    if "," in base64_str:
                        base64_str = base64_str.split(",", 1)[1]
                    
                    try:
                        img_bytes = base64.b64decode(base64_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except Exception:
                        continue

                    if frame is None:
                        continue

                    t = time.perf_counter() - start_time
                    ts_ = max(int(t * 1000), last_ts + 1)
                    last_ts = ts_

                    # Update Audio RMS
                    rms = float(msg.get("rms", 0.0))
                    audio_tracker.update_rms(rms)

                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = lmk.detect_for_video(mp_img, ts_)

                    score = engine.score
                    n_cheat = len(engine.cheating_lines)
                    n_clean = len(engine.clean_speaking)
                    status = "No face detected"
                    label = "WAIT"
                    tel = {"drift_x": 0.0, "step": 0.0, "pending": False, "streak": 0}
                    lip_moving = False
                    lip_mar = 0.0
                    gaze_x, gaze_y = 0.5, 0.5

                    if result.face_landmarks:
                        lms = result.face_landmarks[0]
                        lip_moving, lip_mar = lip_tracker.update(lms, t)
                        is_speaking = audio_tracker.evaluate_speech(lip_moving, t)

                        gaze, (pitch, yaw) = extract_gaze(lms, w, h, engine.base_pose)
                        if t < engine.warmup:
                            engine.record_warmup_pose(pitch, yaw)
                        if gaze is not None:
                            gaze_x, gaze_y = gaze
                            n_cheat, score, status, label, tel = engine.update(t, gaze_x, gaze_y, is_speaking)
                    else:
                        is_speaking = audio_tracker.evaluate_speech(False, t)

                    await websocket.send_json({
                        "type": "telemetry",
                        "elapsed_formatted": fmt_time(t),
                        "elapsed_seconds": round(t, 2),
                        "score": round(score, 4),
                        "score_pct": f"{score*100:.1f}%",
                        "status": status,
                        "label": label,
                        "flow_streak": tel.get("streak", 0),
                        "n_cheat": n_cheat,
                        "n_clean": n_clean,
                        "is_speaking": bool(is_speaking),
                        "lip_moving": bool(lip_moving),
                        "lip_mar": round(float(lip_mar), 4),
                        "mic_rms": round(float(rms), 4),
                        "mic_threshold": round(float(audio_tracker.sensitive_threshold), 4),
                        "gaze_x": round(float(gaze_x), 3),
                        "gaze_y": round(float(gaze_y), 3),
                        "step": round(float(tel.get("step", 0.0)), 3),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# REST Endpoints for History & Reports
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/sessions")
def api_list_sessions():
    """Returns metadata for all persisted sessions stored in RESULTS_DIR."""
    sessions = []
    for report_file in RESULTS_DIR.glob("*_report.json"):
        test_name = report_file.name.replace("_report.json", "")
        graph_file = RESULTS_DIR / f"{test_name}_graph.png"
        summary_file = RESULTS_DIR / f"{test_name}_summary.txt"

        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("test_metadata", {})
            sessions.append({
                "test_name": meta.get("test_name", test_name),
                "timestamp": meta.get("timestamp", ""),
                "duration_formatted": meta.get("duration_formatted", ""),
                "duration_seconds": meta.get("duration_seconds", 0.0),
                "final_verdict": meta.get("final_verdict", "N/A"),
                "confidence_score": meta.get("confidence_score", 0.0),
                "confidence_percentage": meta.get("confidence_percentage", "0.0%"),
                "risk_level": meta.get("risk_level", "CLEAN"),
                "total_cheating_lines": meta.get("total_cheating_lines", 0),
                "has_graph": graph_file.exists(),
                "graph_url": f"/results/{test_name}_graph.png",
                "report_url": f"/results/{test_name}_report.json",
                "summary_url": f"/results/{test_name}_summary.txt" if summary_file.exists() else None,
            })
        except Exception:
            continue

    sessions.sort(key=lambda s: s["timestamp"], reverse=True)
    return {"total_sessions": len(sessions), "sessions": sessions}


@app.get("/api/session/{test_name}")
def api_get_session(test_name: str):
    report_file = RESULTS_DIR / f"{test_name}_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_file = RESULTS_DIR / f"{test_name}_summary.txt"
    summary_text = ""
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_text = f.read()

    return {
        "report": data,
        "summary_text": summary_text,
        "graph_url": f"/results/{test_name}_graph.png",
        "summary_url": f"/results/{test_name}_summary.txt",
        "report_url": f"/results/{test_name}_report.json",
    }


# Mount Static Files
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("   MINIMALIST GAZE PROCTOR SERVER (INTERNET READY)")
    print("="*70)
    print("   Local Address: http://127.0.0.1:8000")
    print("   Camera & Mic: Captured natively in client browser via getUserMedia")
    print("   Press Ctrl+C to terminate.")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
