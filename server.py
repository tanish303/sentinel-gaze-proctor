"""
FastAPI Server for Remote & Internet-Deployed Anti-Cheat Proctoring
Provides:
  - Exact OpenCV Proctoring HUD streamed directly to candidate browser
  - In-browser camera and microphone via navigator.mediaDevices.getUserMedia
  - Real-time WebSocket frame & audio streaming (/ws/session)
  - Processing candidate frames with MediaPipe Face Landmarker & ReadingPatternEngine
  - Instant session persistence to results/ (*_report.json, *_graph.png, *_summary.txt)
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

app = FastAPI(title="Sentinel Gaze Proctor Server")

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
# Real-Time WebSocket Proctoring Endpoint with Exact Local OpenCV HUD
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await websocket.accept()

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
                try:
                    msg_text = await websocket.receive_text()
                    msg = json.loads(msg_text)
                    action = msg.get("action")

                    if action == "stop":
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

                    if action == "telemetry" or "gaze_x" in msg:
                        t = float(msg.get("t", time.perf_counter() - start_time))
                        t = max(0.001, t)
                        gx = float(msg.get("gaze_x", 0.5))
                        gy = float(msg.get("gaze_y", 0.5))
                        is_speaking = bool(msg.get("is_speaking", False))

                        n_cheat, score, status, label, tel = engine.update(t, gx, gy, is_speaking)
                        streak = tel.get("streak", 0)
                        drift = tel.get("drift_x", 0.0)
                        if np.isnan(drift): drift = 0.0
                        n_clean = len(engine.clean_speaking)

                        await websocket.send_json({
                            "type": "telemetry",
                            "score": round(score, 4),
                            "score_pct": f"{score*100:.1f}%",
                            "status": status,
                            "label": label,
                            "flow_streak": streak,
                            "drift_x": round(drift, 4),
                            "n_cheat": n_cheat,
                            "n_clean": n_clean,
                        })
                        continue

                    if "frame" in msg:
                        seq = int(msg.get("seq", 0))
                        base64_str = msg.get("frame", "")
                        if "," in base64_str:
                            base64_str = base64_str.split(",", 1)[1]

                        img_bytes = base64.b64decode(base64_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is None:
                            continue

                        t = max(0.001, time.perf_counter() - start_time)
                        ts_ = max(int(t * 1000), last_ts + 1)
                        last_ts = ts_

                        # Update audio RMS from candidate browser mic
                        rms = float(msg.get("rms", 0.0))
                        if np.isnan(rms): rms = 0.0
                        audio_tracker.update_rms(rms)

                        # Mirror frame horizontally for user view (just like OpenCV local)
                        disp = cv2.flip(frame, 1)
                        h, w, _ = frame.shape

                        score = float(np.clip(engine.score, 0.0, 1.0))
                        n_cheat = len(engine.cheating_lines)
                        n_clean = len(engine.clean_speaking)
                        status = "No face detected"
                        label = "WAIT"
                        tel = {"drift_x": 0.0, "step": 0.0, "pending": False, "streak": 0}
                        lip_moving = False
                        lip_mar = 0.0
                        gaze_x, gaze_y = 0.5, 0.5
                        is_speaking = False

                        try:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            result = lmk.detect_for_video(mp_img, ts_)

                            if result and result.face_landmarks:
                                lms = result.face_landmarks[0]
                                lip_moving, lip_mar = lip_tracker.update(lms, t)
                                is_speaking = audio_tracker.evaluate_speech(lip_moving, t)

                                gaze, (pitch, yaw) = extract_gaze(lms, w, h, engine.base_pose)
                                if t < engine.warmup:
                                    engine.record_warmup_pose(pitch, yaw)
                                if gaze is not None:
                                    gaze_x, gaze_y = gaze
                                    if not (np.isnan(gaze_x) or np.isnan(gaze_y)):
                                        n_cheat, score, status, label, tel = engine.update(t, gaze_x, gaze_y, is_speaking)

                                        # Draw Gaze crosshairs on flipped image (1.0 - gaze_x mirrors to disp)
                                        gx_px = int(np.clip((1.0 - gaze_x) * w, 5, w - 5))
                                        gy_px = int(np.clip(gaze_y * h, 5, h - 5))
                                        cv2.circle(disp, (gx_px, gy_px), 10, (0, 220, 255), 2, cv2.LINE_AA)
                                        cv2.circle(disp, (gx_px, gy_px), 3, (0, 220, 255), -1, cv2.LINE_AA)
                            else:
                                is_speaking = audio_tracker.evaluate_speech(False, t)
                        except Exception as proc_err:
                            print(f"[PROC ERR] {proc_err}", file=sys.stderr)

                        # ── Status colour (exact same as local anti_cheat_detector.py) ─────
                        if score >= 0.70:
                            scol = (0, 0, 255)
                        elif score >= 0.40:
                            scol = (0, 215, 255)
                        elif tel.get("pending", False) or n_cheat >= 1:
                            scol = (0, 255, 255)
                        else:
                            scol = (0, 220, 0)

                        # ── Exact OpenCV HUD (matching lines 998-1044) ────────────────────
                        cv2.rectangle(disp, (0, 0), (w, 115), (16, 16, 16), -1)
                        cv2.line(disp, (0, 115), (w, 115), (55, 55, 55), 1)

                        # Row 1: test name and live clock [MM:SS]
                        time_str = fmt_time(t) if t >= 0 else "00:00.0"
                        cv2.putText(disp, f"TEST: {clean_name}  [{time_str}]", (16, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

                        # Row 2: Status
                        cv2.putText(disp, f"STATUS: {status}", (16, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, scol, 2, cv2.LINE_AA)

                        # Row 3: line counts + telemetry
                        streak = tel.get("streak", 0)
                        drift = tel.get("drift_x", 0.0)
                        if np.isnan(drift): drift = 0.0
                        row3 = (f"Cheat-lines: {n_cheat}  |  Clean-speak: {n_clean}  |  "
                                f"Flow: {streak}L  |  Drift: +{drift:.3f}  |  {label}")
                        cv2.putText(disp, row3, (16, 78),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

                        # Row 4: Speech & Lip VAD status (bottom row)
                        vad_col = (0, 240, 255) if is_speaking else (160, 160, 160)
                        vad_str = "TALKING" if is_speaking else "SILENT"
                        lip_str = "ACTIVE" if lip_moving else "STILL"
                        c_rms = float(audio_tracker.current_rms) if not np.isnan(audio_tracker.current_rms) else 0.0
                        s_thr = float(audio_tracker.sensitive_threshold) if not np.isnan(audio_tracker.sensitive_threshold) else 0.01
                        cv2.putText(disp, f"SPEECH: {vad_str} [LIPS: {lip_str} | MIC: {c_rms:.3f}/{s_thr:.3f}]",
                                    (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, vad_col, 1, cv2.LINE_AA)

                        # Audio RMS bar
                        bx, by, bw, bh = 320, 93, 105, 10
                        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
                        fr = int(bw * float(np.clip(c_rms / 0.050, 0, 1)))
                        cv2.rectangle(disp, (bx, by), (bx + fr, by + bh), vad_col, -1)
                        tx = int(bx + bw * float(np.clip(s_thr / 0.050, 0, 1)))
                        cv2.line(disp, (tx, by - 2), (tx, by + bh + 2), (0, 0, 255), 2)
                        cv2.putText(disp, "[-/+] Sens", (bx + bw + 6, 102),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

                        # Confidence bar (top-right)
                        cbw = 175; cbx = w - cbw - 20; cby = 16
                        cv2.rectangle(disp, (cbx, cby), (cbx + cbw, cby + 16), (40, 40, 40), -1)
                        fc = int(cbw * float(np.clip(score, 0, 1)))
                        mc = (0, 0, 255) if score >= 0.70 else ((0, 215, 255) if score >= 0.40 else (0, 220, 0))
                        cv2.rectangle(disp, (cbx, cby), (cbx + fc, cby + 16), mc, -1)

                        state_tag = "FROZEN" if not is_speaking else ("ALERT" if "READING_CONFIRMED" in label else ("COOLING" if "SPEAKING_NATURAL" in label else ""))
                        tag_str = f" [{state_tag}]" if state_tag else ""
                        cv2.putText(disp, f"Confidence: {score*100:.0f}%{tag_str}", (cbx, cby + 33),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

                        # Encode annotated disp frame to JPEG
                        ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8") if ok else ""

                        await websocket.send_json({
                            "type": "frame",
                            "seq": seq,
                            "frame": annotated_b64,
                            "elapsed_formatted": time_str,
                            "score": round(score, 4),
                            "score_pct": f"{score*100:.1f}%",
                            "status": status,
                            "label": label,
                            "flow_streak": streak,
                            "n_cheat": n_cheat,
                            "n_clean": n_clean,
                            "is_speaking": bool(is_speaking),
                            "lip_moving": bool(lip_moving),
                        })

                except WebSocketDisconnect:
                    break
                except Exception as frame_loop_err:
                    import traceback
                    print(f"[WS FRAME LOOP ERR] {frame_loop_err}\n{traceback.format_exc()}", file=sys.stderr)
                    continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# REST Endpoints for History & Reports
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/sessions")
def api_list_sessions():
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


@app.get("/api/diagnose")
def api_diagnose():
    import traceback
    info = {"platform": sys.platform, "python": sys.version}
    try:
        info["model_exists"] = MODEL_PATH.exists()
        info["model_size"] = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0
    except Exception as e:
        info["model_err"] = str(e)

    try:
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
        )
        with vision.FaceLandmarker.create_from_options(opts) as lmk:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect_for_video(mp_img, 100)
            info["landmarker_detect_ok"] = True
    except Exception as e:
        info["landmarker_err"] = str(e)
        info["traceback"] = traceback.format_exc()

    return info


# Mount Static Files
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("   SENTINEL GAZE // EXACT LOCAL HUD STREAMING SERVER")
    print("="*70)
    port = int(os.environ.get("PORT", 8000))
    print(f"   Listening on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
