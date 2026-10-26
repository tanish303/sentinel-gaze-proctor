"""
Anti-Cheat Gaze & Speech Proctoring Engine — v3.1 (High Precision)
Location: Desktop/anti cheating/anti_cheat_detector.py

Architecture:
  - 100% MediaPipe Face Landmarker for 478 3D facial & iris landmarks
  - Dual-eye coherent normalization: Left & right irises mapped synchronously
  - Head-pose correction (solvePnP) relative to neutral seated baseline
  - Velocity-based return saccade detection (separates real saccades from head drift)
  - Reading pattern: [L->R sweep] -> [fast return saccade] -> [downward step] -> [next L->R sweep]
  - Score rules (strictly enforced per user requirements):
      * When SILENT: score is 100% FROZEN (no decrease / no decay)
      * When SPEAKING CLEANLY: score cools down gradually (~1.0%/sec)
      * When READING: score increases (+0.12 aloud / +0.10 silent) per confirmed line
  - Detailed forensic report with timestamps, line numbers, and speech correlation
"""

import json
import shutil
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from multi_pillar_engine import MultiPillarProctorEngine, save_multi_pillar_deliverables

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = PROJECT_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ── MediaPipe landmark indices ───────────────────────────────────────────────
IRIS_L, IRIS_R = 468, 473          # left/right iris centers
EYE_L = dict(inner=133, outer=33, top=159, bot=145)
EYE_R = dict(inner=362, outer=263, top=386, bot=374)

# 6 stable skull points for solvePnP head-pose estimation
PNP_IDS = [1, 152, 33, 263, 61, 291]  # nose-tip, chin, eye corners, mouth corners
PNP_OBJ_PTS = np.array([
    (0.0,    0.0,    0.0),     # nose tip
    (0.0, -330.0,  -65.0),    # chin
    (-225.0, 170.0, -135.0),  # left eye outer
    ( 225.0, 170.0, -135.0),  # right eye outer
    (-150.0,-150.0, -125.0),  # left mouth corner
    ( 150.0,-150.0, -125.0),  # right mouth corner
], dtype=np.float64)

AUDIO_AVAILABLE = False
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except (ImportError, OSError, Exception):
    AUDIO_AVAILABLE = False


def fmt_time(seconds):
    """Formats seconds into MM:SS.s (e.g. 01:23.4) human-readable format."""
    sec = max(0.0, float(seconds))
    m = int(sec) // 60
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


def setup_model():
    if MODEL_PATH.exists():
        return
    alt = PROJECT_DIR.parent / "proctor" / "proctor" / "experiments" / "face_landmarker.task"
    if alt.exists():
        shutil.copy(alt, MODEL_PATH)
        return
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Visual Lip Motion Tracker (Visual VAD)
# ─────────────────────────────────────────────────────────────────────────────
class LipMotionTracker:
    """
    Tracks real-time phonetic lip modulation using MediaPipe inner/outer lip contours.
    Features:
      - 5-frame median filtering to eliminate single-frame landmark flicker and camera sensor noise.
      - Physical speech aperture thresholding:
          * Resting mouth (inner_ratio <= 0.035) -> STRICTLY STILL (100% immune to facial hair & noise).
          * Speech mouth (inner_ratio >= 0.055 with syllable velocity/range, or >= 0.090 open vowel) -> ACTIVE.
    """
    def __init__(self):
        self._raw_buf       = deque(maxlen=5)    # 5-frame median filter
        self._history_inner = deque(maxlen=24)   # ~0.8s sliding window of smoothed values
        self.is_moving      = False
        self.current_mar    = 0.0
        self.inner_ratio    = 0.0
        self.mar_velocity   = 0.0
        self.dyn_range      = 0.0
        self.hangover_until = 0.0

    def update(self, lm, t):
        if not lm:
            self.is_moving = False
            return False, 0.0

        try:
            # Lip landmark anchors:
            # 61: left mouth corner, 291: right mouth corner
            # 13, 14: inner upper/lower center
            # 82, 87: inner upper/lower left
            # 312, 317: inner upper/lower right
            # 0, 17: outer upper/lower center
            mouth_w = max(abs(lm[291].x - lm[61].x), 1e-4)

            inner_h = (abs(lm[14].y - lm[13].y) * 0.50 +
                       abs(lm[87].y - lm[82].y) * 0.25 +
                       abs(lm[317].y - lm[312].y) * 0.25)
            outer_h = abs(lm[17].y - lm[0].y)

            mar = (inner_h * 0.75 + outer_h * 0.25) / mouth_w
            raw_inner = inner_h / mouth_w
            self.current_mar = mar

            # 5-frame median filter eliminates single-frame pixel jitter & sensor noise
            self._raw_buf.append(raw_inner)
            smoothed_inner = float(np.median(self._raw_buf))
            self.inner_ratio = smoothed_inner
            self._history_inner.append(smoothed_inner)

            if len(self._history_inner) >= 6:
                vals = list(self._history_inner)
                diffs = np.abs(np.diff(vals[-6:]))
                self.mar_velocity = float(np.mean(diffs))
                self.dyn_range    = float(np.max(vals[-10:]) - np.min(vals[-10:]))

                # Phonetic speech articulation criteria:
                # 1. Mouth open for speech (>= 0.055) WITH syllable articulation dynamics
                speaking_opened = (smoothed_inner >= 0.055 and (self.mar_velocity >= 0.0030 or self.dyn_range >= 0.020))
                # 2. Wide open vowel articulation (>= 0.090)
                mouth_wide      = (smoothed_inner >= 0.090)
                # 3. Dynamic syllable oscillation (opening/closing across phonemes >= 0.045)
                syllable_active = (self.dyn_range >= 0.045 and self.mar_velocity >= 0.0040 and smoothed_inner >= 0.035)

                active = speaking_opened or mouth_wide or syllable_active

                if active:
                    self.hangover_until = t + 0.35
                    self.is_moving = True
                else:
                    self.is_moving = (t < self.hangover_until)
            else:
                self.is_moving = False
        except Exception:
            self.is_moving = False

        return self.is_moving, self.current_mar


# ─────────────────────────────────────────────────────────────────────────────
# Microphone tracker — sensitive audio with Audio-Visual VAD fusion
# ─────────────────────────────────────────────────────────────────────────────
class LiveMicrophoneTracker:
    def __init__(self):
        self.current_rms = 0.0
        self.ambient = 0.007
        self.margin = 0.0020         # Gentle offset: quiet/normal speaking voice detected effortlessly when lips move
        self.is_speaking = False
        self.lip_moving = False
        self.cal_samples = []
        self._noise_buf = deque(maxlen=40)
        self.calibrated = False
        self.hangover_until = 0.0
        self.stream = None

        if AUDIO_AVAILABLE:
            try:
                self.stream = sd.InputStream(
                    samplerate=16000, channels=1, blocksize=1600,
                    callback=self._cb)
                self.stream.start()
                print("[MIC] Microphone live. Audio-Visual Multimodal VAD active.")
            except Exception as e:
                print(f"[MIC] Warning: {e}")

    @property
    def sensitive_threshold(self):
        return max(0.006, self.ambient + self.margin)

    @property
    def threshold(self):
        return self.sensitive_threshold

    def _cb(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        self.current_rms = rms
        self._noise_buf.append(rms)

        # Calibrate for initial 1.5s: 20th percentile captures true quiet floor
        if not self.calibrated:
            self.cal_samples.append(rms)
            if len(self.cal_samples) >= 15:
                self.ambient = float(np.percentile(self.cal_samples, 20))
                self.calibrated = True
                print(f"[MIC] Calibrated -> ambient={self.ambient:.5f}, threshold={self.sensitive_threshold:.5f}")
            return

        # Adaptive ambient tracking: ONLY track the quietest 15% of the noise buffer
        # This prevents candidate speech volume from inflating the ambient baseline!
        if len(self._noise_buf) >= 20:
            quiet_p15 = float(np.percentile(self._noise_buf, 15))
            self.ambient = float(np.clip(0.98 * self.ambient + 0.02 * quiet_p15, 0.001, 0.025))

    def evaluate_speech(self, lip_moving, t):
        """
        Multimodal Audio-Visual Fusion VAD:
        - STRICT PHYSICAL GATE: If lips are STILL / CLOSED, candidate CANNOT be speaking words.
          Blocks 100% of fan noise, keyboard typing, chair squeaks, room echo, and external sounds.
        - When lips ARE moving: Sensitive low threshold confirms voice immediately, even for quiet voice.
        - Hangover (400ms) bridges natural pauses between syllables/words.
        """
        self.lip_moving = bool(lip_moving)

        if not AUDIO_AVAILABLE or self.stream is None:
            self.is_speaking = self.lip_moving
            return self.is_speaking

        # If lips are NOT moving, candidate is NOT speaking words.
        # Zero noise can trigger speech when lips are closed/resting!
        if not self.lip_moving:
            if t > self.hangover_until:
                self.is_speaking = False
            return self.is_speaking

        # Lips ARE actively moving: check if voice energy is detected
        has_voice = (self.current_rms >= self.sensitive_threshold)

        if has_voice:
            self.hangover_until = t + 0.40
            self.is_speaking = True
        else:
            self.is_speaking = (t < self.hangover_until)

        return self.is_speaking

    def adjust(self, delta):
        self.margin = float(np.clip(self.margin + delta, 0.0008, 0.025))
        print(f"[MIC] Adjusted: Margin={self.margin:.5f}  Threshold={self.sensitive_threshold:.5f}")

    def close(self):
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except: pass


# ─────────────────────────────────────────────────────────────────────────────
# Head-pose & Gaze extraction
# ─────────────────────────────────────────────────────────────────────────────
def get_head_pose(lm, img_w, img_h):
    """Returns (pitch, yaw, roll) in degrees using solvePnP."""
    img_pts = np.array(
        [(lm[i].x * img_w, lm[i].y * img_h) for i in PNP_IDS], dtype=np.float64)
    fl = float(img_w)
    cam = np.array([[fl, 0, img_w/2], [0, fl, img_h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(PNP_OBJ_PTS, img_pts, cam,
                                   np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    pm = cv2.hconcat([rmat, tvec])
    _, _, _, _, _, _, ea = cv2.decomposeProjectionMatrix(pm)
    return float(ea[0, 0]), float(ea[1, 0]), float(ea[2, 0])


def get_eye_gaze(lm, iris_idx, eye_dict):
    """
    Returns (h_ratio, v_ratio, openness) — iris position relative to eye socket.
    Normalized consistently so that for BOTH eyes:
      h_ratio: 0.0 = iris at leftmost corner in camera image, 1.0 = iris at rightmost corner
      v_ratio: 0.0 = iris at top eyelid, 1.0 = iris at bottom eyelid
    """
    iris = lm[iris_idx]
    inner = lm[eye_dict["inner"]]
    outer = lm[eye_dict["outer"]]
    top   = lm[eye_dict["top"]]
    bot   = lm[eye_dict["bot"]]

    left_x  = min(inner.x, outer.x)
    right_x = max(inner.x, outer.x)
    top_y   = min(top.y, bot.y)
    bot_y   = max(top.y, bot.y)

    w = right_x - left_x
    h = bot_y - top_y
    if w < 1e-5 or h < 1e-5:
        return None

    hr = (iris.x - left_x) / w      # 0..1 left-to-right in image
    vr = (iris.y - top_y)  / h      # 0..1 top-to-bottom
    op = h / max(w, 1e-5)
    return hr, vr, op


def normalize_angle_diff(a, b):
    diff = (a - b + 180.0) % 360.0 - 180.0
    return float(np.clip(diff, -35.0, 35.0))


def extract_gaze(lm, img_w, img_h, base_pose=None):
    """
    Returns head-pose-corrected (gaze_x, gaze_y) and raw (pitch, yaw).
      gaze_x increases when eyes sweep LEFT→RIGHT on physical screen
      gaze_y increases when eyes step DOWNWARD on physical screen
    Returns (None, (0,0)) if face/eyes not usable.
    """
    le = get_eye_gaze(lm, IRIS_L, EYE_L)
    re = get_eye_gaze(lm, IRIS_R, EYE_R)
    if le is None or re is None:
        return None, (0.0, 0.0)
    if le[2] < 0.18 or re[2] < 0.18:   # blink & eyelid occlusion guard
        return None, (0.0, 0.0)

    # Average both eyes
    raw_x = (le[0] + re[0]) / 2.0   # 0=left-in-camera, 1=right-in-camera
    raw_y = (le[1] + re[1]) / 2.0

    pitch, yaw = 0.0, 0.0
    try:
        pitch, yaw, _ = get_head_pose(lm, img_w, img_h)
    except Exception:
        pass

    if base_pose is not None:
        bp, by = base_pose
        dy = normalize_angle_diff(yaw, by)
        dp = normalize_angle_diff(pitch, bp)
        K_YAW   = 0.004
        K_PITCH = 0.004
        corr_x = raw_x - dy * K_YAW
        corr_y = raw_y - dp * K_PITCH
    else:
        corr_x, corr_y = raw_x, raw_y

    # Clamp to valid screen coordinates [0, 1]
    screen_x = float(np.clip(1.0 - corr_x, 0.0, 1.0))
    screen_y = float(np.clip(corr_y, 0.0, 1.0))
    return (screen_x, screen_y), (pitch, yaw)


# ─────────────────────────────────────────────────────────────────────────────
# Reading pattern engine — velocity-based, head-pose-aware
# ─────────────────────────────────────────────────────────────────────────────
class ReadingPatternEngine:
    """
    Detects the biological reading sawtooth pattern:
      1. Smooth L→R sweep (0.35–4.5 s, drift >= 0.020, nearly flat Y)
      2. Fast return saccade (velocity < -0.18 units/s OR sharp drop >= 0.018)
      3. Downward step to next line (0.004 <= dY <= 0.055)

    Score rules (strictly per user requirements):
      - Increases by 0.12 (aloud) / 0.10 (silent) per confirmed reading line
      - STRICTLY FROZEN when SILENT (not speaking) — zero decay
      - COOLS DOWN when SPEAKING CLEANLY (talking to camera, no reading) at ~1.0%/sec
      - HUD shows exact line numbers, clean speaking count, and freeze status
    """
    def __init__(self, warmup=2.5):
        self.warmup = warmup

        # Gaze history ringbuffer for velocity calculation
        self._hist_t = deque(maxlen=60)
        self._hist_x = deque(maxlen=60)
        self._hist_y = deque(maxlen=60)

        # Baseline pose calibration
        self._warmup_poses = []
        self.base_pose     = None

        # Full timeline for audit chart
        self.tl_t    = []
        self.tl_x    = []
        self.tl_y    = []
        self.tl_conf = []
        self.tl_spk  = []

        # Sweep tracker
        self.sw_start_x  = None
        self.sw_start_y  = None
        self.sw_start_t  = 0.0
        self.sw_peak_x   = None
        self.sw_peak_t   = 0.0
        self.sw_ys       = []       # Y history during current sweep
        self.sw_spk_cnt  = 0        # Number of speaking frames during sweep
        self.sw_tot_cnt  = 0        # Total frames during sweep

        # Saccade return tracker
        self.in_saccade    = False
        self.sacc_trough_x = None
        self.sacc_trough_t = 0.0
        self.sacc_trough_y = None

        # Multi-line state
        self.pending_line    = None   # Line 1 anchored, waiting for Line 2
        self.pending_age     = 0.0    # Time pending line was last confirmed
        self.flow_streak     = 0      # Continuous lines in teleprompter flow
        self.unconfirmed_rec = None   # Holds Line 2 until Line 3 confirms

        # Results
        self.cheating_lines  = []   # Confirmed reading incidents
        self.clean_speaking  = []   # Normal speech intervals
        self.score           = 0.0

        self.last_t          = 0.0
        self.last_read_t     = 0.0
        self.last_clean_t    = 0.0
        self.drift_x         = 0.0
        self.last_step       = 0.0
        self.confirmed_hold_until = 0.0
        self.confirmed_msg        = ""

    def record_warmup_pose(self, pitch, yaw):
        self._warmup_poses.append((pitch, yaw))
        if len(self._warmup_poses) >= 20 and self.base_pose is None:
            ps = [p[0] for p in self._warmup_poses]
            ys = [p[1] for p in self._warmup_poses]
            self.base_pose = (float(np.median(ps)), float(np.median(ys)))

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _smooth(self, buf, n=5):
        vals = list(buf)
        return float(np.median(vals[-n:])) if len(vals) >= n else vals[-1] if vals else 0.0

    def _velocity_x(self, window_s=0.18):
        """Returns horizontal gaze velocity (units/sec) over last ~window_s seconds."""
        if len(self._hist_t) < 4:
            return 0.0
        ts = np.array(self._hist_t)
        xs = np.array(self._hist_x)
        now = ts[-1]
        mask = ts >= (now - window_s)
        if mask.sum() < 2:
            return 0.0
        dt = ts[mask][-1] - ts[mask][0]
        if dt < 1e-4:
            return 0.0
        return float((xs[mask][-1] - xs[mask][0]) / dt)

    def _reset_sweep(self, x, y, t):
        self.sw_start_x = x
        self.sw_start_y = y
        self.sw_start_t = t
        self.sw_peak_x  = x
        self.sw_peak_t  = t
        self.sw_ys      = [y]
        self.sw_spk_cnt = 0
        self.sw_tot_cnt = 0
        self.in_saccade = False

    # ── Main update ──────────────────────────────────────────────────────────
    def update(self, t, gx, gy, speaking):
        self.tl_t.append(t)
        self.tl_x.append(gx)
        self.tl_y.append(gy)
        self.tl_spk.append(speaking)

        if t < self.warmup:
            self.tl_conf.append(0.0)
            self.last_t = t
            rem = max(0.0, self.warmup - t)
            return self._ret(f"Calibrating posture… ({rem:.1f}s)", "CALIBRATING")

        dt = max(t - self.last_t, 0.001)
        self.last_t = t

        # Push to history
        self._hist_t.append(t)
        self._hist_x.append(gx)
        self._hist_y.append(gy)

        sx = self._smooth(self._hist_x, 5)
        sy = self._smooth(self._hist_y, 5)

        if self.sw_start_x is None:
            self._reset_sweep(sx, sy, t)

        # Pending line anchor timeout: allow up to 14.0 seconds (natural 20-30 word sentence pace)
        if self.pending_line is not None and (t - self.pending_age) > 14.0:
            self.pending_line    = None
            self.flow_streak     = 0
            self.unconfirmed_rec = None

        vel_x = self._velocity_x()
        label  = "NORMAL"
        status = "Normal conversational posture"

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 1: READING SWEEP (eye drifts L→R across sentence)
        # ═════════════════════════════════════════════════════════════════════
        if not self.in_saccade:
            if sy > 0.20:
                self.sw_ys.append(sy)
                if len(self.sw_ys) > 250:
                    self.sw_ys.pop(0)

            self.sw_tot_cnt += 1
            if speaking:
                self.sw_spk_cnt += 1

            if sx > self.sw_peak_x:
                self.sw_peak_x = sx
                self.sw_peak_t = t

            self.drift_x   = self.sw_peak_x - self.sw_start_x
            sweep_dur      = t - self.sw_start_t
            drop_from_peak = self.sw_peak_x - sx

            if self.flow_streak >= 2:
                label  = "FLOW_TRACKING"
                status = f"Teleprompter flow active (Streak {self.flow_streak} lines) — tracking next line"
            elif self.pending_line:
                label  = "LINE_ANCHORED"
                status = f"Line 1 anchored — tracking Line 2 (Drift +{self.drift_x:.3f})"
            elif self.drift_x >= 0.030:
                label  = "SWEEP"
                status = f"Tracking reading sweep (Drift +{self.drift_x:.3f}, vel {vel_x:+.2f})"
            else:
                label  = "GAZE_CENTER"
                status = "Conversational eye posture"

            # ── Return saccade detection (snap back left) ────────────────
            fast_snap = (vel_x < -0.12 and self.drift_x >= 0.035)
            drop_back = (drop_from_peak >= 0.024 and (t - self.sw_peak_t) <= 0.80 and self.drift_x >= 0.035)
            rel_drop  = (drop_from_peak >= 0.45 * self.drift_x and self.drift_x >= 0.035)

            if fast_snap or drop_back or rel_drop:
                self.in_saccade    = True
                self.sacc_trough_x = sx
                self.sacc_trough_t = t
                self.sacc_trough_y = sy
                self.last_read_t   = t
                label              = "SACCADE_RETURN"
                status             = "Return saccade (snap left & down)"

            elif sweep_dur > 10.0 or (sx < self.sw_start_x - 0.06):
                self._reset_sweep(sx, sy, t)

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2: SACCADE RETURNING (Snap Down-Left to next line)
        # ═════════════════════════════════════════════════════════════════════
        else:
            if sx < self.sacc_trough_x:
                self.sacc_trough_x = sx
                self.sacc_trough_t = t
                self.sacc_trough_y = sy

            sacc_time = t - self.sacc_trough_t
            label  = "SACCADE_RETURN"
            status = "Return saccade (snap left & down to next line)"

            # Saccade ends when eye turns right or 320ms timeout
            if (sx > self.sacc_trough_x + 0.005) or sacc_time >= 0.32:
                completed_drift = self.sw_peak_x - self.sw_start_x
                valid_ys = [y for y in self.sw_ys if y > 0.20]
                y_iqr    = float(np.percentile(valid_ys, 75) - np.percentile(valid_ys, 25)) if len(valid_ys) >= 4 else 0.01
                y_median = float(np.median(valid_ys)) if valid_ys else sy
                dur      = self.sw_peak_t - self.sw_start_t

                # Did candidate speak during this sweep? (at least 35% of sweep vocalized)
                spk_ratio = (self.sw_spk_cnt / max(self.sw_tot_cnt, 1))
                was_speaking = spk_ratio >= 0.35

                # Duration: 0.75s to 10.0s (accommodates fast and normal reading paces)
                valid_sweep = (
                    completed_drift >= 0.035 and
                    y_iqr <= 0.050 and
                    0.75 <= dur <= 10.0
                )

                if valid_sweep:
                    cand = {
                        "span_x": round(completed_drift, 4),
                        "y":      round(y_median, 4),
                        "t":      round(self.sw_peak_t, 2),
                        "dur":    round(dur, 2),
                        "speaking": was_speaking,
                    }

                    # RULE: ONLY INCREASE CONFIDENCE WHEN CANDIDATE IS SPEAKING ALOUD!
                    # Silent reading/browsing (IDE, prompt, thinking) NEVER scores!
                    if not was_speaking:
                        self.pending_line    = None   # Silent reading breaks teleprompter flow
                        self.flow_streak     = 0
                        self.unconfirmed_rec = None
                        label  = "SILENT_READING"
                        status = "Silent reading/browsing (not delivering answer)"

                    elif self.pending_line is None:
                        # Line 1 read aloud: anchor it, flow streak = 1 (ZERO score increase)
                        self.pending_line = cand
                        self.pending_age  = t
                        self.flow_streak  = 1
                        label  = "LINE_ANCHORED"
                        status = f"Line 1 read aloud (span +{completed_drift:.3f}) -> tracking Line 2"
                        self.last_read_t = t

                    else:
                        step = cand["y"] - self.pending_line["y"]
                        self.last_step = step

                        # Accommodates laptop screens, auto-scrolling teleprompters, and natural gaze jitter (-0.025 to +0.085)
                        if -0.025 <= step <= 0.085:
                            self.flow_streak += 1
                            ln = self.flow_streak
                            rec = {
                                "line_number":       ln,
                                "timestamp_seconds": round(t, 2),
                                "description":       f"AI Teleprompter reading confirmed on Line #{ln} (step {step:+.3f}, span {completed_drift:.3f})",
                                "horizontal_span":   round(completed_drift, 4),
                                "vertical_step":     round(step, 4),
                                "duration_seconds":  round(cand["dur"], 2),
                                "speaking":          True,
                            }

                            if self.flow_streak == 2:
                                # 2 continuous lines in flow: FIRST SCORE INCREASE (+10%)!
                                self.cheating_lines.append(rec)
                                self.pending_line = cand
                                self.pending_age  = t
                                inc = 0.10
                                self.score = min(0.99, self.score + inc)
                                self.last_read_t = t

                                label  = "READING_CONFIRMED"
                                status = f">>> AI TELEPROMPTER FLOW Line #2 ({step:+.3f}) [+10%] <<<"
                                self.confirmed_hold_until = t + 2.5
                                self.confirmed_msg = status
                                print(f"\n[t={t:.1f}s] AI TELEPROMPTER FLOW CONFIRMED Line #2! Score={self.score*100:.1f}%\n")

                            elif self.flow_streak >= 3:
                                # Sustained continuous teleprompter flow: +10% per subsequent line
                                self.cheating_lines.append(rec)
                                self.pending_line = cand
                                self.pending_age  = t
                                inc = 0.10
                                self.score = min(0.99, self.score + inc)
                                self.last_read_t = t

                                label  = "READING_CONFIRMED"
                                status = f">>> AI TELEPROMPTER FLOW Line #{ln} ({step:+.3f}) [+10%] <<<"
                                self.confirmed_hold_until = t + 2.5
                                self.confirmed_msg = status
                                print(f"\n[t={t:.1f}s] AI TELEPROMPTER FLOW CONFIRMED Line #{ln}! Streak={self.flow_streak} | Score={self.score*100:.1f}%\n")

                        else:
                            # Step out of range: reset streak to 1
                            self.pending_line    = cand
                            self.pending_age     = t
                            self.flow_streak     = 1
                            self.unconfirmed_rec = None
                            label  = "LINE_RESET"
                            status = f"Gaze reset — new Line 1 (step {step:+.3f})"
                            self.last_read_t = t

                self._reset_sweep(self.sacc_trough_x, self.sacc_trough_y or sy, t)

        # ═════════════════════════════════════════════════════════════════════
        # SCORING DYNAMICS
        # ═════════════════════════════════════════════════════════════════════
        if label == "READING_CONFIRMED":
            pass

        elif not speaking:
            # ── SILENT: score is 100% STRICTLY FROZEN (zero decay, zero increase) ─
            if label not in ["LINE_ANCHORED", "VERIFYING_FLOW", "FLOW_TRACKING", "SWEEP", "SACCADE_RETURN"]:
                label  = "SILENT (Score frozen)"
                status = f"Candidate silent — score frozen at {self.score*100:.0f}%"

        else:
            # ── SPEAKING NATURALLY: candidate looking at camera/interviewer ─
            # Reading sweeps, pending lines, and return saccades are actively protected from decay!
            is_reading_active = (
                (label in ["SWEEP", "LINE_ANCHORED", "VERIFYING_FLOW", "FLOW_TRACKING", "READING_CONFIRMED", "SACCADE_RETURN"]) or
                (self.pending_line is not None) or
                (self.drift_x >= 0.025) or
                (t < self.confirmed_hold_until)
            )
            gaze_central = abs(sx - 0.50) < 0.18 and abs(sy - 0.50) < 0.20

            # Natural speaking: candidate looking directly at camera without reading sweeps
            if not is_reading_active and gaze_central:
                old_score = self.score
                if self.score > 0.0:
                    talk_decay = 0.010 * dt   # ~1.0%/sec
                    self.score = max(0.0, self.score - talk_decay)
                label  = "SPEAKING_NATURAL (Normal Conversational Speech)"
                status = f"Speaking naturally to interviewer (score {self.score*100:.0f}%)"

                if (t - self.last_clean_t) > 3.5:
                    self.last_clean_t = t
                    desc = f"Natural conversational speech (score cooled to {self.score*100:.1f}%)" if old_score > 0 else "Natural conversational speech to interviewer"
                    self.clean_speaking.append({
                        "timestamp_seconds": round(t, 2),
                        "description": desc
                    })

        # Retain confirmed alert on HUD for 2.5 seconds
        if t < self.confirmed_hold_until and self.confirmed_msg:
            status = self.confirmed_msg
            label  = "READING_CONFIRMED"

        self.tl_conf.append(self.score)
        return self._ret(status, label)

    def _ret(self, status, label):
        return (
            len(self.cheating_lines),
            self.score,
            status,
            label,
            {
                "drift_x": self.drift_x,
                "step": self.last_step,
                "pending": self.pending_line is not None,
                "streak": self.flow_streak,
            }
        )

    # ── Forensic chart ────────────────────────────────────────────────────────
    def save_chart(self, test_name, path):
        if len(self.tl_t) < 20:
            return
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        plt.subplots_adjust(hspace=0.30)

        t   = np.array(self.tl_t)
        x   = np.array(self.tl_x)
        y   = np.array(self.tl_y)
        c   = np.array(self.tl_conf) * 100

        ax_x, ax_y, ax_c = axes

        # Panel 1 — Horizontal (sawtooth)
        ax_x.plot(t, x, color="#2563eb", lw=1.1, label="Gaze X (Screen)")
        for r in self.cheating_lines:
            ax_x.axvline(r["timestamp_seconds"], color="#ef4444", ls="--", alpha=0.8)
            ax_x.text(r["timestamp_seconds"], float(np.max(x)), f" L{r['line_number']}", color="#dc2626", fontsize=8, fontweight="bold")
        ax_x.set_ylabel("Horizontal X", fontsize=9)
        ax_x.set_title(f"Anti-Cheat Forensic Analysis — {test_name}", fontsize=11, fontweight="bold")
        ax_x.legend(fontsize=8); ax_x.grid(True, ls=":", alpha=0.5)

        # Panel 2 — Vertical (staircase)
        ax_y.plot(t, y, color="#059669", lw=1.1, label="Gaze Y (Screen)")
        for r in self.cheating_lines:
            ax_y.axvline(r["timestamp_seconds"], color="#ef4444", ls="--", alpha=0.8)
        ax_y.set_ylabel("Vertical Y", fontsize=9)
        ax_y.legend(fontsize=8); ax_y.grid(True, ls=":", alpha=0.5)

        # Panel 3 — Confidence
        ax_c.plot(t, c, color="#dc2626", lw=1.5, label="Confidence Score (%)")
        for ev in self.clean_speaking:
            ax_c.axvline(ev["timestamp_seconds"], color="#10b981", ls=":", alpha=0.45, label="_nolegend_")
        ax_c.axhline(70, color="#f59e0b", ls=":", lw=1.2, label="Alert Threshold (70%)")
        ax_c.set_ylim(-2, 105)
        ax_c.set_ylabel("Confidence %", fontsize=9)
        ax_c.set_xlabel("Time (seconds)", fontsize=9)
        ax_c.legend(fontsize=8); ax_c.grid(True, ls=":", alpha=0.5)

        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[REPORT] Chart -> {path}")


def save_session_deliverables(test_name, duration, engine):
    """
    Saves JSON report, TXT audit summary, and PNG forensic chart to RESULTS_DIR.
    Returns (report_dict, json_path, txt_path, graph_path, verdict, risk).
    """
    final_score = engine.score
    n_cheat     = len(engine.cheating_lines)

    if final_score >= 0.70:
        verdict = "CHEATING / READING DETECTED (High Confidence)"
        risk    = "CRITICAL"
    elif final_score >= 0.40:
        verdict = "SUSPICIOUS ACTIVITY"
        risk    = "HIGH"
    elif n_cheat >= 1:
        verdict = "LOW RISK (Isolated Pattern)"
        risk    = "LOW"
    else:
        verdict = "CLEAN (Normal Conversational Speech)"
        risk    = "CLEAN"

    # JSON report
    rpt = {
        "test_metadata": {
            "test_name":              test_name,
            "timestamp":              datetime.now().isoformat(),
            "duration_formatted":     fmt_time(duration),
            "duration_seconds":       round(duration, 2),
            "final_verdict":          verdict,
            "confidence_score":       round(final_score, 4),
            "confidence_percentage":  f"{final_score*100:.1f}%",
            "risk_level":             risk,
            "total_cheating_lines":   n_cheat,
            "total_clean_speak_events": len(engine.clean_speaking),
        },
        "cheating_reading_lines": [
            {
                "incident_number":    idx + 1,
                "line_number":        r["line_number"],
                "timestamp_formatted": fmt_time(r["timestamp_seconds"]),
                "timestamp_seconds":  r["timestamp_seconds"],
                "description":        r["description"],
                "horizontal_span":    r["horizontal_span"],
                "vertical_step":      r["vertical_step"],
                "duration_seconds":   r["duration_seconds"],
                "candidate_speaking": r["speaking"],
            }
            for idx, r in enumerate(engine.cheating_lines)
        ],
        "clean_speaking_timeline": engine.clean_speaking,
    }
    json_path = RESULTS_DIR / f"{test_name}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rpt, f, indent=2)

    # TXT summary
    txt_path = RESULTS_DIR / f"{test_name}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        sep = "="*65
        f.write(sep + "\n")
        f.write(f"       AUDIT SUMMARY: {test_name}\n")
        f.write(sep + "\n")
        f.write(f"Timestamp        : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Duration         : {fmt_time(duration)} ({duration:.1f} s)\n")
        f.write(f"Final Verdict    : {verdict}\n")
        f.write(f"Confidence       : {final_score*100:.1f}% ({risk})\n\n")
        f.write("-"*65 + "\n")
        f.write(f"CHEATING / READING LINES DETECTED: {n_cheat}\n")
        f.write("-"*65 + "\n")
        if engine.cheating_lines:
            for idx, r in enumerate(engine.cheating_lines):
                sp = "Aloud" if r["speaking"] else "Silent"
                f.write(f"  Incident #{idx+1:>2} (Line #{r['line_number']:>2}) @ {fmt_time(r['timestamp_seconds'])} (t={r['timestamp_seconds']:>5.1f}s) | "
                        f"span={r['horizontal_span']:.3f}  step={r['vertical_step']:+.3f}  "
                        f"read_dur={r['duration_seconds']:.2f}s  [{sp}]\n")
        else:
            f.write("  None detected.\n")
        f.write("\n" + "-"*65 + "\n")
        f.write(f"CLEAN SPEAKING INTERVALS: {len(engine.clean_speaking)}\n")
        f.write("-"*65 + "\n")
        for ev in engine.clean_speaking[:25]:
            f.write(f"  @ {fmt_time(ev['timestamp_seconds'])} (t={ev['timestamp_seconds']:>5.1f}s): {ev['description']}\n")
        if not engine.clean_speaking:
            f.write("  None recorded.\n")
        f.write(sep + "\n")

    # Graph
    graph_path = RESULTS_DIR / f"{test_name}_graph.png"
    engine.save_chart(test_name, graph_path)

    return rpt, json_path, txt_path, graph_path, verdict, risk


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    print("\n" + "="*65)
    print("   ANTI-CHEAT 2D READING & SPEECH PROCTORING SYSTEM v3.1")
    print("="*65)

    default = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw     = input(f"Enter Test Name (Enter='{default}'): ").strip()
    test_name = (raw.replace(" ", "_") if raw else default)
    print(f"\n[CONFIG] Test: '{test_name}'  ->  Results: {RESULTS_DIR.resolve()}\n")

    setup_model()

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return

    mic         = LiveMicrophoneTracker()
    lip_tracker = LipMotionTracker()
    engine      = ReadingPatternEngine(warmup=2.5)
    multi_engine = MultiPillarProctorEngine(engine)
    start       = None
    last_ts     = -1

    # ── Build test-text canvas (Wide, long full-width lines) ──────────────────
    LINES = [
        "Line 1: When candidates take online technical assessments or live proctored interviews, eye gaze patterns reflect cognitive focus and visual attention across the screen.",
        "Line 2: As your eyes read continuously across each written sentence from left to right, the ocular muscles maintain a steady horizontal sweep speed across the full window.",
        "Line 3: Upon reaching the end of the line on the right side of the screen, the visual cortex initiates a rapid ballistic return saccade back to the beginning of the next sentence.",
        "Line 4: This rapid return jump is accompanied by a coordinated downward vertical step of several pixels to precisely align your focus with the start of the next line.",
        "Line 5: In contrast to natural conversational speaking where people look directly at the webcam, reading off an external script creates repeated consecutive staircase steps.",
        "Line 6: When you pause or remain completely silent between thoughts, the anti-cheat confidence score is strictly frozen in place without any premature decrease.",
        "Line 7: However, when you speak naturally and maintain direct conversational eye contact with the interviewer, the confidence score cools down safely over time.",
        "Line 8: Once you complete your reading evaluation or interview session, press Q or ESC to immediately finalize the detailed audit report and forensic graphs.",
    ]
    canvas = np.full((520, 1320, 3), (250, 250, 250), np.uint8)

    # Header banner
    cv2.rectangle(canvas, (0, 0), (1320, 48), (235, 238, 245), -1)
    cv2.line(canvas, (0, 48), (1320, 48), (200, 205, 215), 1)
    cv2.putText(canvas, "AI TELEPROMPTER READ-ALOUD TEST (Cheating Daddy / Parakeet AI Proctoring Engine):",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 30, 20), 2, cv2.LINE_AA)

    for i, ln in enumerate(LINES):
        y_pos = 92 + i * 48
        if i % 2 == 1:
            cv2.rectangle(canvas, (10, y_pos - 28), (1310, y_pos + 12), (242, 244, 248), -1)
        cv2.putText(canvas, ln, (22, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 1, cv2.LINE_AA)

    # Footer instructions
    cv2.line(canvas, (0, 480), (1320, 480), (210, 215, 225), 1)
    cv2.putText(canvas, "[READ ALOUD: Speak lines aloud into mic. Silent reading = FROZEN (0% up). Natural talk to webcam = COOLS DOWN.]",
                (20, 504), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1, cv2.LINE_AA)

    print("\nDETECTOR LIVE. Rules:")
    print("  • Audio-Visual VAD: Lip tracking + sensitive volume (no shouting needed)")
    print("  • Score up (+10% per line) ONLY when reciting aloud with L->R sweep in continuous flow")
    print("  • Silent candidate (thinking, browsing IDE, listening): Score 100% STRICTLY FROZEN (zero increase)")
    print("  • Natural conversational speaking (looking at webcam): Score COOLS DOWN at ~1.0%/s")
    print("  • [ / - = increase mic sensitivity;  ] / + = decrease")
    print("  • Q / ESC / Ctrl-C to finish\n" + "="*65 + "\n")

    try:
        with vision.FaceLandmarker.create_from_options(opts) as lmk:
            while True:
                ok, frame = cap.read()
                if not ok: break

                if start is None:
                    start = time.perf_counter()
                t    = time.perf_counter() - start
                ts_  = max(int(t * 1000), last_ts + 1)
                last_ts = ts_

                h, w, _ = frame.shape
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = lmk.detect_for_video(mp_img, ts_)
                disp     = cv2.flip(frame, 1)

                n_cheat = len(engine.cheating_lines)
                n_clean = len(engine.clean_speaking)
                score   = engine.score
                label   = "WAIT"
                status  = "No face detected"
                tel     = {"drift_x": 0.0, "step": 0.0, "pending": False}
                lip_moving = False
                lip_mar    = 0.0

                gx, gy = 0.5, 0.5
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    lip_moving, lip_mar = lip_tracker.update(lms, t)
                    is_speaking = mic.evaluate_speech(lip_moving, t)

                    gaze, (pitch, yaw) = extract_gaze(lms, w, h, engine.base_pose)
                    if t < engine.warmup:
                        engine.record_warmup_pose(pitch, yaw)
                    if gaze is not None:
                        gx, gy = gaze
                else:
                    is_speaking = mic.evaluate_speech(False, t)

                # Active prompt line from test text
                cur_cheat_lines = len(engine.cheating_lines)
                active_line_idx = min(len(LINES) - 1, max(0, cur_cheat_lines))
                active_prompt_text = LINES[active_line_idx]

                # Update 5-Pillar Engine
                tel_res = multi_engine.update(
                    t, gx, gy, is_speaking,
                    transcript="", wpm=0, active_prompt=active_prompt_text, rms=mic.current_rms
                )

                comp_score = tel_res["composite_score"]
                comp_pct = tel_res["composite_score_pct"]
                label = tel_res["label"]
                status = tel_res["status"]
                n_cheat = tel_res["n_cheat"]
                n_clean = tel_res["n_clean"]
                pillar_details = tel_res.get("pillar_details", {})

                # ── Multi-Pillar HUD (Top Composite + 5 Separate Horizontal Cards) ──
                hud_h = 135
                cv2.rectangle(disp, (0, 0), (w, hud_h), (14, 17, 23), -1)
                cv2.line(disp, (0, hud_h), (w, hud_h), (45, 52, 64), 1)

                # 1. Top Row: Final Composite Score (Mixed Score)
                cv2.putText(disp, f"TEST: {test_name}  [{fmt_time(t)}]", (12, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

                comp_col = (0, 220, 0)
                comp_tag = "CLEAN"
                if comp_score >= 0.70:
                    comp_col = (0, 0, 240)
                    comp_tag = "CRITICAL"
                elif comp_score >= 0.40:
                    comp_col = (0, 215, 255)
                    comp_tag = "REVIEW"
                elif comp_score >= 0.20:
                    comp_col = (0, 240, 255)
                    comp_tag = "MONITOR"

                cv2.putText(disp, f"FINAL COMPOSITE RISK: {comp_pct} [{comp_tag}]", (165, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, comp_col, 2, cv2.LINE_AA)

                # Top-Right Meter Bar
                mb_x, mb_y, mb_w, mb_h = w - 150, 8, 138, 12
                cv2.rectangle(disp, (mb_x, mb_y), (mb_x + mb_w, mb_y + mb_h), (35, 38, 45), -1)
                fill_w = int(mb_w * float(np.clip(comp_score, 0.0, 1.0)))
                cv2.rectangle(disp, (mb_x, mb_y), (mb_x + fill_w, mb_y + mb_h), comp_col, -1)

                # 2. Horizontal 5-Parameter Grid Row (y: 28 to 98)
                cw = (w - 18) // 5
                p_keys = [("gaze", "1. GAZE"), ("wpm", "2. WPM"), ("llm", "3. LLM"), ("echo", "4. ECHO"), ("cadence", "5. CADENCE")]

                for idx, (pkey, title) in enumerate(p_keys):
                    bx = 8 + idx * (cw + 2)
                    by = 28
                    bw = cw
                    bh = 70
                    pdata = pillar_details.get(pkey, {})
                    psc = pdata.get("score", 0.0)
                    psev = pdata.get("severity", "CLEAN")
                    pstat = pdata.get("status", "Normal")[:13]
                    pmetric = pdata.get("metric", "")[:14]

                    bcol = (50, 60, 75)
                    txtcol = (0, 220, 0)
                    if psev == "CRITICAL":
                        bcol = (0, 0, 230)
                        txtcol = (0, 0, 255)
                    elif psev == "WARN":
                        bcol = (0, 180, 240)
                        txtcol = (0, 215, 255)
                    elif psev == "MONITOR":
                        bcol = (200, 160, 60)
                        txtcol = (230, 180, 80)

                    cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (22, 26, 34), -1)
                    cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), bcol, 1)

                    cv2.putText(disp, title, (bx + 5, by + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.putText(disp, f"{psc*100:.0f}%", (bx + bw - 32, by + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, txtcol, 1, cv2.LINE_AA)

                    mbar_y = by + 22
                    mbar_w = bw - 10
                    cv2.rectangle(disp, (bx + 5, mbar_y), (bx + 5 + mbar_w, mbar_y + 4), (35, 40, 50), -1)
                    mbar_fill = int(mbar_w * float(np.clip(psc, 0.0, 1.0)))
                    if mbar_fill > 0:
                        cv2.rectangle(disp, (bx + 5, mbar_y), (bx + 5 + mbar_fill, mbar_y + 4), txtcol, -1)

                    cv2.putText(disp, pstat, (bx + 5, by + 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.31, (180, 180, 180), 1, cv2.LINE_AA)
                    cv2.putText(disp, pmetric, (bx + 5, by + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 140, 155), 1, cv2.LINE_AA)

                # 3. Row 3: Speech, Lips, Mic VU meter (y: 104 to 132)
                vad_col = (0, 240, 255) if mic.is_speaking else (160, 160, 160)
                vad_str = "TALKING" if mic.is_speaking else "SILENT"
                lip_str = "ACTIVE" if lip_moving else "STILL"
                cv2.putText(disp, f"SPEECH: {vad_str} [LIPS: {lip_str} | MIC: {mic.current_rms:.3f}/{mic.sensitive_threshold:.3f}]",
                            (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, vad_col, 1, cv2.LINE_AA)

                bx, by, bw, bh = 320, 112, 110, 10
                cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (35, 40, 48), -1)
                fr = int(bw * float(np.clip(mic.current_rms / 0.050, 0, 1)))
                cv2.rectangle(disp, (bx, by), (bx + fr, by + bh), vad_col, -1)
                tx = int(bx + bw * float(np.clip(mic.sensitive_threshold / 0.050, 0, 1)))
                cv2.line(disp, (tx, by - 2), (tx, by + bh + 2), (0, 0, 255), 2)
                cv2.putText(disp, "[-/+] Sens", (bx + bw + 8, 121),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.31, (130, 130, 130), 1, cv2.LINE_AA)

                cv2.imshow("Anti-Cheat Proctoring", disp)
                cv2.imshow("Test Text", canvas)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                elif k in (ord('['), ord('-')):
                    mic.adjust(-0.001)
                elif k in (ord(']'), ord('+'), ord('=')):
                    mic.adjust(+0.001)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mic.close()
        cap.release()
        cv2.destroyAllWindows()
        duration = (time.time() - start) if start else 0.0
        rpt, json_path, txt_path, graph_path, verdict, risk = save_multi_pillar_deliverables(test_name, duration, multi_engine, RESULTS_DIR)

        # Terminal scorecard
        print("\n" + "="*65)
        print(f"          FINAL 5-PILLAR SCORECARD — {test_name}")
        print("="*65)
        print(f"  Duration          : {fmt_time(duration)} ({duration:.1f} s)")
        print(f"  Final Verdict     : {verdict}")
        print(f"  Composite Score   : {multi_engine.composite_score*100:.1f}% ({risk})")
        print("  " + "-"*50)
        print("  Pillar Sub-Scores :")
        for k, v in rpt["pillar_scorecard"].items():
            print(f"    - {k:28}: {v['sub_score_percentage']} [{v['status']}]")
        print("-"*65)
        print(f"  [1] JSON Report   : {json_path}")
        print(f"  [2] Text Summary  : {txt_path}")
        print(f"  [3] Forensic Chart: {graph_path}")
        print("="*65 + "\n")


if __name__ == "__main__":
    main()
