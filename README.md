# Anti-Cheating Reading & Speech Proctoring Engine — v3.1 (High Precision)

**Directory:** `C:\Users\tanis\OneDrive\Desktop\anti cheating`

---

## The Threat Model: AI Teleprompter Cheating (Cheating Daddy, Parakeet AI)

During live technical interviews and coding rounds, dishonest candidates use transparent overlay tools (such as **Cheating Daddy**, **Parakeet AI**, or **Interview Coder**) to display AI-generated solutions on their screen.

To deliver the answer, the candidate **must speak the AI text aloud into the microphone** while scanning each line across their screen.

### The 3 Proctoring States:

| State | Candidate Behavior | Physical Kinematics | Proctoring Score Action |
|---|---|---|---|
| **1. AI Teleprompter Reading** | Candidate recites the AI answer aloud into the mic | **Continuous multi-line cadence**: Sweep L $\to$ R across Line 1 $\to$ Snap Down-Left to Line 2 $\to$ Snap Down-Left to Line 3 while vocalizing aloud ($\Delta Y > 0$). | **CONFIDENCE INCREASES (+20% on Line 3 confirmation, +15% per subsequent line)**. Requires sustained cadence (3+ lines) to eliminate false triggers from casual 2-line glances. |
| **2. Silent Thinking / Browsing** | Candidate reading problem statement, reading code in IDE, API docs, or thinking | Gaze may scan text, but candidate is **NOT SPEAKING** (`is_speaking == False`). | **CONFIDENCE 100% STRICTLY FROZEN** (0% increase, 0% decrease). Silent work is normal and never penalized. |
| **3. Natural Conversational Speech** | Candidate answering from memory / thoughts | Candidate is **SPEAKING** (`is_speaking == True`) while maintaining gaze towards the webcam/interviewer. No line-by-line staircase. | **CONFIDENCE COOLS DOWN (~1.0%/s)**. Rewards honest, conversational interview delivery. HUD displays `[COOLING]`. |

---

## How Speech & Silence Detection Works (Audio-Visual Fusion)

Detecting speech purely by audio volume often fails in interview environments:
- If threshold is low: Laptop fan hum or room noise falsely triggers "Speaking" even during complete silence.
- If threshold is high: The candidate has to shout or speak very loudly to be detected.

**The Solution: Multimodal Audio-Visual Fusion**
1. **MediaPipe Lip Movement Tracking (Visual VAD):**
   - Continuously tracks the candidate's inner and outer lip contours (landmarks 13, 14, 82, 87, 312, 317, 61, 291).
   - Computes Mouth Aspect Ratio (MAR) and syllable modulation dynamics (2–6 Hz).
   - When candidate's lips are closed or resting, **`LIPS: STILL`** is enforced.
2. **Ultra-Sensitive Audio Fusion Rule:**
   - **Candidate Silent:** If lips are still, the engine guarantees **`SILENT`** state. Background fan noise and room hum are 100% rejected. Score is strictly **FROZEN** (zero decay).
   - **Candidate Speaking:** When lips are actively moving, the microphone uses an **ultra-sensitive threshold** (ambient + 0.0012). Candidate can speak in a quiet, normal conversational voice or whisper without shouting!

---

## Kinematics & Mathematical Calibration

1. **Horizontal Reading Sweep:**
   - Monotonic Left-to-Right drift: $\Delta X \ge 0.035$ (3.5% of normalized gaze coordinate space).
   - Duration: **$1.40\text{s} \le \text{dur} \le 10.0\text{s}$** (natural human reading speed: 150–250 words/min; rejects casual < 1.4s eye glances).
   - Line Flatness: Vertical Interquartile Range $Y_{\text{IQR}} \le 0.050$.
   - Speech Correlation: Candidate must be vocalizing for $\ge 35\%$ of the sweep.
2. **Return Saccade (Snap Left & Down):**
   - Ballistic snap back to the left: $v_x < -0.12\text{ units/s}$ OR sharp drop $\ge 45\%$ of sweep span within 0.8s.
3. **Downward Staircase Step & Continuous Cadence:**
   - Step to the next line is strictly **DOWNWARD**: $+0.005 \le \Delta Y \le +0.065$.
   - **Continuous Flow Verification**:
     - Line 1: Anchored (0% score increase).
     - Line 2: Flow verification (0% score increase).
     - Line 3: Sustained teleprompter flow confirmed (**+20% score jump**).
     - Line 4+: Continued teleprompter recitation (**+15% per subsequent line**).
     - Any pause > 6.0s or silent reading resets the streak back to 0.
4. **Clean HUD Display:**
   - Row 1: Test Name & Live Clock `[MM:SS.s]`.
   - Row 2: Live Status & Warning color (`ALERT`, `NORMAL`, `SILENT`).
   - Row 3: Cheating line count, Clean speaking count, Sweep drift, Line step.
   - Row 4: Real-time `SPEECH: TALKING [LIPS: ACTIVE | MIC]` vs `SPEECH: SILENT [LIPS: STILL | MIC]`.
   - Top-Right: Confidence bar with `[FROZEN]`, `[COOLING]`, or `[ALERT]` tag.
5. **Detailed Audit Deliverables in `results/`:**
   - `results/{test_name}_report.json`: Exact line numbers where candidate cheated, timestamps formatted in `MM:SS.s`, audio & speech telemetry.
   - `results/{test_name}_summary.txt`: Executive scorecard summary text.
   - `results/{test_name}_graph.png`: Multi-axis Matplotlib forensic chart showing X-sawtooth, Y-staircase, and confidence timeline.

---

## How to Run

Open PowerShell:

```powershell
cd "C:\Users\tanis\OneDrive\Desktop\anti cheating"
python anti_cheat_detector.py
```

1. Enter your test name (e.g. `test_interview` or `candidate_john`).
2. **Test Silent Reading / Browsing:** Read lines or look at the screen silently. Confidence stays **100% FROZEN at 0%** (zero increase).
3. **Test AI Teleprompter Reading:** Read each line **ALOUD** into the mic across the screen, then snap down-left to Line 2 and read aloud. Confidence jumps **+14% per line**.
4. **Test Natural Speaking:** Look towards the webcam and speak naturally to the interviewer. Confidence **COOLS DOWN** (~1.0%/s) towards 0%.
5. Press `Q`, `ESC`, or `Ctrl+C` to finalize and view results in `results/`.

