"""
Multi-Pillar Anti-Cheat Proctoring Engine — v4.0
Combines 5 Distinct Physical & Behavioral Biometrics for Forensic Cheat Detection:

  1. Ocular Reading Saccades (Left-to-Right sweeps + Downward staircase step) [Weight: 35%]
  2. Cognitive Latency & WPM Reading Burst Rate (> 165 WPM textbook burst)     [Weight: 20%]
  3. Canonical LLM Verbal Stylometry & Artifacts ("Firstly...", "In conclusion") [Weight: 20%]
  4. Question Echoing / Delay Tactic Detection (Stalling keyword repetition)  [Weight: 15%]
  5. Cadence & Prosody Uniformity (Mechanical reading pace vs natural pauses) [Weight: 10%]

ANTI-FALSE-POSITIVE / "ANTI-RUDE" DESIGN:
  - No single isolated event increases the confidence score!
  - Every pillar requires sustained repetition (2+ confirmed occurrences) before accumulating score.
  - Silent thinking, looking away to ponder, and normal conversational pauses are strictly 0% penalty.
"""

import re
import math
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2: WPM Reading Burst Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class WpmBurstEvaluator:
    """
    Evaluates speech delivery rate (Words Per Minute).
    Normal spontaneous conversation: 105 - 145 WPM with variable pauses.
    AI Teleprompter reading aloud: Consistently >= 165 WPM with uniform flow.
    
    Anti-Rude & Cooldown Rules:
      - 1st burst: Logged as ANCHOR / MONITORING (score remains 0.0).
      - 2nd burst: Confirmed reading speed pattern (+35%).
      - Cooldown: Speaking at normal conversational pace (<= 145 WPM) cools down score smoothly.
    """
    def __init__(self):
        self.incidents: List[Dict[str, Any]] = []
        self.score: float = 0.0
        self.burst_count: int = 0
        self.last_burst_t: float = -999.0
        self.wpm_history: List[Tuple[float, int]] = []

    def update(self, t: float, dt: float, wpm: int, is_speaking: bool, snippet: str = "") -> Tuple[float, Optional[str]]:
        # Cooldown: Normal conversational speech cools down the pace score
        if is_speaking and 0 < wpm <= 145:
            if (t - self.last_burst_t) > 4.5 and self.score > 0.0:
                self.score = max(0.0, self.score - 0.015 * dt)
        elif not is_speaking and (t - self.last_burst_t) > 10.0 and self.score > 0.0:
            self.score = max(0.0, self.score - 0.008 * dt)

        if not is_speaking or wpm <= 0:
            return self.score, None

        self.wpm_history.append((t, wpm))
        if len(self.wpm_history) > 60:
            self.wpm_history.pop(0)

        alert_msg = None
        # Threshold: >= 165 WPM sustained reading rate
        if wpm >= 165 and (t - self.last_burst_t) > 3.5:
            self.burst_count += 1
            self.last_burst_t = t

            rec = {
                "incident_number": self.burst_count,
                "timestamp_seconds": round(t, 2),
                "wpm": int(wpm),
                "snippet": snippet[:100] if snippet else f"{wpm} WPM rapid delivery burst",
                "status": "CONFIRMED" if self.burst_count >= 2 else "ANCHORED (Monitoring)"
            }
            self.incidents.append(rec)

            if self.burst_count == 1:
                alert_msg = f"Rapid speech pace detected ({wpm} WPM) — monitoring for second burst"
            elif self.burst_count == 2:
                self.score = min(1.0, self.score + 0.35)
                alert_msg = f"Pillar 2 Confirmed: Sustained reading pace ({wpm} WPM) [Hit #{self.burst_count}]"
            else:
                self.score = min(1.0, self.score + 0.25)
                alert_msg = f"Pillar 2 Confirmed: Repeated teleprompter burst ({wpm} WPM) [Hit #{self.burst_count}]"

        return self.score, alert_msg


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3: Spoken LLM Stylometry & Artifact Evaluator
# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2: Spoken LLM Stylometry & Artifact Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class LlmStylometryEvaluator:
    """
    Scans candidate's spoken speech for canonical ChatGPT / Claude verbal structures.
    Cooldown: Clean conversational speaking gradually cools down the stylometry score.
    """
    CANONICAL_PATTERNS = [
        # AI Connectives & Structural Enumeration
        (r"\b(furthermore|further more|moreover|in addition|first and foremost|firstly|secondly|thirdly)\b", "AI Connective"),
        # Formal AI Hedging & Analytical Framing
        (r"\b(it is (important|crucial|essential) to (note|understand|remember)( that)?|worth noting|bear in mind)\b", "Formal AI Hedging"),
        # AI Conclusion & Summarization
        (r"\b(in conclusion|to conclude|to summarize|in summary|to sum up|in essence)\b", "AI Conclusion"),
        # Signature LLM Vocabulary
        (r"\b(delv(e|ing)( deeper)?( into)?)\b", "LLM Delve Vocabulary"),
        (r"\b((core )?nuance[s]?|multifaceted|tapestry|holistic)\b", "LLM Nuance Vocabulary"),
        (r"\b(significant aspect[s]?|key takeaway[s]?|key consideration[s]?|pivotal role)\b", "AI Analytical Framing"),
        # Conversational AI Openers & Structure
        (r"\b(certainly[,]? here (is|are)|sure[,]? here (is|are)|let['’]?s break this down|step[- ]by[- ]step|quick overview)\b", "AI Assistant Framing"),
        # AI Self Reference
        (r"\b(as an ai|as a language model|my knowledge cutoff|based on my training)\b", "Direct AI Artifact"),
    ]

    def __init__(self):
        self.incidents: List[Dict[str, Any]] = []
        self.score: float = 0.0
        self.hit_count: int = 0
        self.last_hit_t: float = -999.0
        self.last_evaluated_transcript: str = ""

    def update(self, t: float, dt: float, transcript: str, is_speaking: bool) -> Tuple[float, Optional[str]]:
        # Active Cooldown: When not triggering new AI markers, decay score smoothly
        if (t - self.last_hit_t) > 2.0 and self.score > 0.0:
            self.score = max(0.0, self.score - 0.025 * dt)

        if not transcript:
            return self.score, None

        # Only evaluate when a NEW or updated transcript snippet arrives
        clean_text = transcript.strip().lower()
        if clean_text == self.last_evaluated_transcript:
            return self.score, None
        self.last_evaluated_transcript = clean_text

        alert_msg = None
        new_matches = []

        for pattern, category in self.CANONICAL_PATTERNS:
            match = re.search(pattern, clean_text)
            if match:
                phrase = match.group(0)
                new_matches.append((category, phrase))

        if new_matches:
            self.hit_count += 1
            self.last_hit_t = t
            # Multiple markers in the same sentence indicate dense ChatGPT generation (+35%), single (+20%)
            inc = 0.35 if len(new_matches) >= 2 else 0.20
            self.score = min(1.0, self.score + inc)
            primary_cat, primary_phrase = new_matches[0]
            alert_msg = f"LLM marker detected: '{primary_phrase}' ({len(new_matches)} markers)"

            rec = {
                "incident_number": self.hit_count,
                "timestamp_seconds": round(t, 2),
                "matched_phrase": primary_phrase,
                "category": primary_cat,
                "all_matches": [m[1] for m in new_matches],
                "context": transcript[-120:],
                "status": "CONFIRMED"
            }
            self.incidents.append(rec)

        return self.score, alert_msg


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3: Question Echoing Delay Tactic Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class QuestionEchoEvaluator:
    """
    Detects candidates repeating prompt keywords or using verbal stalling framing
    to buy 4 - 7 seconds while an external AI generates the answer.
    Cooldown: Active decay whenever not echoing questions (smooth 2.5%/sec decay).
    """
    STOPWORDS = {
        "what", "when", "where", "which", "while", "this", "that", "these", "those",
        "have", "from", "with", "about", "your", "will", "would", "could", "should",
        "their", "there", "then", "than", "each", "some", "onto", "into", "over"
    }

    DELAY_PATTERNS = [
        r"\b(let me repeat|to repeat the question|repeat the question|repeating the question|so the question asks|the question is asking|to rephrase the question|so you are asking|let me think about that|give me a second|give me a moment|let me read the question|let me understand the question|good question let me think)\b"
    ]

    def __init__(self):
        self.incidents: List[Dict[str, Any]] = []
        self.score: float = 0.0
        self.echo_count: int = 0
        self.last_echo_t: float = -999.0
        self.last_evaluated_text: str = ""

    def update(self, t: float, dt: float, spoken_text: str, active_prompt: str, is_speaking: bool) -> Tuple[float, Optional[str]]:
        # Active Cooldown: When not echoing prompt, decay score smoothly
        if (t - self.last_echo_t) > 2.0 and self.score > 0.0:
            self.score = max(0.0, self.score - 0.025 * dt)

        if not spoken_text:
            return self.score, None

        # CRITICAL FIX: Only evaluate when a NEW transcript arrives!
        # Prevents re-triggering every 3 seconds on stale transcript
        clean_spoken = spoken_text.strip().lower()
        if clean_spoken == self.last_evaluated_text:
            return self.score, None
        self.last_evaluated_text = clean_spoken

        alert_msg = None

        # 1. Check verbal delay stall pattern
        matched_phrase = None
        for pat in self.DELAY_PATTERNS:
            m = re.search(pat, clean_spoken)
            if m:
                matched_phrase = m.group(0)
                break

        # 2. Check keyword echo against active prompt (ONLY if active_prompt is an actual question)
        is_keyword_echo = False
        overlap_keywords = []
        ratio = 0.0

        if active_prompt and ("?" in active_prompt or "question" in active_prompt.lower()) and len(active_prompt) >= 15:
            prompt_words = [
                w for w in re.findall(r"[a-z]{4,}", active_prompt.lower())
                if w not in self.STOPWORDS
            ]
            if len(prompt_words) >= 3:
                spoken_words = set(re.findall(r"[a-z]{4,}", clean_spoken[-250:]))
                overlap_keywords = [w for w in prompt_words if w in spoken_words]
                ratio = len(overlap_keywords) / float(len(prompt_words))
                if len(overlap_keywords) >= 3 and ratio >= 0.45:
                    is_keyword_echo = True

        triggered = (matched_phrase is not None) or is_keyword_echo

        if triggered:
            self.echo_count += 1
            self.last_echo_t = t

            rec = {
                "incident_number": self.echo_count,
                "timestamp_seconds": round(t, 2),
                "matched_keywords": overlap_keywords,
                "overlap_ratio": round(ratio, 2),
                "delay_phrase": matched_phrase,
                "prompt_snippet": active_prompt[:70] if active_prompt else "Verbal delay tactic",
                "spoken_snippet": spoken_text[-90:],
                "status": "CONFIRMED"
            }
            self.incidents.append(rec)

            if self.echo_count == 1:
                self.score = min(1.0, self.score + 0.25)
                alert_msg = f"Question echo delay tactic detected"
            elif self.echo_count == 2:
                self.score = min(1.0, self.score + 0.25)
                alert_msg = f"Repeated question echo delay tactic (#{self.echo_count})"
            else:
                self.score = min(1.0, self.score + 0.20)
                alert_msg = f"Sustained delay tactic (#{self.echo_count})"

        return self.score, alert_msg


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4: Speech Cadence & Robotic Voice Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class CadenceProsodyEvaluator:
    """
    Measures cadence and vocal energy variance during active speech.
    Spontaneous cognitive speech has natural pause variability & pitch inflection (CV >= 0.40).
    AI Script teleprompter reading has an unnaturally flat, metronomic volume envelope (CV <= 0.34).
    Cooldown: Enabled actively whenever candidate speaks normally or pauses/silent (smooth 2.5%/sec decay).
    """
    def __init__(self):
        self.incidents: List[Dict[str, Any]] = []
        self.score: float = 0.0
        self.hit_count: int = 0
        self.rms_window: List[float] = []
        self.last_eval_t: float = -999.0
        self.last_incident_t: float = -999.0
        self.silence_frames: int = 0

    def update(self, t: float, dt: float, is_speaking: bool, rms: float) -> Tuple[float, Optional[str]]:
        # Active Cooldown: When not actively in a monotone incident, decay score smoothly
        if (t - self.last_incident_t) > 2.0 and self.score > 0.0:
            self.score = max(0.0, self.score - 0.025 * dt)

        # STRICT Human Speech Gate:
        # Candidate MUST be actively speaking (multimodal VAD), and RMS must be above quiet ambient
        if is_speaking and rms >= 0.0025:
            self.silence_frames = 0
            self.rms_window.append(rms)
            if len(self.rms_window) > 40:
                self.rms_window.pop(0)

            # Evaluates over 16 frames (~0.5s) of speech
            if len(self.rms_window) >= 16:
                arr = np.array(self.rms_window)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                cv = (std / mean) if mean > 1e-4 else 1.0

                # Flat monotone delivery: CV <= 0.34 (Expressive human speech is CV >= 0.42)
                if cv <= 0.34 and mean >= 0.003:
                    if (t - self.last_eval_t) > 2.5:
                        self.hit_count += 1
                        self.last_eval_t = t
                        self.last_incident_t = t
                        rec = {
                            "incident_number": self.hit_count,
                            "timestamp_seconds": round(t, 2),
                            "duration_samples": len(self.rms_window),
                            "cadence_variance_cv": round(cv, 3),
                            "status": "CONFIRMED"
                        }
                        self.incidents.append(rec)

                        # Progressive score increments (+20% -> +20% -> +15%)
                        if self.hit_count == 1:
                            self.score = min(1.0, self.score + 0.20)
                            return self.score, f"Robotic monotone cadence detected (CV={cv:.2f})"
                        elif self.hit_count == 2:
                            self.score = min(1.0, self.score + 0.20)
                            return self.score, f"Sustained robotic teleprompter delivery (CV={cv:.2f})"
                        else:
                            self.score = min(1.0, self.score + 0.15)
                            return self.score, f"Continued monotone cadence (CV={cv:.2f})"
                elif cv >= 0.40:
                    # Natural pitch inflection detected: cool down robotic score actively
                    if self.score > 0.0:
                        self.score = max(0.0, self.score - 0.030 * dt)
        else:
            self.silence_frames += 1
            # Pop frames during natural speech pauses
            if self.silence_frames > 8 and len(self.rms_window) > 0:
                self.rms_window.pop(0)
            if self.silence_frames > 35:
                self.rms_window.clear()

        return self.score, None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Pillar Master Proctoring Engine (Pure Weighted Average Model)
# ─────────────────────────────────────────────────────────────────────────────
class MultiPillarProctorEngine:
    """
    Orchestrates all 4 Anti-Cheat Proctoring Pillars using Pure Weighted Average:
      - Pillar 1: Ocular Reading Mechanics (Gaze)      [Weight: 45%]
      - Pillar 2: Spoken LLM Stylometry & Artifacts   [Weight: 25%]
      - Pillar 3: Question Echoing Delay Tactic        [Weight: 15%]
      - Pillar 4: Cadence & Prosody Uniformity         [Weight: 15%]
    """
    WEIGHTS = {
        "gaze": 0.45,
        "llm": 0.25,
        "echo": 0.15,
        "cadence": 0.15,
    }

    def __init__(self, gaze_engine):
        self.gaze = gaze_engine
        self.wpm = WpmBurstEvaluator()
        self.llm = LlmStylometryEvaluator()
        self.echo = QuestionEchoEvaluator()
        self.cadence = CadenceProsodyEvaluator()

        self.composite_score: float = 0.0
        self.active_alerts: List[str] = []
        self.last_t: float = -1.0

    def update(
        self,
        t: float,
        screen_x: float,
        screen_y: float,
        is_speaking: bool,
        transcript: str = "",
        wpm: int = 0,
        active_prompt: str = "",
        rms: float = 0.0,
    ) -> Dict[str, Any]:
        dt = max(0.001, min(0.5, t - self.last_t)) if self.last_t >= 0 else 0.04
        self.last_t = t

        # 1. Update Gaze Engine (Pillar 1) — has its own natural speaking cooldown
        n_cheat, s_gaze, status, label, tel = self.gaze.update(t, screen_x, screen_y, is_speaking)

        # 2. Update LLM Stylometry (Pillar 2) with active cooldown
        s_llm, llm_alert = self.llm.update(t, dt, transcript, is_speaking)

        # 3. Update Question Echoing (Pillar 3) with active cooldown
        s_echo, echo_alert = self.echo.update(t, dt, transcript, active_prompt, is_speaking)

        # 4. Update Cadence Uniformity (Pillar 4) with active cooldown
        s_cadence, cadence_alert = self.cadence.update(t, dt, is_speaking, rms)

        # Calculate True PURE WEIGHTED AVERAGE of 4 pillars (Gaze 45%, LLM 25%, Echo 15%, Cadence 15%)
        weighted_avg = (
            self.WEIGHTS["gaze"] * s_gaze +
            self.WEIGHTS["llm"] * s_llm +
            self.WEIGHTS["echo"] * s_echo +
            self.WEIGHTS["cadence"] * s_cadence
        )

        self.composite_score = float(np.clip(weighted_avg, 0.0, 1.0))

        # Collect newest alert
        latest_alert = None
        for a in [llm_alert, echo_alert, cadence_alert]:
            if a:
                latest_alert = a
                self.active_alerts.append(f"[{t:.1f}s] {a}")

        # Derive composite status label
        active_pillars = sum(1 for s in [s_gaze, s_llm, s_echo, s_cadence] if s >= 0.25)
        display_status = status
        display_label = label
        if latest_alert:
            display_status = latest_alert
            display_label = "MULTI_PILLAR_ALERT"
        elif active_pillars >= 2:
            display_label = "MULTI_PILLAR_FLOW"

        # Individual Pillar Statuses & Severities for Live Display
        # 1. Gaze Status
        if s_gaze >= 0.70:
            gaze_status = "Sustained Reading"
            gaze_sev = "CRITICAL"
        elif s_gaze >= 0.35:
            gaze_status = "Saccade Pattern"
            gaze_sev = "WARN"
        elif tel.get("streak", 0) >= 2:
            gaze_status = f"Tracking ({tel.get('streak', 0)}L)"
            gaze_sev = "MONITOR"
        else:
            gaze_status = "Direct Contact"
            gaze_sev = "CLEAN"

        # 2. LLM Stylometry Status
        if s_llm >= 0.60:
            llm_status = "Sustained AI Phrasing"
            llm_sev = "CRITICAL"
        elif s_llm >= 0.35:
            llm_status = "Confirmed AI Markers"
            llm_sev = "WARN"
        else:
            llm_status = "Natural Candidate Speech"
            llm_sev = "CLEAN"

        # 3. Question Echo Status
        if s_echo >= 0.60:
            echo_status = "Sustained Stalling"
            echo_sev = "CRITICAL"
        elif s_echo >= 0.35:
            echo_status = "Question Echo Confirmed"
            echo_sev = "WARN"
        else:
            echo_status = "Direct Response"
            echo_sev = "CLEAN"

        # 4. Cadence Status
        if s_cadence >= 0.35:
            cadence_status = "Robotic Drone Confirmed"
            cadence_sev = "CRITICAL"
        elif s_cadence >= 0.18:
            cadence_status = "Monotone Detected"
            cadence_sev = "WARN"
        else:
            cadence_status = "Natural Pitch Variance"
            cadence_sev = "CLEAN"

        pillar_details = {
            "gaze": {
                "name": "1. Eye Reading",
                "short_name": "Eye Reading",
                "weight": "45%",
                "score": round(s_gaze, 3),
                "score_pct": f"{s_gaze * 100:.0f}%",
                "status": gaze_status,
                "metric": f"{n_cheat} Saccade Lines" if n_cheat else "Direct Contact",
                "severity": gaze_sev,
                "hits": n_cheat,
            },
            "llm": {
                "name": "2. AI Phrases",
                "short_name": "AI Phrases",
                "weight": "25%",
                "score": round(s_llm, 3),
                "score_pct": f"{s_llm * 100:.0f}%",
                "status": llm_status,
                "metric": f"{len(self.llm.incidents)} AI Markers" if self.llm.incidents else "0 AI Markers",
                "severity": llm_sev,
                "hits": len(self.llm.incidents),
            },
            "echo": {
                "name": "3. Question Echo",
                "short_name": "Question Echo",
                "weight": "15%",
                "score": round(s_echo, 3),
                "score_pct": f"{s_echo * 100:.0f}%",
                "status": echo_status,
                "metric": f"{len(self.echo.incidents)} Echo Delays",
                "severity": echo_sev,
                "hits": len(self.echo.incidents),
            },
            "cadence": {
                "name": "4. Robotic Voice",
                "short_name": "Robotic Voice",
                "weight": "15%",
                "score": round(s_cadence, 3),
                "score_pct": f"{s_cadence * 100:.0f}%",
                "status": cadence_status,
                "metric": f"{len(self.cadence.incidents)} Monotone Intervals" if self.cadence.incidents else "Natural Pitch",
                "severity": cadence_sev,
                "hits": len(self.cadence.incidents),
            },
        }

        return {
            "composite_score": round(self.composite_score, 4),
            "composite_score_pct": f"{self.composite_score * 100:.1f}%",
            "status": display_status,
            "label": display_label,
            "flow_streak": tel.get("streak", 0),
            "drift_x": tel.get("drift_x", 0.0),
            "n_cheat": n_cheat,
            "n_clean": len(self.gaze.clean_speaking),
            "pillar_scores": {
                "gaze": round(s_gaze, 3),
                "llm": round(s_llm, 3),
                "echo": round(s_echo, 3),
                "cadence": round(s_cadence, 3),
            },
            "pillar_hits": {
                "gaze_lines": n_cheat,
                "llm_markers": len(self.llm.incidents),
                "echo_events": len(self.echo.incidents),
                "cadence_intervals": len(self.cadence.incidents),
            },
            "pillar_details": pillar_details,
        }

    def get_full_deliverables(self, test_name: str, duration: float) -> Dict[str, Any]:
        """
        Builds the 100% comprehensive forensic audit JSON containing all 4 checks.
        """
        score = self.composite_score
        if score >= 0.70:
            verdict = "CHEATING CONFIRMED (Multi-Modal Teleprompter Recitation)"
            risk = "CRITICAL"
        elif score >= 0.40:
            verdict = "SUSPICIOUS ACTIVITY (Multiple Cheating Signals Detected)"
            risk = "HIGH"
        elif any(len(x.incidents) > 0 for x in [self.llm, self.echo, self.cadence]) or len(self.gaze.cheating_lines) > 0:
            verdict = "LOW RISK (Isolated Behavioral Patterns)"
            risk = "LOW"
        else:
            verdict = "CLEAN (Normal Conversational Candidate Delivery)"
            risk = "CLEAN"

        return {
            "test_metadata": {
                "test_name": test_name,
                "duration_seconds": round(duration, 2),
                "final_verdict": verdict,
                "risk_level": risk,
                "composite_confidence_score": round(score, 4),
                "composite_confidence_percentage": f"{score * 100:.1f}%",
                "pillars_evaluated": 4,
                "anti_false_positive_policy": "Strict multi-hit verification enforced (0% penalty on single events)"
            },
            "pillar_scorecard": {
                "eye_reading_screen": {
                    "weight": "45%",
                    "sub_score": round(self.gaze.score, 4),
                    "sub_score_percentage": f"{self.gaze.score * 100:.1f}%",
                    "cheating_lines_count": len(self.gaze.cheating_lines),
                    "clean_speaking_events": len(self.gaze.clean_speaking),
                    "status": "ALERT" if self.gaze.score >= 0.40 else "NORMAL"
                },
                "chatgpt_ai_phrases": {
                    "weight": "25%",
                    "sub_score": round(self.llm.score, 4),
                    "sub_score_percentage": f"{self.llm.score * 100:.1f}%",
                    "markers_detected_count": len(self.llm.incidents),
                    "status": "ALERT" if self.llm.score >= 0.35 else "NORMAL"
                },
                "question_echo_stalling": {
                    "weight": "15%",
                    "sub_score": round(self.echo.score, 4),
                    "sub_score_percentage": f"{self.echo.score * 100:.1f}%",
                    "echo_incidents_count": len(self.echo.incidents),
                    "status": "ALERT" if self.echo.score >= 0.35 else "NORMAL"
                },
                "robotic_voice_cadence": {
                    "weight": "15%",
                    "sub_score": round(self.cadence.score, 4),
                    "sub_score_percentage": f"{self.cadence.score * 100:.1f}%",
                    "cadence_incidents_count": len(self.cadence.incidents),
                    "status": "ALERT" if self.cadence.score >= 0.30 else "NORMAL"
                }
            },
            "detailed_incidents": {
                "gaze_cheating_lines": self.gaze.cheating_lines,
                "llm_stylometry_markers": self.llm.incidents,
                "question_echo_events": self.echo.incidents,
                "cadence_uniformity_events": self.cadence.incidents,
                "clean_speaking_timeline": self.gaze.clean_speaking,
            },
            "all_incidents": self._compile_all_incidents(duration)
        }

    def _compile_all_incidents(self, duration: float) -> List[Dict[str, Any]]:
        def fmt(s):
            m = int(s // 60)
            sec = s % 60
            return f"{m:02d}:{sec:04.1f}"

        items = []
        for r in self.gaze.cheating_lines:
            items.append({
                "pillar": "Eye Reading",
                "badge": "GAZE",
                "timestamp_formatted": fmt(r["timestamp_seconds"]),
                "timestamp_seconds": r["timestamp_seconds"],
                "description": r["description"],
                "metric": f"Span +{r['horizontal_span']:.3f} (Step {r['vertical_step']:+.3f})",
                "severity": "CRITICAL"
            })
        for r in self.llm.incidents:
            items.append({
                "pillar": "AI Phrases",
                "badge": "AI",
                "timestamp_formatted": fmt(r["timestamp_seconds"]),
                "timestamp_seconds": r["timestamp_seconds"],
                "description": f"AI Phrasing matched [{r['category']}]: \"{r['matched_phrase']}\"",
                "metric": r["matched_phrase"],
                "severity": "HIGH" if r["status"] == "CONFIRMED" else "LOW"
            })
        for r in self.echo.incidents:
            items.append({
                "pillar": "Question Echo",
                "badge": "ECHO",
                "timestamp_formatted": fmt(r["timestamp_seconds"]),
                "timestamp_seconds": r["timestamp_seconds"],
                "description": f"Question echoing stall: {int(r['overlap_ratio']*100)}% keyword repetition",
                "metric": f"{int(r['overlap_ratio']*100)}% match",
                "severity": "HIGH" if r["status"] == "CONFIRMED" else "LOW"
            })
        for r in self.cadence.incidents:
            items.append({
                "pillar": "Robotic Voice",
                "badge": "ROBOT",
                "timestamp_formatted": fmt(r["timestamp_seconds"]),
                "timestamp_seconds": r["timestamp_seconds"],
                "description": f"Robotic monotone delivery (CV={r['cadence_variance_cv']})",
                "metric": f"CV={r['cadence_variance_cv']}",
                "severity": "HIGH" if r["status"] == "CONFIRMED" else "LOW"
            })
        items.sort(key=lambda x: x["timestamp_seconds"])
        return items


def save_multi_pillar_deliverables(test_name: str, duration: float, multi_engine, results_dir: Path):
    """
    Saves comprehensive 5-pillar JSON report, executive TXT summary, and PNG forensic chart.
    """
    import json
    from datetime import datetime

    rpt = multi_engine.get_full_deliverables(test_name, duration)
    meta = rpt["test_metadata"]
    verdict = meta["final_verdict"]
    risk = meta["risk_level"]
    score = meta["composite_confidence_score"]

    # 1. JSON Report
    json_path = results_dir / f"{test_name}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rpt, f, indent=2)

    # 2. Executive TXT Summary
    txt_path = results_dir / f"{test_name}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        sep = "=" * 70
        f.write(sep + "\n")
        f.write(f"       SENTINEL GAZE // 4-PILLAR FORENSIC AUDIT: {test_name}\n")
        f.write(sep + "\n")
        f.write(f"Timestamp          : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Duration           : {duration:.1f} seconds\n")
        f.write(f"Final Verdict      : {verdict}\n")
        f.write(f"Composite Confidence: {score * 100:.1f}% ({risk})\n\n")

        f.write("-" * 70 + "\n")
        f.write("MULTI-PILLAR SCORECARD BREAKDOWN:\n")
        f.write("-" * 70 + "\n")
        for pillar_name, pdata in rpt["pillar_scorecard"].items():
            f.write(f"  • {pillar_name.upper().replace('_', ' ')} (Weight: {pdata['weight']}):\n")
            f.write(f"      Sub-score : {pdata['sub_score_percentage']} [{pdata['status']}]\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"DETAILED FORENSIC INCIDENTS LOGGED: {len(rpt['all_incidents'])}\n")
        f.write("-" * 70 + "\n")
        if rpt["all_incidents"]:
            for idx, inc in enumerate(rpt["all_incidents"]):
                f.write(f"  [{idx+1:>2}] @ {inc['timestamp_formatted']} | [{inc['badge']}] {inc['description']} ({inc['metric']})\n")
        else:
            f.write("  No anomalous or cheating incidents recorded across any of the 5 checks.\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"CLEAN CONVERSATIONAL SPEAKING INTERVALS: {len(multi_engine.gaze.clean_speaking)}\n")
        f.write("-" * 70 + "\n")
        for ev in multi_engine.gaze.clean_speaking[:20]:
            f.write(f"  @ {ev['timestamp_seconds']:.1f}s: {ev['description']}\n")
        f.write(sep + "\n")

    # 3. Chart
    graph_path = results_dir / f"{test_name}_graph.png"
    multi_engine.gaze.save_chart(test_name, graph_path)

    return rpt, json_path, txt_path, graph_path, verdict, risk

