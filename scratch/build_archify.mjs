import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, 'archify_repo');
const { compileWorkflow } = await import(pathToFileURL(path.join(repoRoot, 'archify/renderers/workflow/workflow-compiler.mjs')).href);
const { applyTemplate, renderCards, esc } = await import(pathToFileURL(path.join(repoRoot, 'archify/renderers/shared/utils.mjs')).href);

const workflowSpec = {
  "schema_version": 2,
  "diagram_type": "workflow",
  "meta": {
    "title": "Sentinel-Gaze Anti-Cheat Proctoring Engine",
    "subtitle": "4-Parameter Multimodal Proctoring Architecture & Incident Flow",
    "animation": "trace",
    "visual_preset": "signal-flow",
    "quality_profile": "standard",
    "output": "results/sentinel_gaze_archify.html"
  },
  "lanes": [
    { "id": "l_01", "label": "01 / Multimodal Sensor Ingestion" },
    { "id": "l_02", "label": "02 / Facial Mesh & Head-Pose Normalization" },
    { "id": "l_03", "label": "03 / Audio-Visual Multimodal VAD Gate" },
    { "id": "l_04", "label": "04 / Pillar 1: Eye Reading Detection (45%)" },
    { "id": "l_05", "label": "05 / Pillar 2: AI Phrases Detection (25%)" },
    { "id": "l_06", "label": "06 / Pillar 3: Question Echo Detection (15%)" },
    { "id": "l_07", "label": "07 / Pillar 4: Robotic Voice Detection (15%)" },
    { "id": "l_08", "label": "08 / Cooldown & Silence Policy Checks" },
    { "id": "l_09", "label": "09 / Pure Weighted Average Fusion" },
    { "id": "l_10", "label": "10 / Threat Severity Decision Gate", "variant": "exception" },
    { "id": "l_11", "label": "11 / Real-Time Dispatch & HUD" },
    { "id": "l_12", "label": "12 / Certified Clean Output (<20%)" },
    { "id": "l_13", "label": "13 / Flagged Cheating Incident (>=70%)", "variant": "exception" },
    { "id": "l_ex", "label": "EX / Forensic Evidence Vault" }
  ],
  "phases": [
    { "id": "ph_in", "label": "Sensor Ingestion & Preprocessing", "fromCol": 0, "toCol": 1 },
    { "id": "ph_eval", "label": "4 Threat Evaluation Pillars", "fromCol": 2, "toCol": 3, "variant": "emphasis" },
    { "id": "ph_out", "label": "Weighted Fusion & Audit Vault", "fromCol": 4, "toCol": 5, "variant": "dashed" }
  ],
  "groups": [
    { "id": "gp_gaze", "label": "Ocular Gaze Pipeline", "lane": "l_04", "fromCol": 1, "toCol": 3, "variant": "emphasis" },
    { "id": "gp_speech", "label": "Speech & Prosody Pipeline", "lane": "l_07", "fromCol": 1, "toCol": 3, "variant": "dashed" },
    { "id": "gp_cooldown", "label": "Active Cooldown Engine", "lane": "l_08", "fromCol": 1, "toCol": 3, "variant": "security" }
  ],
  "nodes": [
    // 01 / Sensors
    { "id": "n_cam", "lane": "l_01", "col": 1, "type": "external", "label": "Webcam Stream", "sublabel": "30 FPS RGB video", "width": 140 },
    { "id": "n_mic", "lane": "l_01", "col": 2, "type": "external", "label": "Microphone Audio", "sublabel": "16kHz 16-bit PCM", "width": 140 },
    { "id": "n_prompt", "lane": "l_01", "col": 3, "type": "external", "label": "Question Anchor", "sublabel": "interviewer text", "width": 140 },

    // 02 / Preprocessing
    { "id": "n_mp", "lane": "l_02", "col": 1, "type": "backend", "label": "MediaPipe Mesh", "sublabel": "478 3D pts + dual iris", "tag": "iris 468/473", "width": 140 },
    { "id": "n_pnp", "lane": "l_02", "col": 2, "type": "backend", "label": "solvePnP Pose", "sublabel": "yaw & pitch correction", "width": 140 },

    // 03 / VAD
    { "id": "n_vad", "lane": "l_03", "col": 2, "type": "security", "label": "Multimodal VAD", "sublabel": "lip MAR + audio RMS", "tag": "noise proof", "width": 140 },
    { "id": "n_stt", "lane": "l_03", "col": 3, "type": "cloud", "label": "Google STT", "sublabel": "speech-to-text", "width": 140 },

    // 04 / Test 1: Eye Reading
    { "id": "n_t1_sweep", "lane": "l_04", "col": 1, "type": "backend", "label": "Gaze Sweep Velocity", "sublabel": "velocity ring buffer", "width": 140 },
    { "id": "n_t1_gate", "lane": "l_04", "col": 2, "type": "security", "label": "Reading Saccade?", "sublabel": "vel < -0.18 + dY step", "tag": "Weight 45%", "width": 140 },

    // 05 / Test 2: AI Phrases
    { "id": "n_t2_ngram", "lane": "l_05", "col": 2, "type": "backend", "label": "N-gram Scanner", "sublabel": "STT phrase tokenizer", "width": 140 },
    { "id": "n_t2_gate", "lane": "l_05", "col": 3, "type": "security", "label": "ChatGPT Words?", "sublabel": "furthermore, conclusion", "tag": "Weight 25%", "width": 140 },

    // 06 / Test 3: Question Echo
    { "id": "n_t3_sim", "lane": "l_06", "col": 2, "type": "backend", "label": "Jaccard Overlap", "sublabel": "Levenshtein distance", "width": 140 },
    { "id": "n_t3_gate", "lane": "l_06", "col": 3, "type": "security", "label": "Stalling Echo?", "sublabel": ">60% overlap in 12s", "tag": "Weight 15%", "width": 140 },

    // 07 / Test 4: Robotic Voice
    { "id": "n_t4_rms", "lane": "l_07", "col": 1, "type": "backend", "label": "16-Frame RMS", "sublabel": "voiced energy buffer", "width": 140 },
    { "id": "n_t4_gate", "lane": "l_07", "col": 2, "type": "security", "label": "Monotone Drone?", "sublabel": "CV = σ/μ <= 0.34", "tag": "Weight 15%", "width": 140 },

    // 08 / Cooldown & Policy
    { "id": "n_silent", "lane": "l_08", "col": 1, "type": "security", "label": "Silence Guard", "sublabel": "freeze score when quiet", "width": 140 },
    { "id": "n_cooldown", "lane": "l_08", "col": 2, "type": "messagebus", "label": "Automatic Decay", "sublabel": "-3.0%/sec natural cooldown", "width": 140 },

    // 09 / Fusion
    { "id": "n_fusion", "lane": "l_09", "col": 2, "type": "backend", "label": "Weighted Average", "sublabel": "0.45·T1+0.25·T2+0.15·T3+0.15·T4", "width": 148 },

    // 10 / Severity
    { "id": "n_sev_gate", "lane": "l_10", "col": 2, "type": "security", "label": "Threat Classifier", "sublabel": "Clean / Monitor / Critical", "tag": "thresholds", "width": 140 },

    // 11 / Dispatch
    { "id": "n_hud", "lane": "l_11", "col": 1, "type": "frontend", "label": "OpenCV Camera HUD", "sublabel": "top meter + 4 cards", "width": 140 },
    { "id": "n_canvas", "lane": "l_11", "col": 2, "type": "frontend", "label": "Test Text Canvas", "sublabel": "statements & guidelines", "width": 140 },
    { "id": "n_ws", "lane": "l_11", "col": 3, "type": "messagebus", "label": "WebSocket API", "sublabel": "telemetry stream :8000", "width": 140 },

    // 12 / Clean
    { "id": "n_clean", "lane": "l_12", "col": 1, "type": "backend", "label": "Clean Integrity", "sublabel": "Score < 20% (Passed)", "width": 140 },

    // 13 / Flagged
    { "id": "n_cheat", "lane": "l_13", "col": 3, "type": "security", "label": "Cheating Flagged", "sublabel": "Score >= 70% (Critical)", "width": 140 },

    // EX / Vault
    { "id": "n_json", "lane": "l_ex", "col": 1, "type": "database", "label": "Session Report", "sublabel": "results/*_report.json", "width": 140 },
    { "id": "n_txt", "lane": "l_ex", "col": 2, "type": "database", "label": "Audit Summary", "sublabel": "results/*_summary.txt", "width": 140 },
    { "id": "n_graph", "lane": "l_ex", "col": 3, "type": "database", "label": "Forensic Graph", "sublabel": "results/*_graph.png", "width": 140 }
  ],
  "edges": [
    // 01 to 02
    { "id": "e_cam_mp", "from": "n_cam", "to": "n_mp", "variant": "default" },
    { "id": "e_mp_pnp", "from": "n_mp", "to": "n_pnp", "variant": "default" },

    // 01 to 03
    { "id": "e_mic_vad", "from": "n_mic", "to": "n_vad", "variant": "default" },
    { "id": "e_vad_stt", "from": "n_vad", "to": "n_stt", "label": "voice active", "variant": "emphasis" },

    // 02/03 to 04 (Test 1)
    { "id": "e_pnp_t1", "from": "n_pnp", "to": "n_t1_sweep", "label": "gaze X,Y", "variant": "emphasis" },
    { "id": "e_t1_step", "from": "n_t1_sweep", "to": "n_t1_gate", "variant": "default" },

    // 03 to 05 (Test 2)
    { "id": "e_stt_t2", "from": "n_stt", "to": "n_t2_ngram", "label": "text", "variant": "emphasis" },
    { "id": "e_t2_step", "from": "n_t2_ngram", "to": "n_t2_gate", "variant": "default" },

    // 01/03 to 06 (Test 3)
    { "id": "e_prompt_t3", "from": "n_prompt", "to": "n_t3_sim", "label": "prompt", "variant": "default" },
    { "id": "e_t3_step", "from": "n_t3_sim", "to": "n_t3_gate", "variant": "default" },

    // 03 to 07 (Test 4)
    { "id": "e_vad_t4", "from": "n_vad", "to": "n_t4_rms", "label": "RMS audio", "variant": "default" },
    { "id": "e_t4_step", "from": "n_t4_rms", "to": "n_t4_gate", "variant": "default" },

    // Cooldown
    { "id": "e_vad_silent", "from": "n_vad", "to": "n_silent", "label": "silent pause", "variant": "dashed" },
    { "id": "e_vad_cool", "from": "n_vad", "to": "n_cooldown", "label": "natural speech", "variant": "dashed" },

    // All Tests feed to Fusion
    { "id": "e_t1_fus", "from": "n_t1_gate", "to": "n_fusion", "label": "line (+10%)", "variant": "emphasis" },
    { "id": "e_t2_fus", "from": "n_t2_gate", "to": "n_fusion", "label": "AI marker (+35%)", "variant": "emphasis" },
    { "id": "e_t3_fus", "from": "n_t3_gate", "to": "n_fusion", "label": "echo hit (+25%)", "variant": "emphasis" },
    { "id": "e_t4_fus", "from": "n_t4_gate", "to": "n_fusion", "label": "drone hit (+20%)", "variant": "emphasis" },
    { "id": "e_cool_fus", "from": "n_cooldown", "to": "n_fusion", "label": "cooldown", "variant": "dashed" },

    // Fusion to Severity
    { "id": "e_fus_sev", "from": "n_fusion", "to": "n_sev_gate", "variant": "emphasis" },

    // Severity to Dispatch
    { "id": "e_sev_hud", "from": "n_sev_gate", "to": "n_hud", "variant": "default" },
    { "id": "e_sev_canvas", "from": "n_sev_gate", "to": "n_canvas", "variant": "default" },
    { "id": "e_sev_ws", "from": "n_sev_gate", "to": "n_ws", "variant": "default" },

    // Severity to Verdict Outcomes
    { "id": "e_sev_clean", "from": "n_sev_gate", "to": "n_clean", "label": "< 20%", "variant": "emphasis" },
    { "id": "e_sev_cheat", "from": "n_sev_gate", "to": "n_cheat", "label": ">= 70%", "variant": "security", "role": "error" },

    // Output to Vault
    { "id": "e_hud_json", "from": "n_hud", "to": "n_json", "label": "key [Q]", "variant": "dashed" },
    { "id": "e_canvas_txt", "from": "n_canvas", "to": "n_txt", "variant": "dashed" },
    { "id": "e_cheat_graph", "from": "n_cheat", "to": "n_graph", "label": "evidence", "variant": "dashed" }
  ],
  "cards": [
    {
      "dot": "cyan",
      "title": "Multimodal Sensor Pipeline",
      "items": [
        "Webcam video stream (30 FPS) extracts 478 3D facial landmarks and dual iris centers (468/473)",
        "Microphone audio (16kHz PCM) passes Multimodal VAD, gating voice while rejecting fan & typing noise"
      ]
    },
    {
      "dot": "emerald",
      "title": "Pure Weighted Average Math",
      "items": [
        "Final Risk = 0.45*(Eye Reading) + 0.25*(AI Phrases) + 0.15*(Question Echo) + 0.15*(Robotic Voice)",
        "Weights sum precisely to 1.00; looking at webcam and speaking naturally decays all scores at -2.5%/sec"
      ]
    },
    {
      "dot": "rose",
      "title": "Forensic Vault & Audit Trail",
      "items": [
        "Every cheating saccade, ChatGPT phrase, and monotone drone interval is indexed with exact timestamps",
        "Audit files written on key [Q] to results/ (*_report.json, *_summary.txt, *_graph.png)"
      ]
    }
  ]
};

console.log("Compiling Archify workflow specification...");
const compiled = compileWorkflow({ workflow: workflowSpec });

if (!compiled.ok) {
  console.error("Compilation error:", compiled.error, compiled.diagnostics);
  process.exit(1);
}

console.log("SVG generated successfully! Size:", compiled.svg.length);

const templatePath = path.join(repoRoot, 'archify/assets/template.html');
const template = fs.readFileSync(templatePath, 'utf8');

const renderedCards = renderCards(workflowSpec.cards);
const renderedHtml = applyTemplate(template, {
  title: workflowSpec.meta.title,
  subtitle: workflowSpec.meta.subtitle,
  svg: compiled.svg,
  cards: renderedCards,
  visualPreset: 'signal-flow'
});

const outPath = path.resolve('results/sentinel_gaze_archify.html');
fs.writeFileSync(outPath, renderedHtml, 'utf8');
console.log("SUCCESS! Saved full interactive Archify viewer to:", outPath);

// Copy to project root for easy opening
fs.copyFileSync(outPath, path.resolve('sentinel_gaze_archify.html'));

// Also save the JSON specification
const jsonPath = path.resolve('results/sentinel_gaze_archify.workflow.json');
fs.writeFileSync(jsonPath, JSON.stringify(workflowSpec, null, 2), 'utf8');
console.log("Saved Archify workflow JSON specification to:", jsonPath);
