import time
from multi_pillar_engine import MultiPillarProctorEngine

class MockGazeEngine:
    def __init__(self):
        self.score = 0.80
        self.clean_speaking = []
        self.cheating_lines = [1, 2]
    def update(self, t, sx, sy, spk):
        return len(self.cheating_lines), self.score, "Mock status", "NORMAL", {"streak": 0, "drift_x": 0.0}

gaze_mock = MockGazeEngine()
engine = MultiPillarProctorEngine(gaze_mock)

print("=== 1. WEIGHTS TEST ===")
print("Weights:", engine.WEIGHTS)
assert abs(sum(engine.WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0!"
print("Weights sum to 1.0 verified.")

print("\n=== 2. TRIGGER TEST (LLM & ECHO) ===")
# Trigger LLM on first hit
res1 = engine.update(1.0, 0.5, 0.5, is_speaking=True, transcript="Furthermore, it is important to note that this is an architecture.")
print("After LLM hit 1:")
print("  LLM Score:", res1["pillar_scores"]["llm"])
assert res1["pillar_scores"]["llm"] >= 0.35, "LLM should trigger on first hit!"

# Trigger Question Echo
res2 = engine.update(2.0, 0.5, 0.5, is_speaking=True, transcript="Let me repeat the question to make sure I understand the question.")
print("After Echo hit 1:")
print("  Echo Score:", res2["pillar_scores"]["echo"])
assert res2["pillar_scores"]["echo"] >= 0.35, "Echo should trigger on first hit!"

print("\n=== 3. CADENCE SILENCE & MONOTONE TEST ===")
# In silence, cadence should NEVER trigger or count drone intervals!
for i in range(10):
    t = 3.0 + i * 0.1
    res_silent = engine.update(t, 0.5, 0.5, is_speaking=False, rms=0.003)
print("After 10 silent frames:")
print("  Cadence Score:", res_silent["pillar_scores"]["cadence"])
print("  Cadence Hits:", len(engine.cadence.incidents))
assert len(engine.cadence.incidents) == 0, "Cadence must NOT trigger while silent!"
print("Cadence silence guard verified.")

print("\n=== 4. COOLDOWN TEST ===")
# Let 10 seconds pass with clean natural speech (or silence)
t_start = 10.0
for i in range(100):
    t = t_start + i * 0.1
    res_cool = engine.update(t, 0.5, 0.5, is_speaking=False, rms=0.003)

print("Scores after 10s cooldown:")
for p, sc in res_cool["pillar_scores"].items():
    print(f"  {p:8}: {sc:.3f}")

assert res_cool["pillar_scores"]["llm"] < res1["pillar_scores"]["llm"], "LLM score must cool down!"
assert res_cool["pillar_scores"]["echo"] < res2["pillar_scores"]["echo"], "Echo score must cool down!"
print("Cooldown active and verified across all tests!")

print("\n=== 5. COMPOSITE SCORE CALCULATION ===")
expected_comp = (
    0.45 * res_cool["pillar_scores"]["gaze"] +
    0.25 * res_cool["pillar_scores"]["llm"] +
    0.15 * res_cool["pillar_scores"]["echo"] +
    0.15 * res_cool["pillar_scores"]["cadence"]
)
print(f"Expected Composite: {expected_comp:.4f} | Actual: {res_cool['composite_score']:.4f}")
assert abs(expected_comp - res_cool["composite_score"]) < 1e-3, "Composite must be pure weighted average!"
print("Pure weighted average verified!")
print("\nALL VERIFICATIONS PASSED!")
