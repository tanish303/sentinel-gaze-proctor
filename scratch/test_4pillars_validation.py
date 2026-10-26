import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from multi_pillar_engine import MultiPillarProctorEngine

class MockGazeEngine:
    def __init__(self):
        self.score = 0.0
        self.clean_speaking = []
        self.cheating_lines = []
    def update(self, t, sx, sy, spk):
        return len(self.cheating_lines), self.score, 'Direct Contact', 'CLEAN', {'streak': 0, 'drift_x': 0.0}

gaze_mock = MockGazeEngine()
engine = MultiPillarProctorEngine(gaze_mock)

print('=== 1. WEIGHTS TEST ===')
print('Weights:', engine.WEIGHTS)
assert abs(sum(engine.WEIGHTS.values()) - 1.0) < 1e-6, 'Weights must sum to 1.0!'
print('Weights sum to 1.0 verified.')

print('\n=== 2. TEST 2: AI PHRASES ===')
res1 = engine.update(
    1.0, 0.5, 0.5, is_speaking=True,
    transcript='Furthermore, it is important to note that in conclusion, delving deeper into the core nuances of this problem reveals significant aspects.'
)
s2 = res1['pillar_scores']['llm']
print(f'Test 2 score on AI statement: {s2:.3f}')
assert s2 >= 0.30, f'Test 2 should increase on AI statement! Got {s2}'

# Ensure subsequent frames with same transcript do NOT boom
for i in range(50):
    t = 1.0 + (i + 1) * 0.033
    res_repeat = engine.update(
        t, 0.5, 0.5, is_speaking=True,
        transcript='Furthermore, it is important to note that in conclusion, delving deeper into the core nuances of this problem reveals significant aspects.'
    )
print(f'Test 2 score after 50 frames of same transcript: {res_repeat["pillar_scores"]["llm"]:.3f}')
assert res_repeat['pillar_scores']['llm'] <= s2, 'Test 2 should NOT boom on same transcript!'
print('Test 2: PASSED')

print('\n=== 3. TEST 3: QUESTION ECHO ===')
res_echo1 = engine.update(
    4.0, 0.5, 0.5, is_speaking=True,
    transcript='Let me repeat the question to make sure I understand correctly... so you are asking about candidate eye gaze patterns during online tests...'
)
s3 = res_echo1['pillar_scores']['echo']
print(f'Test 3 score on stalling statement: {s3:.3f}')
assert s3 >= 0.20, f'Test 3 should increase on stalling statement! Got {s3}'

# Ensure subsequent frames (3s later, 6s later) do NOT boom without reason!
for i in range(120):
    t = 4.0 + (i + 1) * 0.033
    res_echo_rpt = engine.update(
        t, 0.5, 0.5, is_speaking=True,
        transcript='Let me repeat the question to make sure I understand correctly... so you are asking about candidate eye gaze patterns during online tests...'
    )
print(f'Test 3 score after 120 frames (4s) of same transcript: {res_echo_rpt["pillar_scores"]["echo"]:.3f}')
assert res_echo_rpt['pillar_scores']['echo'] < s3, 'Test 3 should decay or hold, NOT boom!'
print('Test 3: PASSED (No booming!)')

print('\n=== 4. TEST 4: ROBOTIC VOICE ===')
# Room silence / fan noise must NOT trigger Test 4
for i in range(40):
    t = 9.0 + i * 0.033
    res_fan = engine.update(t, 0.5, 0.5, is_speaking=False, rms=0.004)
print(f'Test 4 score in room silence/fan noise: {res_fan["pillar_scores"]["cadence"]:.3f}')
assert res_fan['pillar_scores']['cadence'] == 0.0, 'Fan noise must NOT trigger Test 4!'

# Monotone reading triggers Test 4 cleanly
for i in range(25):
    t = 11.0 + i * 0.033
    rms = 0.012 + 0.002 * np.sin(i * 0.4)
    res_robot = engine.update(t, 0.5, 0.5, is_speaking=True, rms=rms)
s4 = res_robot['pillar_scores']['cadence']
print(f'Test 4 score after monotone reading: {s4:.3f}')
assert s4 >= 0.15, f'Test 4 should increase on monotone voice! Got {s4}'
print('Test 4: PASSED')

print('\n=== 5. PURE WEIGHTED AVERAGE ===')
expected = (
    0.45 * res_robot['pillar_scores']['gaze'] +
    0.25 * res_robot['pillar_scores']['llm'] +
    0.15 * res_robot['pillar_scores']['echo'] +
    0.15 * res_robot['pillar_scores']['cadence']
)
print(f'Expected composite: {expected:.4f} | Actual composite: {res_robot["composite_score"]:.4f}')
assert abs(expected - res_robot['composite_score']) < 1e-3, 'Composite must match pure weighted average!'
print('Pure weighted average: PASSED')

print('\n=== 6. ACTIVE COOLDOWN VERIFICATION ===')
t_cool_start = 13.0
for i in range(120):
    t = t_cool_start + i * 0.033
    res_cooled = engine.update(t, 0.5, 0.5, is_speaking=False, rms=0.002)

print('Scores after 4s clean cooldown:')
for p, sc in res_cooled['pillar_scores'].items():
    print(f'  {p:8}: {sc:.3f}')
assert res_cooled['composite_score'] < res_robot['composite_score'], 'Composite score must cool down smoothly!'
print('Cooldown across all pillars: PASSED')

print('\nALL 4 PILLAR TESTS VERIFIED AND PASSED PERFECTLY!')
