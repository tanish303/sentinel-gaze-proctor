import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Users/tanis/OneDrive/Desktop/anti cheating")
sys.path.insert(0, str(PROJECT_ROOT))

from anti_cheat_detector import ReadingPatternEngine
from multi_pillar_engine import MultiPillarProctorEngine

def test_multi_pillar_engine():
    gaze_engine = ReadingPatternEngine(warmup=2.0)
    engine = MultiPillarProctorEngine(gaze_engine)

    print("[TEST] Testing Anti-Rude / Grace Period (Single Occurrences)...")
    
    # 1. Warmup
    for i in range(25):
        t = 0.1 + i * 0.08
        res = engine.update(t, 0.5, 0.5, is_speaking=False)
        assert res["composite_score"] == 0.0

    # 2. Test Pillar 2: Single WPM burst (should NOT increase score)
    res = engine.update(3.0, 0.5, 0.5, is_speaking=True, wpm=185, transcript="talking very fast right now")
    assert engine.wpm.burst_count == 1
    assert engine.wpm.score == 0.0, f"Expected 0.0 for first burst, got {engine.wpm.score}"
    print("[TEST] Passed: Single WPM burst did NOT penalize candidate.")

    # 3. Test Pillar 3: Single LLM marker (should NOT increase score)
    res = engine.update(3.5, 0.5, 0.5, is_speaking=True, transcript="firstly we should consider the architecture")
    assert engine.llm.hit_count == 1
    assert engine.llm.score == 0.0, f"Expected 0.0 for first LLM marker, got {engine.llm.score}"
    print("[TEST] Passed: Single LLM marker did NOT penalize candidate.")

    # 4. Test Pillar 4: Single question echo (should NOT increase score)
    prompt_1 = "Line 1: When candidates take online technical assessments or live proctored interviews"
    res = engine.update(4.0, 0.5, 0.5, is_speaking=True, active_prompt=prompt_1, transcript="When candidates take online technical assessments")
    assert engine.echo.echo_count == 1
    assert engine.echo.score == 0.0, f"Expected 0.0 for first echo, got {engine.echo.score}"
    print("[TEST] Passed: Single Question Echo did NOT penalize candidate.")

    # 5. Composite score must still be 0.0 after single occurrences!
    assert res["composite_score"] == 0.0, f"Expected composite 0.0, got {res['composite_score']}"
    print("[TEST] Passed: Composite score strictly protected against isolated single triggers!")

    print("\n[TEST] Testing Confirmed Repetition (2nd Occurrences)...")
    
    # 6. Second WPM burst (should confirm Pillar 2)
    res = engine.update(8.0, 0.5, 0.5, is_speaking=True, wpm=190, transcript="continuing to deliver textbook answer without pausing")
    assert engine.wpm.burst_count == 2
    assert engine.wpm.score > 0.0, "Expected positive score on 2nd burst"
    print(f"[TEST] Passed: 2nd WPM burst confirmed! WPM Sub-score = {engine.wpm.score*100:.1f}%")

    # 7. Second LLM marker (should confirm Pillar 3)
    res = engine.update(9.0, 0.5, 0.5, is_speaking=True, transcript="in conclusion, the above points demonstrate the result")
    assert engine.llm.hit_count == 2
    assert engine.llm.score > 0.0, "Expected positive score on 2nd LLM marker"
    print(f"[TEST] Passed: 2nd LLM marker confirmed! LLM Sub-score = {engine.llm.score*100:.1f}%")

    # 8. Second Question echo (should confirm Pillar 4)
    prompt_2 = "Line 2: As your eyes read continuously across each written sentence from left to right"
    res = engine.update(15.0, 0.5, 0.5, is_speaking=True, active_prompt=prompt_2, transcript="As your eyes read continuously across each written sentence")
    assert engine.echo.echo_count == 2
    assert engine.echo.score > 0.0, "Expected positive score on 2nd question echo"
    print(f"[TEST] Passed: 2nd Question Echo confirmed! Echo Sub-score = {engine.echo.score*100:.1f}%")

    # Verify composite score has increased proportionately
    print(f"[TEST] Composite Score = {res['composite_score_pct']}")
    assert res["composite_score"] > 0.10

    # 9. Test Deliverables generation
    deliverables = engine.get_full_deliverables("test_candidate", 30.0)
    assert deliverables["test_metadata"]["pillars_evaluated"] == 5
    assert len(deliverables["pillar_scorecard"]) == 5
    assert "detailed_incidents" in deliverables
    print("[TEST] Full 5-pillar forensic deliverables generated successfully:")
    for pillar, data in deliverables["pillar_scorecard"].items():
        print(f"  - {pillar} ({data['weight']}): {data['sub_score_percentage']}")

    print("\n>>> ALL MULTI-PILLAR & ANTI-RUDE TESTS PASSED 100%! <<<\n")

if __name__ == "__main__":
    test_multi_pillar_engine()
