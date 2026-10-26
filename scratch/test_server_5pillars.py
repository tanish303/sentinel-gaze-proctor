import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Users/tanis/OneDrive/Desktop/anti cheating")
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from server import app

def test_server_5pillars_e2e():
    client = TestClient(app)
    
    with client.websocket_connect("/ws/session?test_name=test_5pillars_audit") as ws:
        print("[TEST] Connected to WebSocket session.")
        
        # 1. Warmup (t: 0.1 to 2.4s)
        for i in range(25):
            t = 0.1 + i * 0.09
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.50,
                "raw_y": 0.50,
                "pitch": 0.0,
                "yaw": 0.0,
                "is_speaking": False,
                "detected": True
            })
            resp = ws.receive_json()
            assert resp["type"] == "telemetry"
        print("[TEST] Warmup completed.")

        # 2. Test Anti-Rude Grace Period: Send single occurrence of WPM burst and LLM marker
        ws.send_json({
            "action": "telemetry",
            "t": 3.0,
            "raw_x": 0.50,
            "raw_y": 0.50,
            "is_speaking": True,
            "detected": True,
            "wpm": 180,
            "transcript": "firstly this is just a quick thought"
        })
        resp = ws.receive_json()
        assert resp["score"] == 0.0, f"Expected 0.0 score on single isolated burst/marker, got {resp['score']}"
        print("[TEST] Anti-Rude Rule verified: Single occurrences produced 0% penalty.")

        # 3. Confirmed occurrences: 2nd WPM burst + 2nd LLM marker
        ws.send_json({
            "action": "telemetry",
            "t": 8.0,
            "raw_x": 0.50,
            "raw_y": 0.50,
            "is_speaking": True,
            "detected": True,
            "wpm": 195,
            "transcript": "secondly furthermore in conclusion here are the key steps"
        })
        resp = ws.receive_json()
        print(f"[TEST] After 2nd triggers: Composite Score = {resp['score_pct']}, Status = {resp['status']}")
        assert resp["score"] > 0.05, f"Expected composite score increase, got {resp['score']}"
        assert "pillar_scores" in resp
        print(f"[TEST] Pillar Sub-scores: {resp['pillar_scores']}")

        # 4. Stop session & receive full deliverables
        ws.send_json({"action": "stop"})
        rpt_msg = ws.receive_json()
        assert rpt_msg["type"] == "report"
        
        print("\n[TEST] Final 5-Pillar Deliverables Received:")
        print(f"  • Verdict: {rpt_msg['verdict']} ({rpt_msg['risk']})")
        print(f"  • Composite Score: {rpt_msg['score_pct']}")
        print(f"  • Pillar Scorecard:")
        for p_name, p_data in rpt_msg["pillar_scorecard"].items():
            print(f"      - {p_name}: {p_data['sub_score_percentage']} [{p_data['status']}]")

        print(f"  • Total Forensic Incidents Logged: {len(rpt_msg['all_incidents'])}")
        for idx, inc in enumerate(rpt_msg['all_incidents']):
            print(f"      [{idx+1}] [{inc['badge']}] {inc['description']} ({inc['metric']})")

    print("\n>>> END-TO-END 5-PILLAR WEBSOCKET AUDIT PASSED 100%! <<<\n")

if __name__ == "__main__":
    test_server_5pillars_e2e()
