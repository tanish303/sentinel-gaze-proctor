import sys
import os
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Users/tanis/OneDrive/Desktop/anti cheating")
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from server import app

def test_telemetry_reading_accuracy():
    client = TestClient(app)
    
    with client.websocket_connect("/ws/session?test_name=test_accuracy") as ws:
        print("[TEST] Connected to WebSocket session.")
        
        # 1. Warmup frames (t: 0.1 to 2.4s)
        for i in range(25):
            t = 0.1 + i * 0.09
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.50,
                "raw_y": 0.50,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": False,
                "detected": True
            })
            resp = ws.receive_json()
            assert resp["type"] == "telemetry"
        print("[TEST] Warmup completed successfully.")
        
        # 2. Line 1: Read aloud left to right (t: 2.6s to 5.0s)
        # raw_x goes from 0.65 down to 0.45 (screen_x goes 0.35 -> 0.55, drift +0.20)
        dt = 0.04
        t = 2.6
        steps = 50
        for i in range(steps):
            ratio = i / float(steps)
            raw_x = 0.65 - ratio * 0.20
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": raw_x,
                "raw_y": 0.40,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            t += dt
            
        print(f"[TEST] Line 1 sweep completed. drift_x: {resp.get('drift_x')}")
        assert resp.get("drift_x", 0) >= 0.035
        
        # Saccade snap back to left: 8 frames at 40ms interval (320ms)
        for i in range(8):
            t += dt
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.65,
                "raw_y": 0.42,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            
        # Saccade finishes when gaze turns right again at start of line 2
        for i in range(5):
            t += dt
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.64,
                "raw_y": 0.42,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            
        print(f"[TEST] After Line 1 saccade: streak={resp['flow_streak']}, label={resp['label']}, status={resp['status']}")
        assert resp["flow_streak"] >= 1, f"Expected flow_streak >= 1, got {resp['flow_streak']}"
        
        # 3. Line 2: Read aloud left to right (t: 5.8s to 8.2s)
        for i in range(steps):
            ratio = i / float(steps)
            raw_x = 0.65 - ratio * 0.20
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": raw_x,
                "raw_y": 0.42,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            t += dt
            
        print(f"[TEST] Line 2 sweep completed. drift_x: {resp.get('drift_x')}")
        
        # Line 2 return saccade: 8 frames
        for i in range(8):
            t += dt
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.65,
                "raw_y": 0.44,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            
        for i in range(5):
            t += dt
            ws.send_json({
                "action": "telemetry",
                "t": t,
                "raw_x": 0.64,
                "raw_y": 0.44,
                "pitch": 2.0,
                "yaw": -1.0,
                "is_speaking": True,
                "detected": True
            })
            resp = ws.receive_json()
            
        print(f"[TEST] After Line 2 saccade: streak={resp['flow_streak']}, cheat_lines={resp['n_cheat']}, score={resp['score']}, label={resp['label']}")
        assert resp["n_cheat"] >= 1, f"Expected n_cheat >= 1, got {resp['n_cheat']}"
        assert resp["score"] >= 0.08, f"Expected score >= 0.08, got {resp['score']}"
        
        # 4. Stop session
        ws.send_json({"action": "stop"})
        final_rpt = ws.receive_json()
        assert final_rpt["type"] == "report"
        print(f"[TEST] Final report received: verdict={final_rpt['verdict']}, risk={final_rpt['risk']}, score={final_rpt['score_pct']}")
        
    print("\n>>> ALL ACCURACY & TELEMETRY PARITY TESTS PASSED 100%! <<<\n")

if __name__ == "__main__":
    test_telemetry_reading_accuracy()
