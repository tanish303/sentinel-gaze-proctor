import asyncio
import websockets
import json

async def test_telemetry():
    uri = "ws://127.0.0.1:8000/ws/session?test_name=test_telemetry"
    print("Connecting...")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        # Send telemetry
        payload = json.dumps({
            "action": "telemetry",
            "t": 3.5,
            "gaze_x": 0.45,
            "gaze_y": 0.52,
            "is_speaking": True
        })
        print("Sending telemetry...")
        await ws.send(payload)
        resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
        data = json.loads(resp)
        print("Received response:", data)
        assert data.get("type") == "telemetry"
        print("TELEMETRY PIPELINE SUCCESSFUL!")

if __name__ == "__main__":
    asyncio.run(test_telemetry())
