import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/session?test_name=diagnostic_test"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        # Create a test 640x480 frame with a synthetic circle
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)
        ok, buf = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        payload = json.dumps({"frame": b64, "rms": 0.035})
        print("Sending frame...")
        await ws.send(payload)

        print("Waiting for response...")
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        print("Received response type:", data.get("type"))
        print("Status:", data.get("status"))
        print("Score:", data.get("score"))
        print("Is speaking:", data.get("is_speaking"))
        print("Lip moving:", data.get("lip_moving"))
        print("Frame returned len:", len(data.get("frame", "")))
        print("SUCCESS!")

if __name__ == "__main__":
    asyncio.run(test_ws())
