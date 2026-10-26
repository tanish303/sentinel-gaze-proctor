import asyncio
import websockets
import json
import cv2
import base64
import numpy as np
import time

async def test_render():
    uri = "wss://sentinel-gaze-proctor.onrender.com/ws/session?test_name=diagnostic_render"
    print(f"Connecting to {uri}...")
    try:
        t0 = time.time()
        async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
            print(f"Connected in {time.time()-t0:.2f}s!")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)
            ok, buf = cv2.imencode(".jpg", img)
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            payload = json.dumps({"frame": b64, "rms": 0.035})
            print("Sending frame to Render...")
            t1 = time.time()
            await ws.send(payload)

            print("Waiting for response from Render...")
            resp = await asyncio.wait_for(ws.recv(), timeout=15.0)
            t2 = time.time()
            data = json.loads(resp)
            print(f"Received in {t2-t1:.2f}s!")
            print("Received response type:", data.get("type"))
            print("Status:", data.get("status"))
            print("Score:", data.get("score"))
            print("SUCCESS ON RENDER!")
    except Exception as e:
        print(f"ERROR ON RENDER: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_render())
