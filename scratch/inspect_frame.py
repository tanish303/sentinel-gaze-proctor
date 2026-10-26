import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

async def inspect_real_frame():
    uri = "wss://sentinel-gaze-proctor.onrender.com/ws/session?test_name=real_frame_test"
    print("Connecting...")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        # Create a real 640x480 frame
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:300, 200:400] = [128, 128, 128]
        ok, buf = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        print("Sending 1st real frame...")
        await ws.send(json.dumps({"frame": b64, "rms": 0.02}))
        
        print("Waiting for response from 1st frame...")
        resp = await ws.recv()
        print("Received response for 1st frame:", resp[:80])

        print("Sending 2nd real frame...")
        await ws.send(json.dumps({"frame": b64, "rms": 0.02}))
        resp2 = await ws.recv()
        print("Received response for 2nd frame:", resp2[:80])

if __name__ == "__main__":
    try:
        asyncio.run(inspect_real_frame())
    except Exception as e:
        print(f"FAILED AT: {type(e).__name__}: {e}")

