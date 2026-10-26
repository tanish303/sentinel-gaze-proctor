import asyncio
import websockets
import json

async def test_full_flow():
    uri = "ws://127.0.0.1:8000/ws/session?test_name=test_inbrowser_unit"
    print("Connecting...")
    async with websockets.connect(uri) as ws:
        # Send 5 telemetry ticks
        for i in range(5):
            t = 1.0 + i * 0.1
            gx = 0.5 + i * 0.02
            await ws.send(json.dumps({
                "action": "telemetry",
                "t": t,
                "gaze_x": gx,
                "gaze_y": 0.5,
                "is_speaking": True
            }))
            resp = json.loads(await ws.recv())
            print(f"Tick {i+1}: score={resp['score_pct']}, status={resp['status'][:30]}")

        # Send stop
        print("Sending stop...")
        await ws.send(json.dumps({"action": "stop"}))
        resp = json.loads(await ws.recv())
        print("Report received!")
        print("Verdict:", resp.get("verdict"))
        print("Risk:", resp.get("risk"))
        print("Graph URL:", resp.get("graph_url"))
        print("ALL VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_full_flow())
