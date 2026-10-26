import asyncio
import websockets
import json
import time

async def inspect():
    uri = "wss://sentinel-gaze-proctor.onrender.com/ws/session?test_name=debug_connection"
    print("1. Connecting...")
    async with websockets.connect(uri) as ws:
        print("2. Connected successfully!")
        
        # Test 1: Send a ping or invalid JSON to see what happens
        print("3. Testing ping...")
        await ws.ping()
        print("4. Ping response received!")

        # Test 2: Send dummy action
        print("5. Sending action: 'ping'...")
        await ws.send(json.dumps({"action": "ping"}))
        await asyncio.sleep(1)
        print("6. Still alive after dummy action!")

        # Test 3: Send frame with empty bytes
        print("7. Sending dummy frame...")
        await ws.send(json.dumps({"frame": "data:image/jpeg;base64,invalid", "rms": 0.01}))
        await asyncio.sleep(1)
        print("8. Still alive after invalid frame!")

if __name__ == "__main__":
    try:
        asyncio.run(inspect())
    except Exception as e:
        print(f"FAILED AT: {type(e).__name__}: {e}")
