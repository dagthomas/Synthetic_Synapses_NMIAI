"""Quick diagnostic: test WebSocket connection to the game server."""
import asyncio, json, sys, websockets

async def test(url):
    print(f"Connecting to {url[:80]}...")
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            print("Connected! Waiting for first message...")
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            print(f"Type: {data['type']}")
            if data["type"] == "game_state":
                print(f"Round: {data['round']}/{data['max_rounds']}")
                print(f"Grid: {data['grid']['width']}x{data['grid']['height']}")
                print(f"Bots: {len(data['bots'])}")
                print(f"Items: {len(data['items'])}")
                print(f"Drop-off: {data['drop_off']}")
                # Send a wait action to keep connection alive
                actions = [{"bot": b["id"], "action": "wait"} for b in data["bots"]]
                await ws.send(json.dumps({"actions": actions}))
                print("Sent wait action. Connection works!")
            await ws.close()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Paste WS URL: ")
    asyncio.run(test(url))
