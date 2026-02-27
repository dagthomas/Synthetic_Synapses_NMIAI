import asyncio, json, sys, websockets

async def test(url):
    print(f"Connecting...")
    async with websockets.connect(url) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(msg)
        print(f"Type: {data['type']}, Bots: {len(data.get('bots',[]))}")
        actions = [{"bot": b["id"], "action": "wait"} for b in data["bots"]]
        await ws.send(json.dumps({"actions": actions}))
        print("OK!")

asyncio.run(test(sys.argv[1]))
