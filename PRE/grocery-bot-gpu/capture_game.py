"""Probe game that captures server state for offline optimization.

Usage:
    python capture_game.py <wss://game.ainm.no/ws?token=...> <difficulty>

Saves capture to captures/<difficulty>_<date>.json.
Reuses existing capture for same day (items/orders are deterministic within a day).
"""
import asyncio
import json
import sys
import os
import time
from datetime import datetime, timezone
from collections import deque


def bfs_first_step(walls_set, width, height, start, goal):
    if start == goal:
        return "wait"
    q = deque([(start, None)])
    visited = {start}
    directions = [(0, -1, "move_up"), (0, 1, "move_down"),
                  (-1, 0, "move_left"), (1, 0, "move_right")]
    while q:
        (x, y), first = q.popleft()
        for dx, dy, action in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                (nx, ny) not in walls_set and (nx, ny) not in visited):
                fa = first if first else action
                if (nx, ny) == goal:
                    return fa
                visited.add((nx, ny))
                q.append(((nx, ny), fa))
    return "wait"


def bfs_distance(walls_set, width, height, start, goal):
    if start == goal:
        return 0
    q = deque([(start, 0)])
    visited = {start}
    while q:
        (x, y), dist = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                (nx, ny) not in walls_set and (nx, ny) not in visited):
                if (nx, ny) == goal:
                    return dist + 1
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))
    return 9999


def find_adj_cells(ix, iy, walls_set, width, height):
    adj = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ax, ay = ix + dx, iy + dy
        if 0 <= ax < width and 0 <= ay < height and (ax, ay) not in walls_set:
            adj.append((ax, ay))
    return adj


def get_order_needs(order):
    needed = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order.get("items_delivered", []):
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def decide_action(bot, data, walls_set, width, height):
    """Greedy action for probe game (coordinates all bots)."""
    bid = bot["id"]
    bx, by = bot["position"]
    inv = bot["inventory"]
    drop = tuple(data["drop_off"])

    active = preview = None
    for o in data["orders"]:
        if o.get("status") == "active" and not o["complete"]:
            active = o
        elif o.get("status") == "preview" and not o["complete"]:
            preview = o

    if not active:
        return {"bot": bid, "action": "wait"}

    needed = get_order_needs(active)

    # Global remaining: subtract ALL bots' inventories
    global_remaining = dict(needed)
    for other_bot in data["bots"]:
        for item in other_bot["inventory"]:
            if item in global_remaining and global_remaining[item] > 0:
                global_remaining[item] -= 1
    still_needed = {k: v for k, v in global_remaining.items() if v > 0}

    has_useful = any(item in needed for item in inv)

    # 1. At dropoff with useful items -> deliver
    if has_useful and (bx, by) == drop:
        return {"bot": bid, "action": "drop_off"}

    # 2. Adjacent to needed item -> pickup
    if len(inv) < 3:
        for item in data["items"]:
            ix, iy = item["position"]
            if abs(bx - ix) + abs(by - iy) == 1:
                if still_needed.get(item["type"], 0) > 0:
                    still_needed[item["type"]] -= 1
                    return {"bot": bid, "action": "pick_up", "item_id": item["id"]}

    # 3. Has useful items -> deliver
    if has_useful:
        act = bfs_first_step(walls_set, width, height, (bx, by), drop)
        return {"bot": bid, "action": act}

    # 4. Go pick nearest needed item
    if len(inv) < 3 and still_needed:
        best_cell = None
        best_dist = 9999
        for item in data["items"]:
            if still_needed.get(item["type"], 0) > 0:
                ix, iy = item["position"]
                for ax, ay in find_adj_cells(ix, iy, walls_set, width, height):
                    dist = bfs_distance(walls_set, width, height, (bx, by), (ax, ay))
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = (ax, ay)
        if best_cell:
            act = bfs_first_step(walls_set, width, height, (bx, by), best_cell)
            return {"bot": bid, "action": act}

    # 5. Preview pickup when active covered
    if len(inv) < 3 and not still_needed and preview:
        pneeded = get_order_needs(preview)
        for item in inv:
            if item in pneeded and pneeded[item] > 0:
                pneeded[item] -= 1

        for item in data["items"]:
            ix, iy = item["position"]
            if abs(bx - ix) + abs(by - iy) == 1:
                if pneeded.get(item["type"], 0) > 0:
                    return {"bot": bid, "action": "pick_up", "item_id": item["id"]}

        best_cell = None
        best_dist = 9999
        for item in data["items"]:
            if pneeded.get(item["type"], 0) > 0:
                ix, iy = item["position"]
                for ax, ay in find_adj_cells(ix, iy, walls_set, width, height):
                    dist = bfs_distance(walls_set, width, height, (bx, by), (ax, ay))
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = (ax, ay)
        if best_cell:
            act = bfs_first_step(walls_set, width, height, (bx, by), best_cell)
            return {"bot": bid, "action": act}

    if inv:
        act = bfs_first_step(walls_set, width, height, (bx, by), drop)
        return {"bot": bid, "action": act}

    return {"bot": bid, "action": "wait"}


def get_capture_path(difficulty, date_str=None):
    """Get capture file path for a given difficulty and date."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    os.makedirs('captures', exist_ok=True)
    return f'captures/{difficulty}_{date_str}.json'


def load_capture(difficulty, date_str=None):
    """Load existing capture file if available for today."""
    path = get_capture_path(difficulty, date_str)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


async def capture_and_play(ws_url, difficulty, save_path=None):
    """Play a probe game and capture server state for offline optimization."""
    import websockets

    if save_path is None:
        save_path = get_capture_path(difficulty)

    captured = {
        'difficulty': difficulty,
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'grid': None,
        'items': None,
        'drop_off': None,
        'num_bots': 0,
        'orders': [],
    }

    seen_order_ids = set()
    walls_set = None
    width = height = 0

    print(f"Probe game: {difficulty}")
    print(f"Connecting to {ws_url[:60]}...")

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                captured['probe_score'] = data['score']
                captured['rounds_used'] = data['rounds_used']
                captured['orders_completed'] = data['orders_completed']
                captured['items_delivered'] = data['items_delivered']
                print(f"\nProbe complete! Score={data['score']}, "
                      f"Orders={data['orders_completed']}, "
                      f"Items={data['items_delivered']}")
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]

            if rnd == 0:
                width = data["grid"]["width"]
                height = data["grid"]["height"]
                captured['grid'] = data['grid']
                captured['items'] = data['items']
                captured['drop_off'] = data['drop_off']
                captured['num_bots'] = len(data['bots'])

                walls_set = set()
                for w_pos in data["grid"]["walls"]:
                    walls_set.add((w_pos[0], w_pos[1]))
                for item in data["items"]:
                    ix, iy = item["position"]
                    walls_set.add((ix, iy))

                print(f"Map: {width}x{height}, {len(data['bots'])} bots, "
                      f"{len(data['items'])} items")

            # Capture new orders as they appear
            for order in data["orders"]:
                oid = order["id"]
                if oid not in seen_order_ids:
                    seen_order_ids.add(oid)
                    captured['orders'].append({
                        'id': oid,
                        'items_required': list(order['items_required']),
                    })

            # Play greedy actions
            actions = []
            for bot in data["bots"]:
                action = decide_action(bot, data, walls_set, width, height)
                actions.append(action)

            await ws.send(json.dumps({"actions": actions}))

            if rnd < 5 or rnd % 50 == 0 or rnd >= 295:
                print(f"  R{rnd}: score={data['score']} orders_seen={len(captured['orders'])}")

    # Save
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(captured, f, indent=2)

    print(f"\nCapture saved: {save_path}")
    print(f"  Items: {len(captured['items'])}")
    print(f"  Orders captured: {len(captured['orders'])}")

    return captured


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture_game.py <ws_url> <difficulty>")
        print("       python capture_game.py --check <difficulty> [date]")
        sys.exit(1)

    if sys.argv[1] == '--check':
        difficulty = sys.argv[2]
        date = sys.argv[3] if len(sys.argv) > 3 else None
        cap = load_capture(difficulty, date)
        if cap:
            print(f"Capture exists for {difficulty} ({cap.get('captured_at', '?')})")
            print(f"  Items: {len(cap['items'])}")
            print(f"  Orders: {len(cap['orders'])}")
            print(f"  Probe score: {cap.get('probe_score', 'N/A')}")
        else:
            path = get_capture_path(difficulty, date)
            print(f"No capture at {path}")
    else:
        ws_url = sys.argv[1]
        difficulty = sys.argv[2]
        asyncio.run(capture_and_play(ws_url, difficulty))
