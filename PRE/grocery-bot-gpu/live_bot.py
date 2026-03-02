"""Real-time bot that plays using server game state.

Usage:
    python live_bot.py <wss://game.ainm.no/ws?token=...>
"""
import asyncio
import json
import sys
import time
from collections import deque


def bfs_first_step(walls_set, width, height, start, goal):
    """BFS pathfinding, returns action string."""
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
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls_set and (nx, ny) not in visited:
                fa = first if first else action
                if (nx, ny) == goal:
                    return fa
                visited.add((nx, ny))
                q.append(((nx, ny), fa))
    return "wait"


def bfs_distance(walls_set, width, height, start, goal):
    """BFS distance between two points."""
    if start == goal:
        return 0
    q = deque([(start, 0)])
    visited = {start}
    while q:
        (x, y), dist = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls_set and (nx, ny) not in visited:
                if (nx, ny) == goal:
                    return dist + 1
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))
    return 9999


def find_adj_cells(ix, iy, walls_set, width, height):
    """Find walkable cells adjacent to an item."""
    adj = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ax, ay = ix + dx, iy + dy
        if 0 <= ax < width and 0 <= ay < height and (ax, ay) not in walls_set:
            adj.append((ax, ay))
    return adj


def get_order_needs(order):
    """Get remaining needs for an order."""
    needed = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def decide_action(bot, data, walls_set, width, height):
    """Decide best action for one bot."""
    bid = bot["id"]
    bx, by = bot["position"]
    inv = bot["inventory"]
    drop = tuple(data["drop_off"])

    # Find active and preview orders
    active = None
    preview = None
    for o in data["orders"]:
        if o.get("status") == "active" and not o["complete"]:
            active = o
        elif o.get("status") == "preview" and not o["complete"]:
            preview = o

    if not active:
        return {"bot": bid, "action": "wait"}

    needed = get_order_needs(active)

    # Count useful items in inventory
    inv_contribution = {}
    for item in inv:
        if item in needed and needed.get(item, 0) > inv_contribution.get(item, 0):
            inv_contribution[item] = inv_contribution.get(item, 0) + 1
    has_useful = sum(inv_contribution.values()) > 0

    # Remaining needs after subtracting inventory
    remaining = dict(needed)
    for item in inv:
        if item in remaining and remaining[item] > 0:
            remaining[item] -= 1
    still_needed = {k: v for k, v in remaining.items() if v > 0}

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

    # 3. Has useful items -> go deliver
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

    # 5. Preview pickup when active order is covered
    if len(inv) < 3 and not still_needed and preview:
        pneeded = get_order_needs(preview)
        # Subtract own inventory
        for item in inv:
            if item in pneeded and pneeded[item] > 0:
                pneeded[item] -= 1

        # Check adjacent first
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

    # 6. Has items that might be useful -> deliver
    if inv:
        act = bfs_first_step(walls_set, width, height, (bx, by), drop)
        return {"bot": bid, "action": act}

    return {"bot": bid, "action": "wait"}


async def play(ws_url):
    """Connect and play a full game."""
    import websockets

    print(f"Connecting to {ws_url[:60]}...")

    async with websockets.connect(ws_url) as ws:
        walls_set = None
        width = height = 0

        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                print(f"\nGAME OVER! Score: {data['score']}, "
                      f"Rounds: {data['rounds_used']}, "
                      f"Items: {data['items_delivered']}, "
                      f"Orders: {data['orders_completed']}")
                return data["score"]

            if data["type"] != "game_state":
                continue

            rnd = data["round"]

            if rnd == 0:
                width = data["grid"]["width"]
                height = data["grid"]["height"]
                walls_set = set()
                for w in data["grid"]["walls"]:
                    walls_set.add((w[0], w[1]))
                # Shelf cells (where items sit) are not walkable
                for item in data["items"]:
                    ix, iy = item["position"]
                    walls_set.add((ix, iy))
                print(f"Map: {width}x{height}, {len(data['bots'])} bots, "
                      f"{len(data['items'])} items")
                print(f"Drop-off: {data['drop_off']}")

            # Decide actions for all bots
            actions = []
            for bot in data["bots"]:
                action = decide_action(bot, data, walls_set, width, height)
                actions.append(action)

            await ws.send(json.dumps({"actions": actions}))

            if rnd < 5 or rnd % 50 == 0 or rnd >= 295:
                print(f"  R{rnd}: score={data['score']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python live_bot.py <wss://game.ainm.no/ws?token=...>")
        sys.exit(1)

    ws_url = sys.argv[1]
    asyncio.run(play(ws_url))
