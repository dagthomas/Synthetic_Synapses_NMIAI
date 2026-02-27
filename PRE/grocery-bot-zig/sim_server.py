#!/usr/bin/env python -u
"""Local game simulator for grocery-bot testing.
Emulates the game server over WebSocket on localhost.
Usage: python sim_server.py [port] [difficulty]
  difficulty: easy(1 bot), medium(3), hard(5), expert(10)
Then connect: grocery-bot ws://localhost:<port>
"""
import asyncio, json, random, sys, websockets

# Difficulty configs
CONFIGS = {
    "easy":   {"w": 12, "h": 10, "bots": 1, "aisles": 2, "types": 4, "order_size": (3, 4)},
    "medium": {"w": 16, "h": 12, "bots": 3, "aisles": 3, "types": 8, "order_size": (3, 5)},
    "hard":   {"w": 22, "h": 14, "bots": 5, "aisles": 4, "types": 12, "order_size": (3, 5)},
    "expert": {"w": 28, "h": 18, "bots": 10, "aisles": 5, "types": 16, "order_size": (4, 6)},
}

ALL_TYPES = ["milk", "bread", "eggs", "butter", "cheese", "pasta", "rice", "juice",
             "yogurt", "cereal", "flour", "sugar", "coffee", "tea", "oil", "salt"]

MAX_ROUNDS = 300
INV_CAP = 3


def build_map(cfg):
    w, h = cfg["w"], cfg["h"]
    walls = set()
    shelves = set()

    # Border walls
    for x in range(w):
        walls.add((x, 0))
        walls.add((x, h - 1))
    for y in range(h):
        walls.add((0, y))
        walls.add((w - 1, y))

    # Build aisles: shelf-walkway-shelf, 3 cols wide
    # Start at x=3, spaced by 4 (shelf, walk, shelf, gap)
    aisle_starts = []
    start_x = 3
    for _ in range(cfg["aisles"]):
        aisle_starts.append(start_x)
        start_x += 4

    # Shelf rows: top section and bottom section with mid corridor
    mid_y = h // 2
    shelf_rows_top = list(range(2, mid_y))
    shelf_rows_bot = list(range(mid_y + 1, h - 2))

    for ax in aisle_starts:
        for y in shelf_rows_top + shelf_rows_bot:
            shelves.add((ax, y))      # left shelf
            shelves.add((ax + 2, y))  # right shelf

    # Drop-off: bottom-left area
    drop_off = (1, h - 2)

    # Bot spawn: bottom-right inside border
    spawn = (w - 2, h - 2)

    # Items on shelves
    item_types = ALL_TYPES[:cfg["types"]]
    items = []
    shelf_list = sorted(shelves)
    for i, (sx, sy) in enumerate(shelf_list):
        itype = item_types[i % len(item_types)]
        items.append({"id": f"item_{i}", "type": itype, "position": [sx, sy], "picked": False})

    return w, h, walls, shelves, drop_off, spawn, items, item_types


def generate_order(idx, item_types, order_size, status, available_counts=None):
    """Generate an order using only items that are available on the map."""
    n = random.randint(order_size[0], order_size[1])
    if available_counts:
        # Only use types that have items remaining
        avail_types = [t for t, c in available_counts.items() if c > 0]
        if not avail_types:
            avail_types = item_types  # fallback
        temp_counts = dict(available_counts)
        required = []
        for _ in range(n):
            usable = [t for t in avail_types if temp_counts.get(t, 0) > 0]
            if not usable:
                usable = avail_types
            t = random.choice(usable)
            required.append(t)
            if t in temp_counts:
                temp_counts[t] -= 1
    else:
        required = [random.choice(item_types) for _ in range(n)]
    return {
        "id": f"order_{idx}",
        "items_required": required,
        "items_delivered": [],
        "complete": False,
        "status": status,
    }


def make_game_state(rnd, max_rounds, w, h, walls, shelves, bots, items, orders, drop_off, score, order_idx, total_orders):
    # Include shelf positions as walls (shelves are always impassable)
    wall_list = [[x, y] for (x, y) in sorted(walls | shelves)]
    item_list = [{"id": it["id"], "type": it["type"], "position": it["position"]}
                 for it in items if not it["picked"]]
    bot_list = [{"id": b["id"], "position": b["position"], "inventory": list(b["inventory"])}
                for b in bots]
    order_list = [{"id": o["id"], "items_required": o["items_required"],
                   "items_delivered": list(o["items_delivered"]),
                   "complete": o["complete"], "status": o["status"]}
                  for o in orders if not o["complete"]][:2]  # active + preview

    return {
        "type": "game_state",
        "round": rnd,
        "max_rounds": max_rounds,
        "grid": {"width": w, "height": h, "walls": wall_list},
        "bots": bot_list,
        "items": item_list,
        "orders": order_list,
        "drop_off": list(drop_off),
        "score": score,
        "active_order_index": order_idx,
        "total_orders": total_orders,
    }


def is_walkable(x, y, w, h, walls, shelves):
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    if (x, y) in walls or (x, y) in shelves:
        return False
    return True


async def run_game(websocket, cfg):
    seed = random.randint(0, 999999)
    random.seed(seed)
    print(f"Seed: {seed}")
    w, h, walls, shelves, drop_off, spawn, items, item_types = build_map(cfg)
    num_bots = cfg["bots"]
    order_size = cfg["order_size"]

    # Track available item counts for order generation
    def get_available_counts():
        counts = {}
        for it in items:
            if not it["picked"]:
                counts[it["type"]] = counts.get(it["type"], 0) + 1
        return counts

    # Initialize bots at spawn
    bots = []
    for i in range(num_bots):
        bots.append({"id": i, "position": list(spawn), "inventory": []})

    # Generate initial orders using available items
    avail = get_available_counts()
    all_orders = []
    all_orders.append(generate_order(0, item_types, order_size, "active", avail))
    all_orders.append(generate_order(1, item_types, order_size, "preview", avail))
    next_order_idx = 2
    active_idx = 0

    score = 0
    total_items_delivered = 0
    total_orders_completed = 0

    print(f"Game: {w}x{h}, {num_bots} bots, {len(items)} items, {len(item_types)} types")
    print(f"Drop-off: {drop_off}, Spawn: {spawn}")
    print(f"Order 0 (active): {all_orders[0]['items_required']}")
    print(f"Order 1 (preview): {all_orders[1]['items_required']}")

    for rnd in range(MAX_ROUNDS):
        # Build visible orders (active + first non-complete preview)
        visible_orders = []
        for o in all_orders:
            if not o["complete"]:
                visible_orders.append(o)
                if len(visible_orders) >= 2:
                    break

        state = make_game_state(rnd, MAX_ROUNDS, w, h, walls, shelves, bots, items,
                                all_orders, drop_off, score, active_idx, next_order_idx)

        msg = json.dumps(state)
        await websocket.send(msg)

        try:
            resp = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            print(f"  Round {rnd}: TIMEOUT - all bots wait")
            continue

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            print(f"  Round {rnd}: invalid JSON")
            continue

        actions = data.get("actions", [])

        # Build bot map for collision: {(x,y): set of bot_ids}
        occupied = {}
        for b in bots:
            pos = (b["position"][0], b["position"][1])
            if pos not in occupied:
                occupied[pos] = set()
            occupied[pos].add(b["id"])

        # Process actions in bot ID order
        action_map = {a["bot"]: a for a in actions}

        for bot in sorted(bots, key=lambda b: b["id"]):
            bid = bot["id"]
            act = action_map.get(bid, {"action": "wait"})
            action = act.get("action", "wait")

            bx, by = bot["position"]

            if action.startswith("move_"):
                dx, dy = 0, 0
                if action == "move_up": dy = -1
                elif action == "move_down": dy = 1
                elif action == "move_left": dx = -1
                elif action == "move_right": dx = 1

                nx, ny = bx + dx, by + dy
                if is_walkable(nx, ny, w, h, walls, shelves):
                    # Check bot collision (spawn exempt)
                    target_occ = occupied.get((nx, ny), set())
                    if len(target_occ) == 0 or (nx, ny) == tuple(spawn):
                        occupied[(bx, by)].discard(bid)
                        if not occupied[(bx, by)]:
                            del occupied[(bx, by)]
                        bot["position"] = [nx, ny]
                        if (nx, ny) not in occupied:
                            occupied[(nx, ny)] = set()
                        occupied[(nx, ny)].add(bid)

            elif action == "pick_up":
                item_id = act.get("item_id")
                if item_id and len(bot["inventory"]) < INV_CAP:
                    for it in items:
                        if it["id"] == item_id and not it["picked"]:
                            ix, iy = it["position"]
                            mdist = abs(bx - ix) + abs(by - iy)
                            if mdist == 1:
                                bot["inventory"].append(it["type"])
                                it["picked"] = True
                                if rnd < 10 or rnd % 50 == 0:
                                    print(f"  R{rnd} Bot{bid}: picked up {it['type']} ({item_id})")
                            break

            elif action == "drop_off":
                if bx == drop_off[0] and by == drop_off[1] and len(bot["inventory"]) > 0:
                    # Find active order
                    active_order = None
                    for o in all_orders:
                        if not o["complete"] and o["status"] == "active":
                            active_order = o
                            break

                    if active_order:
                        # Deliver matching items
                        remaining_inv = []
                        for inv_item in bot["inventory"]:
                            # Check if active order still needs this item
                            needed = list(active_order["items_required"])
                            for d in active_order["items_delivered"]:
                                if d in needed:
                                    needed.remove(d)
                            if inv_item in needed:
                                active_order["items_delivered"].append(inv_item)
                                score += 1
                                total_items_delivered += 1
                                print(f"  R{rnd} Bot{bid}: DELIVERED {inv_item} (+1, score={score})")
                            else:
                                remaining_inv.append(inv_item)
                        bot["inventory"] = remaining_inv

                        # Check if order is complete
                        needed_after = list(active_order["items_required"])
                        for d in active_order["items_delivered"]:
                            if d in needed_after:
                                needed_after.remove(d)

                        if len(needed_after) == 0:
                            active_order["complete"] = True
                            score += 5
                            total_orders_completed += 1
                            print(f"  R{rnd}: ORDER COMPLETE! +5 bonus (score={score})")

                            # Activate next order
                            for o in all_orders:
                                if not o["complete"] and o["status"] == "preview":
                                    o["status"] = "active"
                                    break

                            # Generate new preview
                            new_order = generate_order(next_order_idx, item_types, order_size, "preview", get_available_counts())
                            all_orders.append(new_order)
                            next_order_idx += 1
                            print(f"  New order: {new_order['items_required']}")

                            # Re-check all bot inventories for auto-delivery
                            new_active = None
                            for o in all_orders:
                                if not o["complete"] and o["status"] == "active":
                                    new_active = o
                                    break
                            if new_active:
                                for b2 in bots:
                                    if b2["position"][0] == drop_off[0] and b2["position"][1] == drop_off[1]:
                                        remaining = []
                                        for inv_item in b2["inventory"]:
                                            needed2 = list(new_active["items_required"])
                                            for d in new_active["items_delivered"]:
                                                if d in needed2:
                                                    needed2.remove(d)
                                            if inv_item in needed2:
                                                new_active["items_delivered"].append(inv_item)
                                                score += 1
                                                total_items_delivered += 1
                                                print(f"  R{rnd} Bot{b2['id']}: AUTO-DELIVERED {inv_item} (+1)")
                                            else:
                                                remaining.append(inv_item)
                                        b2["inventory"] = remaining

        if rnd % 50 == 0 or rnd < 20:
            bot_info = ", ".join(f"B{b['id']}@({b['position'][0]},{b['position'][1]}) inv={b['inventory']}" for b in bots)
            active_order = next((o for o in all_orders if not o["complete"] and o["status"] == "active"), None)
            preview_order = next((o for o in all_orders if not o["complete"] and o["status"] == "preview"), None)
            act_str = f"Active: need={active_order['items_required']} del={active_order['items_delivered']}" if active_order else "No active"
            prev_str = f"Preview: {preview_order['items_required']}" if preview_order else "No preview"
            avail_types = {}
            for it in items:
                if not it["picked"]:
                    avail_types[it["type"]] = avail_types.get(it["type"], 0) + 1
            print(f"R{rnd} | Score:{score} | {bot_info}")
            print(f"  {act_str}")
            print(f"  {prev_str}")
            print(f"  Available items: {avail_types}")
            act_list = []
            for a in actions:
                s = f"B{a['bot']}:{a['action']}"
                if 'item_id' in a:
                    s += f"({a['item_id']})"
                act_list.append(s)
            print(f"  Bot actions: {act_list}")

    # Send game_over
    game_over = {
        "type": "game_over",
        "score": score,
        "rounds_used": MAX_ROUNDS,
        "items_delivered": total_items_delivered,
        "orders_completed": total_orders_completed,
    }
    await websocket.send(json.dumps(game_over))
    print(f"\n{'='*50}")
    print(f"GAME OVER")
    print(f"  Score: {score}")
    print(f"  Items delivered: {total_items_delivered}")
    print(f"  Orders completed: {total_orders_completed}")
    print(f"  Rounds used: {MAX_ROUNDS}")
    print(f"{'='*50}")


async def handler(websocket):
    print("Bot connected!")
    await run_game(websocket, active_cfg)
    print("Game finished.")


async def main_server(port, cfg):
    global active_cfg
    active_cfg = cfg
    print(f"Starting sim server on ws://localhost:{port}")
    print(f"Config: {cfg['w']}x{cfg['h']}, {cfg['bots']} bots, {cfg['types']} item types")
    print(f"Connect your bot: grocery-bot ws://localhost:{port}")
    print()
    async with websockets.serve(handler, "localhost", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    diff = sys.argv[2] if len(sys.argv) > 2 else "easy"
    if diff not in CONFIGS:
        print(f"Unknown difficulty: {diff}. Use: easy, medium, hard, expert")
        sys.exit(1)
    asyncio.run(main_server(port, CONFIGS[diff]))
