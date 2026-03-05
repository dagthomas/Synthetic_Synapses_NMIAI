#!/usr/bin/env python3
"""Records grocery bot games to PostgreSQL for replay visualization.

Usage:
    # Record a single game
    python recorder.py easy --seed 1001

    # Record a sweep of 40 seeds
    python recorder.py expert --seeds 40

    # Custom port and DB
    python recorder.py hard --seeds 10 --port 9870 --db "$GROCERY_DB_URL"
"""
import asyncio
import json
import os
import random
import subprocess  # nosec B404
import sys
import time

import psycopg2
from psycopg2.extras import execute_values
import websockets

# Add parent dir to path for sim_server imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim_server import (
    CONFIGS, ALL_TYPES, MAX_ROUNDS, INV_CAP,
    build_map, generate_order, is_walkable, make_game_state
)

BOT_PATH = os.path.join(os.path.dirname(__file__), "..", "zig-out", "bin", "grocery-bot.exe")
DEFAULT_DB = os.environ.get("GROCERY_DB_URL", "postgres://grocery@localhost:5433/grocery_bot")


async def run_game_recorded(websocket, cfg, seed):
    """Run a game and return full recording data."""
    random.seed(seed)  # nosec B311
    w, h, walls, shelves, drop_off, spawn, items, item_types = build_map(cfg)
    num_bots = cfg["bots"]
    order_size = cfg["order_size"]

    def get_available_counts():
        counts = {}
        for it in items:
            counts[it["type"]] = counts.get(it["type"], 0) + 1
        return counts

    bots = [{"id": i, "position": list(spawn), "inventory": []} for i in range(num_bots)]

    avail = get_available_counts()
    all_orders = [
        generate_order(0, item_types, order_size, "active", avail),
        generate_order(1, item_types, order_size, "preview", avail),
    ]
    next_order_idx = 2
    active_idx = 0
    score = 0
    total_items_delivered = 0
    total_orders_completed = 0

    # Separate walls and shelves for recording
    wall_positions = sorted([list(p) for p in walls])
    shelf_positions = sorted([list(p) for p in shelves])
    item_list_initial = [{"id": it["id"], "type": it["type"], "position": it["position"]} for it in items]

    round_records = []

    for rnd in range(MAX_ROUNDS):
        visible_orders = [o for o in all_orders if not o["complete"]][:2]
        state = make_game_state(rnd, MAX_ROUNDS, w, h, walls, shelves, bots, items,
                                all_orders, drop_off, score, active_idx, next_order_idx)
        msg = json.dumps(state)
        await websocket.send(msg)

        try:
            resp = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            actions = []
            round_records.append({
                "round": rnd,
                "bots": [{"id": b["id"], "position": list(b["position"]), "inventory": list(b["inventory"])} for b in bots],
                "orders": [{"id": o["id"], "items_required": o["items_required"],
                            "items_delivered": list(o["items_delivered"]),
                            "status": o["status"]} for o in visible_orders],
                "actions": [],
                "score": score,
                "events": [],
            })
            continue

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            round_records.append({
                "round": rnd,
                "bots": [{"id": b["id"], "position": list(b["position"]), "inventory": list(b["inventory"])} for b in bots],
                "orders": [{"id": o["id"], "items_required": o["items_required"],
                            "items_delivered": list(o["items_delivered"]),
                            "status": o["status"]} for o in visible_orders],
                "actions": [],
                "score": score,
                "events": [],
            })
            continue

        actions = data.get("actions", [])
        events = []

        # Build occupied map
        occupied = {}
        for b in bots:
            pos = (b["position"][0], b["position"][1])
            occupied.setdefault(pos, set()).add(b["id"])

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
                    target_occ = occupied.get((nx, ny), set())
                    if len(target_occ) == 0 or (nx, ny) == tuple(spawn):
                        occupied[(bx, by)].discard(bid)
                        if not occupied[(bx, by)]:
                            del occupied[(bx, by)]
                        bot["position"] = [nx, ny]
                        occupied.setdefault((nx, ny), set()).add(bid)

            elif action == "pick_up":
                item_id = act.get("item_id")
                if item_id and len(bot["inventory"]) < INV_CAP:
                    for it in items:
                        if it["id"] == item_id:
                            ix, iy = it["position"]
                            if abs(bx - ix) + abs(by - iy) == 1:
                                bot["inventory"].append(it["type"])
                                events.append({"type": "pickup", "bot": bid, "item_type": it["type"], "item_id": item_id})
                            break

            elif action == "drop_off":
                if bx == drop_off[0] and by == drop_off[1] and len(bot["inventory"]) > 0:
                    active_order = next((o for o in all_orders if not o["complete"] and o["status"] == "active"), None)
                    if active_order:
                        remaining_inv = []
                        for inv_item in bot["inventory"]:
                            needed = list(active_order["items_required"])
                            for d in active_order["items_delivered"]:
                                if d in needed:
                                    needed.remove(d)
                            if inv_item in needed:
                                active_order["items_delivered"].append(inv_item)
                                score += 1
                                total_items_delivered += 1
                                events.append({"type": "deliver", "bot": bid, "item_type": inv_item})
                            else:
                                remaining_inv.append(inv_item)
                        bot["inventory"] = remaining_inv

                        needed_after = list(active_order["items_required"])
                        for d in active_order["items_delivered"]:
                            if d in needed_after:
                                needed_after.remove(d)

                        if len(needed_after) == 0:
                            active_order["complete"] = True
                            score += 5
                            total_orders_completed += 1
                            events.append({"type": "order_complete", "order_id": active_order["id"], "score": score})

                            for o in all_orders:
                                if not o["complete"] and o["status"] == "preview":
                                    o["status"] = "active"
                                    break

                            new_order = generate_order(next_order_idx, item_types, order_size, "preview", get_available_counts())
                            all_orders.append(new_order)
                            next_order_idx += 1

                            new_active = next((o for o in all_orders if not o["complete"] and o["status"] == "active"), None)
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
                                                events.append({"type": "auto_deliver", "bot": b2["id"], "item_type": inv_item})
                                            else:
                                                remaining.append(inv_item)
                                        b2["inventory"] = remaining

        # Record this round AFTER processing actions (so positions reflect movement)
        visible_orders_now = [o for o in all_orders if not o["complete"]][:2]
        round_records.append({
            "round": rnd,
            "bots": [{"id": b["id"], "position": list(b["position"]), "inventory": list(b["inventory"])} for b in bots],
            "orders": [{"id": o["id"], "items_required": o["items_required"],
                        "items_delivered": list(o["items_delivered"]),
                        "status": o["status"]} for o in visible_orders_now],
            "actions": actions,
            "score": score,
            "events": events,
        })

    # Send game_over
    game_over = {
        "type": "game_over",
        "score": score,
        "rounds_used": MAX_ROUNDS,
        "items_delivered": total_items_delivered,
        "orders_completed": total_orders_completed,
    }
    await websocket.send(json.dumps(game_over))

    return {
        "seed": seed,
        "grid_width": w,
        "grid_height": h,
        "bot_count": num_bots,
        "item_types": len(item_types),
        "order_size_min": order_size[0],
        "order_size_max": order_size[1],
        "walls": wall_positions,
        "shelves": shelf_positions,
        "items": item_list_initial,
        "drop_off": list(drop_off),
        "spawn": list(spawn),
        "final_score": score,
        "items_delivered": total_items_delivered,
        "orders_completed": total_orders_completed,
        "rounds": round_records,
    }


def save_to_db(db_url, difficulty, record):
    """Insert a game record into PostgreSQL."""
    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, spawn,
                              final_score, items_delivered, orders_completed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            record["seed"], difficulty,
            record["grid_width"], record["grid_height"], record["bot_count"],
            record["item_types"], record["order_size_min"], record["order_size_max"],
            json.dumps(record["walls"]), json.dumps(record["shelves"]),
            json.dumps(record["items"]), json.dumps(record["drop_off"]),
            json.dumps(record["spawn"]),
            record["final_score"], record["items_delivered"], record["orders_completed"],
        ))
        run_id = cur.fetchone()[0]

        # Batch insert rounds
        if record["rounds"]:
            round_tuples = [
                (run_id, r["round"], json.dumps(r["bots"]), json.dumps(r["orders"]),
                 json.dumps(r["actions"]), r["score"], json.dumps(r["events"]))
                for r in record["rounds"]
            ]
            execute_values(cur, """
                INSERT INTO rounds (run_id, round_number, bots, orders, actions, score, events)
                VALUES %s
            """, round_tuples, page_size=100)

        conn.commit()
        return run_id
    finally:
        conn.close()


async def record_single(port, difficulty, seed, db_url):
    """Record a single game."""
    cfg = CONFIGS[difficulty]
    result = {"data": None}
    done = asyncio.Event()

    async def handler(ws):
        try:
            result["data"] = await run_game_recorded(ws, cfg, seed)
        except websockets.exceptions.ConnectionClosed:
            pass  # Bot disconnected after game ended; recording is already captured
        finally:
            done.set()

    server = await websockets.serve(handler, "localhost", port)

    # Start the bot
    bot_url = f"ws://localhost:{port}"
    proc = await asyncio.create_subprocess_exec(  # nosec B603 B607
        BOT_PATH, bot_url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    await done.wait()
    try:
        await asyncio.wait_for(proc.wait(), timeout=300)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    server.close()
    await server.wait_closed()

    if result["data"]:
        run_id = save_to_db(db_url, difficulty, result["data"])
        print(f"  Seed {seed}: score={result['data']['final_score']}, "
              f"orders={result['data']['orders_completed']}, "
              f"items={result['data']['items_delivered']} -> run_id={run_id}")
        return result["data"]["final_score"]
    return 0


async def record_sweep(port_base, difficulty, seeds, db_url):
    """Record multiple seeds sequentially."""
    scores = []
    for i, seed in enumerate(seeds):
        port = port_base + (i % 10)
        score = await record_single(port, difficulty, seed, db_url)
        scores.append(score)

    print(f"\n{'='*50}")
    print(f"Sweep complete: {difficulty}")
    print(f"  Seeds: {len(scores)}")
    print(f"  Max:  {max(scores)}")
    print(f"  Mean: {sum(scores)/len(scores):.1f}")
    print(f"  Min:  {min(scores)}")
    print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record grocery bot games to PostgreSQL")
    parser.add_argument("difficulty", choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=None, help="Single seed to record")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds (1000-1000+N)")
    parser.add_argument("--port", type=int, default=9900, help="Base port for game server")
    parser.add_argument("--db", default=DEFAULT_DB, help="PostgreSQL connection URL")
    args = parser.parse_args()

    if args.seed is not None:
        asyncio.run(record_single(args.port, args.difficulty, args.seed, args.db))
    elif args.seeds is not None:
        seed_list = list(range(1000, 1000 + args.seeds))
        asyncio.run(record_sweep(args.port, args.difficulty, seed_list, args.db))
    else:
        # Default: record 5 seeds
        seed_list = list(range(1000, 1005))
        asyncio.run(record_sweep(args.port, args.difficulty, seed_list, args.db))


if __name__ == "__main__":
    main()
