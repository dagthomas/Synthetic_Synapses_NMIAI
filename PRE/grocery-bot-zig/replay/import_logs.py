#!/usr/bin/env python3
"""Import game_log JSONL files into PostgreSQL for replay visualization.

Usage:
    # Import all game_log*.jsonl files from parent directory
    python import_logs.py

    # Import specific files
    python import_logs.py ../game_log_1772318764.jsonl ../game_log_1772318765.jsonl

    # Import with custom DB
    python import_logs.py --db "$GROCERY_DB_URL" ../game_log.jsonl
"""
import argparse
import glob
import json
import os
import sys

import psycopg2
from psycopg2.extras import execute_values

DEFAULT_DB = os.environ.get("GROCERY_DB_URL", "postgres://grocery:grocery123@localhost:5433/grocery_bot")

# Difficulty detection by grid size and bot count
DIFFICULTY_MAP = {
    (12, 10, 1): "easy",
    (16, 12, 3): "medium",
    (22, 14, 5): "hard",
    (28, 18, 10): "expert",
    (30, 18, 20): "nightmare",
}


def detect_difficulty(width, height, bot_count):
    return DIFFICULTY_MAP.get((width, height, bot_count), "unknown")


def parse_log_file(path):
    """Parse a game_log JSONL file into a structured record for DB insertion."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return None

    # Separate game_state lines, action lines, and game_over
    game_states = []
    action_lines = []
    game_over = None

    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        if d.get("type") == "game_state":
            game_states.append(d)
        elif d.get("type") == "game_over":
            game_over = d
        elif "actions" in d and "type" not in d:
            action_lines.append(d)

    if not game_states:
        return None

    # Extract map info from round 0
    r0 = game_states[0]
    grid = r0["grid"]
    width = grid["width"]
    height = grid["height"]
    walls = grid["walls"]
    bot_count = len(r0["bots"])
    items = r0["items"]
    drop_off = r0["drop_off"]
    drop_off_zones = r0.get("drop_off_zones", [drop_off])  # Nightmare has 3 zones
    spawn = [width - 2, height - 2]  # Standard spawn position

    # Derive shelves from item positions (items sit adjacent to shelves)
    # The actual shelf positions are not in the game log — we store item positions instead
    # Shelves = unique positions of all items
    shelf_set = set()
    for item in items:
        pos = item["position"]
        shelf_set.add((pos[0], pos[1]))
    shelves = sorted([list(p) for p in shelf_set])

    # Item types
    item_type_set = set(it["type"] for it in items)
    item_types_count = len(item_type_set)

    # Detect order size from orders in round 0
    order_sizes = [len(o["items_required"]) for o in r0["orders"]]
    order_size_min = min(order_sizes) if order_sizes else 3
    order_size_max = max(order_sizes) if order_sizes else 5

    difficulty = detect_difficulty(width, height, bot_count)

    # Extract final score from game_over or last game_state
    if game_over:
        final_score = game_over["score"]
        items_delivered = game_over.get("items_delivered", 0)
        orders_completed = game_over.get("orders_completed", 0)
    else:
        last_state = game_states[-1]
        final_score = last_state["score"]
        items_delivered = 0
        orders_completed = 0

    # Build round records by pairing game_states with actions
    # Use dict to deduplicate rounds (keep last occurrence in case of desync)
    round_map = {}
    for i, gs in enumerate(game_states):
        rnd = gs["round"]
        # Actions for this round come right after this game_state
        actions = action_lines[i]["actions"] if i < len(action_lines) else []

        # Build bot state
        bots = [{"id": b["id"], "position": b["position"], "inventory": b.get("inventory", [])}
                for b in gs["bots"]]

        # Orders
        orders = []
        for o in gs["orders"]:
            orders.append({
                "id": o["id"],
                "items_required": o["items_required"],
                "items_delivered": o.get("items_delivered", []),
                "status": o.get("status", "active"),
            })

        round_map[rnd] = {
            "round": rnd,
            "bots": bots,
            "orders": orders,
            "actions": actions,
            "score": gs["score"],
            "events": [],  # Events not captured in game_log format
        }

    round_records = [round_map[k] for k in sorted(round_map.keys())]

    # Try to extract seed from filename (e.g. game_log_1772318764.jsonl -> use as pseudo-seed)
    basename = os.path.basename(path)
    seed = 0
    if "_" in basename:
        parts = basename.replace(".jsonl", "").split("_")
        for p in reversed(parts):
            try:
                seed = int(p)
                break
            except ValueError:
                continue

    # Items in DB format — position must be [x, y] array (Grid.svelte reads item.position[0/1])
    items_db = [{"id": it["id"], "type": it["type"], "position": it["position"]}
                for it in items]

    # Remove shelf cells from walls so they render as shelves (not walls) in the frontend.
    # The game log puts all non-walkable cells in grid.walls, including shelf cells.
    # The frontend checks: dropoff → spawn → wall → shelf, so shelf cells must NOT be in walls.
    shelf_coords = {(p[0], p[1]) for p in shelves}
    walls = [w for w in walls if tuple(w) not in shelf_coords]

    return {
        "seed": seed,
        "difficulty": difficulty,
        "grid_width": width,
        "grid_height": height,
        "bot_count": bot_count,
        "item_types": item_types_count,
        "order_size_min": order_size_min,
        "order_size_max": order_size_max,
        "walls": walls,
        "shelves": shelves,
        "items": items_db,
        "drop_off": drop_off,
        "drop_off_zones": drop_off_zones,
        "spawn": spawn,
        "final_score": final_score,
        "items_delivered": items_delivered,
        "orders_completed": orders_completed,
        "rounds": round_records,
    }


def save_to_db(db_url, record, run_type='live'):
    """Insert a parsed game record into PostgreSQL."""
    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, drop_off_zones, spawn,
                              final_score, items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            record["seed"], record["difficulty"],
            record["grid_width"], record["grid_height"], record["bot_count"],
            record["item_types"], record["order_size_min"], record["order_size_max"],
            json.dumps(record["walls"]), json.dumps(record["shelves"]),
            json.dumps(record["items"]), json.dumps(record["drop_off"]),
            json.dumps(record["drop_off_zones"]),
            json.dumps(record["spawn"]),
            record["final_score"], record["items_delivered"], record["orders_completed"],
            run_type,
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


def main():
    parser = argparse.ArgumentParser(description="Import game_log JSONL files into PostgreSQL")
    parser.add_argument("files", nargs="*", help="JSONL files to import (default: all game_log*.jsonl in parent dir)")
    parser.add_argument("--db", default=DEFAULT_DB, help="PostgreSQL connection URL")
    parser.add_argument("--run-type", default="live", choices=["live", "replay", "synthetic"], help="Run type: live, replay, or synthetic")
    args = parser.parse_args()

    # Determine files to import
    if args.files:
        files = args.files
    else:
        parent = os.path.join(os.path.dirname(__file__), "..")
        files = sorted(glob.glob(os.path.join(parent, "game_log*.jsonl")))

    if not files:
        print("No game_log files found.")
        return

    # Test DB connection
    try:
        conn = psycopg2.connect(args.db)
        conn.close()
    except Exception as e:
        print(f"Cannot connect to DB: {e}")
        return

    print(f"Importing {len(files)} log file(s) into PostgreSQL")
    print()

    imported = 0
    failed = 0

    for path in files:
        basename = os.path.basename(path)
        record = parse_log_file(path)
        if record is None:
            print(f"  {basename}: SKIP (no valid game data)")
            failed += 1
            continue

        try:
            run_id = save_to_db(args.db, record, run_type=args.run_type)
            print(f"  {basename}: {record['difficulty']} {record['grid_width']}x{record['grid_height']}, "
                  f"score={record['final_score']}, orders={record['orders_completed']}, "
                  f"rounds={len(record['rounds'])}, type={args.run_type} -> run_id={run_id}")
            imported += 1
        except Exception as e:
            print(f"  {basename}: ERROR - {e}")
            failed += 1

    print()
    print(f"Done: {imported} imported, {failed} failed/skipped")


if __name__ == "__main__":
    main()
