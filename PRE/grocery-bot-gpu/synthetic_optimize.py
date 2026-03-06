#!/usr/bin/env python3
"""Batch GPU sequential solver with progressive budgets and DB recording.

Runs solve_sequential across many seeds, escalating budget until a target score
is reached (or the max budget is exhausted). Every result is recorded to PostgreSQL.

Usage:
    python synthetic_optimize.py hard --seeds 7001-7020 --target 250
    python synthetic_optimize.py expert --seeds 42,7001 --target 200
    python synthetic_optimize.py hard --seeds 3 --budgets 200000,500000,1000000
    python synthetic_optimize.py hard --seeds 1 --max-time 300  # 5min per seed
"""
import argparse
import json
import os
import sys
import time

import psycopg2
from psycopg2.extras import execute_values
from configs import parse_seeds
from gpu_sequential_solver import solve_sequential

DB_URL = os.environ.get("GROCERY_DB_URL", "postgres://grocery@localhost:5433/grocery_bot")

DEFAULT_BUDGETS = {
    'easy':   [200_000, 500_000],
    'medium': [200_000, 500_000, 1_000_000, 2_000_000],
    'hard':   [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
    'expert': [200_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
}

DEFAULT_REFINE = {
    'easy': 0, 'medium': 3, 'hard': 6, 'expert': 12,
}

DEFAULT_ORDERINGS = {
    'easy': 1, 'medium': 1, 'hard': 3, 'expert': 5,
}



_map_cache = {}  # difficulty -> map data dict

def _get_map_data(difficulty):
    """Build and cache map data for a difficulty (walls, shelves, items, etc.)."""
    if difficulty in _map_cache:
        return _map_cache[difficulty]

    from game_engine import build_map, CONFIGS, CELL_WALL, CELL_SHELF
    ms = build_map(difficulty)
    cfg = CONFIGS[difficulty]

    walls = []
    shelves = []
    for y in range(ms.height):
        for x in range(ms.width):
            c = int(ms.grid[y, x])
            if c == CELL_WALL:
                walls.append([x, y])
            elif c == CELL_SHELF:
                shelves.append([x, y])

    items = [{"id": it["id"], "type": it["type"], "position": list(it["position"])}
             for it in ms.items]

    data = {
        'w': ms.width, 'h': ms.height, 'bots': cfg['bots'],
        'walls': walls, 'shelves': shelves, 'items': items,
        'drop_off': list(ms.drop_off), 'spawn': list(ms.spawn),
        'item_types': ms.num_types,
    }
    _map_cache[difficulty] = data
    return data


def _replay_rounds(seed, difficulty, actions, no_filler=True):
    """Replay actions through game_engine to produce per-round state for DB.

    Returns list of round dicts with bots, orders, actions, score, events.
    """
    from game_engine import (
        init_game, step, state_to_ws_format, actions_to_ws_format, MAX_ROUNDS,
    )
    num_orders = None  # let init_game use default
    gs, all_orders = init_game(seed, difficulty)
    ms = gs.map_state

    rounds = []
    for rnd in range(min(MAX_ROUNDS, len(actions))):
        gs.round = rnd
        ws_data = state_to_ws_format(gs, all_orders)
        ws_acts = actions_to_ws_format(actions[rnd], ms)

        bots = [{"id": b["id"], "position": b["position"], "inventory": b.get("inventory", [])}
                for b in ws_data["bots"]]
        orders = []
        for o in ws_data.get("orders", []):
            orders.append({
                "id": o["id"],
                "items_required": o["items_required"],
                "items_delivered": o.get("items_delivered", []),
                "status": o.get("status", "active"),
            })

        rounds.append({
            "round": rnd,
            "bots": bots,
            "orders": orders,
            "actions": ws_acts,
            "score": ws_data["score"],
            "events": [],
        })

        step(gs, actions[rnd], all_orders)

    return rounds, gs.score, gs.orders_completed, gs.items_delivered


def record_synthetic_score(difficulty, seed, score, max_states, refine_iters,
                           time_secs, orders_completed=0, items_delivered=0,
                           metadata=None, actions=None):
    """Insert synthetic run result into PostgreSQL runs + rounds tables.

    If actions is provided, replays through game_engine to populate rounds.
    """
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        md = _get_map_data(difficulty)

        # Replay to get round data and accurate delivery counts
        round_records = []
        if actions is not None:
            try:
                round_records, replay_score, replay_orders, replay_items = _replay_rounds(seed, difficulty, actions)
                orders_completed = replay_orders
                items_delivered = replay_items
            except Exception as e:
                print(f"  Replay error (rounds will be empty): {e}", file=sys.stderr)

        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, spawn,
                              final_score, items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            seed, difficulty, md['w'], md['h'], md['bots'],
            md['item_types'], 3, 5,
            json.dumps(md['walls']), json.dumps(md['shelves']),
            json.dumps(md['items']), json.dumps(md['drop_off']),
            json.dumps(md['spawn']),
            score, items_delivered, orders_completed,
            'synthetic',
        ))
        run_id = cur.fetchone()[0]

        # Insert rounds in batches
        if round_records:
            round_tuples = [
                (run_id, r["round"], json.dumps(r["bots"]), json.dumps(r["orders"]),
                 json.dumps(r["actions"]), r["score"], json.dumps(r["events"]))
                for r in round_records
            ]
            execute_values(cur, """
                INSERT INTO rounds (run_id, round_number, bots, orders, actions, score, events)
                VALUES %s
            """, round_tuples, page_size=100)

        conn.commit()
        conn.close()
        return run_id
    except Exception as e:
        print(f"  DB error: {e}", file=sys.stderr)
        return None


def run_seed(seed, difficulty, budgets, target, device, refine_iters,
             num_orderings, max_time_s, verbose):
    """Run progressive budgets for a single seed. Returns (best_score, best_budget, total_time)."""
    best_score = 0
    best_budget = 0
    total_time = 0

    for budget in budgets:
        t0 = time.time()
        try:
            score, actions = solve_sequential(
                seed=seed,
                difficulty=difficulty,
                device=device,
                max_states=budget,
                verbose=verbose,
                max_refine_iters=refine_iters,
                num_pass1_orderings=num_orderings,
                no_filler=True,
                max_time_s=max_time_s,
            )
        except Exception as e:
            print(f"  ERROR seed={seed} budget={budget}: {e}", file=sys.stderr)
            score = -1

        elapsed = time.time() - t0
        total_time += elapsed

        if score > 0:
            # Record to DB (pass actions for round-level replay data)
            run_id = record_synthetic_score(
                difficulty, seed, score, budget, refine_iters, elapsed,
                metadata={'num_orderings': num_orderings},
                actions=actions if score > 0 else None,
            )
            db_str = f" -> db#{run_id}" if run_id else ""
            print(f"  seed={seed} budget={budget:>10,} -> score={score:>4}  "
                  f"({elapsed:.1f}s){db_str}", file=sys.stderr)

            if score > best_score:
                best_score = score
                best_budget = budget

        if score >= target:
            print(f"  TARGET {target} reached! Skipping remaining budgets.", file=sys.stderr)
            break

    return best_score, best_budget, total_time


def main():
    parser = argparse.ArgumentParser(
        description='Batch GPU solver with progressive budgets and DB recording')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seeds', default='7001-7003',
                        help='Seeds: "7001-7003", "42,7001", or count "3"')
    parser.add_argument('--target', type=int, default=250,
                        help='Stop escalating budget when this score is reached (default: 250)')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list (e.g. "200000,500000,1000000")')
    parser.add_argument('--refine-iters', type=int, default=None,
                        help='Override refinement iterations')
    parser.add_argument('--orderings', type=int, default=None,
                        help='Override num_pass1_orderings')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-time', type=int, default=None,
                        help='Max seconds per solve call')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-db', action='store_true',
                        help='Skip DB recording')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    diff = args.difficulty

    if args.budgets:
        budgets = [int(b.strip()) for b in args.budgets.split(',')]
    else:
        budgets = DEFAULT_BUDGETS[diff]

    refine_iters = args.refine_iters if args.refine_iters is not None else DEFAULT_REFINE[diff]
    num_orderings = args.orderings if args.orderings is not None else DEFAULT_ORDERINGS[diff]

    print(f"Synthetic optimize: {diff}", file=sys.stderr)
    print(f"  Seeds: {seeds}", file=sys.stderr)
    print(f"  Budgets: {budgets}", file=sys.stderr)
    print(f"  Target: {args.target}", file=sys.stderr)
    print(f"  Refine iters: {refine_iters}, Orderings: {num_orderings}", file=sys.stderr)
    print(f"  Device: {args.device}", file=sys.stderr)
    print(file=sys.stderr)

    # Test DB connection
    if not args.no_db:
        try:
            conn = psycopg2.connect(DB_URL)
            conn.close()
            print("  DB: connected", file=sys.stderr)
        except Exception as e:
            print(f"  DB: NOT available ({e}). Use --no-db to skip.", file=sys.stderr)
            return

    results = {}
    t_total = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed {seed}", file=sys.stderr)
        best_score, best_budget, seed_time = run_seed(
            seed, diff, budgets, args.target, args.device,
            refine_iters, num_orderings, args.max_time, args.verbose,
        )
        results[seed] = (best_score, best_budget, seed_time)

    total_time = time.time() - t_total

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SYNTHETIC RESULTS: {diff} (target={args.target})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"{'Seed':>8} {'Best':>6} {'Budget':>12} {'Time':>8}", file=sys.stderr)
    print(f"{'-'*38}", file=sys.stderr)

    scores = []
    for seed in seeds:
        sc, bud, t = results[seed]
        hit = ' *' if sc >= args.target else ''
        print(f"{seed:>8} {sc:>6} {bud:>12,} {t:>7.1f}s{hit}", file=sys.stderr)
        if sc > 0:
            scores.append(sc)

    if scores:
        print(f"\nMean: {sum(scores)/len(scores):.1f}  Max: {max(scores)}  "
              f"Min: {min(scores)}  Hits: {sum(1 for s in scores if s >= args.target)}/{len(scores)}",
              file=sys.stderr)
    print(f"Total time: {total_time:.0f}s", file=sys.stderr)


if __name__ == '__main__':
    main()
