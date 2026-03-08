"""Nightmare live client: reactive solve, train offline, replay trained solution.

Pipeline: run → capture orders → train → replay → repeat
Game is deterministic per day — same orders every time.

Usage:
    python nightmare_live.py "wss://...?token=..."              # reactive V4
    python nightmare_live.py "wss://...?token=..." --replay     # replay trained solution
    python nightmare_live.py "wss://...?token=..." --loop       # auto: run, train, replay, repeat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import websockets

from game_engine import (
    MapState, CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF,
    ACT_WAIT, ACT_PICKUP, ACT_DROPOFF, actions_to_ws_format,
    build_map_from_capture, init_game_from_capture,
)
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables


def build_map_state(data: dict) -> MapState:
    """Build MapState from initial game data."""
    import numpy as np

    ms = MapState()
    grid_data = data['grid']
    ms.width = grid_data['width']
    ms.height = grid_data['height']

    ms.grid = np.zeros((ms.height, ms.width), dtype=np.int8)
    for wall in grid_data.get('walls', []):
        ms.grid[wall[1], wall[0]] = CELL_WALL

    items = data.get('items', [])
    ms.items = items
    ms.num_items = len(items)

    type_names = sorted(set(it['type'] for it in items))
    ms.type_name_to_id = {name: i for i, name in enumerate(type_names)}
    ms.item_type_names = type_names
    ms.num_types = len(type_names)

    ms.item_positions = np.zeros((ms.num_items, 2), dtype=np.int16)
    ms.item_types = np.zeros(ms.num_items, dtype=np.int8)

    for idx, it in enumerate(items):
        if 'position' in it:
            ix, iy = it['position'][0], it['position'][1]
            it['x'] = ix
            it['y'] = iy
        else:
            ix, iy = it['x'], it['y']
        ms.item_positions[idx] = [ix, iy]
        ms.item_types[idx] = ms.type_name_to_id.get(it['type'], 0)
        ms.grid[iy, ix] = CELL_SHELF

    drop_offs = data.get('drop_off_zones', [])
    if not drop_offs and 'drop_off' in data:
        d = data['drop_off']
        drop_offs = [[d['x'], d['y']]]

    ms.drop_off_zones = []
    for dz in drop_offs:
        if isinstance(dz, dict):
            x, y = dz['x'], dz['y']
        else:
            x, y = dz[0], dz[1]
        ms.drop_off_zones.append((x, y))
        ms.grid[y, x] = CELL_DROPOFF
    ms.drop_off = ms.drop_off_zones[0] if ms.drop_off_zones else (0, 0)

    spawn = data.get('spawn', data.get('spawn_point',
                     {'x': ms.width - 2, 'y': ms.height - 2}))
    if isinstance(spawn, dict):
        ms.spawn = (spawn['x'], spawn['y'])
    else:
        ms.spawn = (spawn[0], spawn[1])
    ms.grid[ms.spawn[1], ms.spawn[0]] = CELL_FLOOR

    ms.item_adjacencies = {}
    for idx in range(ms.num_items):
        ix = int(ms.item_positions[idx, 0])
        iy = int(ms.item_positions[idx, 1])
        adj = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < ms.width and 0 <= ny < ms.height:
                cell = ms.grid[ny, nx]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    adj.append((nx, ny))
        ms.item_adjacencies[idx] = adj

    return ms


def save_live_data(data: dict, capture: dict):
    """Save init data and capture to disk and PostgreSQL (merge orders)."""
    os.makedirs('data', exist_ok=True)
    with open('data/live_init.json', 'w') as f:
        json.dump(data, f)
    try:
        from solution_store import merge_capture
        merged, num_new, total = merge_capture('nightmare', capture)
        # Save merged version locally
        with open('data/nightmare_capture.json', 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"DB: merged {len(capture['orders'])} new → {total} total orders "
              f"(+{num_new} new)", file=sys.stderr)
    except Exception as e:
        print(f"DB save error: {e}", file=sys.stderr)
        # Fallback: save local only
        with open('data/nightmare_capture.json', 'w') as f:
            json.dump(capture, f, indent=2)


def build_item_id_map(live_ms: MapState, train_ms: MapState) -> dict[int, str]:
    """Map trained item indices → live server item IDs by position."""
    pos_to_live_id = {}
    for idx in range(live_ms.num_items):
        pos = (int(live_ms.item_positions[idx, 0]),
               int(live_ms.item_positions[idx, 1]))
        pos_to_live_id[pos] = live_ms.items[idx]['id']

    idx_to_id = {}
    for idx in range(train_ms.num_items):
        pos = (int(train_ms.item_positions[idx, 0]),
               int(train_ms.item_positions[idx, 1]))
        if pos in pos_to_live_id:
            idx_to_id[idx] = pos_to_live_id[pos]
    return idx_to_id


def trained_actions_to_ws(actions, item_id_map: dict[int, str],
                          num_bots: int) -> list[dict]:
    """Convert trained action tuples to WS format with live item IDs."""
    action_names = ['wait', 'move_up', 'move_down', 'move_left',
                    'move_right', 'pick_up', 'drop_off']
    ws_actions = []
    for bid in range(min(len(actions), num_bots)):
        act_type, item_idx = actions[bid]
        a = {'bot': bid, 'action': action_names[act_type]}
        if act_type == ACT_PICKUP:
            live_id = item_id_map.get(item_idx)
            if live_id:
                a['item_id'] = live_id
            else:
                a['action'] = 'wait'
        ws_actions.append(a)
    return ws_actions


def load_trained_solution():
    """Load best trained solution from DB or file."""
    # Try file first (most recent training)
    path = 'data/nightmare_best_live.json'
    if os.path.exists(path):
        with open(path) as f:
            sol = json.load(f)
        score = sol.get('score', 0)
        log = sol.get('action_log', [])
        if log:
            print(f"Loaded trained solution: score={score}, "
                  f"rounds={len(log)}", file=sys.stderr)
            return score, log
    return 0, []


def train_offline(max_time=300):
    """Train on captured data, save best solution."""
    from solution_store import load_capture
    cap = load_capture('nightmare')
    if not cap or not cap.get('orders'):
        print("No captured orders for training", file=sys.stderr)
        return 0, []

    # Fix format
    do = cap.get('drop_off', {})
    if isinstance(do, dict):
        cap['drop_off'] = (do.get('x', 1), do.get('y', 16))
    for it in cap['items']:
        if 'position' not in it:
            it['position'] = [it['x'], it['y']]

    print(f"Training: {len(cap['orders'])} orders, {max_time}s budget...",
          file=sys.stderr)

    from nightmare_offline import NightmareTrainer
    trainer = NightmareTrainer(capture_data=cap, verbose=True)
    best_score, best_log = trainer.train(max_time=max_time,
                                         num_restarts=max(10, max_time // 10))

    # Save
    os.makedirs('data', exist_ok=True)
    with open('data/nightmare_best_live.json', 'w') as f:
        json.dump({'score': best_score, 'action_log': best_log}, f)
    print(f"Trained: score={best_score}", file=sys.stderr)
    return best_score, best_log


def get_trained_map_state():
    """Build MapState from capture data (same as trainer uses)."""
    from solution_store import load_capture
    cap = load_capture('nightmare')
    if not cap:
        return None
    do = cap.get('drop_off', {})
    if isinstance(do, dict):
        cap['drop_off'] = (do.get('x', 1), do.get('y', 16))
    for it in cap['items']:
        if 'position' not in it:
            it['position'] = [it['x'], it['y']]
    return build_map_from_capture(cap)


async def run_game(ws_url: str, replay: bool = False):
    """Play one game. Returns (score, num_orders_captured)."""
    t_start = time.time()
    mode = "REPLAY" if replay else "REACTIVE"
    print(f"[{mode}] Connecting...", file=sys.stderr)

    async with websockets.connect(ws_url, max_size=10_000_000) as ws:
        init_msg = await ws.recv()
        data = json.loads(init_msg)

        if 'error' in data:
            print(f"ERROR: {data['error']}", file=sys.stderr)
            return 0, 0

        print(f"Connected! Bots: {len(data.get('bots', []))}, "
              f"Rounds: {data.get('max_rounds', '?')}", file=sys.stderr)

        ms = build_map_state(data)
        tables = PrecomputedTables.get(ms)
        solver = LMAPFSolver(ms, tables)

        print(f"Map: {ms.width}x{ms.height}, Items: {ms.num_items}",
              file=sys.stderr)

        # Build capture
        normalized_items = []
        for it in data['items']:
            nit = dict(it)
            if 'position' in nit and 'x' not in nit:
                nit['x'] = nit['position'][0]
                nit['y'] = nit['position'][1]
            normalized_items.append(nit)

        capture = {
            'difficulty': 'nightmare',
            'grid': data['grid'],
            'items': normalized_items,
            'drop_off': data.get('drop_off', {}),
            'drop_off_zones': data.get('drop_off_zones', []),
            'spawn': data.get('spawn', {}),
            'num_bots': len(data.get('bots', [])),
            'orders': [],
        }
        seen_orders = set()

        # Load trained solution for replay
        action_log = []
        item_id_map = {}
        using_replay = False
        if replay:
            train_score, action_log = load_trained_solution()
            if action_log:
                train_ms = get_trained_map_state()
                if train_ms:
                    item_id_map = build_item_id_map(ms, train_ms)
                    using_replay = True
                    print(f"Replay mode: {train_score}pts, "
                          f"{len(action_log)} rounds, "
                          f"{len(item_id_map)} items mapped",
                          file=sys.stderr)

        max_rounds = data.get('max_rounds', 500)
        rnd = 0
        score = 0
        desync = False
        num_bots = len(data.get('bots', []))

        while True:
            if rnd == 0:
                game_data = data
            else:
                msg = await ws.recv()
                game_data = json.loads(msg)

            if game_data.get('type') == 'game_over' or 'game_over' in game_data:
                score = game_data.get('score', score)
                print(f"\nGAME_OVER Score: {score}", file=sys.stderr)
                break

            # Track orders
            for order in game_data.get('orders', []):
                oid = order.get('id', f'o{len(seen_orders)}')
                if oid not in seen_orders:
                    seen_orders.add(oid)
                    capture['orders'].append({
                        'items_required': order['items_required'],
                    })

            score = game_data.get('score', score)
            live_bots = game_data.get('bots', [])

            # Choose action source
            if using_replay and not desync and rnd < len(action_log):
                ws_actions = trained_actions_to_ws(
                    action_log[rnd], item_id_map, num_bots)
                source = 'replay'
            else:
                # Reactive fallback
                try:
                    ws_actions = solver.ws_action(live_bots, game_data, ms)
                except Exception as e:
                    print(f"R{rnd} solver error: {e}", file=sys.stderr)
                    ws_actions = [{'bot': b['id'], 'action': 'wait'}
                                  for b in live_bots]
                source = 'V4'

            await ws.send(json.dumps({'actions': ws_actions}))

            # Progress logging
            if rnd < 5 or rnd % 50 == 0 or rnd >= max_rounds - 5:
                n_inv = sum(1 for b in live_bots if b.get('inventory'))
                print(f"R{rnd:3d}/{max_rounds} Score:{score:3d}"
                      f" Ord:{len(seen_orders)} Inv:{n_inv} [{source}]",
                      file=sys.stderr)

            rnd += 1

        elapsed = time.time() - t_start
        n_orders = len(capture['orders'])
        print(f"\nFinal: Score={score}, Orders={n_orders}, "
              f"Time={elapsed:.1f}s", file=sys.stderr)

        save_live_data(data, capture)
        return score, n_orders


async def run_loop(ws_url_template: str, fetch_token_fn=None):
    """Full pipeline: run → train → replay → repeat.

    Since game is deterministic per day, each run captures same orders.
    Training improves the action plan. Replay uses the trained plan.
    """
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ITERATION {iteration}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Step 1: Get token
        if fetch_token_fn:
            ws_url = fetch_token_fn()
            if not ws_url:
                print("Could not get token, stopping", file=sys.stderr)
                break
        else:
            ws_url = ws_url_template

        # Step 2: Run (reactive or replay)
        use_replay = iteration > 1  # First run is reactive, rest replay
        score, n_orders = await run_game(ws_url, replay=use_replay)

        if score == 0 and use_replay:
            # Desync — replay failed, run reactive
            print("Replay got 0, retrying reactive...", file=sys.stderr)
            if fetch_token_fn:
                ws_url = fetch_token_fn()
                if ws_url:
                    score, n_orders = await run_game(ws_url, replay=False)

        # Step 3: Train offline
        print(f"\nTraining with {n_orders} orders...", file=sys.stderr)
        train_score, _ = train_offline(max_time=300)

        print(f"\nIteration {iteration}: live={score}, "
              f"trained={train_score}", file=sys.stderr)

        # Step 4: Check if we should continue
        if not fetch_token_fn:
            print("No token fetcher — stopping after 1 iteration",
                  file=sys.stderr)
            print("Run again with a new token to replay the trained solution",
                  file=sys.stderr)
            break


def fetch_token():
    """Try to fetch a nightmare token."""
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'fetch_token.py', 'nightmare'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            token_data = json.loads(result.stdout)
            return token_data.get('ws_url', '')
    except Exception as e:
        print(f"Token fetch error: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description='Nightmare live solver')
    parser.add_argument('ws_url', nargs='?', help='WebSocket URL with token')
    parser.add_argument('--replay', action='store_true',
                        help='Replay trained solution')
    parser.add_argument('--loop', action='store_true',
                        help='Run → train → replay loop')
    parser.add_argument('--train-only', action='store_true',
                        help='Just train offline (no live game)')
    parser.add_argument('--train-time', type=int, default=300,
                        help='Training time budget (seconds)')
    args = parser.parse_args()

    if args.train_only:
        score, _ = train_offline(max_time=args.train_time)
        print(json.dumps({'score': score, 'mode': 'train'}))
        return

    if not args.ws_url:
        print("Need ws_url or --train-only", file=sys.stderr)
        sys.exit(1)

    if args.loop:
        asyncio.run(run_loop(args.ws_url))
    else:
        score, _ = asyncio.run(run_game(args.ws_url, replay=args.replay))
        print(json.dumps({'score': score, 'difficulty': 'nightmare'}))


if __name__ == '__main__':
    main()
