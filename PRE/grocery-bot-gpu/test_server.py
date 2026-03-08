"""Local WebSocket server that mimics the live game server format.

Runs a nightmare game locally so nightmare_live.py can connect and play
without needing a real token. Uses game_engine.py for simulation.

Usage:
    python test_server.py [--port 8765] [--seed 7005] [--difficulty nightmare]

Then connect:
    python nightmare_live.py "ws://localhost:8765/ws?token=test"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np
import websockets

from game_engine import (
    init_game, step, build_map, generate_all_orders,
    GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, DX, DY, INV_CAP,
    CELL_WALL, CELL_SHELF,
)
from configs import CONFIGS, DIFF_ROUNDS


def state_to_server_msg(state: GameState, ms: MapState,
                        max_rounds: int, difficulty: str,
                        is_init: bool = False) -> dict:
    """Convert GameState to server-format JSON message."""
    num_bots = len(state.bot_positions)

    # Bots: server format {'id': int, 'position': [x, y], 'inventory': [type_name, ...]}
    bots = []
    for bid in range(num_bots):
        inv_names = []
        for i in range(INV_CAP):
            tid = int(state.bot_inventories[bid, i])
            if tid >= 0:
                inv_names.append(ms.item_type_names[tid])
        bots.append({
            'id': bid,
            'position': [int(state.bot_positions[bid, 0]),
                         int(state.bot_positions[bid, 1])],
            'inventory': inv_names,
        })

    # Orders: server format
    orders = []
    for o in state.orders:
        if o.complete:
            continue
        req_names = [ms.item_type_names[int(r)] for r in o.required]
        del_names = []
        for i in range(len(o.required)):
            if o.delivered[i]:
                del_names.append(ms.item_type_names[int(o.required[i])])
        orders.append({
            'id': o.id,
            'status': o.status,
            'items_required': req_names,
            'items_delivered': del_names,
        })

    msg = {
        'round': state.round,
        'score': state.score,
        'bots': bots,
        'orders': orders,
        'max_rounds': max_rounds,
    }

    if is_init:
        # Initial message includes grid, items, spawn, dropoffs
        walls = []
        for y in range(ms.height):
            for x in range(ms.width):
                if ms.grid[y, x] == CELL_WALL:
                    walls.append([x, y])

        msg['grid'] = {
            'width': ms.width,
            'height': ms.height,
            'walls': walls,
        }

        # Items: server format {'id': str, 'type': str, 'position': [x, y]}
        items = []
        for idx in range(ms.num_items):
            items.append({
                'id': f'item_{idx}',
                'type': ms.items[idx]['type'],
                'position': [int(ms.item_positions[idx, 0]),
                             int(ms.item_positions[idx, 1])],
            })
        msg['items'] = items

        # Dropoff zones: [[x, y], ...]
        msg['drop_off_zones'] = [[dz[0], dz[1]] for dz in ms.drop_off_zones]
        msg['drop_off'] = {'x': ms.drop_off[0], 'y': ms.drop_off[1]}

        # Spawn
        msg['spawn'] = {'x': ms.spawn[0], 'y': ms.spawn[1]}

        msg['difficulty'] = difficulty

    return msg


def parse_client_actions(actions_json: list[dict], ms: MapState) -> list[tuple[int, int]]:
    """Parse client actions from server format to game_engine format.

    Client sends: [{'bot': id, 'action': 'move_up/pickup/dropoff/wait', 'item_id': ...}, ...]
    Game engine wants: [(action_type, item_idx), ...] indexed by bot ID
    """
    num_bots = max(a['bot'] for a in actions_json) + 1
    result = [(ACT_WAIT, -1)] * num_bots

    # Build item_id -> idx lookup
    item_id_to_idx = {}
    for idx, item in enumerate(ms.items):
        item_id_to_idx[item['id']] = idx

    action_map = {
        'wait': ACT_WAIT,
        'move_up': ACT_MOVE_UP,
        'move_down': ACT_MOVE_DOWN,
        'move_left': ACT_MOVE_LEFT,
        'move_right': ACT_MOVE_RIGHT,
        'pickup': ACT_PICKUP,
        'pick_up': ACT_PICKUP,
        'dropoff': ACT_DROPOFF,
        'drop_off': ACT_DROPOFF,
    }

    for a in actions_json:
        bid = a['bot']
        act_name = a.get('action', 'wait')
        act_type = action_map.get(act_name, ACT_WAIT)

        item_idx = -1
        if act_type == ACT_PICKUP:
            item_id = a.get('item_id', '')
            item_idx = item_id_to_idx.get(item_id, -1)

        if bid < num_bots:
            result[bid] = (act_type, item_idx)

    return result


async def handle_game(ws, seed: int, difficulty: str, from_live: str = None):
    """Run one game session over WebSocket."""
    max_rounds = DIFF_ROUNDS.get(difficulty, 300)
    num_orders = 200

    if from_live:
        # Load captured live init data
        from game_engine import init_game_from_capture
        with open(from_live) as f:
            capture = json.load(f)
        # Build capture_data format for init_game_from_capture
        # Try to load captured orders from PostgreSQL
        captured_orders = []
        try:
            from solution_store import load_capture
            db_cap = load_capture(difficulty)
            if db_cap and db_cap.get('orders'):
                captured_orders = db_cap['orders']
                print(f"Loaded {len(captured_orders)} orders from DB", file=sys.stderr)
        except Exception as e:
            print(f"Could not load orders from DB: {e}", file=sys.stderr)

        cap = {
            'difficulty': difficulty,
            'grid': capture['grid'],
            'items': capture['items'],
            'drop_off': capture.get('drop_off', {}),
            'drop_off_zones': capture.get('drop_off_zones', []),
            'spawn': capture.get('spawn', {}),
            'num_bots': len(capture.get('bots', [])),
            'orders': captured_orders,
        }
        # Normalize items to have 'position' key
        for it in cap['items']:
            if 'position' not in it:
                it['position'] = [it['x'], it['y']]
        # Handle drop_off format
        do = cap['drop_off']
        if isinstance(do, dict):
            cap['drop_off'] = (do.get('x', 1), do.get('y', 16))
        state, all_orders = init_game_from_capture(cap, num_orders=num_orders)
        ms = state.map_state
    else:
        state, all_orders = init_game(seed, difficulty, num_orders=num_orders)
        ms = state.map_state

    print(f"Game started: {difficulty} seed={seed} "
          f"bots={len(state.bot_positions)} rounds={max_rounds} "
          f"items={ms.num_items} types={ms.num_types}", file=sys.stderr)
    print(f"Dropoffs: {[tuple(dz) for dz in ms.drop_off_zones]}", file=sys.stderr)
    print(f"Spawn: {ms.spawn}", file=sys.stderr)

    # Send initial state
    init_msg = state_to_server_msg(state, ms, max_rounds, difficulty, is_init=True)
    await ws.send(json.dumps(init_msg))

    for rnd in range(max_rounds):
        state.round = rnd

        # Wait for client actions
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            print(f"R{rnd}: Client timeout, using wait for all bots", file=sys.stderr)
            actions = [(ACT_WAIT, -1)] * len(state.bot_positions)
        else:
            try:
                parsed = json.loads(raw)
                # Accept both {"actions": [...]} and bare [...]
                if isinstance(parsed, dict) and 'actions' in parsed:
                    client_actions = parsed['actions']
                else:
                    client_actions = parsed
                actions = parse_client_actions(client_actions, ms)
            except Exception as e:
                print(f"R{rnd}: Parse error: {e}", file=sys.stderr)
                actions = [(ACT_WAIT, -1)] * len(state.bot_positions)

        # Debug: show parsed actions + pickup details
        if rnd < 10 or rnd % 100 == 0:
            act_names_map = {ACT_WAIT: 'wait', ACT_MOVE_UP: 'up', ACT_MOVE_DOWN: 'dn',
                            ACT_MOVE_LEFT: 'lt', ACT_MOVE_RIGHT: 'rt',
                            ACT_PICKUP: 'PU', ACT_DROPOFF: 'DO'}
            parsed = [(act_names_map.get(a, '?'), idx) for a, idx in actions[:5]]
            # Show bot inventories
            inv_sample = []
            for b in range(min(5, len(state.bot_positions))):
                inv = [int(state.bot_inventories[b, i]) for i in range(INV_CAP)
                       if state.bot_inventories[b, i] >= 0]
                if inv:
                    inv_sample.append(f"b{b}:{inv}")
            # Debug pickup attempts
            pu_debug = []
            for bid, (act, idx) in enumerate(actions):
                if act == ACT_PICKUP and idx >= 0:
                    bx = int(state.bot_positions[bid, 0])
                    by = int(state.bot_positions[bid, 1])
                    ix = int(ms.item_positions[idx, 0])
                    iy = int(ms.item_positions[idx, 1])
                    mdist = abs(bx - ix) + abs(by - iy)
                    ic = state.bot_inv_count(bid)
                    pu_debug.append(f"b{bid}@({bx},{by})->item{idx}@({ix},{iy}) d={mdist} inv={ic}")
            print(f"  SRV R{rnd}: parsed={parsed} inv={inv_sample}", file=sys.stderr)
            if pu_debug:
                for pd in pu_debug[:3]:
                    print(f"    PU: {pd}", file=sys.stderr)

        # Apply actions
        score_before = state.score
        step(state, actions, all_orders)
        delta = state.score - score_before

        if rnd < 5 or rnd % 50 == 0 or delta > 0 or rnd >= max_rounds - 5:
            print(f"R{rnd:3d}/{max_rounds} Score:{state.score:3d} "
                  f"Ord:{state.orders_completed} Delta:{delta:+d}",
                  file=sys.stderr)

        # Send round result (or game_over on last round)
        if rnd == max_rounds - 1:
            game_over = {
                'type': 'game_over',
                'game_over': True,
                'score': state.score,
                'orders_completed': state.orders_completed,
                'items_delivered': state.items_delivered,
            }
            await ws.send(json.dumps(game_over))
            print(f"\nGAME OVER: Score={state.score} "
                  f"Orders={state.orders_completed} "
                  f"Items={state.items_delivered}", file=sys.stderr)
        else:
            round_msg = state_to_server_msg(state, ms, max_rounds, difficulty)
            await ws.send(json.dumps(round_msg))


async def server_main(port: int, seed: int, difficulty: str, from_live: str = None):
    """Start WebSocket server."""
    async def handler(ws, path=None):
        print(f"Client connected from {ws.remote_address}", file=sys.stderr)
        try:
            await handle_game(ws, seed, difficulty, from_live=from_live)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected", file=sys.stderr)
        except Exception as e:
            print(f"Game error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    mode = f"from {from_live}" if from_live else f"seed={seed}"
    print(f"Test server starting on ws://localhost:{port}/ws", file=sys.stderr)
    print(f"Difficulty: {difficulty}, {mode}", file=sys.stderr)
    print(f"Connect with: python nightmare_live.py \"ws://localhost:{port}/ws?token=test\"",
          file=sys.stderr)

    async with websockets.serve(handler, "localhost", port, max_size=10_000_000):
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description='Local test game server')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--difficulty', default='nightmare')
    parser.add_argument('--from-live', default=None,
                        help='Load map from captured live init JSON (e.g. data/live_init.json)')
    args = parser.parse_args()

    asyncio.run(server_main(args.port, args.seed, args.difficulty,
                            from_live=args.from_live))


if __name__ == '__main__':
    main()
