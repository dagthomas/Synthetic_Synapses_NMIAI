"""Lightweight nightmare WebSocket client — runs V3 solver in-process.

No subprocess, no PostgreSQL, no GPU threads. Just:
1. Connect to WS
2. Run V3 solver per round
3. Capture all orders seen
4. Save capture + solution to local_store

Usage:
    python nightmare_ws.py "wss://game.ainm.no/ws?token=..."
    python nightmare_ws.py "wss://..." --replay  # replay existing solution
"""
from __future__ import annotations

import _shared  # noqa: F401 — redirects solution_store to local_store

import asyncio
import json
import sys
import time

import websockets

from game_engine import (
    build_map_from_capture, ACT_WAIT,
    MapState, Order,
)
from precompute import PrecomputedTables
from nightmare_solver_v2 import NightmareSolverV3
from live_solver import ws_to_capture
from solution_store import save_capture, merge_capture, save_solution, load_solution, load_meta


def _build_replay_actions(solution, ms):
    """Convert solution [(action, item_idx), ...] to WS format per round."""
    ws_rounds = []
    for round_actions in solution:
        ws_acts = []
        for bid, (act, item_idx) in enumerate(round_actions):
            act = int(act)
            if act == 0:
                ws_acts.append({'bot': bid, 'action': 'wait'})
            elif act == 1:
                ws_acts.append({'bot': bid, 'action': 'move', 'direction': 'up'})
            elif act == 2:
                ws_acts.append({'bot': bid, 'action': 'move', 'direction': 'down'})
            elif act == 3:
                ws_acts.append({'bot': bid, 'action': 'move', 'direction': 'left'})
            elif act == 4:
                ws_acts.append({'bot': bid, 'action': 'move', 'direction': 'right'})
            elif act == 5:
                # Pickup — need item_id from map
                if 0 <= item_idx < len(ms.items):
                    ws_acts.append({'bot': bid, 'action': 'pick_up',
                                    'item_id': ms.items[item_idx]['id']})
                else:
                    ws_acts.append({'bot': bid, 'action': 'wait'})
            elif act == 6:
                ws_acts.append({'bot': bid, 'action': 'drop_off'})
            else:
                ws_acts.append({'bot': bid, 'action': 'wait'})
        ws_rounds.append(ws_acts)
    return ws_rounds


async def run_v3_live(ws_url: str, replay: bool = False, verbose: bool = True) -> tuple[int, dict | None]:
    """Play a live nightmare game with V3 solver. Returns (score, capture_data)."""
    t0 = time.time()
    solver = None
    ms = None
    capture = None
    seen_order_ids = set()
    final_score = 0
    action_log = []
    replay_actions = None

    print(f"Connecting to {ws_url[:80]}...", file=sys.stderr)

    try:
        ws = await asyncio.wait_for(
            websockets.connect(ws_url), timeout=15)
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 0, None

    try:
        async with ws:
            game_over = False
            while not game_over:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    print(f"Recv timeout", file=sys.stderr)
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}", file=sys.stderr)
                    break

                data = json.loads(message)

                if data['type'] == 'game_over':
                    final_score = data.get('score', 0)
                    print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                    break

                if data['type'] != 'game_state':
                    continue

                # Drain stale messages
                while True:
                    try:
                        peek = await asyncio.wait_for(ws.recv(), timeout=0.002)
                        peek_data = json.loads(peek)
                        if peek_data.get('type') == 'game_over':
                            final_score = peek_data.get('score', 0)
                            print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                            game_over = True
                            break
                        if peek_data.get('type') == 'game_state':
                            data = peek_data
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        break
                if game_over:
                    break

                rnd = data['round']
                score = data.get('score', 0)
                max_rounds = data.get('max_rounds', 500)

                # Round 0: init solver + capture
                if rnd == 0:
                    capture = ws_to_capture(data)
                    ms = build_map_from_capture(capture)
                    tables = PrecomputedTables.get(ms)

                    # Load existing capture orders for future-order awareness
                    from solution_store import load_capture as _load_cap
                    existing_cap = _load_cap('nightmare')
                    future_orders = []
                    if existing_cap:
                        for o in existing_cap.get('orders', []):
                            req_ids = [ms.type_name_to_id.get(n, 0) for n in o.get('items_required', [])]
                            future_orders.append(Order(0, req_ids, 'pending'))
                        print(f"  Loaded {len(future_orders)} future orders from capture",
                              file=sys.stderr)

                    solver = NightmareSolverV3(ms, tables, future_orders=future_orders)

                    if replay:
                        sol = load_solution('nightmare')
                        if sol:
                            replay_actions = _build_replay_actions(sol, ms)
                            meta = load_meta('nightmare')
                            expected = meta.get('score', 0) if meta else 0
                            print(f"  Replay mode: {len(replay_actions)} rounds, "
                                  f"expected={expected}", file=sys.stderr)
                        else:
                            print(f"  No solution to replay, using V3 live", file=sys.stderr)
                            replay = False

                # Track new orders (skip round 0 — already in capture from ws_to_capture)
                if rnd > 0:
                    for order in data.get('orders', []):
                        oid = order.get('id', f'order_{len(seen_order_ids)}')
                        if oid not in seen_order_ids:
                            seen_order_ids.add(oid)
                            if capture is not None:
                                capture['orders'].append({
                                    'id': oid,
                                    'items_required': order['items_required'],
                                    'items_delivered': list(order.get('items_delivered', [])),
                                    'status': order['status'],
                                })
                else:
                    # Mark round-0 orders as seen
                    for order in data.get('orders', []):
                        oid = order.get('id', f'order_{len(seen_order_ids)}')
                        seen_order_ids.add(oid)

                # Get actions
                if replay and replay_actions and rnd < len(replay_actions):
                    ws_actions = replay_actions[rnd]
                elif solver:
                    ws_actions = solver.ws_action(data['bots'], data, ms)
                else:
                    ws_actions = [{'bot': i, 'action': 'wait'}
                                  for i in range(len(data['bots']))]

                # Build action_log entry for solution saving
                if solver and ms:
                    round_acts = []
                    for wa in ws_actions:
                        bid = wa['bot']
                        act_str = wa['action']
                        item_idx = 0
                        if act_str == 'wait':
                            a = 0
                        elif act_str == 'move':
                            d = wa.get('direction', 'up')
                            a = {'up': 1, 'down': 2, 'left': 3, 'right': 4}[d]
                        elif act_str == 'pick_up':
                            a = 5
                            iid = wa.get('item_id', '')
                            for idx, item in enumerate(ms.items):
                                if item['id'] == iid:
                                    item_idx = idx
                                    break
                        elif act_str == 'drop_off':
                            a = 6
                        else:
                            a = 0
                        round_acts.append((a, item_idx))
                    action_log.append(round_acts)

                # Log
                if verbose and (rnd < 10 or rnd % 50 == 0 or rnd >= max_rounds - 3):
                    mode = "replay" if (replay and replay_actions and rnd < len(replay_actions)) else "v3"
                    # Show bot positions + action summary
                    bots = data.get('bots', [])
                    act_counts = {}
                    for a in ws_actions:
                        act_counts[a['action']] = act_counts.get(a['action'], 0) + 1
                    bot0_pos = bots[0]['position'] if bots else '?'
                    bot0_inv = bots[0].get('inventory', []) if bots else []
                    orders_info = ""
                    for od in data.get('orders', []):
                        s = od.get('status', '?')[0]
                        req = len(od.get('items_required', []))
                        deld = len(od.get('items_delivered', []))
                        orders_info += f" {s}:{deld}/{req}"
                    print(f"  R{rnd}/{max_rounds} Score:{score} [{mode}] "
                          f"b0={bot0_pos} inv={bot0_inv} acts={act_counts}{orders_info}",
                          file=sys.stderr)

                response = {'actions': ws_actions}
                try:
                    await asyncio.wait_for(ws.send(json.dumps(response)), timeout=5)
                except Exception as e:
                    print(f"Send error R{rnd}: {e}", file=sys.stderr)
                    break

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    elapsed = time.time() - t0
    n_orders = len(capture.get('orders', [])) if capture else 0
    print(f"\nResult: score={final_score}, orders_seen={n_orders}, time={elapsed:.1f}s",
          file=sys.stderr)

    # Save capture
    if capture and capture.get('orders'):
        merged, num_new, total = merge_capture('nightmare', capture)
        print(f"Capture saved: {total} orders ({num_new} new)", file=sys.stderr)

    # Save solution if we have action_log and score > 0
    if action_log and final_score > 0:
        saved = save_solution('nightmare', final_score, action_log)
        if saved:
            print(f"Solution saved: score={final_score}", file=sys.stderr)
        else:
            meta = load_meta('nightmare')
            existing = meta.get('score', 0) if meta else 0
            print(f"Solution not saved (existing={existing} >= {final_score})", file=sys.stderr)

    return final_score, capture


def run_live(ws_url: str, replay: bool = False, verbose: bool = True) -> tuple[int, dict | None]:
    """Synchronous wrapper for run_v3_live."""
    return asyncio.run(run_v3_live(ws_url, replay=replay, verbose=verbose))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare WS client (V3 in-process)')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('--replay', action='store_true', help='Replay existing solution')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()

    score, cap = run_live(args.ws_url, replay=args.replay, verbose=args.verbose)
    print(f"\nFinal: {score}", file=sys.stderr)
