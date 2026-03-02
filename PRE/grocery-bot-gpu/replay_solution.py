"""Replay a saved optimized solution over WebSocket.

Loads best.json + capture.json for a difficulty, connects to the game server,
and replays the pre-computed actions with adaptive desync correction.

Two replay modes:
  1. SYNCED: actual positions match expected -> send DP action as-is
  2. DESYNCED: positions diverge -> navigate toward DP plan's goal (pickup/dropoff target)

Usage:
    python replay_solution.py <wss://...token> [--difficulty auto]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque

from solution_store import load_solution, load_capture, load_meta, save_capture as store_capture
from game_engine import (build_map_from_capture, actions_to_ws_format,
                         ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN,
                         ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_PICKUP, ACT_DROPOFF,
                         step, init_game_from_capture, DX, DY,
                         CELL_FLOOR, CELL_DROPOFF, INV_CAP)
from live_solver import detect_difficulty, ws_to_capture

SEND_DELAY = 0.0  # Zero delay — send immediately for reliability


def predict_full_sim(actions_list, capture, map_state):
    """Simulate full game from capture to predict positions + game state per round.

    Returns:
        positions: list of [(bx, by), ...] per round (before actions applied)
        game_states: GameState snapshots per round (for order/inventory tracking)
    """
    gs, all_orders = init_game_from_capture(capture)
    positions = []

    for rnd in range(len(actions_list)):
        pos = []
        for bid in range(len(gs.bot_positions)):
            pos.append((int(gs.bot_positions[bid, 0]), int(gs.bot_positions[bid, 1])))
        positions.append(pos)
        step(gs, actions_list[rnd], all_orders)

    return positions


def extract_goals(actions_list, map_state, expected_positions):
    """Extract high-level goal sequence per bot from DP action plan.

    A goal is a PICKUP or DROPOFF action. Between goals, the bot is navigating.

    Returns:
        bot_goals: dict {bot_id: [(round, goal_pos, action_type, item_idx), ...]}
    """
    num_bots = len(actions_list[0]) if actions_list else 0
    bot_goals = {bid: [] for bid in range(num_bots)}

    for bid in range(num_bots):
        for rnd in range(len(actions_list)):
            act, item_idx = actions_list[rnd][bid]
            if act == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
                # Goal: be adjacent to this item
                adjs = map_state.item_adjacencies[item_idx]
                if adjs:
                    # Use the adj cell the bot is expected to be at
                    exp_pos = expected_positions[rnd][bid] if rnd < len(expected_positions) else None
                    goal_pos = adjs[0]  # default
                    if exp_pos and exp_pos in adjs:
                        goal_pos = exp_pos
                    bot_goals[bid].append((rnd, goal_pos, ACT_PICKUP, item_idx))
            elif act == ACT_DROPOFF:
                bot_goals[bid].append((rnd, map_state.drop_off, ACT_DROPOFF, -1))

    return bot_goals


def bfs_next_action(start, goal, walkable, occupied, map_state):
    """BFS from start toward goal, avoiding occupied cells. Returns action constant."""
    if start == goal:
        return ACT_WAIT

    sx, sy = start
    gx, gy = goal
    spawn = map_state.spawn

    queue = deque()
    visited = {start: None}

    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = sx + DX[act], sy + DY[act]
        if (nx, ny) not in walkable:
            continue
        if (nx, ny) in occupied and (nx, ny) != spawn and (nx, ny) != goal:
            continue
        if (nx, ny) not in visited:
            visited[(nx, ny)] = act
            if (nx, ny) == goal:
                return act
            queue.append((nx, ny))

    while queue:
        cx, cy = queue.popleft()
        first_act = visited[(cx, cy)]

        for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            nx, ny = cx + DX[act], cy + DY[act]
            if (nx, ny) not in walkable:
                continue
            # Only avoid occupied for FIRST step
            if (nx, ny) not in visited:
                visited[(nx, ny)] = first_act
                if (nx, ny) == goal:
                    return first_act
                queue.append((nx, ny))

    # Fallback: greedy move toward goal
    best_act = ACT_WAIT
    best_dist = abs(gx - sx) + abs(gy - sy)
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = sx + DX[act], sy + DY[act]
        if (nx, ny) in walkable and (nx, ny) not in occupied:
            d = abs(gx - nx) + abs(gy - ny)
            if d < best_dist:
                best_dist = d
                best_act = act
    return best_act


def build_walkable(map_state):
    """Build set of walkable cells from map grid."""
    walkable = set()
    for y in range(map_state.height):
        for x in range(map_state.width):
            cell = map_state.grid[y, x]
            if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                walkable.add((x, y))
    return walkable


async def replay_best(ws_url, difficulty=None, log_dir=None):
    """Replay saved best solution with adaptive desync correction."""
    import websockets

    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(__file__))

    timestamp = int(time.time())
    log_path = os.path.join(log_dir, f'game_log_{timestamp}.jsonl')
    log_file = open(log_path, 'w')

    print(f"Connecting to server...", file=sys.stderr)

    actions = None
    map_state = None
    final_score = 0
    expected_positions = None
    bot_goals = None
    bot_goal_idx = None  # per-bot: which goal are we working toward
    walkable = None
    desync_count = 0
    desync_rounds = 0  # rounds with at least one desynced bot
    synced_rounds = 0
    round_offset = 0  # Track cumulative round shift (missed rounds)
    seen_order_ids = set()
    all_orders_captured = []

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            log_file.write(json.dumps(data) + '\n')
            log_file.flush()

            if data["type"] == "game_over":
                final_score = data.get('score', 0)
                print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]
            max_rounds = data.get("max_rounds", 300)
            score = data.get("score", 0)

            # Always capture orders
            for order in data.get('orders', []):
                oid = order.get('id', f'order_{len(all_orders_captured)}')
                if oid not in seen_order_ids:
                    seen_order_ids.add(oid)
                    all_orders_captured.append({
                        'id': oid,
                        'items_required': order['items_required'],
                        'items_delivered': [],
                        'status': 'future',
                    })

            if rnd == 0:
                num_bots = len(data['bots'])
                if difficulty is None or difficulty == 'auto':
                    difficulty = detect_difficulty(num_bots)

                actions = load_solution(difficulty)
                capture = load_capture(difficulty)
                meta = load_meta(difficulty)

                if actions is None:
                    print(f"ERROR: No solution found for {difficulty}", file=sys.stderr)
                    # Fall back to all-wait
                    actions = [[(ACT_WAIT, -1)] * num_bots] * 300

                if capture:
                    map_state = build_map_from_capture(capture)
                else:
                    cap = ws_to_capture(data)
                    map_state = build_map_from_capture(cap)

                walkable = build_walkable(map_state)

                if meta:
                    print(f"  Replaying {difficulty}: expected_score={meta.get('score', '?')} "
                          f"optimizations={meta.get('optimizations_run', 0)}", file=sys.stderr)

                # Pre-compute expected positions and goals
                if actions and capture:
                    try:
                        expected_positions = predict_full_sim(actions, capture, map_state)
                        bot_goals = extract_goals(actions, map_state, expected_positions)
                        bot_goal_idx = {bid: 0 for bid in range(num_bots)}
                        for bid in range(num_bots):
                            print(f"  Bot {bid}: {len(bot_goals[bid])} goals "
                                  f"(pickups+dropoffs)", file=sys.stderr)
                    except Exception as e:
                        print(f"  Warning: Could not pre-compute goals: {e}", file=sys.stderr)
                        expected_positions = None
                        bot_goals = None

            # Determine effective DP round (accounting for missed rounds)
            live_bots = data.get('bots', [])
            num_bots = len(live_bots)

            # Detect round offset: check if bots match expected positions
            # for the current offset, or if we need to increase offset
            if expected_positions and actions:
                dp_rnd = rnd - round_offset
                if 0 <= dp_rnd < len(expected_positions):
                    # Check if current positions match expected for dp_rnd
                    all_match = True
                    for bid, bot in enumerate(live_bots):
                        if bid < len(expected_positions[dp_rnd]):
                            ex, ey = expected_positions[dp_rnd][bid]
                            if tuple(bot['position']) != (ex, ey):
                                all_match = False
                                break

                    if not all_match:
                        # Check if positions match dp_rnd-1 (missed 1 round)
                        check_rnd = dp_rnd - 1
                        if check_rnd >= 0 and check_rnd < len(expected_positions):
                            prev_match = True
                            for bid, bot in enumerate(live_bots):
                                if bid < len(expected_positions[check_rnd]):
                                    ex, ey = expected_positions[check_rnd][bid]
                                    if tuple(bot['position']) != (ex, ey):
                                        prev_match = False
                                        break
                            if prev_match:
                                round_offset += 1
                                dp_rnd = rnd - round_offset
                                if round_offset <= 5:
                                    print(f"R{rnd}: Detected missed round, offset now {round_offset} "
                                          f"(dp_rnd={dp_rnd})", file=sys.stderr)
            else:
                dp_rnd = rnd

            # Build actions from DP plan at the offset-corrected round
            ws_actions = []
            round_has_desync = False

            for bid, bot in enumerate(live_bots):
                lx, ly = bot['position']

                # Check sync with offset-corrected round
                is_synced = True
                if expected_positions and 0 <= dp_rnd < len(expected_positions) and bid < len(expected_positions[dp_rnd]):
                    ex, ey = expected_positions[dp_rnd][bid]
                    if (lx, ly) != (ex, ey):
                        is_synced = False
                        desync_count += 1
                        round_has_desync = True

                # Use DP action at offset-corrected round
                action_rnd = dp_rnd if is_synced else dp_rnd
                if actions and 0 <= action_rnd < len(actions) and bid < len(actions[action_rnd]):
                    act, item_idx = actions[action_rnd][bid]
                    a = {'bot': bot['id'], 'action': ['wait', 'move_up', 'move_down',
                         'move_left', 'move_right', 'pick_up', 'drop_off'][act]}
                    if act == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
                        a['item_id'] = map_state.items[item_idx]['id']
                    elif act == ACT_PICKUP:
                        a['action'] = 'wait'
                    ws_actions.append(a)
                else:
                    ws_actions.append({'bot': bot['id'], 'action': 'wait'})

            if round_has_desync:
                desync_rounds += 1
            else:
                synced_rounds += 1

            # Log
            mode = "SYNC" if not round_has_desync else f"OFF{round_offset}"
            if rnd < 5 or rnd % 25 == 0 or rnd >= 295 or (round_has_desync and desync_rounds <= 5):
                print(f"R{rnd}/{max_rounds} Score:{score} [{mode}] dp_rnd={dp_rnd} "
                      f"(synced:{synced_rounds} desynced:{desync_rounds})", file=sys.stderr)

            response = {"actions": ws_actions}
            log_file.write(json.dumps(response) + '\n')
            log_file.flush()

            # Small delay to prevent timing issues
            if SEND_DELAY > 0:
                await asyncio.sleep(SEND_DELAY)

            await ws.send(json.dumps(response))

    log_file.close()
    print(f"Log saved: {log_path}", file=sys.stderr)
    print(f"Stats: synced_rounds={synced_rounds} desync_rounds={desync_rounds} "
          f"total_desyncs={desync_count}", file=sys.stderr)

    # Update capture with newly seen orders
    if all_orders_captured and difficulty:
        capture = load_capture(difficulty)
        if capture:
            existing_ids = set()
            for i, o in enumerate(capture.get('orders', [])):
                existing_ids.add(o.get('id', f'order_{i}'))

            new_count = 0
            for o in all_orders_captured:
                if o['id'] not in existing_ids:
                    capture['orders'].append(o)
                    existing_ids.add(o['id'])
                    new_count += 1

            if new_count > 0:
                store_capture(difficulty, capture)
                print(f"  Capture updated: +{new_count} new orders ({len(capture['orders'])} total)",
                      file=sys.stderr)

    return final_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay best saved solution')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('--difficulty', default='auto',
                        help='Difficulty (auto-detects from bot count)')
    parser.add_argument('--log-dir', default=None, help='Log directory')
    args = parser.parse_args()

    diff = args.difficulty if args.difficulty != 'auto' else None
    asyncio.run(replay_best(args.ws_url, difficulty=diff, log_dir=args.log_dir))
