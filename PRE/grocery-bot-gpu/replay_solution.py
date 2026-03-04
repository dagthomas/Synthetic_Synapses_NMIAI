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

from solution_store import load_solution, load_capture, load_meta, merge_capture
from game_engine import (build_map_from_capture, actions_to_ws_format,
                         ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN,
                         ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_PICKUP, ACT_DROPOFF,
                         step, init_game_from_capture, DX, DY,
                         CELL_FLOOR, CELL_DROPOFF, INV_CAP)
from live_solver import detect_difficulty, ws_to_capture

SEND_DELAY = 0.005  # 5ms delay — fast enough to avoid missed rounds, enough to prevent server racing


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


def _find_item_center(map_state, walkable):
    """Find walkable cell closest to the centroid of all items — good pre-position."""
    if not map_state.items:
        return None
    cx = sum(it['position'][0] for it in map_state.items) / len(map_state.items)
    cy = sum(it['position'][1] for it in map_state.items) / len(map_state.items)
    best = None
    best_d = 9999
    for (wx, wy) in walkable:
        d = abs(wx - cx) + abs(wy - cy)
        if d < best_d:
            best_d = d
            best = (wx, wy)
    return best


def _find_nearest_item(bx, by, map_state, walkable, exclude_types=None):
    """Find the nearest item of any type (or excluding certain types)."""
    best_item = None
    best_dist = 9999
    for idx, it in enumerate(map_state.items):
        if exclude_types and it['type'] in exclude_types:
            continue
        adj_cells = map_state.item_adjacencies.get(idx, [])
        if not adj_cells:
            ix, iy = it['position']
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ix + ddx, iy + ddy
                if (nx, ny) in walkable:
                    adj_cells.append((nx, ny))
        for adj in adj_cells:
            d = abs(adj[0] - bx) + abs(adj[1] - by)
            if d < best_dist:
                best_dist = d
                best_item = (it, adj)
    return best_item


_ACT_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']


def greedy_action(bot, data, map_state, walkable, live_bots):
    """Greedy fallback: pick needed items, deliver to dropoff. Used after DP plan exhausted."""
    bid = bot['id']
    bx, by = bot['position']
    bpos = (bx, by)
    inv = bot.get('inventory', [])
    drop_off = tuple(map_state.drop_off)
    occupied = {tuple(b['position']) for b in live_bots if b['id'] != bid}

    # Build set of item types needed by active order
    needed_types = set()
    active_order = None
    for order in data.get('orders', []):
        if order.get('status') == 'active':
            active_order = order
            break
    if active_order:
        delivered = active_order.get('items_delivered', [])
        for i, item_type in enumerate(active_order.get('items_required', [])):
            if i >= len(delivered) or not delivered[i]:
                needed_types.add(item_type)

    # Also consider preview order items for pre-fetching
    preview_types = set()
    for order in data.get('orders', []):
        if order.get('status') == 'preview':
            for item_type in order.get('items_required', []):
                preview_types.add(item_type)

    # Map item types in bot inventory (by item ID -> type lookup)
    inv_types = []
    for item_id in inv:
        for it in map_state.items:
            if it['id'] == item_id:
                inv_types.append(it['type'])
                break

    all_target_types = needed_types | preview_types

    # 1. At dropoff with ANY items -> always drop off (even non-matching — frees inventory,
    #    and items might match a future order via auto-delivery)
    if bpos == drop_off and len(inv) > 0:
        return {'bot': bid, 'action': 'drop_off'}

    # 2. Has items matching active order -> deliver to dropoff
    if len(inv) > 0:
        has_active_match = any(t in needed_types for t in inv_types)
        if has_active_match or len(inv) >= INV_CAP:
            nav = bfs_next_action(bpos, drop_off, walkable, occupied, map_state)
            if nav == ACT_WAIT and bpos == drop_off:
                return {'bot': bid, 'action': 'drop_off'}
            return {'bot': bid, 'action': _ACT_NAMES[nav]}

    # 3. Find nearest needed/preview item to pick up
    if len(inv) < INV_CAP:
        target_types = needed_types if needed_types else preview_types
        if target_types:
            best_item = None
            best_dist = 9999
            for idx, it in enumerate(map_state.items):
                if it['type'] in target_types:
                    adj_cells = map_state.item_adjacencies.get(idx, [])
                    if not adj_cells:
                        ix, iy = it['position']
                        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            nx, ny = ix + ddx, iy + ddy
                            if (nx, ny) in walkable:
                                adj_cells.append((nx, ny))
                    for adj in adj_cells:
                        d = abs(adj[0] - bx) + abs(adj[1] - by)
                        if d < best_dist:
                            best_dist = d
                            best_item = (it, adj)

            if best_item:
                item, adj_pos = best_item
                adj_pos = tuple(adj_pos)
                if bpos == adj_pos:
                    return {'bot': bid, 'action': 'pick_up', 'item_id': item['id']}
                else:
                    nav = bfs_next_action(bpos, adj_pos, walkable, occupied, map_state)
                    return {'bot': bid, 'action': _ACT_NAMES[nav]}

    # 4. Has items but no match -> deliver anyway (frees inv for future orders)
    if len(inv) > 0:
        nav = bfs_next_action(bpos, drop_off, walkable, occupied, map_state)
        if nav == ACT_WAIT and bpos == drop_off:
            return {'bot': bid, 'action': 'drop_off'}
        return {'bot': bid, 'action': _ACT_NAMES[nav]}

    # 5. No orders, no items -> speculatively pick up nearest item (any type).
    #    Having items ready means faster delivery when a new order appears.
    if len(inv) < INV_CAP:
        best = _find_nearest_item(bx, by, map_state, walkable)
        if best:
            item, adj_pos = best
            adj_pos = tuple(adj_pos)
            if bpos == adj_pos:
                return {'bot': bid, 'action': 'pick_up', 'item_id': item['id']}
            else:
                nav = bfs_next_action(bpos, adj_pos, walkable, occupied, map_state)
                return {'bot': bid, 'action': _ACT_NAMES[nav]}

    # 6. Inventory full, no orders -> pre-position near item center
    center = _find_item_center(map_state, walkable)
    if center and bpos != center:
        nav = bfs_next_action(bpos, center, walkable, occupied, map_state)
        return {'bot': bid, 'action': _ACT_NAMES[nav]}

    return {'bot': bid, 'action': 'wait'}


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
    bot_goal_idx = None  # per-bot: which goal index we are working toward
    walkable = None
    desync_count = 0
    desync_rounds = 0  # rounds with at least one desynced bot
    synced_rounds = 0
    round_offset = 0  # Track cumulative round shift (genuine missed rounds only)
    last_round_synced = True  # Only detect missed rounds when transitioning from synced
    bot_consecutive_desync = {}  # bid -> consecutive desynced rounds (for greedy fallback)
    MAX_DESYNC_BEFORE_GREEDY = 8  # Fall back to greedy after this many consecutive desyncs
    seen_order_ids = set()
    all_orders_captured = []
    bot_prev_inv = {}       # bid -> list of item types in inventory last round
    bot_prev_score = 0      # score last round (for delivery detection)
    greedy_mode_logged = False  # log once when switching to greedy

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            log_file.write(json.dumps(data) + '\n')

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

                # Reconcile item IDs: capture may have stale IDs from a different
                # game session. The actual game's items (by position) are authoritative.
                if capture and 'items' in data:
                    actual_items = {}
                    for it in data['items']:
                        pos = tuple(it['position'])
                        actual_items[pos] = it['id']
                    mismatches = 0
                    for i, item in enumerate(map_state.items):
                        pos = tuple(item['position'])
                        actual_id = actual_items.get(pos)
                        if actual_id and actual_id != item['id']:
                            item['id'] = actual_id
                            mismatches += 1
                    if mismatches > 0:
                        print(f"  WARNING: {mismatches} item ID mismatches reconciled "
                              f"(capture vs actual game)", file=sys.stderr)
                    else:
                        print(f"  Item IDs validated: 0 mismatches", file=sys.stderr)

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

                    # Only detect missed rounds when TRANSITIONING from synced to desynced.
                    # When already desynced, the bot's position can coincidentally match
                    # expected[dp_rnd-1], causing false positive offset increments that
                    # cascade and make recovery impossible.
                    if not all_match and last_round_synced:
                        check_rnd = dp_rnd - 1
                        if check_rnd >= 0 and check_rnd < len(expected_positions):
                            expected_moved = False
                            for bid in range(len(expected_positions[dp_rnd])):
                                if bid < len(expected_positions[check_rnd]):
                                    ex_cur = expected_positions[dp_rnd][bid]
                                    ex_prv = expected_positions[check_rnd][bid]
                                    if ex_cur != ex_prv:
                                        expected_moved = True
                                        break
                            if expected_moved:
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
                                    if round_offset <= 10:
                                        print(f"R{rnd}: Detected missed round, offset now {round_offset} "
                                              f"(dp_rnd={dp_rnd})", file=sys.stderr)

                    last_round_synced = all_match
            else:
                dp_rnd = rnd

            # ---- Goal advancement: detect completed goals from last round ----
            if bot_goals and bot_goal_idx is not None:
                for bid, bot in enumerate(live_bots):
                    cur_inv = bot.get('inventory', [])
                    prev_inv = bot_prev_inv.get(bid, cur_inv)  # default to cur on first round
                    gi = bot_goal_idx.get(bid, 0)
                    goals = bot_goals.get(bid, [])

                    while gi < len(goals):
                        goal_rnd, goal_pos, goal_act, goal_item = goals[gi]

                        # Pickup completed: bot's inventory grew
                        if goal_act == ACT_PICKUP and len(cur_inv) > len(prev_inv):
                            gi += 1
                            continue

                        # Dropoff completed: bot's inventory shrank
                        if goal_act == ACT_DROPOFF and len(cur_inv) < len(prev_inv):
                            gi += 1
                            continue

                        # Skip dropoff goals when inventory is empty (nothing to deliver)
                        if goal_act == ACT_DROPOFF and len(cur_inv) == 0:
                            gi += 1
                            continue

                        break

                    bot_goal_idx[bid] = gi

            # ---- Build per-bot actions ----
            ws_actions = []
            round_has_desync = False

            for bid, bot in enumerate(live_bots):
                lx, ly = bot['position']
                bpos = (lx, ly)
                inv = bot.get('inventory', [])

                # Check sync with offset-corrected round
                is_synced = True
                if expected_positions and 0 <= dp_rnd < len(expected_positions) and bid < len(expected_positions[dp_rnd]):
                    ex, ey = expected_positions[dp_rnd][bid]
                    if (lx, ly) != (ex, ey):
                        is_synced = False
                        desync_count += 1
                        round_has_desync = True

                # Track consecutive desync per bot
                if is_synced:
                    bot_consecutive_desync[bid] = 0
                else:
                    bot_consecutive_desync[bid] = bot_consecutive_desync.get(bid, 0) + 1

                if is_synced:
                    # SYNCED: execute raw DP action
                    has_dp = actions and 0 <= dp_rnd < len(actions) and bid < len(actions[dp_rnd])
                    if has_dp:
                        act, item_idx = actions[dp_rnd][bid]
                        # Check if bot's DP plan has remaining goals
                        gi = bot_goal_idx.get(bid, 0) if bot_goal_idx else 0
                        goals = (bot_goals or {}).get(bid, [])
                        plan_exhausted = gi >= len(goals) and act == ACT_WAIT

                        if plan_exhausted:
                            # DP says wait but no more goals -> greedy for new orders
                            if not greedy_mode_logged:
                                greedy_mode_logged = True
                                print(f"R{rnd}: All bots switched to GREEDY mode "
                                      f"(DP plan exhausted, score={score})", file=sys.stderr)
                            ws_actions.append(greedy_action(bot, data, map_state, walkable, live_bots))
                        else:
                            a = {'bot': bot['id'], 'action': ['wait', 'move_up', 'move_down',
                                 'move_left', 'move_right', 'pick_up', 'drop_off'][act]}
                            if act == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
                                a['item_id'] = map_state.items[item_idx]['id']
                            elif act == ACT_PICKUP:
                                a['action'] = 'wait'
                            ws_actions.append(a)
                    else:
                        # DP plan exhausted -> greedy fallback for new orders
                        ws_actions.append(greedy_action(bot, data, map_state, walkable, live_bots))
                elif bot_consecutive_desync.get(bid, 0) > MAX_DESYNC_BEFORE_GREEDY:
                    # PERSISTENT DESYNC: goal-based recovery wastes rounds navigating
                    # toward stale goals. Greedy is more productive.
                    if bot_consecutive_desync[bid] == MAX_DESYNC_BEFORE_GREEDY + 1:
                        print(f"R{rnd}: Bot {bid} switched to GREEDY (desync>{MAX_DESYNC_BEFORE_GREEDY} rounds, "
                              f"score={score})", file=sys.stderr)
                    ws_actions.append(greedy_action(bot, data, map_state, walkable, live_bots))
                else:
                    # SHORT DESYNC: navigate toward current goal via BFS
                    gi = bot_goal_idx.get(bid, 0) if bot_goal_idx else 0
                    goals = (bot_goals or {}).get(bid, [])

                    if gi < len(goals):
                        goal_rnd, goal_pos, goal_act, goal_item = goals[gi]
                        goal_pos_t = tuple(goal_pos)
                        occupied = {tuple(b['position']) for b in live_bots if b['id'] != bot['id']}
                        drop_off = tuple(map_state.drop_off)

                        # Full inventory at pickup goal -> deliver first to clear space
                        if goal_act == ACT_PICKUP and len(inv) >= INV_CAP:
                            if bpos == drop_off:
                                ws_actions.append({'bot': bot['id'], 'action': 'drop_off'})
                            else:
                                nav_act = bfs_next_action(bpos, drop_off, walkable, occupied, map_state)
                                ws_actions.append({'bot': bot['id'],
                                                   'action': _ACT_NAMES[nav_act]})
                        elif bpos == goal_pos_t:
                            # Already at goal position — execute goal action
                            a = {'bot': bot['id'], 'action': ['wait', 'move_up', 'move_down',
                                 'move_left', 'move_right', 'pick_up', 'drop_off'][goal_act]}
                            if goal_act == ACT_PICKUP and 0 <= goal_item < len(map_state.items):
                                a['item_id'] = map_state.items[goal_item]['id']
                            elif goal_act == ACT_PICKUP:
                                a['action'] = 'wait'
                            ws_actions.append(a)
                        else:
                            # Navigate toward goal position
                            nav_act = bfs_next_action(bpos, goal_pos_t, walkable, occupied, map_state)
                            ws_actions.append({'bot': bot['id'],
                                               'action': _ACT_NAMES[nav_act]})
                    else:
                        # No more goals -> greedy fallback for new orders
                        ws_actions.append(greedy_action(bot, data, map_state, walkable, live_bots))

            if round_has_desync:
                desync_rounds += 1
            else:
                synced_rounds += 1

            # Log
            mode = "SYNC" if not round_has_desync else "DESYNC"
            if rnd < 5 or rnd % 25 == 0 or rnd >= 295 or (round_has_desync and desync_rounds <= 10):
                gi_str = ''
                if bot_goal_idx and round_has_desync:
                    gi_str = ' goals=[' + ','.join(str(bot_goal_idx.get(i, 0)) for i in range(num_bots)) + ']'
                print(f"R{rnd}/{max_rounds} Score:{score} [{mode}] dp_rnd={dp_rnd} "
                      f"(synced:{synced_rounds} desynced:{desync_rounds}){gi_str}", file=sys.stderr)

            response = {"actions": ws_actions}
            log_file.write(json.dumps(response) + '\n')
            # Flush periodically instead of every line to avoid blocking asyncio
            if rnd % 25 == 0 or rnd >= max_rounds - 5:
                log_file.flush()

            # Send response with small delay to prevent server from racing ahead
            await asyncio.sleep(SEND_DELAY)
            await ws.send(json.dumps(response))

            # Update state tracking for next round's goal advancement
            bot_prev_score = score
            for bid, bot in enumerate(live_bots):
                bot_prev_inv[bid] = list(bot.get('inventory', []))

    log_file.flush()
    log_file.close()
    print(f"Log saved: {log_path}", file=sys.stderr)
    print(f"Stats: synced_rounds={synced_rounds} desync_rounds={desync_rounds} "
          f"total_desyncs={desync_count}", file=sys.stderr)

    # Auto-import to PostgreSQL
    try:
        import subprocess
        import_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'grocery-bot-zig', 'replay', 'import_logs.py')
        result = subprocess.run(
            [sys.executable, import_script, '--run-type', 'replay', log_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  DB import: {result.stdout.strip()}", file=sys.stderr)
        else:
            print(f"  DB import failed: {result.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"  DB import error: {e}", file=sys.stderr)

    # Update capture with newly seen orders (merge, never lose existing orders)
    if all_orders_captured and difficulty:
        capture = load_capture(difficulty)
        if capture:
            new_capture = dict(capture)
            new_capture['orders'] = [{'items_required': o['items_required']} for o in all_orders_captured]
            merged, num_new, total = merge_capture(difficulty, new_capture)
            if num_new > 0:
                print(f"  Capture updated: +{num_new} new orders ({total} total)",
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
