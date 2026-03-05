"""Replay a saved optimized solution over WebSocket.

Loads best.json + capture.json for a difficulty, connects to the game server,
and replays the pre-computed actions with adaptive desync correction.

Two replay modes:
  1. SYNCED: actual positions match expected -> send DP action as-is
  2. DESYNCED: positions diverge -> navigate toward DP plan's goal (pickup/dropoff target)

Usage:
    python replay_solution.py <wss://...token> [--difficulty auto]
"""
from __future__ import annotations

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
                         CELL_FLOOR, CELL_DROPOFF, INV_CAP,
                         MapState, CaptureData)
from configs import detect_difficulty
from live_solver import ws_to_capture

SEND_DELAY = 0.0  # Zero delay — cached responses are pre-built, no computation in hot loop


def predict_full_sim(actions_list: list[list[tuple[int, int]]],
                     capture: CaptureData, map_state: MapState) -> list[list[tuple[int, int]]]:
    """Simulate full game from capture to predict positions per round.

    Returns:
        positions: list of [(bx, by), ...] per round (before actions applied)
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


def extract_goals(actions_list: list[list[tuple[int, int]]], map_state: MapState,
                  expected_positions: list[list[tuple[int, int]]]) -> dict[int, list[tuple]]:
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


def bfs_next_action(start: tuple[int, int], goal: tuple[int, int],
                    walkable: set[tuple[int, int]],
                    occupied: set[tuple[int, int]],
                    map_state: MapState) -> int:
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


def build_walkable(map_state: MapState) -> set[tuple[int, int]]:
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


def goal_following_action(bid: int, bot: dict, data: dict, map_state: MapState,
                          walkable: set[tuple[int, int]], live_bots: list[dict],
                          bot_goals: dict, bot_goal_idx: dict,
                          current_round: int) -> dict:
    """Follow the DP plan's goal sequence using BFS navigation.

    Instead of blind greedy, this follows the same pickup/dropoff sequence
    as the DP plan but navigates from the bot's actual position. Much better
    coordination than greedy since each bot targets different items.
    """
    bx, by = bot['position']
    bpos = (bx, by)
    inv = bot.get('inventory', [])
    drop_off = tuple(map_state.drop_off)
    occupied = {tuple(b['position']) for b in live_bots if b['id'] != bid}

    goals = bot_goals.get(bid, [])
    gidx = bot_goal_idx.get(bid, 0)

    # Skip past goals that are far in the past (bot may have already done them)
    while gidx < len(goals):
        goal_rnd, goal_pos, goal_act, goal_item_idx = goals[gidx]
        # Skip if this goal's round is way past AND bot isn't near it
        if goal_rnd < current_round - 10:
            # Check if we should skip: if it's a pickup and we're far from it
            dist_to_goal = abs(bx - goal_pos[0]) + abs(by - goal_pos[1])
            if dist_to_goal > 3:
                gidx += 1
                continue
        break
    bot_goal_idx[bid] = gidx

    if gidx >= len(goals):
        # No more goals — fall back to greedy
        return greedy_action(bot, data, map_state, walkable, live_bots)

    goal_rnd, goal_pos, goal_act, goal_item_idx = goals[gidx]

    # At dropoff with items -> always drop off first
    if bpos == drop_off and len(inv) > 0:
        # Advance past dropoff goals since we're delivering
        while gidx < len(goals) and goals[gidx][2] == ACT_DROPOFF:
            gidx += 1
        bot_goal_idx[bid] = gidx
        return {'bot': bid, 'action': 'drop_off'}

    # Goal is PICKUP: navigate to the pickup position, then pick up
    if goal_act == ACT_PICKUP and 0 <= goal_item_idx < len(map_state.items):
        item = map_state.items[goal_item_idx]
        ix, iy = item['position']
        mdist = abs(bx - ix) + abs(by - iy)

        if mdist == 1 and len(inv) < INV_CAP:
            # Adjacent to item — pick it up and advance goal
            bot_goal_idx[bid] = gidx + 1
            return {'bot': bid, 'action': 'pick_up', 'item_id': item['id']}

        if len(inv) >= INV_CAP:
            # Full inventory but goal is pickup — need to deliver first
            nav = bfs_next_action(bpos, drop_off, walkable, occupied, map_state)
            if nav == ACT_WAIT and bpos == drop_off:
                return {'bot': bid, 'action': 'drop_off'}
            return {'bot': bid, 'action': _ACT_NAMES[nav]}

        # Navigate toward the pickup adjacency cell
        # Use closest adjacency cell to current position
        adjs = map_state.item_adjacencies.get(goal_item_idx, [])
        if adjs:
            best_adj = min(adjs, key=lambda a: abs(a[0] - bx) + abs(a[1] - by))
            nav = bfs_next_action(bpos, best_adj, walkable, occupied, map_state)
        else:
            nav = bfs_next_action(bpos, goal_pos, walkable, occupied, map_state)
        return {'bot': bid, 'action': _ACT_NAMES[nav]}

    # Goal is DROPOFF: navigate to dropoff
    if goal_act == ACT_DROPOFF:
        if bpos == drop_off:
            if len(inv) > 0:
                bot_goal_idx[bid] = gidx + 1
                return {'bot': bid, 'action': 'drop_off'}
            else:
                # At dropoff with empty inv — skip this goal
                bot_goal_idx[bid] = gidx + 1
                return goal_following_action(bid, bot, data, map_state, walkable,
                                             live_bots, bot_goals, bot_goal_idx,
                                             current_round)

        if len(inv) > 0:
            nav = bfs_next_action(bpos, drop_off, walkable, occupied, map_state)
            return {'bot': bid, 'action': _ACT_NAMES[nav]}
        else:
            # No items to deliver — skip to next goal
            bot_goal_idx[bid] = gidx + 1
            return goal_following_action(bid, bot, data, map_state, walkable,
                                         live_bots, bot_goals, bot_goal_idx,
                                         current_round)

    # Fallback
    return greedy_action(bot, data, map_state, walkable, live_bots)


def find_dp_last_meaningful_round(actions: list) -> int:
    """Find the last round where any bot does a PICKUP or DROPOFF.

    After this round, the DP plan has no more scoring events and bots should
    switch to greedy mode to continue scoring. We look for pickup/dropoff
    specifically (not just any move) because movement after the last scoring
    action is wasted effort — greedy would be more productive.
    """
    last_meaningful = -1
    for rnd in range(len(actions)):
        for act, item_idx in actions[rnd]:
            if act == ACT_PICKUP or act == ACT_DROPOFF:
                last_meaningful = rnd
                break
    return last_meaningful


def build_cached_responses(actions: list, map_state: MapState) -> list[str]:
    """Pre-build all WS response JSON strings from DP actions.

    Returns list of JSON strings, one per round. Sent directly via ws.send()
    with zero computation in the hot loop.
    """
    num_rounds = len(actions)
    num_bots = len(actions[0]) if actions else 0
    cached = []

    for rnd in range(num_rounds):
        ws_actions = []
        for bid in range(num_bots):
            act, item_idx = actions[rnd][bid]
            a = {'bot': bid, 'action': _ACT_NAMES[act]}
            if act == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
                a['item_id'] = map_state.items[item_idx]['id']
            elif act == ACT_PICKUP:
                a['action'] = 'wait'  # invalid item index -> wait
            ws_actions.append(a)
        cached.append(json.dumps({"actions": ws_actions}))

    return cached


def greedy_action(bot: dict, data: dict, map_state: MapState,
                  walkable: set[tuple[int, int]], live_bots: list[dict],
                  claimed_items: dict | None = None) -> dict:
    """Greedy fallback: pick needed items, deliver to dropoff. Used after DP plan exhausted.

    Args:
        claimed_items: Optional dict mapping item_idx -> bot_id. Bots skip items
            claimed by other bots (avoids all bots targeting same item).
            Caller should pass a shared dict across all bots in the same round
            and update it with this bot's target after calling.
    """
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
        # items_delivered is a list of type names already delivered
        delivered_counts = {}
        for t in active_order.get('items_delivered', []):
            delivered_counts[t] = delivered_counts.get(t, 0) + 1
        for item_type in active_order.get('items_required', []):
            remaining = delivered_counts.get(item_type, 0)
            if remaining > 0:
                delivered_counts[item_type] = remaining - 1
            else:
                needed_types.add(item_type)

    # Also consider preview order items for pre-fetching
    preview_types = set()
    for order in data.get('orders', []):
        if order.get('status') == 'preview':
            for item_type in order.get('items_required', []):
                preview_types.add(item_type)

    # Bot inventory already contains type names (e.g. "cream", "eggs")
    inv_types = list(inv)

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

    # 3. Find nearest needed/preview item to pick up (skip items claimed by other bots)
    if len(inv) < INV_CAP:
        target_types = needed_types if needed_types else preview_types
        if target_types:
            best_item = None
            best_item_idx = -1
            best_dist = 9999
            for idx, it in enumerate(map_state.items):
                if it['type'] not in target_types:
                    continue
                # Skip items already claimed by another bot this round
                if claimed_items and idx in claimed_items and claimed_items[idx] != bid:
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
                        best_item_idx = idx

            if best_item:
                item, adj_pos = best_item
                adj_pos = tuple(adj_pos)
                # Claim this item for this bot
                if claimed_items is not None and best_item_idx >= 0:
                    claimed_items[best_item_idx] = bid
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


async def replay_best(ws_url: str, difficulty: str | None = None,
                      log_dir: str | None = None) -> int:
    """Replay saved best solution with adaptive desync correction."""
    import websockets

    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(__file__))

    timestamp = int(time.time())
    log_path = os.path.join(log_dir, f'game_log_{timestamp}.jsonl')
    log_lines = []  # Buffer log in memory, write to disk after game ends

    # ── Pre-compute BEFORE connecting (avoids server timeout at R0) ────
    # For expert (10 bots), predict_full_sim takes ~10s inside the WS loop,
    # causing the server to advance 5+ rounds with WAIT actions.
    actions = None
    map_state = None
    expected_positions = None
    bot_goals = None
    walkable = None
    capture = None
    meta = None
    num_bots = 0

    if difficulty and difficulty != 'auto':
        actions = load_solution(difficulty)
        capture = load_capture(difficulty)
        meta = load_meta(difficulty)

        if actions is None:
            print(f"ERROR: No solution found for {difficulty}", file=sys.stderr)

        if capture:
            map_state = build_map_from_capture(capture)
            walkable = build_walkable(map_state)

            if meta:
                print(f"Replaying {difficulty}: expected_score={meta.get('score', '?')} "
                      f"optimizations={meta.get('optimizations_run', 0)}", file=sys.stderr)

            if actions:
                try:
                    t0 = time.time()
                    expected_positions = predict_full_sim(actions, capture, map_state)
                    bot_goals = extract_goals(actions, map_state, expected_positions)
                    num_bots = len(actions[0]) if actions else 0
                    dp_last_meaningful = find_dp_last_meaningful_round(actions)
                    for bid in range(num_bots):
                        print(f"  Bot {bid}: {len(bot_goals[bid])} goals "
                              f"(pickups+dropoffs)", file=sys.stderr)
                    print(f"  DP plan ends at round {dp_last_meaningful} "
                          f"(greedy after)", file=sys.stderr)
                    print(f"  Pre-compute done in {time.time() - t0:.1f}s", file=sys.stderr)
                except Exception as e:
                    print(f"  Warning: Could not pre-compute goals: {e}", file=sys.stderr)
                    expected_positions = None
                    bot_goals = None

    print(f"Connecting to server...", file=sys.stderr)

    final_score = 0
    bot_goal_idx = None
    desync_count = 0
    desync_rounds = 0
    synced_rounds = 0
    round_offset = 0
    last_round_synced = True
    bot_consecutive_desync = {}
    MAX_DESYNC_BEFORE_GREEDY = 1  # Switch to greedy immediately — wrong DP actions create dead inventory
    seen_order_ids = set()
    all_orders_captured = []
    bot_prev_inv = {}
    bot_prev_score = 0
    greedy_mode_logged = False
    greedy_after_dp_logged = False
    cached_responses = None  # Pre-built JSON strings for zero-delay sending
    dp_last_meaningful = -1  # Last round with non-WAIT DP action; greedy after this (computed later)
    prev_server_rnd = -1  # Track actual server round numbers to detect exact gaps

    WS_CONNECT_TIMEOUT = 15  # seconds to establish connection
    WS_RECV_TIMEOUT = 10     # seconds to wait for each message (server has 2s round timeout)
    WS_SEND_TIMEOUT = 5      # seconds to send a message

    try:
        ws = await asyncio.wait_for(
            websockets.connect(ws_url),
            timeout=WS_CONNECT_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"ERROR: WebSocket connection timed out after {WS_CONNECT_TIMEOUT}s", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"ERROR: WebSocket connection failed: {e}", file=sys.stderr)
        return 0

    print(f"WebSocket connected", file=sys.stderr)
    ws_start = time.time()
    recv_count = 0
    last_recv = time.time()

    try:
      async with ws:
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
            except asyncio.TimeoutError:
                elapsed_since = time.time() - last_recv
                print(f"WARNING: No message for {elapsed_since:.1f}s (recv_count={recv_count}, "
                      f"score={final_score})", file=sys.stderr)
                if elapsed_since > 30:
                    print(f"ERROR: WebSocket stalled for {elapsed_since:.1f}s, aborting", file=sys.stderr)
                    break
                continue
            except websockets.exceptions.ConnectionClosed as e:
                print(f"WebSocket closed: {e} (recv_count={recv_count})", file=sys.stderr)
                break
            except Exception as e:
                print(f"WebSocket recv error: {e}", file=sys.stderr)
                break

            recv_count += 1
            last_recv = time.time()
            data = json.loads(message)

            log_lines.append(message)

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

                # If not pre-computed (difficulty was auto), do it now
                if actions is None:
                    if difficulty is None or difficulty == 'auto':
                        difficulty = detect_difficulty(num_bots)
                    actions = load_solution(difficulty)
                    capture = load_capture(difficulty)
                    meta = load_meta(difficulty)

                if actions is None:
                    actions = [[(ACT_WAIT, -1)] * num_bots] * 300

                if map_state is None:
                    if capture:
                        map_state = build_map_from_capture(capture)
                    else:
                        cap = ws_to_capture(data)
                        map_state = build_map_from_capture(cap)
                    walkable = build_walkable(map_state)

                # Reconcile item IDs (fast — just a dict lookup per item)
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

                # Build cached responses AFTER item ID reconciliation
                # This is the key optimization: all 300 JSON strings pre-built,
                # zero computation in the hot WS loop
                cached_responses = build_cached_responses(actions, map_state)
                dp_last_meaningful = find_dp_last_meaningful_round(actions)
                print(f"  Cached {len(cached_responses)} responses, "
                      f"DP plan ends at round {dp_last_meaningful}", file=sys.stderr)

                # Compute goals/positions for desync detection (lightweight, used for logging)
                if expected_positions is None and actions and capture:
                    try:
                        expected_positions = predict_full_sim(actions, capture, map_state)
                        bot_goals = extract_goals(actions, map_state, expected_positions)
                    except Exception as e:
                        print(f"  Warning: Could not compute goals: {e}", file=sys.stderr)

                if bot_goals:
                    bot_goal_idx = {bid: 0 for bid in range(num_bots)}

            # Determine effective DP round (accounting for missed rounds)
            live_bots = data.get('bots', [])
            num_bots = len(live_bots)

            # Detect round gaps from server round numbers
            server_round_gap = rnd - prev_server_rnd - 1 if prev_server_rnd >= 0 else 0
            prev_server_rnd = rnd

            # ── Fast path: use cached response (zero computation) ──
            # Detect round offset first
            if expected_positions and actions:
                dp_rnd = rnd - round_offset
                if 0 <= dp_rnd < len(expected_positions):
                    all_match = True
                    for bid, bot in enumerate(live_bots):
                        if bid < len(expected_positions[dp_rnd]):
                            ex, ey = expected_positions[dp_rnd][bid]
                            if tuple(bot['position']) != (ex, ey):
                                all_match = False
                                break

                    # Detect missed rounds: use server round gap if available,
                    # otherwise fall back to position-based detection
                    if not all_match and last_round_synced and server_round_gap > 0:
                        # Server told us exactly how many rounds were missed
                        found_offset = server_round_gap
                        round_offset += found_offset
                        dp_rnd = rnd - round_offset
                        print(f"R{rnd}: Server round gap={server_round_gap}, "
                              f"offset now {round_offset} (dp_rnd={dp_rnd})",
                              file=sys.stderr)

                        # Re-simulate with WAITs inserted at missed rounds
                        if actions and capture and map_state:
                            try:
                                wait_act = [(ACT_WAIT, -1)] * num_bots
                                modified_actions = list(actions[:dp_rnd])
                                for _ in range(round_offset):
                                    modified_actions.append(wait_act)
                                modified_actions.extend(actions[dp_rnd:])
                                modified_actions = modified_actions[:300]
                                expected_positions = predict_full_sim(
                                    modified_actions, capture, map_state)
                                cached_responses = build_cached_responses(
                                    modified_actions, map_state)
                                dp_last_meaningful = find_dp_last_meaningful_round(
                                    modified_actions)
                                actions = modified_actions
                                round_offset = 0
                                dp_rnd = rnd
                                print(f"  Re-simulated with {found_offset} WAIT(s) "
                                      f"inserted. Plan ends at round "
                                      f"{dp_last_meaningful}. Offset reset to 0.",
                                      file=sys.stderr)
                            except Exception as e:
                                print(f"  WARNING: Re-sim failed ({e})",
                                      file=sys.stderr)

                        all_match = True
                        check_dp = rnd - round_offset
                        for bid, bot in enumerate(live_bots):
                            if bid < len(expected_positions[check_dp]):
                                ex, ey = expected_positions[check_dp][bid]
                                if tuple(bot['position']) != (ex, ey):
                                    all_match = False
                                    break

                    # Legacy: position-based offset detection (fallback)
                    elif not all_match and last_round_synced:
                        found_offset = 0
                        for k in range(1, min(6, dp_rnd + 1)):
                            check_rnd = dp_rnd - k
                            if check_rnd < 0:
                                break
                            moved_count = 0
                            for bid2 in range(min(len(expected_positions[dp_rnd]),
                                                  len(expected_positions[check_rnd]))):
                                if expected_positions[dp_rnd][bid2] != expected_positions[check_rnd][bid2]:
                                    moved_count += 1
                            min_moved = max(num_bots // 4, 2) if num_bots >= 5 else 1
                            if moved_count < min_moved:
                                continue
                            prev_match = True
                            for bid, bot in enumerate(live_bots):
                                if bid < len(expected_positions[check_rnd]):
                                    ex, ey = expected_positions[check_rnd][bid]
                                    if tuple(bot['position']) != (ex, ey):
                                        prev_match = False
                                        break
                            if prev_match:
                                found_offset = k
                                break
                        if found_offset > 0:
                            round_offset += found_offset
                            dp_rnd = rnd - round_offset
                            print(f"R{rnd}: Detected {found_offset} missed round(s), "
                                  f"offset now {round_offset} (dp_rnd={dp_rnd})",
                                  file=sys.stderr)

                            # Re-simulate expected positions with WAIT inserted
                            # at the missed round(s). This fixes collision timing.
                            if actions and capture and map_state:
                                try:
                                    wait_act = [(ACT_WAIT, -1)] * num_bots
                                    modified_actions = list(actions[:dp_rnd])
                                    for _ in range(round_offset):
                                        modified_actions.append(wait_act)
                                    modified_actions.extend(actions[dp_rnd:])
                                    # Trim to 300 rounds
                                    modified_actions = modified_actions[:300]
                                    expected_positions = predict_full_sim(
                                        modified_actions, capture, map_state)
                                    # Rebuild cached responses with modified actions
                                    cached_responses = build_cached_responses(
                                        modified_actions, map_state)
                                    dp_last_meaningful = find_dp_last_meaningful_round(
                                        modified_actions)
                                    # Update actions to the modified plan
                                    actions = modified_actions
                                    # Now dp_rnd maps to the ACTUAL round (no more offset)
                                    round_offset = 0
                                    dp_rnd = rnd
                                    print(f"  Re-simulated with {found_offset} WAIT(s) "
                                          f"inserted. New plan ends at round "
                                          f"{dp_last_meaningful}. Offset reset to 0.",
                                          file=sys.stderr)
                                except Exception as e:
                                    print(f"  WARNING: Re-sim failed ({e}), "
                                          f"falling back to offset mode",
                                          file=sys.stderr)

                            all_match = True
                            check_dp = rnd - round_offset
                            for bid, bot in enumerate(live_bots):
                                if bid < len(expected_positions[check_dp]):
                                    ex, ey = expected_positions[check_dp][bid]
                                    if tuple(bot['position']) != (ex, ey):
                                        all_match = False
                                        break

                    # Update dp_rnd after potential resim (offset may have changed)
                    dp_rnd = rnd - round_offset

                    if all_match:
                        synced_rounds += 1
                        last_round_synced = True
                        for bid in range(num_bots):
                            bot_consecutive_desync[bid] = 0
                    else:
                        desync_rounds += 1
                        last_round_synced = False
                        if desync_rounds <= 3:
                            # Log first few desyncs with position details
                            for bid, bot in enumerate(live_bots):
                                if bid < len(expected_positions[dp_rnd]):
                                    ex, ey = expected_positions[dp_rnd][bid]
                                    ax, ay = bot['position']
                                    if (ax, ay) != (ex, ey):
                                        print(f"  R{rnd} bot{bid}: actual=({ax},{ay}) "
                                              f"expected=({ex},{ey}) dp_rnd={dp_rnd}",
                                              file=sys.stderr)
                        for bid, bot in enumerate(live_bots):
                            if bid < len(expected_positions[dp_rnd]):
                                ex, ey = expected_positions[dp_rnd][bid]
                                if tuple(bot['position']) != (ex, ey):
                                    bot_consecutive_desync[bid] = bot_consecutive_desync.get(bid, 0) + 1
                                    desync_count += 1
                                else:
                                    bot_consecutive_desync[bid] = 0
                else:
                    dp_rnd = rnd
                    synced_rounds += 1
                    last_round_synced = True
            else:
                dp_rnd = rnd
                synced_rounds += 1

            # Decide per-bot: DP or greedy
            # A desynced bot getting DP actions picks up wrong items → dead inventory
            dp_plan_exhausted = (dp_rnd > dp_last_meaningful)
            any_bot_desynced = any(
                bot_consecutive_desync.get(bid, 0) > MAX_DESYNC_BEFORE_GREEDY
                for bid in range(num_bots)
            ) if num_bots > 0 else False

            # Pure cached fast-path: all synced AND DP plan still active
            use_pure_cached = (cached_responses
                               and not any_bot_desynced
                               and not dp_plan_exhausted
                               and 0 <= dp_rnd < len(cached_responses))

            if use_pure_cached:
                # FAST PATH: all bots synced, send pre-built response
                response_str = cached_responses[dp_rnd]
            elif not dp_plan_exhausted and cached_responses and 0 <= dp_rnd < len(cached_responses):
                # HYBRID: some bots synced (follow DP), some desynced (goal-following)
                dp_response = json.loads(cached_responses[dp_rnd])
                dp_actions_by_bot = {a['bot']: a for a in dp_response['actions']}
                ws_actions = []
                claimed = {}
                for bid, bot in enumerate(live_bots):
                    if bot_consecutive_desync.get(bid, 0) > MAX_DESYNC_BEFORE_GREEDY:
                        # Desynced bot → goal-following (uses DP goal sequence)
                        if bot_goals and bot_goal_idx is not None:
                            ws_actions.append(goal_following_action(
                                bid, bot, data, map_state, walkable,
                                live_bots, bot_goals, bot_goal_idx, rnd))
                        else:
                            ws_actions.append(greedy_action(bot, data, map_state, walkable,
                                                            live_bots, claimed_items=claimed))
                    elif bid in dp_actions_by_bot:
                        ws_actions.append(dp_actions_by_bot[bid])
                    else:
                        if bot_goals and bot_goal_idx is not None:
                            ws_actions.append(goal_following_action(
                                bid, bot, data, map_state, walkable,
                                live_bots, bot_goals, bot_goal_idx, rnd))
                        else:
                            ws_actions.append(greedy_action(bot, data, map_state, walkable,
                                                            live_bots, claimed_items=claimed))
                response_str = json.dumps({"actions": ws_actions})
                if not greedy_mode_logged and any_bot_desynced:
                    n_greedy = sum(1 for bid in range(num_bots)
                                  if bot_consecutive_desync.get(bid, 0) > MAX_DESYNC_BEFORE_GREEDY)
                    greedy_mode_logged = True
                    print(f"R{rnd}: HYBRID mode — {n_greedy}/{num_bots} bots goal-following "
                          f"(score={score})", file=sys.stderr)
            else:
                # ALL GOAL-FOLLOWING or GREEDY: DP plan exhausted or no cached data
                if not greedy_after_dp_logged and dp_plan_exhausted:
                    greedy_after_dp_logged = True
                    print(f"R{rnd}: DP plan exhausted (last meaningful={dp_last_meaningful}), "
                          f"switching to goal-following/greedy (score={score})", file=sys.stderr)
                elif not greedy_mode_logged and any_bot_desynced:
                    greedy_mode_logged = True
                    print(f"R{rnd}: All bots goal-following (score={score})", file=sys.stderr)

                ws_actions = []
                claimed = {}
                for bid, bot in enumerate(live_bots):
                    if bot_goals and bot_goal_idx is not None:
                        ws_actions.append(goal_following_action(
                            bid, bot, data, map_state, walkable,
                            live_bots, bot_goals, bot_goal_idx, rnd))
                    else:
                        ws_actions.append(greedy_action(bot, data, map_state, walkable,
                                                        live_bots, claimed_items=claimed))
                response_str = json.dumps({"actions": ws_actions})

            # Log status
            round_has_desync = not last_round_synced
            mode = "SYNC" if not round_has_desync else "DESYNC"
            if rnd < 5 or rnd % 25 == 0 or rnd >= 295 or (round_has_desync and desync_rounds <= 10):
                gi_str = ''
                if bot_goal_idx and round_has_desync:
                    gi_str = ' goals=[' + ','.join(str(bot_goal_idx.get(i, 0)) for i in range(num_bots)) + ']'
                print(f"R{rnd}/{max_rounds} Score:{score} [{mode}] dp_rnd={dp_rnd} "
                      f"(synced:{synced_rounds} desynced:{desync_rounds}){gi_str}", file=sys.stderr)

            log_lines.append(response_str)

            # Send immediately — no delay, response is pre-built
            try:
                await asyncio.wait_for(ws.send(response_str), timeout=WS_SEND_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"WARNING: WebSocket send timed out at R{rnd}", file=sys.stderr)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"WebSocket closed during send at R{rnd}: {e}", file=sys.stderr)
                break
            except Exception as e:
                print(f"WebSocket send error at R{rnd}: {e}", file=sys.stderr)
                break

            # Track inventory for goal advancement (lightweight)
            bot_prev_score = score
            for bid, bot in enumerate(live_bots):
                bot_prev_inv[bid] = list(bot.get('inventory', []))

    except Exception as e:
        print(f"WebSocket unexpected error: {e}", file=sys.stderr)

    ws_elapsed = time.time() - ws_start
    print(f"WebSocket session: {ws_elapsed:.1f}s, {recv_count} messages received", file=sys.stderr)

    # Write buffered log to disk after game ends (zero I/O during hot loop)
    with open(log_path, 'w') as log_file:
        log_file.write('\n'.join(log_lines) + '\n')
    print(f"Log saved: {log_path}", file=sys.stderr)
    print(f"Stats: synced_rounds={synced_rounds} desync_rounds={desync_rounds} "
          f"total_desyncs={desync_count}", file=sys.stderr)

    # Auto-import to PostgreSQL
    try:
        import subprocess  # nosec B404
        import_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'grocery-bot-zig', 'replay', 'import_logs.py')
        result = subprocess.run(  # nosec B603 B607
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

            # Persist to order_lists for cross-session reuse
            order_lists_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'order_lists')
            os.makedirs(order_lists_dir, exist_ok=True)
            order_file = os.path.join(order_lists_dir, f'{difficulty}_orders.json')
            order_data = {
                'difficulty': difficulty,
                'date': time.strftime('%Y-%m-%d'),
                'total_orders': total,
                'orders': [{'index': i, 'items_required': o['items_required']}
                           for i, o in enumerate(merged['orders'])],
            }
            # Only update if we have more orders
            try:
                if os.path.exists(order_file):
                    existing = json.loads(open(order_file).read())
                    if len(existing.get('orders', [])) >= total:
                        order_data = None  # Don't overwrite
                if order_data:
                    with open(order_file, 'w') as f:
                        json.dump(order_data, f, indent=2)
                    print(f"  Order list saved: {order_file} ({total} orders)",
                          file=sys.stderr)
            except Exception as e:
                print(f"  Order list save error: {e}", file=sys.stderr)

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
