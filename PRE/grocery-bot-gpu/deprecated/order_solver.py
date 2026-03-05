"""DEPRECATED: Order-level solver with route planning + iterative optimization.

This module is not part of the active production pipeline. Kept for reference.

Different from Zig (reactive) and beam search (action-level):
1. Pre-generates ALL orders (full foresight)
2. Plans ROUTES (sequences of 1-3 pickups) for each bot per order
3. Uses mini-TSP to find optimal pickup sequences
4. Pipelines: pre-picks next order items while delivering current
5. Iterative optimization: randomly perturbs routes and keeps improvements

The search operates at the ROUTE ASSIGNMENT level, not individual actions.
"""
import time
import random as py_random
import numpy as np
from itertools import permutations
from game_engine import (
    init_game, step, GameState, MapState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
)
from pathfinding import (
    precompute_all_distances, get_distance, get_first_step,
    get_nearest_item_cell,
)
from action_gen import find_items_of_type


# ── Bot State (persistent across rounds) ─────────────────────────────

class BotState:
    __slots__ = ['route', 'route_pos', 'phase', 'stall_count', 'last_pos',
                 'last_order_id']

    def __init__(self):
        self.route = []        # list of (item_idx, adj_pos) for pickup sequence
        self.route_pos = 0     # current position in route
        self.phase = 'idle'    # 'pickup', 'deliver', 'idle', 'escape'
        self.stall_count = 0
        self.last_pos = (-1, -1)
        self.last_order_id = -1


# ── Route Planning ───────────────────────────────────────────────────

def find_best_route(bot_pos, item_list, dist_maps, ms):
    """Find the best-ordered pickup route for a list of items.

    item_list: [(item_idx, type_id), ...]
    Returns: [(item_idx, adj_pos), ...] in optimal order, and total cost.
    """
    if not item_list:
        return [], 9999

    drop_pos = ms.drop_off

    # Get adjacency cells for each item
    items_with_adj = []
    for item_idx, type_id in item_list:
        result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
        if result:
            adj = (result[0], result[1])
            items_with_adj.append((item_idx, adj))

    if not items_with_adj:
        return [], 9999

    n = len(items_with_adj)

    if n == 1:
        item_idx, adj = items_with_adj[0]
        cost = (int(get_distance(dist_maps, bot_pos, adj)) +
                int(get_distance(dist_maps, adj, drop_pos)))
        return [(item_idx, adj)], cost

    # For 2-3 items, try all permutations (2! = 2, 3! = 6)
    best_route = None
    best_cost = 9999

    for perm in permutations(range(n)):
        cost = int(get_distance(dist_maps, bot_pos, items_with_adj[perm[0]][1]))
        for i in range(len(perm) - 1):
            cost += int(get_distance(dist_maps,
                                      items_with_adj[perm[i]][1],
                                      items_with_adj[perm[i+1]][1]))
        cost += int(get_distance(dist_maps, items_with_adj[perm[-1]][1], drop_pos))

        if cost < best_cost:
            best_cost = cost
            best_route = [items_with_adj[perm[i]] for i in perm]

    return best_route or [], best_cost


def plan_order_routes(state, dist_maps, all_orders, bot_states):
    """Plan routes for all bots for the current active order.

    Assigns items to bots optimally, plans pickup routes with mini-TSP.
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    preview = state.get_preview_order()

    if not active:
        return

    # What does the active order still need?
    active_needs = {}
    for tid in active.needs():
        tid = int(tid)
        active_needs[tid] = active_needs.get(tid, 0) + 1

    # Subtract items already in ALL bots' inventories
    inv_coverage = {}  # type_id -> count already carried
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            inv_coverage[t] = inv_coverage.get(t, 0) + 1

    pickup_needs = {}
    for tid, cnt in active_needs.items():
        covered = min(cnt, inv_coverage.get(tid, 0))
        remaining = cnt - covered
        if remaining > 0:
            pickup_needs[tid] = remaining
        if tid in inv_coverage:
            inv_coverage[tid] -= covered

    # Identify delivering bots (have active items, NOT at dropoff)
    delivering = set()
    for bid in range(num_bots):
        inv = state.bot_inv_list(bid)
        if inv and any(active.needs_type(t) for t in inv):
            bot_states[bid].phase = 'deliver'
            delivering.add(bid)

    # Build list of all items to pick up
    items_to_pick = []  # [(type_id, count_needed)]
    for tid, cnt in pickup_needs.items():
        for _ in range(cnt):
            items_to_pick.append(tid)

    if not items_to_pick:
        # All items covered! Assign preview if possible
        _assign_preview_routes(state, dist_maps, all_orders, bot_states, delivering)
        return

    # Find all shelf instances for needed types
    type_to_items = {}  # type_id -> [item_idx, ...]
    for tid in set(items_to_pick):
        type_to_items[tid] = find_items_of_type(ms, tid)

    # Balanced bot-to-items assignment
    # Key: spread items across bots for parallel execution (minimize makespan)
    available_bots = []
    for bid in range(num_bots):
        if bid in delivering:
            continue
        if bot_states[bid].phase == 'escape':
            continue
        inv_count = state.bot_inv_count(bid)
        if inv_count >= INV_CAP:
            continue
        slots = INV_CAP - inv_count
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        available_bots.append((bid, slots, (bx, by)))

    type_assigned = {}
    for tid in set(items_to_pick):
        type_assigned[tid] = 0

    # Score all possible bot-item assignments
    candidates = []  # (cost, bid, item_idx, adj, type_id)
    for bid, slots, bot_pos in available_bots:
        for tid in set(items_to_pick):
            needed = items_to_pick.count(tid)
            if type_assigned.get(tid, 0) >= needed:
                continue
            for item_idx in type_to_items.get(tid, []):
                result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                if result:
                    adj = (result[0], result[1])
                    dist_to = result[2]
                    dist_back = int(get_distance(dist_maps, adj, ms.drop_off))
                    cost = dist_to + dist_back
                    candidates.append((cost, bid, item_idx, adj, tid))

    candidates.sort(key=lambda x: x[0])

    bot_assigned_items = {}  # bid -> [(item_idx, type_id)]
    bot_assigned_count = {}

    # Phase 1: Assign ONE item per bot first (spread workload)
    for cost, bid, item_idx, adj, tid in candidates:
        needed = items_to_pick.count(tid)
        if type_assigned.get(tid, 0) >= needed:
            continue
        if bot_assigned_count.get(bid, 0) >= 1:
            continue  # already has an item in phase 1
        inv_count = state.bot_inv_count(bid)
        if inv_count >= INV_CAP:
            continue

        if bid not in bot_assigned_items:
            bot_assigned_items[bid] = []
        bot_assigned_items[bid].append((item_idx, tid))
        bot_assigned_count[bid] = 1
        type_assigned[tid] = type_assigned.get(tid, 0) + 1

    # Phase 2: Assign remaining items to bots with spare slots
    for cost, bid, item_idx, adj, tid in candidates:
        needed = items_to_pick.count(tid)
        if type_assigned.get(tid, 0) >= needed:
            continue
        slots_used = bot_assigned_count.get(bid, 0)
        inv_count = state.bot_inv_count(bid)
        if inv_count + slots_used >= INV_CAP:
            continue

        if bid not in bot_assigned_items:
            bot_assigned_items[bid] = []
        bot_assigned_items[bid].append((item_idx, tid))
        bot_assigned_count[bid] = slots_used + 1
        type_assigned[tid] = type_assigned.get(tid, 0) + 1

    # Plan routes for assigned bots
    for bid, item_list in bot_assigned_items.items():
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        route, cost = find_best_route((bx, by), item_list, dist_maps, ms)

        bot_states[bid].route = route
        bot_states[bid].route_pos = 0
        bot_states[bid].phase = 'pickup'

    # Assign preview to idle bots
    _assign_preview_routes(state, dist_maps, all_orders, bot_states, delivering)


def _assign_preview_routes(state, dist_maps, all_orders, bot_states, delivering):
    """Assign preview item pickups to idle bots (pipelining)."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    preview = state.get_preview_order()

    if not preview:
        return

    # What does preview need?
    preview_needs = {}
    for tid in preview.needs():
        tid = int(tid)
        preview_needs[tid] = preview_needs.get(tid, 0) + 1

    # Subtract carried items
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in preview_needs and preview_needs[t] > 0:
                preview_needs[t] -= 1

    # Find idle bots
    max_preview_bots = max(1, num_bots // 3)
    preview_assigned = 0

    for bid in range(num_bots):
        if preview_assigned >= max_preview_bots:
            break
        if bid in delivering:
            continue
        if bot_states[bid].phase != 'idle':
            continue
        if state.bot_inv_count(bid) >= INV_CAP:
            continue

        # Find nearest preview item
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)

        items_for_bot = []
        slots = INV_CAP - state.bot_inv_count(bid)

        for tid, cnt in preview_needs.items():
            if cnt <= 0:
                continue
            for item_idx in find_items_of_type(ms, tid):
                result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                if result and len(items_for_bot) < slots:
                    items_for_bot.append((item_idx, tid))
                    preview_needs[tid] -= 1
                    break

        if items_for_bot:
            route, cost = find_best_route(bot_pos, items_for_bot, dist_maps, ms)
            bot_states[bid].route = route
            bot_states[bid].route_pos = 0
            bot_states[bid].phase = 'pickup'
            preview_assigned += 1


# ── Per-Round Action Decision ─────────────────────────────────────────

def decide_actions(state, dist_maps, all_orders, bot_states):
    """Decide actions for all bots based on their current state and routes."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    actions = [(ACT_WAIT, -1)] * num_bots

    # Track target cells for collision avoidance
    target_cells = set()
    for bid in range(num_bots):
        target_cells.add((int(state.bot_positions[bid, 0]),
                          int(state.bot_positions[bid, 1])))

    # Process bots in priority order: deliver > pickup > idle
    phase_priority = {'deliver': 0, 'pickup': 1, 'escape': 2, 'idle': 3}
    sorted_bots = sorted(range(num_bots),
                         key=lambda b: phase_priority.get(bot_states[b].phase, 3))

    for bid in sorted_bots:
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)
        inv = state.bot_inv_list(bid)
        inv_count = len(inv)
        bs = bot_states[bid]

        # Stall detection
        if bot_pos == bs.last_pos:
            bs.stall_count += 1
        else:
            bs.stall_count = 0
        bs.last_pos = bot_pos

        if bs.stall_count >= 6:
            bs.stall_count = 0
            bs.phase = 'escape'

        # === DROPOFF ===
        if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
                inv_count > 0 and active and any(active.needs_type(t) for t in inv)):
            actions[bid] = (ACT_DROPOFF, -1)
            bs.phase = 'idle'
            bs.route = []
            bs.route_pos = 0
            continue

        # === ESCAPE ===
        if bs.phase == 'escape':
            act = _escape(state, bid, dist_maps, target_cells)
            if act:
                actions[bid] = act
                _mark_target(act, bx, by, target_cells)
            bs.phase = 'idle'
            continue

        # === DELIVER ===
        if bs.phase == 'deliver':
            if not inv or not (active and any(active.needs_type(t) for t in inv)):
                bs.phase = 'idle'
            else:
                dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
                if dist_to_drop > 0:
                    act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                    if act and act > 0:
                        dx = [0, 0, 0, -1, 1][act]
                        dy = [0, -1, 1, 0, 0][act]
                        tc = (bx + dx, by + dy)
                        if tc not in target_cells or tc == ms.spawn:
                            actions[bid] = (act, -1)
                            _mark_target(actions[bid], bx, by, target_cells)
                            continue
                        else:
                            # Blocked - try alternate directions
                            alt = _find_unblocked_toward(state, bid, ms.drop_off,
                                                          dist_maps, target_cells)
                            if alt:
                                actions[bid] = alt
                                _mark_target(alt, bx, by, target_cells)
                                continue

        # === PICKUP ROUTE ===
        if bs.phase == 'pickup' and bs.route and bs.route_pos < len(bs.route):
            target_item, target_adj = bs.route[bs.route_pos]

            # Check if adjacent to target item
            ix = int(ms.item_positions[target_item, 0])
            iy = int(ms.item_positions[target_item, 1])
            if abs(bx - ix) + abs(by - iy) == 1 and inv_count < INV_CAP:
                actions[bid] = (ACT_PICKUP, target_item)
                bs.route_pos += 1
                # If route complete, switch to deliver
                if bs.route_pos >= len(bs.route):
                    bs.phase = 'deliver'
                continue

            # Move toward target adjacency cell
            dist = int(get_distance(dist_maps, bot_pos, target_adj))
            if dist > 0:
                act = get_first_step(dist_maps, bot_pos, target_adj)
                if act and act > 0:
                    dx = [0, 0, 0, -1, 1][act]
                    dy = [0, -1, 1, 0, 0][act]
                    tc = (bx + dx, by + dy)
                    if tc not in target_cells or tc == ms.spawn:
                        actions[bid] = (act, -1)
                        _mark_target(actions[bid], bx, by, target_cells)
                        continue
                    else:
                        alt = _find_unblocked_toward(state, bid, target_adj,
                                                      dist_maps, target_cells)
                        if alt:
                            actions[bid] = alt
                            _mark_target(alt, bx, by, target_cells)
                            continue
            else:
                # Already at target adj but not adjacent to item?
                # This shouldn't happen. Move on to deliver.
                bs.route_pos += 1
                if bs.route_pos >= len(bs.route):
                    bs.phase = 'deliver'

        # === DELIVER FALLBACK ===
        if inv_count > 0 and active:
            has_active = any(active.needs_type(t) for t in inv)
            if has_active:
                bs.phase = 'deliver'
                dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
                if dist_to_drop > 0:
                    act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                    if act and act > 0:
                        actions[bid] = (act, -1)
                        _mark_target(actions[bid], bx, by, target_cells)
                        continue

        # === IDLE: move away from dropoff if blocking ===
        if bs.phase == 'idle' and num_bots > 1:
            dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
            if dist_to_drop <= 2:
                act = _escape(state, bid, dist_maps, target_cells)
                if act:
                    actions[bid] = act
                    _mark_target(act, bx, by, target_cells)

    return actions


def _mark_target(action, bx, by, target_cells):
    """Mark the target cell of a move action."""
    act_type = action[0]
    if 1 <= act_type <= 4:
        dx = [0, 0, 0, -1, 1][act_type]
        dy = [0, -1, 1, 0, 0][act_type]
        target_cells.add((bx + dx, by + dy))


def _escape(state, bot_id, dist_maps, target_cells):
    """Move away from dropoff, avoiding target cells."""
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    dist_to_drop = int(get_distance(dist_maps, (bx, by), ms.drop_off))

    best_act = None
    best_dist = -1

    for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:
                if (nx, ny) not in target_cells:
                    d2 = int(get_distance(dist_maps, (nx, ny), ms.drop_off))
                    if d2 > best_dist:
                        best_dist = d2
                        best_act = (act_id, -1)

    return best_act


def _find_unblocked_toward(state, bot_id, target, dist_maps, target_cells):
    """Find an unblocked step toward target."""
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    dist_to_target = int(get_distance(dist_maps, (bx, by), target))

    for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:
                if (nx, ny) not in target_cells:
                    d2 = int(get_distance(dist_maps, (nx, ny), target))
                    if d2 < dist_to_target:
                        return (act_id, -1)
    return None


# ── Main Solver ───────────────────────────────────────────────────────

def solve(seed, difficulty, verbose=True):
    """Solve using order-level route planning."""
    t0 = time.time()
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Order solver: {difficulty} seed={seed} bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  Distance maps: {len(dist_maps)} cells ({time.time()-t0:.1f}s)")

    bot_states = [BotState() for _ in range(num_bots)]
    action_log = []
    last_active_id = -1

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        active_id = active.id if active else -1

        # Replan routes when order changes or any bot is idle
        need_replan = False
        if active_id != last_active_id:
            need_replan = True
            # Reset bots based on new order
            for bid in range(num_bots):
                inv = state.bot_inv_list(bid)
                if active and any(active.needs_type(t) for t in inv):
                    bot_states[bid].phase = 'deliver'
                    bot_states[bid].route = []
                    bot_states[bid].route_pos = 0
                else:
                    bot_states[bid].phase = 'idle'
                    bot_states[bid].route = []
                    bot_states[bid].route_pos = 0
            last_active_id = active_id

        # Replan if any bot is idle and there are items to pick
        any_idle = any(bs.phase == 'idle' for bs in bot_states)
        if any_idle:
            need_replan = True

        if need_replan:
            plan_order_routes(state, dist_maps, all_orders, bot_states)

        # Decide actions
        actions = decide_actions(state, dist_maps, all_orders, bot_states)
        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})'
                f'inv={state.bot_inv_list(bid)}'
                f'[{bot_states[bid].phase[:3]}]'
                for bid in range(num_bots)
            )
            print(f'  R{rnd:3d}: score={state.score:3d} orders={state.orders_completed} | {bot_info}')

    if verbose:
        print(f'\nFinal score: {state.score} ({time.time()-t0:.1f}s)')

    return state.score, action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    solve(seed, difficulty)
