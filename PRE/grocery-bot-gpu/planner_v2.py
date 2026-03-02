"""Reservation-based MAPF planner for expert-level multi-bot coordination.

Uses time-space A*: each bot plans its path through (x, y, t) space,
avoiding cells reserved by higher-priority bots. Short-horizon reservations
(8 steps) prevent corridor deadlocks while allowing corridor sharing.

Key difference from planner.py: uses reservation table for multi-step
collision avoidance, enabling 5+ bots to work simultaneously on expert maps.
"""
import time
import numpy as np
from collections import deque
from itertools import permutations
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    DX, DY,
)
from pathfinding import precompute_all_distances, get_distance, get_first_step
from action_gen import find_items_of_type
from planner import (
    BotController, ST_IDLE, ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF, ST_PARKED,
    find_best_adj_cell, optimize_trip_order,
    find_parking_spots, compute_remaining_needs,
    move_toward_avoiding, move_away_from,
)


RESERVE_WINDOW = 8  # Only reserve this many steps ahead (re-plan every round)


def time_space_astar(start, goal, ms, reservations, start_time, max_steps=50):
    """A* in time-space: find path avoiding reserved cells at specific timesteps.

    Returns list of action IDs, or empty list if no path found.
    """
    import heapq

    if start == goal:
        return []

    gx, gy = goal
    sx, sy = start

    h0 = abs(sx - gx) + abs(sy - gy)
    heap = [(h0, 0, sx, sy, start_time, [])]
    visited = set()

    while heap:
        f, g, x, y, t, actions = heapq.heappop(heap)
        if g >= max_steps:
            continue

        state_key = (x, y, t)
        if state_key in visited:
            continue
        visited.add(state_key)

        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_WAIT]:
            if act_id == ACT_WAIT:
                nx, ny = x, y
            else:
                dx, dy = DX[act_id], DY[act_id]
                nx, ny = x + dx, y + dy

            if not (0 <= nx < ms.width and 0 <= ny < ms.height):
                continue
            cell = ms.grid[ny, nx]
            if cell != 0 and cell != 3:
                continue

            nt = t + 1
            if (nx, ny, nt) in reservations and (nx, ny) != ms.spawn:
                continue

            if (nx, ny, nt) in visited:
                continue

            new_actions = actions + [act_id]
            if (nx, ny) == goal:
                return new_actions

            ng = g + 1
            nh = abs(nx - gx) + abs(ny - gy)
            nf = ng + nh
            heapq.heappush(heap, (nf, ng, nx, ny, nt, new_actions))

    return []  # No path found


def reserve_path(reservations, start_pos, actions, start_time):
    """Add a bot's planned path to the reservation table.

    Only reserves up to RESERVE_WINDOW steps to avoid over-constraining.
    """
    x, y = start_pos
    reservations[(x, y, start_time)] = True
    steps_to_reserve = min(len(actions), RESERVE_WINDOW)
    for i in range(steps_to_reserve):
        act = actions[i]
        t = start_time + i + 1
        if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
            dx, dy = DX[act], DY[act]
            x, y = x + dx, y + dy
        reservations[(x, y, t)] = True
    # Reserve final position for 2 extra steps (bot will still be near there)
    for dt in range(1, 3):
        reservations[(x, y, start_time + steps_to_reserve + dt)] = True


def plan_bot_move(bot_pos, target, ms, reservations, current_round, dist_maps, occ_after):
    """Plan a move for a bot toward target using reservations with v1 fallback.

    Returns: (action, planned_path)
    """
    if bot_pos == target:
        return ACT_WAIT, []

    # Try time-space A* first
    path_actions = time_space_astar(
        bot_pos, target, ms, reservations, current_round, max_steps=RESERVE_WINDOW + 5
    )

    if path_actions:
        return path_actions[0], path_actions

    # Fallback: use v1's move_toward_avoiding (single-step collision avoidance)
    act, _ = move_toward_avoiding(bot_pos, target, dist_maps, ms, occ_after)
    if act > 0:
        return act, [act]

    return ACT_WAIT, []


def assign_items_v2(state, dist_maps, all_orders, controllers, max_active_bots=None):
    """Assign items to idle bots, ensuring unique target cells.

    Like assign_items_globally but tracks used_adj_cells to prevent
    multiple bots from being sent to the same physical cell.
    """
    ms = state.map_state
    active = state.get_active_order()
    preview = state.get_preview_order()
    num_bots = len(state.bot_positions)

    if not active:
        return {}

    active_needs = compute_remaining_needs(state, controllers, active, ms)

    remaining = []
    for tid, count in active_needs.items():
        for _ in range(count):
            remaining.append(tid)

    if not remaining:
        if preview:
            preview_needs = compute_remaining_needs(state, controllers, preview, ms)
            remaining = []
            for tid, count in preview_needs.items():
                for _ in range(count):
                    remaining.append(tid)

    if not remaining:
        return {}

    if max_active_bots is None:
        max_active_bots = num_bots

    active_count = sum(1 for bc in controllers if bc.is_busy())

    available = []
    for bc in controllers:
        if bc.state in (ST_IDLE, ST_PARKED):
            inv_count = state.bot_inv_count(bc.bot_id)
            if inv_count < INV_CAP and active_count < max_active_bots:
                available.append(bc.bot_id)
                active_count += 1

    if not available:
        return {}

    # Collect current target cells from busy bots to avoid conflicts
    used_adj_cells = set()
    for bc in controllers:
        if bc.is_busy() and bc.target:
            used_adj_cells.add(bc.target)

    use_round_trip = num_bots <= 1
    candidates = []
    for ni, type_id in enumerate(remaining):
        for item_idx in find_items_of_type(ms, type_id):
            for bid in available:
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                adj_cell, d_to_item = find_best_adj_cell(dist_maps, (bx, by), item_idx, ms)
                if adj_cell and d_to_item < 9999:
                    if use_round_trip:
                        d_to_drop = int(get_distance(dist_maps, adj_cell, ms.drop_off))
                        cost = d_to_item + d_to_drop
                    else:
                        cost = d_to_item
                    candidates.append((cost, ni, bid, item_idx, adj_cell))

    candidates.sort()

    assignments = {}
    assigned_needs = set()
    bot_slots = {}
    for bid in available:
        inv_count = state.bot_inv_count(bid)
        bot_slots[bid] = INV_CAP - inv_count
        assignments[bid] = []

    for d, ni, bid, item_idx, adj_cell in candidates:
        if ni in assigned_needs:
            continue
        if len(assignments.get(bid, [])) >= bot_slots.get(bid, 0):
            continue
        # KEY FIX: Avoid sending two bots to the same cell
        if adj_cell in used_adj_cells:
            # Try alternative adj cells for this item
            alt_adj = ms.item_adjacencies.get(item_idx, [])
            found_alt = False
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            for ax, ay in alt_adj:
                if (ax, ay) not in used_adj_cells:
                    adj_cell = (ax, ay)
                    found_alt = True
                    break
            if not found_alt:
                continue  # Skip this candidate, all adj cells are taken

        assignments[bid].append((item_idx, adj_cell))
        assigned_needs.add(ni)
        used_adj_cells.add(adj_cell)

        if len(assigned_needs) == len(remaining):
            break

    return {bid: items for bid, items in assignments.items() if items}


def solve_v2(seed=None, difficulty=None, verbose=True, max_active_bots=None, game_factory=None):
    """Solve using reservation-based MAPF planning."""
    t0 = time.time()
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"MAPF-v2 Planner: bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)
    parking_spots = find_parking_spots(ms, dist_maps, num_spots=num_bots * 2)

    if max_active_bots is None:
        max_active_bots = min(num_bots, 5)

    controllers = [BotController(bid) for bid in range(num_bots)]
    action_log = []
    last_orders_completed = 0
    last_active_order_id = -1

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()
        active_id = active.id if active else -1

        # Order change detection
        order_changed = False
        if state.orders_completed > last_orders_completed:
            order_changed = True
            last_orders_completed = state.orders_completed
        if active_id != last_active_order_id:
            order_changed = True
            last_active_order_id = active_id

        if order_changed:
            for bc in controllers:
                if bc.state == ST_MOVING_TO_ITEM:
                    bc.set_idle()

        # Stuck detection
        for bc in controllers:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)
            if bc.last_pos == bot_pos and bc.is_busy():
                bc.stuck_count += 1
            else:
                bc.stuck_count = 0
            bc.last_pos = bot_pos
            if bc.stuck_count > 10:
                bc.set_idle()

        # Assign items to idle bots (v2: unique target cells)
        assignments = assign_items_v2(
            state, dist_maps, all_orders, controllers, max_active_bots
        )
        for bid, items in assignments.items():
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            optimized = optimize_trip_order((bx, by), items, ms.drop_off, dist_maps)
            bc.assign_trip(optimized)

        # Check if idle bots should deliver (limit concurrent deliverers)
        current_deliverers = sum(1 for bc2 in controllers
                                 if bc2.state == ST_MOVING_TO_DROPOFF)
        for bc in controllers:
            if bc.state in (ST_IDLE, ST_PARKED):
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    if current_deliverers < 4:
                        bc.assign_deliver(ms.drop_off)
                        current_deliverers += 1

        # Auto-delivery staging: only send preview bots when active deliverers are close
        if preview:
            active_uncovered = 0
            if active:
                active_needs_left = {}
                for tid in active.needs():
                    active_needs_left[tid] = active_needs_left.get(tid, 0) + 1
                for bid2 in range(num_bots):
                    for t in state.bot_inv_list(bid2):
                        if t in active_needs_left and active_needs_left[t] > 0:
                            active_needs_left[t] -= 1
                for bc2 in controllers:
                    if bc2.state == ST_MOVING_TO_ITEM:
                        for item_idx, _ in bc2.trip_items[bc2.trip_idx:]:
                            tid = int(ms.item_types[item_idx])
                            if tid in active_needs_left and active_needs_left[tid] > 0:
                                active_needs_left[tid] -= 1
                active_uncovered = sum(v for v in active_needs_left.values() if v > 0)

            if active_uncovered == 0:
                # Check if active deliverers are close to dropoff
                max_active_dist = 999
                for bc2 in controllers:
                    if bc2.state == ST_MOVING_TO_DROPOFF:
                        inv2 = state.bot_inv_list(bc2.bot_id)
                        if inv2 and active and any(active.needs_type(t) for t in inv2):
                            b2x = int(state.bot_positions[bc2.bot_id, 0])
                            b2y = int(state.bot_positions[bc2.bot_id, 1])
                            d = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                            max_active_dist = min(max_active_dist, d)

                # Only stage preview bots when active deliverers within 5 cells
                # or no active deliverers exist (order about to complete)
                should_stage = max_active_dist <= 5 or max_active_dist == 999
                # Also limit concurrent deliverers to avoid corridor congestion
                current_deliverers = sum(1 for bc2 in controllers
                                         if bc2.state == ST_MOVING_TO_DROPOFF)

                if should_stage and current_deliverers < 4:
                    for bc in controllers:
                        if bc.state in (ST_IDLE, ST_PARKED):
                            bid = bc.bot_id
                            inv = state.bot_inv_list(bid)
                            if inv and any(preview.needs_type(t) for t in inv):
                                if current_deliverers < 4:
                                    bc.assign_deliver(ms.drop_off)
                                    current_deliverers += 1

        # Park idle bots
        parked_spots = set()
        for bc in controllers:
            if bc.state == ST_PARKED and bc.park_target:
                parked_spots.add(bc.park_target)
        for bc in controllers:
            if bc.state == ST_IDLE:
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                dist_drop = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                if inv and preview and any(preview.needs_type(t) for t in inv):
                    continue
                blocking = False
                for bc2 in controllers:
                    if bc2.state == ST_MOVING_TO_DROPOFF and bc2.bot_id != bid:
                        b2x = int(state.bot_positions[bc2.bot_id, 0])
                        b2y = int(state.bot_positions[bc2.bot_id, 1])
                        d_b2_drop = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                        d_b2_me = int(get_distance(dist_maps, (b2x, b2y), (bx, by)))
                        if d_b2_me + dist_drop <= d_b2_drop + 2:
                            blocking = True
                            break
                if blocking or dist_drop <= 3:
                    for spot in parking_spots:
                        if spot not in parked_spots:
                            bc.assign_park(spot)
                            parked_spots.add(spot)
                            break

        # === RESERVATION-BASED MOVE PLANNING ===
        def bot_priority(bc):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            if bc.state == ST_MOVING_TO_DROPOFF:
                if (bx, by) == ms.drop_off:
                    return (0, 0)
                d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                return (1, d)
            if bc.state == ST_MOVING_TO_ITEM:
                d = int(get_distance(dist_maps, (bx, by), bc.target)) if bc.target else 99
                return (2, d)
            if bc.state == ST_PARKED:
                return (3, 0)
            return (4, 0)

        sorted_bots = sorted(controllers, key=bot_priority)
        reservations = {}
        actions = [(ACT_WAIT, -1)] * num_bots
        committed_positions = {}

        for bc in sorted_bots:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

            # Build occ_after for v1 fallback collision avoidance
            occ_after = set()
            for bid2 in range(num_bots):
                if bid2 in committed_positions:
                    occ_after.add(committed_positions[bid2])
                elif bid2 != bid:
                    occ_after.add((int(state.bot_positions[bid2, 0]),
                                   int(state.bot_positions[bid2, 1])))

            if bc.state == ST_MOVING_TO_DROPOFF:
                if bot_pos == ms.drop_off:
                    inv = state.bot_inv_list(bid)
                    if inv and active and any(active.needs_type(t) for t in inv):
                        actions[bid] = (ACT_DROPOFF, -1)
                        committed_positions[bid] = bot_pos
                        reserve_path(reservations, bot_pos, [], rnd)
                        bc.set_idle()
                        continue
                    elif inv and preview and any(preview.needs_type(t) for t in inv):
                        must_yield = False
                        for bc2 in controllers:
                            if bc2.bot_id == bid:
                                continue
                            if bc2.state == ST_MOVING_TO_DROPOFF:
                                inv2 = state.bot_inv_list(bc2.bot_id)
                                if inv2 and active and any(active.needs_type(t) for t in inv2):
                                    b2x = int(state.bot_positions[bc2.bot_id, 0])
                                    b2y = int(state.bot_positions[bc2.bot_id, 1])
                                    d = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                                    if d <= 2:
                                        must_yield = True
                                        break
                        if must_yield:
                            act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                            actions[bid] = act
                            if act[0] != ACT_WAIT:
                                dx, dy = DX[act[0]], DY[act[0]]
                                np_ = (bx + dx, by + dy)
                                committed_positions[bid] = np_
                                reserve_path(reservations, bot_pos, [act[0]], rnd)
                            else:
                                committed_positions[bid] = bot_pos
                                reserve_path(reservations, bot_pos, [], rnd)
                        else:
                            actions[bid] = (ACT_WAIT, -1)
                            committed_positions[bid] = bot_pos
                            reserve_path(reservations, bot_pos, [], rnd)
                        continue
                    else:
                        bc.set_idle()

                if bc.state == ST_MOVING_TO_DROPOFF:
                    first_act, planned = plan_bot_move(
                        bot_pos, ms.drop_off, ms, reservations, rnd, dist_maps, occ_after
                    )
                    if first_act != ACT_WAIT:
                        actions[bid] = (first_act, -1)
                        dx, dy = DX[first_act], DY[first_act]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        actions[bid] = (ACT_WAIT, -1)
                        committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, planned if planned else [], rnd)
                    continue

            if bc.state == ST_MOVING_TO_ITEM:
                if bot_pos == bc.target and bc.item_to_pick >= 0:
                    actions[bid] = (ACT_PICKUP, bc.item_to_pick)
                    committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, [], rnd)
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        next_item, next_cell = bc.trip_items[bc.trip_idx]
                        bc.target = next_cell
                        bc.item_to_pick = next_item
                    else:
                        # Trip complete. Always go deliver — we have items.
                        # The delivery logic at dropoff will check what to do.
                        bc.assign_deliver(ms.drop_off)
                    continue
                else:
                    first_act, planned = plan_bot_move(
                        bot_pos, bc.target, ms, reservations, rnd, dist_maps, occ_after
                    )
                    if first_act != ACT_WAIT:
                        actions[bid] = (first_act, -1)
                        dx, dy = DX[first_act], DY[first_act]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        actions[bid] = (ACT_WAIT, -1)
                        committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, planned if planned else [], rnd)
                    continue

            if bc.state == ST_PARKED:
                target = bc.park_target
                if target and bot_pos != target:
                    first_act, planned = plan_bot_move(
                        bot_pos, target, ms, reservations, rnd, dist_maps, occ_after
                    )
                    if first_act != ACT_WAIT:
                        actions[bid] = (first_act, -1)
                        dx, dy = DX[first_act], DY[first_act]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        actions[bid] = (ACT_WAIT, -1)
                        committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, planned if planned else [], rnd)
                elif target is None:
                    act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        new_pos = (bx + dx, by + dy)
                        committed_positions[bid] = new_pos
                        d_new = int(get_distance(dist_maps, new_pos, ms.drop_off))
                        if d_new > 4:
                            bc.set_idle()
                    else:
                        committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, [act[0]] if act[0] != ACT_WAIT else [], rnd)
                else:
                    committed_positions[bid] = bot_pos
                    reserve_path(reservations, bot_pos, [], rnd)
                continue

            # IDLE
            committed_positions[bid] = bot_pos
            reserve_path(reservations, bot_pos, [], rnd)

        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 5 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            active_bots = sum(1 for bc in controllers if bc.is_busy())
            print(f'  R{rnd:3d}: score={state.score:3d} ord={state.orders_completed} '
                  f'active={active_bots}/{num_bots}')

    if verbose:
        print(f'\nFinal score: {state.score} ({time.time()-t0:.1f}s)')

    return state.score, action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'expert'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    mab = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    score, actions = solve_v2(seed, difficulty, max_active_bots=mab)
