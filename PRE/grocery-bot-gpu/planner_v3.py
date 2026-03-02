"""Delivery-throttled MAPF planner for expert-level multi-bot coordination.

Key insight: corridor congestion happens at DELIVERY (all bots -> dropoff),
not at PICKING (bots spread across map). This planner allows many bots to
pick simultaneously but limits concurrent deliverers to prevent deadlock.

Changes from planner.py:
- max_active_bots -> controls pickers only (can be high, e.g. 8-10)
- max_deliverers -> limits concurrent delivery bots (2-3)
- After trip completion, bot queues for delivery if too many already delivering
- Better stuck resolution: random jitter after 4 rounds stuck
"""
import time
import random as rand_module
import numpy as np
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
    find_best_adj_cell, optimize_trip_order, assign_items_globally,
    find_parking_spots, compute_remaining_needs,
    move_toward_avoiding, move_away_from, bfs_first_step_avoiding,
)

# New state: waiting to deliver (has items, queued for dropoff)
ST_WAITING_TO_DELIVER = 10


def count_deliverers(controllers):
    """Count bots currently heading to or at dropoff."""
    return sum(1 for bc in controllers if bc.state == ST_MOVING_TO_DROPOFF)


def solve_v3(seed=None, difficulty=None, verbose=True, max_active_bots=None,
             max_deliverers=None, game_factory=None):
    """Solve with delivery throttling."""
    t0 = time.time()
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)
    rng = rand_module.Random(seed if seed else 42)

    if verbose:
        print(f"v3 Planner: bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)
    parking_spots = find_parking_spots(ms, dist_maps, num_spots=num_bots * 2)

    if max_active_bots is None:
        if num_bots <= 5:
            max_active_bots = min(num_bots, 3)
        else:
            max_active_bots = min(num_bots, 8)

    if max_deliverers is None:
        if num_bots <= 3:
            max_deliverers = num_bots
        elif num_bots <= 5:
            max_deliverers = 3
        else:
            max_deliverers = 3

    if verbose:
        print(f"  max_pickers={max_active_bots} max_deliverers={max_deliverers}")

    controllers = [BotController(bid) for bid in range(num_bots)]
    # Track bots waiting to deliver
    waiting_to_deliver = set()  # bot IDs with items, waiting for delivery slot

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
                    waiting_to_deliver.discard(bc.bot_id)

        # Stuck detection with jitter
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
            if bc.stuck_count > 8:
                bc.set_idle()
                waiting_to_deliver.discard(bid)

        # Promote waiting-to-deliver bots if delivery slots available
        current_del = count_deliverers(controllers)
        for bid in sorted(waiting_to_deliver):
            if current_del >= max_deliverers:
                break
            bc = controllers[bid]
            inv = state.bot_inv_list(bid)
            if inv and (active and any(active.needs_type(t) for t in inv)):
                bc.assign_deliver(ms.drop_off)
                waiting_to_deliver.discard(bid)
                current_del += 1
            elif inv and (preview and any(preview.needs_type(t) for t in inv)):
                # Only deliver preview items if active order is covered
                active_needs = compute_remaining_needs(state, controllers, active, ms) if active else {}
                uncovered = sum(v for v in active_needs.values() if v > 0)
                if uncovered == 0:
                    bc.assign_deliver(ms.drop_off)
                    waiting_to_deliver.discard(bid)
                    current_del += 1
            else:
                # Items no longer useful — clear waiting
                waiting_to_deliver.discard(bid)

        # Assign items to idle bots
        assignments = assign_items_globally(
            state, dist_maps, all_orders, controllers, max_active_bots
        )
        for bid, items in assignments.items():
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            optimized = optimize_trip_order((bx, by), items, ms.drop_off, dist_maps)
            bc.assign_trip(optimized)
            waiting_to_deliver.discard(bid)

        # Check if idle bots should deliver
        current_del = count_deliverers(controllers)
        for bc in controllers:
            if bc.state in (ST_IDLE, ST_PARKED) and bc.bot_id not in waiting_to_deliver:
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    if current_del < max_deliverers:
                        bc.assign_deliver(ms.drop_off)
                        current_del += 1
                    else:
                        waiting_to_deliver.add(bid)

        # Auto-delivery staging for preview items
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
                current_del = count_deliverers(controllers)
                for bc in controllers:
                    if bc.state in (ST_IDLE, ST_PARKED) and bc.bot_id not in waiting_to_deliver:
                        bid = bc.bot_id
                        inv = state.bot_inv_list(bid)
                        if inv and any(preview.needs_type(t) for t in inv):
                            if current_del < max_deliverers:
                                bc.assign_deliver(ms.drop_off)
                                current_del += 1
                            else:
                                waiting_to_deliver.add(bid)

        # Park idle bots (not waiting-to-deliver)
        parked_spots = set()
        for bc in controllers:
            if bc.state == ST_PARKED and bc.park_target:
                parked_spots.add(bc.park_target)
        for bc in controllers:
            if bc.state == ST_IDLE and bc.bot_id not in waiting_to_deliver:
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

        # === Generate actions ===
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
                return (2, 0)
            if bc.state == ST_PARKED:
                return (3, 0)
            return (4, 0)

        sorted_bots = sorted(controllers, key=bot_priority)
        occupied = set()
        for bid in range(num_bots):
            pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
            occupied.add(pos)

        actions = [(ACT_WAIT, -1)] * num_bots
        committed_positions = {}

        for bc in sorted_bots:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

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
                        bc.set_idle()
                        waiting_to_deliver.discard(bid)
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
                                committed_positions[bid] = (bx + dx, by + dy)
                            else:
                                committed_positions[bid] = bot_pos
                        else:
                            actions[bid] = (ACT_WAIT, -1)
                            committed_positions[bid] = bot_pos
                        continue
                    else:
                        bc.set_idle()
                        waiting_to_deliver.discard(bid)

                if bc.state == ST_MOVING_TO_DROPOFF:
                    act = move_toward_avoiding(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        committed_positions[bid] = bot_pos
                    continue

            if bc.state == ST_MOVING_TO_ITEM:
                if bot_pos == bc.target and bc.item_to_pick >= 0:
                    actions[bid] = (ACT_PICKUP, bc.item_to_pick)
                    committed_positions[bid] = bot_pos
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        next_item, next_cell = bc.trip_items[bc.trip_idx]
                        bc.target = next_cell
                        bc.item_to_pick = next_item
                    else:
                        # Trip complete — queue for delivery
                        current_del = count_deliverers(controllers)
                        if current_del < max_deliverers:
                            bc.assign_deliver(ms.drop_off)
                        else:
                            bc.set_idle()
                            waiting_to_deliver.add(bid)
                    continue
                else:
                    # Move toward pickup, with stuck jitter
                    if bc.stuck_count >= 4:
                        # Jitter: try random valid move to break deadlock
                        valid_moves = []
                        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                            dx, dy = DX[act_id], DY[act_id]
                            nx, ny = bx + dx, by + dy
                            if 0 <= nx < ms.width and 0 <= ny < ms.height:
                                cell = ms.grid[ny, nx]
                                if (cell == 0 or cell == 3) and ((nx, ny) not in occ_after or (nx, ny) == ms.spawn):
                                    valid_moves.append(act_id)
                        if valid_moves:
                            jitter_act = rng.choice(valid_moves)
                            actions[bid] = (jitter_act, -1)
                            dx, dy = DX[jitter_act], DY[jitter_act]
                            committed_positions[bid] = (bx + dx, by + dy)
                            continue

                    act = move_toward_avoiding(bot_pos, bc.target, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        committed_positions[bid] = bot_pos
                    continue

            if bc.state == ST_PARKED:
                if bc.park_target is None:
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
                elif bot_pos != bc.park_target:
                    act = move_toward_avoiding(bot_pos, bc.park_target, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        committed_positions[bid] = (bx + dx, by + dy)
                    else:
                        committed_positions[bid] = bot_pos
                else:
                    any_heading_here = any(
                        bc2.bot_id != bid and bc2.target == bot_pos
                        for bc2 in controllers
                    )
                    if any_heading_here:
                        act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                        actions[bid] = act
                        if act[0] != ACT_WAIT:
                            dx, dy = DX[act[0]], DY[act[0]]
                            committed_positions[bid] = (bx + dx, by + dy)
                        else:
                            committed_positions[bid] = bot_pos
                    else:
                        committed_positions[bid] = bot_pos
                continue

            # IDLE or waiting to deliver: just wait
            committed_positions[bid] = bot_pos

        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            active_bots = sum(1 for bc in controllers if bc.is_busy())
            waiting = len(waiting_to_deliver)
            print(f'  R{rnd:3d}: score={state.score:3d} ord={state.orders_completed} '
                  f'active={active_bots}/{num_bots} waiting={waiting}')

    if verbose:
        print(f'\nFinal score: {state.score} ({time.time()-t0:.1f}s)')

    return state.score, action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'expert'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    mab = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    md = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    score, actions = solve_v3(seed, difficulty, max_active_bots=mab, max_deliverers=md)
