"""Sequential solver: decide each bot's action in order, each seeing
previous bots' committed moves. Natural collision avoidance.

For Easy (1 bot): equivalent to beam search greedy.
For Multi-bot: each bot decides considering all earlier bots' actions.

Then use iterative optimization to improve the sequence.
"""
import time
import numpy as np
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    DX, DY,
)
from pathfinding import (
    precompute_all_distances, get_distance, get_first_step,
    get_nearest_item_cell,
)
from action_gen import (
    find_items_of_type, get_active_needed_types,
    get_preview_needed_types, get_future_needed_types,
)


def eval_action(state, bot_id, action, dist_maps, all_orders, occupied_after):
    """Score a single bot action. Higher is better.

    occupied_after: set of cells occupied after previous bots have moved.
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    active = state.get_active_order()
    preview = state.get_preview_order()
    inv = state.bot_inv_list(bot_id)
    inv_count = len(inv)

    act_type, item_idx = action
    score = 0

    # Get needs
    active_needs = get_active_needed_types(state)
    # Subtract OTHER bots' inventories
    for bid2 in range(num_bots):
        if bid2 == bot_id:
            continue
        for t in state.bot_inv_list(bid2):
            if t in active_needs and active_needs[t] > 0:
                active_needs[t] -= 1

    active_still_needed = sum(v for v in active_needs.values() if v > 0)

    # Compute new position after action
    if 1 <= act_type <= 4:
        nx, ny = bx + DX[act_type], by + DY[act_type]
        # Check if actually valid move
        if not (0 <= nx < ms.width and 0 <= ny < ms.height):
            return -99999
        cell = ms.grid[ny, nx]
        if cell != 0 and cell != 3:
            return -99999
        # Check collision with committed bots
        if (nx, ny) in occupied_after and (nx, ny) != ms.spawn:
            return -99999
        new_pos = (nx, ny)
    else:
        new_pos = (bx, by)

    dist_to_drop = int(get_distance(dist_maps, new_pos, ms.drop_off))

    # DROPOFF action
    if act_type == ACT_DROPOFF:
        if bx == ms.drop_off[0] and by == ms.drop_off[1]:
            if inv_count > 0 and active:
                matching = sum(1 for t in inv if active.needs_type(t))
                if matching > 0:
                    return 100000 + matching * 10000  # Highest priority
        return -99999

    # PICKUP action
    if act_type == ACT_PICKUP:
        if item_idx >= 0 and inv_count < INV_CAP:
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                type_id = int(ms.item_types[item_idx])
                if active and active_needs.get(type_id, 0) > 0:
                    return 90000  # Active item pickup
                if preview and active_still_needed == 0:
                    preview_needs = get_preview_needed_types(state)
                    for bid2 in range(num_bots):
                        if bid2 == bot_id:
                            continue
                        for t in state.bot_inv_list(bid2):
                            if t in preview_needs and preview_needs[t] > 0:
                                preview_needs[t] -= 1
                    if preview_needs.get(type_id, 0) > 0:
                        return 30000  # Preview item pickup
        return -99999

    # MOVE actions
    if 1 <= act_type <= 4:
        has_active_items = active and any(active.needs_type(t) for t in inv)

        # Moving toward dropoff with active items
        if has_active_items:
            old_dist = int(get_distance(dist_maps, (bx, by), ms.drop_off))
            if dist_to_drop < old_dist:
                # Closer to dropoff with items to deliver
                prio = 70000 + (INV_CAP - inv_count) * 100 - dist_to_drop
                if inv_count >= INV_CAP:
                    prio = 85000 - dist_to_drop  # Full inventory
                return prio

        # Moving toward nearest active item
        if inv_count < INV_CAP and active_still_needed > 0:
            best_item_dist = 9999
            for type_id, count in active_needs.items():
                if count <= 0:
                    continue
                for iidx in find_items_of_type(ms, type_id):
                    result = get_nearest_item_cell(dist_maps, new_pos, iidx, ms)
                    if result and int(result[2]) < best_item_dist:
                        best_item_dist = int(result[2])

            old_best = 9999
            for type_id, count in active_needs.items():
                if count <= 0:
                    continue
                for iidx in find_items_of_type(ms, type_id):
                    result = get_nearest_item_cell(dist_maps, (bx, by), iidx, ms)
                    if result and int(result[2]) < old_best:
                        old_best = int(result[2])

            if best_item_dist < old_best:
                return 60000 - int(best_item_dist) * 10

        # Moving toward dropoff with any items
        if inv_count > 0:
            old_dist = int(get_distance(dist_maps, (bx, by), ms.drop_off))
            if dist_to_drop < old_dist:
                return 20000 - int(dist_to_drop)

        # Moving toward preview item
        if inv_count < INV_CAP and active_still_needed == 0 and preview:
            preview_needs = get_preview_needed_types(state)
            for bid2 in range(num_bots):
                if bid2 == bot_id:
                    continue
                for t in state.bot_inv_list(bid2):
                    if t in preview_needs and preview_needs[t] > 0:
                        preview_needs[t] -= 1

            best_prev_dist = 9999
            for type_id, count in preview_needs.items():
                if count <= 0:
                    continue
                for iidx in find_items_of_type(ms, type_id):
                    result = get_nearest_item_cell(dist_maps, new_pos, iidx, ms)
                    if result and int(result[2]) < best_prev_dist:
                        best_prev_dist = int(result[2])

            old_prev = 9999
            for type_id, count in preview_needs.items():
                if count <= 0:
                    continue
                for iidx in find_items_of_type(ms, type_id):
                    result = get_nearest_item_cell(dist_maps, (bx, by), iidx, ms)
                    if result and int(result[2]) < old_prev:
                        old_prev = int(result[2])

            if best_prev_dist < old_prev:
                return 15000 - int(best_prev_dist) * 10

        # Moving away from dropoff when blocking (no active items, near dropoff)
        if not has_active_items and dist_to_drop > int(get_distance(dist_maps, (bx, by), ms.drop_off)):
            # Check if any other bot is delivering
            any_delivering = False
            for bid2 in range(num_bots):
                if bid2 == bot_id:
                    continue
                inv2 = state.bot_inv_list(bid2)
                if active and any(active.needs_type(t) for t in inv2):
                    any_delivering = True
                    break
            if any_delivering:
                old_dist = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                if old_dist <= 3:
                    return 80000 + dist_to_drop * 100  # High priority: get out of the way

        # Generic move: slight penalty for going away from useful things
        return 1000 - dist_to_drop

    # WAIT
    return 0


def generate_valid_actions(state, bot_id):
    """Generate all valid actions for one bot."""
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    inv_count = state.bot_inv_count(bot_id)
    active = state.get_active_order()

    actions = [(ACT_WAIT, -1)]

    # Moves
    for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = bx + DX[act_id], by + DY[act_id]
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:
                actions.append((act_id, -1))

    # Pickups
    if inv_count < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                actions.append((ACT_PICKUP, item_idx))

    # Dropoff
    if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
            inv_count > 0 and active):
        actions.append((ACT_DROPOFF, -1))

    return actions


def sequential_greedy(state, dist_maps, all_orders):
    """Generate joint action using sequential per-bot greedy selection.

    Each bot picks its best action considering previous bots' committed moves.
    """
    num_bots = len(state.bot_positions)
    ms = state.map_state

    # Start with current occupancy
    occupied = set()
    for bid in range(num_bots):
        occupied.add((int(state.bot_positions[bid, 0]),
                      int(state.bot_positions[bid, 1])))

    joint_actions = []

    # Decide bot processing order: delivering bots first, then by distance to dropoff
    active = state.get_active_order()
    bot_priority = []
    for bid in range(num_bots):
        inv = state.bot_inv_list(bid)
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        has_active = active and any(active.needs_type(t) for t in inv) if inv else False
        at_dropoff = bx == ms.drop_off[0] and by == ms.drop_off[1]
        dist = int(get_distance(dist_maps, (bx, by), ms.drop_off))

        if at_dropoff and has_active:
            prio = 0  # Highest: at dropoff with items
        elif has_active:
            prio = 1  # Second: delivering
        else:
            prio = 2  # Rest

        bot_priority.append((prio, dist, bid))

    bot_priority.sort()
    bot_order = [bid for _, _, bid in bot_priority]

    # Sequential action selection
    actions_by_bot = [None] * num_bots

    for bid in bot_order:
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])

        # Remove this bot from occupied (so it can "move")
        occupied_without_me = occupied - {(bx, by)}

        valid = generate_valid_actions(state, bid)

        # Score each action
        best_action = (ACT_WAIT, -1)
        best_score = -999999

        for action in valid:
            s = eval_action(state, bid, action, dist_maps, all_orders,
                            occupied_without_me)
            if s > best_score:
                best_score = s
                best_action = action

        actions_by_bot[bid] = best_action

        # Update occupied: remove old position, add new
        act_type = best_action[0]
        if 1 <= act_type <= 4:
            nx, ny = bx + DX[act_type], by + DY[act_type]
            occupied.discard((bx, by))
            occupied.add((nx, ny))

    return actions_by_bot


def solve(seed, difficulty, verbose=True):
    """Solve using sequential greedy + iterative optimization."""
    t0 = time.time()
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Sequential solver: {difficulty} seed={seed} "
              f"bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  Distance maps: {len(dist_maps)} cells ({time.time()-t0:.1f}s)")

    action_log = []

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        actions = sequential_greedy(state, dist_maps, all_orders)
        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})'
                f'inv={state.bot_inv_list(bid)}'
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
