"""DEPRECATED: Smart coordinated policy for multi-bot action selection.

This module is not part of the active production pipeline. Kept for reference.

Instead of generating independent per-bot candidates and combining them,
this assigns tasks to bots centrally (like the Zig orchestrator) and
generates coordinated actions.
"""
import numpy as np
from game_engine import (
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, GameState,
)
from pathfinding import (
    get_distance, get_first_step, get_nearest_item_cell, precompute_all_distances,
)
from action_gen import find_items_of_type


def smart_policy(state, all_orders, dist_maps):
    """Generate a single coordinated action set for all bots.

    Uses a centralized orchestrator approach:
    1. Assign items to bots based on distance
    2. Direct bots toward their assignments
    3. Handle dropoff and pickup
    4. Clear dropoff congestion
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    preview = state.get_preview_order()

    actions = [(ACT_WAIT, -1)] * num_bots

    if not active:
        return actions

    # Compute remaining active needs after delivered items
    active_needs = {}
    for tid in active.needs():
        active_needs[tid] = active_needs.get(tid, 0) + 1

    # Track what's already in all bots' inventories (global view)
    all_inv = {}  # {type_id: count across all bots}
    bot_inv = {}  # {bot_id: [type_ids]}
    for bid in range(num_bots):
        inv = state.bot_inv_list(bid)
        bot_inv[bid] = inv
        for t in inv:
            all_inv[t] = all_inv.get(t, 0) + 1

    # Calculate what items still need to be picked up
    pickup_needs = dict(active_needs)
    for t, c in all_inv.items():
        if t in pickup_needs:
            pickup_needs[t] = max(0, pickup_needs[t] - c)

    # Phase 1: Handle bots at dropoff
    for bid in range(num_bots):
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        inv = bot_inv[bid]

        if bx == ms.drop_off[0] and by == ms.drop_off[1]:
            # At dropoff
            if inv and any(active.needs_type(t) for t in inv):
                actions[bid] = (ACT_DROPOFF, -1)
            # If no matching items, will be handled in Phase 3 (move away)

    # Phase 2: Assign remaining items to bots
    # For each still-needed item type, find the nearest bot that can pick it up
    assigned = set()  # bot_ids already assigned a task
    assignments = {}  # bot_id -> (task_type, target)

    # First: bots with active items should deliver
    delivering_bots = set()
    for bid in range(num_bots):
        if bid in assigned:
            continue
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        inv = bot_inv[bid]
        has_active = any(active.needs_type(t) for t in inv)

        if has_active and not (bx == ms.drop_off[0] and by == ms.drop_off[1]):
            assignments[bid] = ('deliver', ms.drop_off)
            assigned.add(bid)
            delivering_bots.add(bid)

    # Mark bots that are blocking delivery path
    blocking_bots = set()
    for bid in range(num_bots):
        if bid in assigned:
            continue
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))

        for dbid in delivering_bots:
            dbx = int(state.bot_positions[dbid, 0])
            dby = int(state.bot_positions[dbid, 1])
            d_db_drop = int(get_distance(dist_maps, (dbx, dby), ms.drop_off))
            d_db_me = int(get_distance(dist_maps, (dbx, dby), bot_pos))
            if d_db_me + dist_to_drop <= d_db_drop + 2:
                blocking_bots.add(bid)
                # Mark as "escape" assignment
                assignments[bid] = ('escape', ms.drop_off)
                assigned.add(bid)
                break

    # Second: assign pickup tasks for remaining needs
    # Track how many pickups each bot has been assigned (don't over-assign)
    bot_pickup_count = {}  # bid -> number of items assigned to pick
    for bid in range(num_bots):
        bot_pickup_count[bid] = 0

    # Collect all needed (type_id, count) pairs and sort by count desc
    pickup_items = []
    for type_id, count in pickup_needs.items():
        if count <= 0:
            continue
        for _ in range(count):
            pickup_items.append(type_id)

    for type_id in pickup_items:
        item_indices = find_items_of_type(ms, type_id)
        if not item_indices:
            continue

        # Find nearest available bot (allow bots with existing assignments
        # if they have inventory space and are heading in compatible direction)
        best_bid = -1
        best_dist = 9999
        best_item = -1
        best_cell = None

        for bid in range(num_bots):
            # Skip bots that are delivering or escaping
            if bid in assignments and assignments[bid][0] in ('deliver', 'escape'):
                continue
            total_assigned = state.bot_inv_count(bid) + bot_pickup_count[bid]
            if total_assigned >= INV_CAP:
                continue
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])

            for item_idx in item_indices:
                result = get_nearest_item_cell(dist_maps, (bx, by), item_idx, ms)
                if result and result[2] < best_dist:
                    best_dist = result[2]
                    best_bid = bid
                    best_item = item_idx
                    best_cell = (result[0], result[1])

        if best_bid >= 0:
            bx = int(state.bot_positions[best_bid, 0])
            by = int(state.bot_positions[best_bid, 1])
            ix = int(ms.item_positions[best_item, 0])
            iy = int(ms.item_positions[best_item, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                # Can pickup immediately — but only if this is the first or closest assignment
                if best_bid not in assignments or assignments[best_bid][0] != 'pickup':
                    assignments[best_bid] = ('pickup', best_item)
            else:
                # Move toward item — prefer the closest unassigned target
                if best_bid not in assignments:
                    assignments[best_bid] = ('move_to', best_cell)
                elif assignments[best_bid][0] == 'move_to':
                    # Keep the closer target
                    old_cell = assignments[best_bid][1]
                    old_dist = int(get_distance(dist_maps, (bx, by), old_cell))
                    if best_dist < old_dist:
                        assignments[best_bid] = ('move_to', best_cell)
            assigned.add(best_bid)
            bot_pickup_count[best_bid] += 1

    # Third: assign preview pickup to bots with spare capacity
    # Allow bots already assigned active pickups to also grab preview items
    if preview and sum(v for v in pickup_needs.values() if v > 0) == 0:
        preview_needs_left = {}
        for tid in preview.needs():
            preview_needs_left[tid] = preview_needs_left.get(tid, 0) + 1
        # Subtract already carried by any bot
        for bid in range(num_bots):
            for t in bot_inv[bid]:
                if t in preview_needs_left and preview_needs_left[t] > 0:
                    preview_needs_left[t] -= 1

        preview_items = []
        for type_id, count in preview_needs_left.items():
            if count <= 0:
                continue
            for _ in range(count):
                preview_items.append(type_id)

        for type_id in preview_items:
            item_indices = find_items_of_type(ms, type_id)
            if not item_indices:
                continue

            best_bid = -1
            best_dist = 9999
            best_item = -1
            best_cell = None

            for bid in range(num_bots):
                # Skip bots delivering or escaping
                if bid in assignments and assignments[bid][0] in ('deliver', 'escape'):
                    continue
                total_assigned = state.bot_inv_count(bid) + bot_pickup_count[bid]
                if total_assigned >= INV_CAP:
                    continue
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                for item_idx in item_indices:
                    result = get_nearest_item_cell(dist_maps, (bx, by), item_idx, ms)
                    if result and result[2] < best_dist:
                        best_dist = result[2]
                        best_bid = bid
                        best_item = item_idx
                        best_cell = (result[0], result[1])

            if best_bid >= 0:
                bx = int(state.bot_positions[best_bid, 0])
                by = int(state.bot_positions[best_bid, 1])
                ix = int(ms.item_positions[best_item, 0])
                iy = int(ms.item_positions[best_item, 1])
                if abs(bx - ix) + abs(by - iy) == 1:
                    if best_bid not in assignments or assignments[best_bid][0] != 'pickup':
                        assignments[best_bid] = ('pickup', best_item)
                else:
                    if best_bid not in assignments:
                        assignments[best_bid] = ('move_to', best_cell)
                    elif assignments[best_bid][0] == 'move_to':
                        old_cell = assignments[best_bid][1]
                        old_dist = int(get_distance(dist_maps, (bx, by), old_cell))
                        if best_dist < old_dist:
                            assignments[best_bid] = ('move_to', best_cell)
                assigned.add(best_bid)
                bot_pickup_count[best_bid] += 1

    # Phase 3: Convert assignments to actions
    # First pass: determine target positions for all bots
    bot_targets = {}  # bid -> (target_x, target_y)
    for bid in range(num_bots):
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])

        if actions[bid][0] != ACT_WAIT:
            continue

        if bid in assignments:
            task_type, target = assignments[bid]
            if task_type == 'deliver':
                bot_targets[bid] = target
            elif task_type == 'pickup':
                ix = int(ms.item_positions[target, 0])
                iy = int(ms.item_positions[target, 1])
                bot_targets[bid] = (ix, iy)  # item position (adjacent)
            elif task_type == 'move_to':
                bot_targets[bid] = target

    # Second pass: set actions with collision awareness
    occupied = set()
    for bid in range(num_bots):
        pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
        occupied.add(pos)

    for bid in range(num_bots):
        if actions[bid][0] != ACT_WAIT:
            continue

        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        bot_pos = (bx, by)

        if bid in assignments:
            task_type, target = assignments[bid]

            if task_type == 'deliver':
                act = get_first_step(dist_maps, bot_pos, target)
                if act > 0:
                    actions[bid] = (act, -1)

            elif task_type == 'pickup':
                actions[bid] = (ACT_PICKUP, target)

            elif task_type == 'move_to':
                act = get_first_step(dist_maps, bot_pos, target)
                if act > 0:
                    actions[bid] = (act, -1)

            elif task_type == 'escape':
                # Move away from dropoff, avoiding occupied cells
                bot_positions_set = set()
                for b3 in range(num_bots):
                    if b3 != bid:
                        bot_positions_set.add((int(state.bot_positions[b3, 0]),
                                               int(state.bot_positions[b3, 1])))
                best_act = ACT_WAIT
                best_dist = -1
                for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
                    nx, ny = bx + dx, by + dy
                    if (0 <= nx < ms.width and 0 <= ny < ms.height and
                        (ms.grid[ny, nx] == 0 or ms.grid[ny, nx] == 3)):
                        if (nx, ny) not in bot_positions_set:
                            d2 = int(get_distance(dist_maps, (nx, ny), ms.drop_off))
                            if d2 > best_dist:
                                best_dist = d2
                                best_act = act_id
                if best_act > 0:
                    actions[bid] = (best_act, -1)
        else:
            # Unassigned bot: check if any delivering bot needs this path
            # Move away from the path between any delivering bot and dropoff
            dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))

            # Check if this bot is blocking a delivering bot
            blocking_someone = False
            for bid2 in range(num_bots):
                if bid2 == bid or bid2 not in assignments:
                    continue
                if assignments[bid2][0] != 'deliver':
                    continue
                # Check if this bot is on the path from bid2 to dropoff
                b2x = int(state.bot_positions[bid2, 0])
                b2y = int(state.bot_positions[bid2, 1])
                d_b2_drop = int(get_distance(dist_maps, (b2x, b2y), ms.drop_off))
                d_b2_me = int(get_distance(dist_maps, (b2x, b2y), bot_pos))
                d_me_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
                # If I'm roughly on the path (within 2 cells of shortest)
                if d_b2_me + d_me_drop <= d_b2_drop + 2:
                    blocking_someone = True
                    break

            if blocking_someone or dist_to_drop <= 3:
                # Move away from dropoff, avoiding occupied cells
                bot_positions = set()
                for b3 in range(num_bots):
                    if b3 != bid:
                        bot_positions.add((int(state.bot_positions[b3, 0]),
                                          int(state.bot_positions[b3, 1])))

                best_act = ACT_WAIT
                best_dist = -1  # prefer any direction that gets away
                for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
                    nx, ny = bx + dx, by + dy
                    if (0 <= nx < ms.width and 0 <= ny < ms.height and
                        (ms.grid[ny, nx] == 0 or ms.grid[ny, nx] == 3)):
                        if (nx, ny) not in bot_positions:  # avoid collision
                            d2 = int(get_distance(dist_maps, (nx, ny), ms.drop_off))
                            if d2 > best_dist:
                                best_dist = d2
                                best_act = act_id
                if best_act > 0:
                    actions[bid] = (best_act, -1)

    return actions


def smart_policy_with_variants(state, all_orders, dist_maps, num_variants=5):
    """Generate the smart policy + a few variants for beam search.

    Returns list of action lists (one per variant).
    """
    base = smart_policy(state, all_orders, dist_maps)
    results = [base]

    num_bots = len(state.bot_positions)

    # Generate variants by changing one bot's action at a time
    for bid in range(num_bots):
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        ms = state.map_state

        # Try all 4 move directions as alternatives
        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            dx = [0, 0, 0, -1, 1][act_id]
            dy = [0, -1, 1, 0, 0][act_id]
            nx, ny = bx + dx, by + dy
            if (0 <= nx < ms.width and 0 <= ny < ms.height and
                (ms.grid[ny, nx] == 0 or ms.grid[ny, nx] == 3)):
                variant = list(base)
                variant[bid] = (act_id, -1)
                results.append(variant)

        # Try wait
        if base[bid][0] != ACT_WAIT:
            variant = list(base)
            variant[bid] = (ACT_WAIT, -1)
            results.append(variant)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:num_variants * num_bots]


if __name__ == '__main__':
    import sys
    from game_engine import init_game, step, MAX_ROUNDS

    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001

    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    dist_maps = precompute_all_distances(ms)

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        actions = smart_policy(state, all_orders, dist_maps)
        step(state, actions, all_orders)

        if rnd < 20 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1:
            num_bots = len(state.bot_positions)
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})inv={state.bot_inv_list(bid)}'
                for bid in range(num_bots)
            )
            print(f'R{rnd:3d}: score={state.score:3d} orders={state.orders_completed} | {bot_info}')

    print(f'\nFinal score: {state.score}')
