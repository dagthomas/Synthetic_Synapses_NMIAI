"""Smart per-bot action candidate generation with full order foresight.

Generates 3-5 candidate actions per bot, then assembles joint actions
as combinations across bots.
"""
import numpy as np
from itertools import product
from game_engine import (
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, GameState,
)
from pathfinding import get_distance, get_first_step, get_nearest_item_cell


def get_future_needed_types(all_orders, current_order_idx, lookahead=None):
    """Get set of item type IDs needed across all future orders.

    With full foresight, we know exactly which types will ever be needed.
    """
    needed = set()
    start = current_order_idx
    end = len(all_orders) if lookahead is None else min(len(all_orders), current_order_idx + lookahead)
    for i in range(start, end):
        o = all_orders[i]
        for r in o.required:
            needed.add(int(r))
    return needed


def get_active_needed_types(state):
    """Get dict of {type_id: count_still_needed} for active order."""
    active = state.get_active_order()
    if active is None:
        return {}
    needed = {}
    for tid in active.needs():
        needed[tid] = needed.get(tid, 0) + 1
    return needed


def get_preview_needed_types(state):
    """Get dict of {type_id: count_still_needed} for preview order."""
    preview = state.get_preview_order()
    if preview is None:
        return {}
    needed = {}
    for tid in preview.needs():
        needed[tid] = needed.get(tid, 0) + 1
    return needed


def find_items_of_type(map_state, type_id):
    """Find all item indices of a given type."""
    indices = []
    for i, t in enumerate(map_state.item_types):
        if int(t) == type_id:
            indices.append(i)
    return indices


def generate_bot_candidates(state, bot_id, dist_maps, all_orders, max_candidates=4):
    """Generate candidate actions for one bot.

    Returns list of (action_type, item_idx, priority) tuples.
    Priority is used for ordering candidates (higher = better heuristic).
    """
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    bot_pos = (bx, by)
    inv_count = state.bot_inv_count(bot_id)
    inv_items = state.bot_inv_list(bot_id)

    candidates = []
    active = state.get_active_order()
    preview = state.get_preview_order()
    active_needs = get_active_needed_types(state)
    preview_needs = get_preview_needed_types(state)

    # Get future-needed types for dead inventory prevention
    current_oi = 0
    for i, o in enumerate(state.orders):
        if not o.complete and o.status == 'active':
            current_oi = o.id
            break
    future_types = get_future_needed_types(all_orders, current_oi)

    # 1. DROPOFF: if at dropoff with items matching active order
    if bx == ms.drop_off[0] and by == ms.drop_off[1] and inv_count > 0 and active:
        has_matching = any(active.needs_type(t) for t in inv_items)
        if has_matching:
            candidates.append((ACT_DROPOFF, -1, 10000))

    # 2. PICKUP: if adjacent to a needed item and have space
    # Compute remaining needs after subtracting ALL bots' inventories
    num_bots = len(state.bot_positions)
    active_remaining = dict(active_needs)
    preview_remaining = dict(preview_needs)
    # First subtract all bots' inventories from needs
    for bid2 in range(num_bots):
        for t in state.bot_inv_list(bid2):
            if t in active_remaining and active_remaining[t] > 0:
                active_remaining[t] -= 1
            elif t in preview_remaining and preview_remaining[t] > 0:
                preview_remaining[t] -= 1
    # But ADD BACK this bot's items (so we see what THIS bot still needs to contribute)
    # Actually we want: what items does the team still need, excluding what all bots carry?
    # This bot should pick items that are still needed GLOBALLY (not already covered by any bot).
    # active_remaining already accounts for all bots' inventories globally.
    # So active_remaining tells us: how many more of each type ANY bot still needs to pick up.

    if inv_count < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                type_id = int(ms.item_types[item_idx])
                if active_remaining.get(type_id, 0) > 0:
                    candidates.append((ACT_PICKUP, item_idx, 9000))
                elif preview_remaining.get(type_id, 0) > 0:
                    # Only pick preview if all active items are already in inventory
                    active_still_needed = sum(v for v in active_remaining.values() if v > 0)
                    if active_still_needed == 0:
                        candidates.append((ACT_PICKUP, item_idx, 3000))

    # 3. MOVE TOWARD DROPOFF: if has items to deliver
    if inv_count > 0 and active:
        has_active = any(active.needs_type(t) for t in inv_items)
        if has_active:
            dist_to_drop = get_distance(dist_maps, bot_pos, ms.drop_off)
            if dist_to_drop > 0:
                act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                if act > 0:
                    # Higher priority when inventory is full or close to dropoff
                    prio = 7000 + (INV_CAP - inv_count) * 100 - dist_to_drop
                    if inv_count >= INV_CAP:
                        prio = 8500  # full inventory -> must deliver
                    candidates.append((act, -1, prio))

    # 4. MOVE TOWARD NEAREST NEEDED ITEM: if have space
    if inv_count < INV_CAP:
        # Active items: high priority (only types still needed after inv)
        best_active_dist = 9999
        best_active_act = None
        for type_id, count in active_remaining.items():
            if count <= 0:
                continue
            for item_idx in find_items_of_type(ms, type_id):
                result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                if result and result[2] < best_active_dist:
                    best_active_dist = result[2]
                    cell = (result[0], result[1])
                    act = get_first_step(dist_maps, bot_pos, cell)
                    if act > 0:
                        best_active_act = (act, -1, 6000 - best_active_dist * 10)

        if best_active_act:
            candidates.append(best_active_act)

        # Preview items: lower priority (only if active covered or spare slots)
        active_still_needed = sum(v for v in active_remaining.values() if v > 0)
        if inv_count < INV_CAP - 1 or active_still_needed == 0:
            best_preview_dist = 9999
            best_preview_act = None
            for type_id, count in preview_remaining.items():
                if count <= 0:
                    continue
                for item_idx in find_items_of_type(ms, type_id):
                    result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                    if result and result[2] < best_preview_dist:
                        best_preview_dist = result[2]
                        cell = (result[0], result[1])
                        act = get_first_step(dist_maps, bot_pos, cell)
                        if act > 0:
                            best_preview_act = (act, -1, 2000 - best_preview_dist * 10)

            if best_preview_act:
                candidates.append(best_preview_act)

    # 5. If has non-active items, move toward dropoff (for preview delivery later)
    if inv_count > 0 and active and not any(active.needs_type(t) for t in inv_items):
        dist_to_drop = get_distance(dist_maps, bot_pos, ms.drop_off)
        if dist_to_drop > 0:
            act = get_first_step(dist_maps, bot_pos, ms.drop_off)
            if act > 0:
                candidates.append((act, -1, 1000))

    # 6. BLOCKING AVOIDANCE: if near dropoff without active items, move away
    # Check if any other bot needs to reach the dropoff
    num_bots = len(state.bot_positions)
    dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
    any_other_delivering = False
    for bid2 in range(num_bots):
        if bid2 == bot_id:
            continue
        inv2 = state.bot_inv_list(bid2)
        if active and any(active.needs_type(t) for t in inv2):
            any_other_delivering = True
            break

    has_active_items = active and any(active.needs_type(t) for t in inv_items)

    if not has_active_items and dist_to_drop <= 3 and any_other_delivering:
        # Move away from dropoff to avoid blocking delivery bots
        for dx, dy, act_id in [(0,-1,1),(0,1,2),(-1,0,3),(1,0,4)]:
            nx, ny = bx+dx, by+dy
            if (0 <= nx < ms.width and 0 <= ny < ms.height and
                (ms.grid[ny, nx] == 0 or ms.grid[ny, nx] == 3)):
                d2 = int(get_distance(dist_maps, (nx, ny), ms.drop_off))
                if d2 > dist_to_drop:
                    # High priority: get out of the way
                    candidates.append((act_id, -1, 8000))

    # 7. WAIT is always a candidate
    candidates.append((ACT_WAIT, -1, 0))

    # 8. For single-bot: add ALL valid moves as low-priority fallbacks
    # This ensures the beam search has genuine branching factor
    num_bots = len(state.bot_positions)
    if num_bots == 1:
        from game_engine import CELL_FLOOR, CELL_DROPOFF
        grid = ms.grid
        for act, dx, dy in [(ACT_MOVE_UP, 0, -1), (ACT_MOVE_DOWN, 0, 1),
                             (ACT_MOVE_LEFT, -1, 0), (ACT_MOVE_RIGHT, 1, 0)]:
            nx, ny = bx + dx, by + dy
            if 0 <= nx < ms.width and 0 <= ny < ms.height:
                cell = grid[ny, nx]
                if cell == CELL_FLOOR or cell == CELL_DROPOFF:
                    candidates.append((act, -1, 500))  # low priority fallback

    # Deduplicate by (action_type, item_idx), keep highest priority
    seen = {}
    for act_type, item_idx, prio in candidates:
        key = (act_type, item_idx)
        if key not in seen or prio > seen[key][2]:
            seen[key] = (act_type, item_idx, prio)

    # Sort by priority descending, take top max_candidates
    result = sorted(seen.values(), key=lambda x: -x[2])[:max_candidates]
    return [(a, i) for a, i, p in result]


def generate_joint_actions(state, dist_maps, all_orders, max_per_bot=3, max_joint=500):
    """Generate candidate joint actions (one action per bot).

    Returns list of lists, each inner list has one (action_type, item_idx) per bot.
    """
    num_bots = len(state.bot_positions)

    # Generate per-bot candidates
    per_bot = []
    for bid in range(num_bots):
        cands = generate_bot_candidates(state, bid, dist_maps, all_orders, max_per_bot)
        if not cands:
            cands = [(ACT_WAIT, -1)]
        per_bot.append(cands)

    # For small bot counts, full Cartesian product
    total_combos = 1
    for c in per_bot:
        total_combos *= len(c)

    if total_combos <= max_joint:
        # Full Cartesian product
        return [list(combo) for combo in product(*per_bot)]
    else:
        # Pruned: take top-1 for each bot + permutations of top-2
        # Start with the greedy (top-1 per bot) action
        base = [cands[0] for cands in per_bot]
        results = [base]

        # For each bot, try its alternative actions with all others at top-1
        for bid in range(num_bots):
            for alt_idx in range(1, len(per_bot[bid])):
                variant = list(base)
                variant[bid] = per_bot[bid][alt_idx]
                results.append(variant)

        # Also try pairwise combinations of top-2 for adjacent bots
        if num_bots <= 10:
            for bid1 in range(num_bots):
                for bid2 in range(bid1 + 1, min(bid1 + 3, num_bots)):
                    for a1 in range(min(2, len(per_bot[bid1]))):
                        for a2 in range(min(2, len(per_bot[bid2]))):
                            if a1 == 0 and a2 == 0:
                                continue  # skip base case
                            variant = list(base)
                            variant[bid1] = per_bot[bid1][a1]
                            variant[bid2] = per_bot[bid2][a2]
                            results.append(variant)

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            key = tuple((a, i) for a, i in r)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:max_joint]
