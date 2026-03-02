"""Trip-based solver mirroring the Zig bot architecture.

Uses mini-TSP trip planning, persistent bot state, and centralized orchestration.
Full order foresight: all orders are pre-generated from the seed.
"""
import numpy as np
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


# ── Persistent Bot State ──────────────────────────────────────────────

class BotTrip:
    """A planned sequence of item pickups."""
    __slots__ = ['items', 'adjs', 'count', 'pos', 'total_cost',
                 'active_count', 'preview_count', 'completes_order']

    def __init__(self):
        self.items = []      # list of item_idx
        self.adjs = []       # list of (x, y) walkable cells adjacent to items
        self.count = 0
        self.pos = 0         # current position in trip (0..count-1)
        self.total_cost = 0
        self.active_count = 0
        self.preview_count = 0
        self.completes_order = False


class PersistentBot:
    __slots__ = ['trip', 'delivering', 'stall_count', 'last_pos',
                 'pos_hist', 'escape_rounds', 'last_active_order_id']

    def __init__(self):
        self.trip = None
        self.delivering = False
        self.stall_count = 0
        self.last_pos = (-1, -1)
        self.pos_hist = []
        self.escape_rounds = 0
        self.last_active_order_id = -1


# ── Trip Planning (Mini-TSP) ─────────────────────────────────────────

def plan_best_trip(state, bot_id, dist_maps, all_orders, claimed,
                   active_needs_left, preview_needs_left, allow_preview, rounds_left):
    """Plan the best 1/2/3-item trip for a bot.

    Returns TripPlan or None.
    """
    ms = state.map_state
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    bot_pos = (bx, by)
    slots_free = INV_CAP - state.bot_inv_count(bot_id)
    if slots_free <= 0:
        return None

    num_bots = len(state.bot_positions)
    drop_pos = ms.drop_off

    # Collect candidates: items matching active/preview needs
    candidates = []  # (item_idx, adj_pos, is_active, dist_from_bot, dist_from_drop)

    for item_idx in range(ms.num_items):
        if claimed.get(item_idx, -1) >= 0 and claimed[item_idx] != bot_id:
            continue

        type_id = int(ms.item_types[item_idx])

        is_active = active_needs_left.get(type_id, 0) > 0
        is_preview = False
        if not is_active and allow_preview:
            is_preview = preview_needs_left.get(type_id, 0) > 0
        if not is_active and not is_preview:
            continue

        # Find best adjacent walkable cell
        result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
        if result is None:
            continue

        adj = (result[0], result[1])
        dist_from_bot = result[2]
        dist_from_drop = int(get_distance(dist_maps, adj, drop_pos))

        candidates.append((item_idx, adj, is_active, dist_from_bot, dist_from_drop))

    if not candidates:
        return None

    # Sort by distance from bot (greedy nearest)
    # For single bot or 8+ bots: use round-trip cost
    use_roundtrip = num_bots <= 1 or num_bots >= 8
    if use_roundtrip:
        candidates.sort(key=lambda c: c[3] + c[4])
    else:
        candidates.sort(key=lambda c: c[3])

    # Limit candidates to top 16
    candidates = candidates[:16]

    active_remaining = sum(v for v in active_needs_left.values() if v > 0)
    best_trip = None
    best_score = 0

    n = len(candidates)

    # Evaluate single-item trips
    for a in range(n):
        item_a, adj_a, is_active_a, dist_a, ddrop_a = candidates[a]
        cost = dist_a + ddrop_a
        if cost + 3 > rounds_left:
            continue
        ac = 1 if is_active_a else 0
        pc = 0 if is_active_a else 1
        if ac == 0 and active_remaining > 0:
            continue
        completes = ac >= active_remaining and active_remaining > 0
        score = trip_score(cost, ac, pc, 1, completes, rounds_left)
        if score > best_score:
            best_score = score
            trip = BotTrip()
            trip.items = [item_a]
            trip.adjs = [adj_a]
            trip.count = 1
            trip.total_cost = cost
            trip.active_count = ac
            trip.preview_count = pc
            trip.completes_order = completes
            best_trip = trip

    if slots_free < 2:
        return best_trip

    # Evaluate 2-item trips
    for a in range(n):
        for b in range(a + 1, n):
            item_a, adj_a, is_active_a, dist_a, ddrop_a = candidates[a]
            item_b, adj_b, is_active_b, dist_b, ddrop_b = candidates[b]

            # Count real types (prevent duplicates)
            ac, pc = count_real_types(
                ms, [item_a, item_b], active_needs_left, preview_needs_left, allow_preview
            )
            if ac + pc < 2:
                continue
            if ac == 0 and active_remaining > 0:
                continue
            completes = ac >= active_remaining and active_remaining > 0

            # Try both orderings: a→b→drop and b→a→drop
            dist_ab = int(get_distance(dist_maps, adj_a, adj_b))
            dist_ba = int(get_distance(dist_maps, adj_b, adj_a))

            cost_ab = dist_a + dist_ab + ddrop_b
            cost_ba = dist_b + dist_ba + ddrop_a

            if cost_ab + 4 <= rounds_left:
                score = trip_score(cost_ab, ac, pc, 2, completes, rounds_left)
                if score > best_score:
                    best_score = score
                    trip = BotTrip()
                    trip.items = [item_a, item_b]
                    trip.adjs = [adj_a, adj_b]
                    trip.count = 2
                    trip.total_cost = cost_ab
                    trip.active_count = ac
                    trip.preview_count = pc
                    trip.completes_order = completes
                    best_trip = trip

            if cost_ba + 4 <= rounds_left:
                score = trip_score(cost_ba, ac, pc, 2, completes, rounds_left)
                if score > best_score:
                    best_score = score
                    trip = BotTrip()
                    trip.items = [item_b, item_a]
                    trip.adjs = [adj_b, adj_a]
                    trip.count = 2
                    trip.total_cost = cost_ba
                    trip.active_count = ac
                    trip.preview_count = pc
                    trip.completes_order = completes
                    best_trip = trip

    if slots_free < 3:
        return best_trip

    # Evaluate 3-item trips (limit to top 12 candidates)
    n3 = min(n, 12)
    perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

    for a in range(n3):
        for b in range(a + 1, n3):
            for c in range(b + 1, n3):
                item_a, adj_a, is_active_a, dist_a, ddrop_a = candidates[a]
                item_b, adj_b, is_active_b, dist_b, ddrop_b = candidates[b]
                item_c, adj_c, is_active_c, dist_c, ddrop_c = candidates[c]

                ac, pc = count_real_types(
                    ms, [item_a, item_b, item_c],
                    active_needs_left, preview_needs_left, allow_preview
                )
                if ac + pc < 3:
                    continue
                if ac == 0 and active_remaining > 0:
                    continue
                completes = ac >= active_remaining and active_remaining > 0

                items = [candidates[a], candidates[b], candidates[c]]
                for perm in perms:
                    p0, p1, p2 = perm
                    i0 = items[p0]
                    i1 = items[p1]
                    i2 = items[p2]

                    cost = (i0[3] +
                            int(get_distance(dist_maps, i0[1], i1[1])) +
                            int(get_distance(dist_maps, i1[1], i2[1])) +
                            i2[4])

                    if cost + 5 > rounds_left:
                        continue

                    score = trip_score(cost, ac, pc, 3, completes, rounds_left)
                    if score > best_score:
                        best_score = score
                        trip = BotTrip()
                        trip.items = [i0[0], i1[0], i2[0]]
                        trip.adjs = [i0[1], i1[1], i2[1]]
                        trip.count = 3
                        trip.total_cost = cost
                        trip.active_count = ac
                        trip.preview_count = pc
                        trip.completes_order = completes
                        best_trip = trip

    return best_trip


def count_real_types(ms, item_indices, active_needs, preview_needs, allow_preview):
    """Count how many items match active/preview needs, respecting per-type counts."""
    check_a = dict(active_needs)
    check_p = dict(preview_needs)
    ac = 0
    pc = 0
    for idx in item_indices:
        type_id = int(ms.item_types[idx])
        if check_a.get(type_id, 0) > 0:
            check_a[type_id] -= 1
            ac += 1
        elif allow_preview and check_p.get(type_id, 0) > 0:
            check_p[type_id] -= 1
            pc += 1
    return ac, pc


def trip_score(cost, ac, pc, count, completes_order, rounds_left):
    """Score a trip. Higher is better. Mirrors Zig tripScore."""
    if cost == 0:
        return 10**9

    preview_val = 18 if completes_order else 3
    value = ac * 20 + pc * preview_val
    if completes_order:
        value += 80
    if completes_order and rounds_left < 60:
        value += 20
    value += count * 2
    if completes_order and pc > 0:
        value += pc * 150
    if rounds_left < 60 and cost * 2 > rounds_left:
        value = value // 2

    return int(value) * 10000 // max(int(cost), 1)


# ── Orchestrator ──────────────────────────────────────────────────────

def compute_needs(state, all_orders):
    """Compute active and preview needs after subtracting all bot inventories."""
    active = state.get_active_order()
    preview = state.get_preview_order()

    active_needs = {}
    if active:
        for tid in active.needs():
            active_needs[tid] = active_needs.get(tid, 0) + 1

    preview_needs = {}
    if preview:
        for tid in preview.needs():
            preview_needs[tid] = preview_needs.get(tid, 0) + 1

    # Subtract items already in bots' inventories
    num_bots = len(state.bot_positions)
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in active_needs and active_needs[t] > 0:
                active_needs[t] -= 1
            elif t in preview_needs and preview_needs[t] > 0:
                preview_needs[t] -= 1

    return active_needs, preview_needs


def orchestrate(state, dist_maps, all_orders, pbots, claimed):
    """Assign items to bots. Updates claimed dict and bot trips."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    preview = state.get_preview_order()

    if not active:
        return

    active_needs, preview_needs = compute_needs(state, all_orders)
    active_remaining = sum(v for v in active_needs.values() if v > 0)
    rounds_left = MAX_ROUNDS - state.round

    # Determine max_pickers
    if num_bots <= 1:
        max_pickers = 1
    elif num_bots <= 4:
        max_pickers = num_bots
    elif num_bots <= 7:
        max_pickers = 3
    else:
        max_pickers = 4

    # Phase 1: Identify bots that should deliver (have active items in inventory)
    delivering = set()
    for bid in range(num_bots):
        inv = state.bot_inv_list(bid)
        if inv and active and any(active.needs_type(t) for t in inv):
            pbots[bid].delivering = True
            delivering.add(bid)

    # Phase 2: Assign pickup trips to available bots
    pickers = 0
    picker_bids = []

    # Sort bots by distance to nearest needed item (closest first)
    bot_dists = []
    for bid in range(num_bots):
        if bid in delivering:
            continue
        if pbots[bid].escape_rounds > 0:
            continue
        if state.bot_inv_count(bid) >= INV_CAP:
            continue
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        # Rough distance estimate: distance to nearest active item
        min_dist = 9999
        for type_id, cnt in active_needs.items():
            if cnt <= 0:
                continue
            for item_idx in find_items_of_type(ms, type_id):
                result = get_nearest_item_cell(dist_maps, (bx, by), item_idx, ms)
                if result:
                    min_dist = min(min_dist, result[2])
        bot_dists.append((min_dist, bid))

    bot_dists.sort()

    for _, bid in bot_dists:
        if pickers >= max_pickers:
            break
        if active_remaining <= 0:
            break

        # Check if bot already has a valid trip
        if pbots[bid].trip and pbots[bid].trip.pos < pbots[bid].trip.count:
            # Valid existing trip - check it's still for the right order
            pickers += 1
            picker_bids.append(bid)
            continue

        allow_preview = active_remaining <= 1 and preview is not None
        trip = plan_best_trip(
            state, bid, dist_maps, all_orders, claimed,
            active_needs, preview_needs, allow_preview, rounds_left,
        )
        if trip:
            pbots[bid].trip = trip
            # Claim items
            for item_idx in trip.items:
                claimed[item_idx] = bid
                type_id = int(ms.item_types[item_idx])
                if type_id in active_needs and active_needs[type_id] > 0:
                    active_needs[type_id] -= 1
                    active_remaining -= 1
                elif type_id in preview_needs and preview_needs[type_id] > 0:
                    preview_needs[type_id] -= 1
            pickers += 1
            picker_bids.append(bid)

    # Phase 3: Assign preview trips to remaining idle bots
    if active_remaining <= 0 and preview:
        preview_remaining = sum(v for v in preview_needs.values() if v > 0)
        max_preview = min(2, num_bots // 2) if num_bots > 1 else 1

        preview_assigned = 0
        for bid in range(num_bots):
            if preview_assigned >= max_preview:
                break
            if bid in delivering or bid in picker_bids:
                continue
            if pbots[bid].escape_rounds > 0:
                continue
            if state.bot_inv_count(bid) >= INV_CAP:
                continue
            if preview_remaining <= 0:
                break

            trip = plan_best_trip(
                state, bid, dist_maps, all_orders, claimed,
                {}, preview_needs, True, rounds_left,
            )
            if trip:
                pbots[bid].trip = trip
                for item_idx in trip.items:
                    claimed[item_idx] = bid
                    type_id = int(ms.item_types[item_idx])
                    if type_id in preview_needs and preview_needs[type_id] > 0:
                        preview_needs[type_id] -= 1
                        preview_remaining -= 1
                preview_assigned += 1


# ── Per-Bot Decision ──────────────────────────────────────────────────

def decide_bot_action(state, bot_id, dist_maps, all_orders, pbots, claimed):
    """Decide action for one bot. Returns (action_type, item_idx)."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    bot_pos = (bx, by)
    inv = state.bot_inv_list(bot_id)
    inv_count = len(inv)
    active = state.get_active_order()
    preview = state.get_preview_order()
    pbot = pbots[bot_id]
    rounds_left = MAX_ROUNDS - state.round

    # 1. DROPOFF: at dropoff with active items → deliver
    if bx == ms.drop_off[0] and by == ms.drop_off[1] and inv_count > 0:
        if active and any(active.needs_type(t) for t in inv):
            return (ACT_DROPOFF, -1)

    # 2. ESCAPE: if in escape mode
    if pbot.escape_rounds > 0:
        pbot.escape_rounds -= 1
        return escape_action(state, bot_id, dist_maps)

    # 3. STALL DETECTION: if bot hasn't moved for too long
    if pbot.last_pos == bot_pos:
        pbot.stall_count += 1
    else:
        pbot.stall_count = 0
    pbot.last_pos = bot_pos

    if pbot.stall_count >= 6:
        pbot.stall_count = 0
        pbot.escape_rounds = 4
        return escape_action(state, bot_id, dist_maps)

    # 4. EVACUATE: at dropoff without active items → move away (multi-bot)
    if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
            num_bots > 1 and not (active and any(active.needs_type(t) for t in inv))):
        # Check if any other bot is delivering
        any_delivering = False
        for bid2 in range(num_bots):
            if bid2 == bot_id:
                continue
            inv2 = state.bot_inv_list(bid2)
            if active and any(active.needs_type(t) for t in inv2):
                any_delivering = True
                break
        if any_delivering or inv_count == 0:
            return escape_action(state, bot_id, dist_maps)

    # 5. PICKUP: adjacent to a needed item → pick it up
    if inv_count < INV_CAP:
        # Check if following a trip and next item is adjacent
        if pbot.trip and pbot.trip.pos < pbot.trip.count:
            next_item = pbot.trip.items[pbot.trip.pos]
            ix = int(ms.item_positions[next_item, 0])
            iy = int(ms.item_positions[next_item, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                pbot.trip.pos += 1
                return (ACT_PICKUP, next_item)

        # Also check for any adjacent active item (opportunistic)
        if active:
            for item_idx in range(ms.num_items):
                ix = int(ms.item_positions[item_idx, 0])
                iy = int(ms.item_positions[item_idx, 1])
                if abs(bx - ix) + abs(by - iy) == 1:
                    type_id = int(ms.item_types[item_idx])
                    if active.needs_type(type_id):
                        # Check if this type is still needed
                        return (ACT_PICKUP, item_idx)

    # 6. DELIVER: has active items → navigate to dropoff
    if pbot.delivering and inv_count > 0 and active:
        has_active = any(active.needs_type(t) for t in inv)
        if has_active:
            dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
            if dist_to_drop > 0:
                # Check for delivery detour: pick up nearby items on the way
                if inv_count < INV_CAP:
                    base_detour = 5 if num_bots == 1 else 1
                    active_needs, _ = compute_needs(state, all_orders)
                    for item_idx in range(ms.num_items):
                        type_id = int(ms.item_types[item_idx])
                        if active_needs.get(type_id, 0) <= 0:
                            continue
                        result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
                        if result is None:
                            continue
                        adj = (result[0], result[1])
                        detour = result[2] + int(get_distance(dist_maps, adj, ms.drop_off)) - dist_to_drop
                        if detour <= base_detour:
                            act = get_first_step(dist_maps, bot_pos, adj)
                            if act > 0:
                                return (act, -1)

                act = get_first_step(dist_maps, bot_pos, ms.drop_off)
                if act > 0:
                    return (act, -1)

    # 7. FOLLOW TRIP: navigate to next pickup in planned trip
    if pbot.trip and pbot.trip.pos < pbot.trip.count:
        next_adj = pbot.trip.adjs[pbot.trip.pos]
        dist = int(get_distance(dist_maps, bot_pos, next_adj))
        if dist > 0:
            act = get_first_step(dist_maps, bot_pos, next_adj)
            if act > 0:
                return (act, -1)

    # 8. DELIVER FALLBACK: has items (maybe preview) → go to dropoff
    if inv_count > 0:
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
        # Only deliver if far enough or have enough items
        # For multi-bot with few items far away, don't deliver
        if num_bots >= 5 and dist_to_drop > 8 and inv_count < 2:
            pass  # skip far-with-few
        elif dist_to_drop > 0:
            pbot.delivering = True
            act = get_first_step(dist_maps, bot_pos, ms.drop_off)
            if act > 0:
                return (act, -1)

    # 9. DEAD INVENTORY: has items that don't match any order → camp near dropoff
    if inv_count > 0:
        dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
        if dist_to_drop > 3:
            act = get_first_step(dist_maps, bot_pos, ms.drop_off)
            if act > 0:
                return (act, -1)

    # 10. WAIT
    return (ACT_WAIT, -1)


def escape_action(state, bot_id, dist_maps):
    """Move away from dropoff / other bots, avoiding collisions."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    bx = int(state.bot_positions[bot_id, 0])
    by = int(state.bot_positions[bot_id, 1])
    bot_pos = (bx, by)

    # Build set of other bots' positions
    other_bots = set()
    for b2 in range(num_bots):
        if b2 != bot_id:
            other_bots.add((int(state.bot_positions[b2, 0]),
                            int(state.bot_positions[b2, 1])))

    dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
    best_act = ACT_WAIT
    best_dist = -1

    for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if cell == 0 or cell == 3:  # floor or dropoff
                if (nx, ny) not in other_bots:
                    d2 = int(get_distance(dist_maps, (nx, ny), ms.drop_off))
                    if d2 > best_dist:
                        best_dist = d2
                        best_act = act_id

    if best_act > 0:
        return (best_act, -1)
    return (ACT_WAIT, -1)


# ── Main Solver ───────────────────────────────────────────────────────

def solve(seed, difficulty, verbose=True):
    """Solve a game using trip-based strategy. Returns (score, action_log)."""
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Trip solver: {difficulty} seed={seed} bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  Distance maps computed: {len(dist_maps)} cells")

    # Initialize persistent bot state
    pbots = [PersistentBot() for _ in range(num_bots)]
    action_log = []

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()

        # Check if active order changed → invalidate trips
        active_id = active.id if active else -1
        for bid in range(num_bots):
            if pbots[bid].last_active_order_id != active_id:
                pbots[bid].trip = None
                pbots[bid].delivering = False
                pbots[bid].last_active_order_id = active_id

        # Orchestrate: assign trips to bots
        claimed = {}
        orchestrate(state, dist_maps, all_orders, pbots, claimed)

        # Decide actions
        actions = []
        for bid in range(num_bots):
            act = decide_bot_action(state, bid, dist_maps, all_orders, pbots, claimed)
            actions.append(act)

        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})'
                f'inv={state.bot_inv_list(bid)}'
                f'{"T" if pbots[bid].trip else ""}'
                for bid in range(num_bots)
            )
            print(f'  R{rnd:3d}: score={state.score:3d} orders={state.orders_completed} | {bot_info}')

    if verbose:
        print(f'\nFinal score: {state.score}')

    return state.score, action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    solve(seed, difficulty)
