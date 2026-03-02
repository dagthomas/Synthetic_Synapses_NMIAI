"""Pipeline solver: parallel pickup + sequential delivery.

Key design:
- Only ONE bot delivers at a time (avoids all dropoff congestion)
- Other bots pick items in parallel
- Full order foresight for dead inventory prevention
- Routes use mini-TSP for optimal pickup sequences

This is fundamentally different from Zig (all bots can deliver simultaneously)
and beam search (no coordination). The pipeline approach trades some theoretical
throughput for zero deadlocks.
"""
import time
import numpy as np
from itertools import permutations
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
)
from pathfinding import (
    precompute_all_distances, get_distance, get_first_step,
    get_nearest_item_cell,
)
from action_gen import find_items_of_type, get_future_needed_types


class Bot:
    """Per-bot persistent state."""
    __slots__ = ['route', 'route_pos', 'phase', 'stall_count', 'last_pos']

    def __init__(self):
        self.route = []        # [(item_idx, adj_pos), ...]
        self.route_pos = 0
        self.phase = 'idle'    # 'pickup', 'deliver', 'wait_deliver', 'idle'
        self.stall_count = 0
        self.last_pos = (-1, -1)


def best_route(bot_pos, items, dist_maps, ms):
    """Plan optimal pickup route using mini-TSP.

    items: list of (item_idx, type_id)
    Returns: [(item_idx, adj_pos), ...], total_cost
    """
    if not items:
        return [], 9999

    with_adj = []
    for item_idx, tid in items:
        result = get_nearest_item_cell(dist_maps, bot_pos, item_idx, ms)
        if result:
            with_adj.append((item_idx, (result[0], result[1])))

    if not with_adj:
        return [], 9999

    n = len(with_adj)
    if n == 1:
        idx, adj = with_adj[0]
        cost = (int(get_distance(dist_maps, bot_pos, adj)) +
                int(get_distance(dist_maps, adj, ms.drop_off)))
        return [(idx, adj)], cost

    best = None
    best_cost = 9999
    for perm in permutations(range(n)):
        cost = int(get_distance(dist_maps, bot_pos, with_adj[perm[0]][1]))
        for i in range(len(perm) - 1):
            cost += int(get_distance(dist_maps,
                                      with_adj[perm[i]][1],
                                      with_adj[perm[i+1]][1]))
        cost += int(get_distance(dist_maps, with_adj[perm[-1]][1], ms.drop_off))
        if cost < best_cost:
            best_cost = cost
            best = [(with_adj[perm[i]][0], with_adj[perm[i]][1]) for i in perm]

    return best or [], best_cost


def assign_pickups(state, dist_maps, all_orders, bots, delivering_bid):
    """Assign pickup routes to idle bots.

    Uses full order foresight to never pick dead inventory.
    Balanced assignment: spreads items across bots.
    """
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    preview = state.get_preview_order()

    if not active:
        return

    # Get future needed types (full foresight)
    current_oi = 0
    for o in state.orders:
        if not o.complete and o.status == 'active':
            current_oi = o.id
            break
    future_types = get_future_needed_types(all_orders, current_oi)

    # What does active order still need?
    active_needs = {}
    for tid in active.needs():
        tid = int(tid)
        active_needs[tid] = active_needs.get(tid, 0) + 1

    # Subtract ALL bots' inventories (items already picked)
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in active_needs and active_needs[t] > 0:
                active_needs[t] -= 1

    # Items still needed for pickup
    pickup_list = []
    for tid, cnt in active_needs.items():
        for _ in range(cnt):
            pickup_list.append(tid)

    # Also count items assigned to bots already picking
    for bid in range(num_bots):
        if bots[bid].phase == 'pickup' and bots[bid].route:
            for item_idx, _ in bots[bid].route[bots[bid].route_pos:]:
                tid = int(ms.item_types[item_idx])
                if tid in active_needs:
                    # This item is already being picked
                    if pickup_list.count(tid) > 0:
                        pickup_list.remove(tid)

    # Find idle bots that can pick
    idle_bots = []
    for bid in range(num_bots):
        if bid == delivering_bid:
            continue
        if bots[bid].phase != 'idle':
            continue
        inv_count = state.bot_inv_count(bid)
        if inv_count >= INV_CAP:
            continue
        idle_bots.append(bid)

    if not idle_bots:
        return

    if not pickup_list:
        # Active order fully covered! Assign preview pickups
        _assign_preview(state, dist_maps, all_orders, bots, idle_bots, future_types)
        return

    # Build candidates: (cost, bot_id, item_idx, type_id)
    type_to_items = {}
    for tid in set(pickup_list):
        type_to_items[tid] = find_items_of_type(ms, tid)

    candidates = []
    for bid in idle_bots:
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        for tid in set(pickup_list):
            for item_idx in type_to_items.get(tid, []):
                result = get_nearest_item_cell(dist_maps, (bx, by), item_idx, ms)
                if result:
                    dist_to = result[2]
                    dist_back = int(get_distance(dist_maps,
                                                  (result[0], result[1]), ms.drop_off))
                    candidates.append((dist_to + dist_back, bid, item_idx, tid))

    candidates.sort()

    # Phase 1: one item per bot (spread work)
    type_assigned = {}
    bot_items = {}  # bid -> [(item_idx, type_id)]
    bot_count = {}  # bid -> count

    for cost, bid, item_idx, tid in candidates:
        needed = pickup_list.count(tid)
        if type_assigned.get(tid, 0) >= needed:
            continue
        if bot_count.get(bid, 0) >= 1:
            continue  # phase 1: max 1 per bot
        inv_count = state.bot_inv_count(bid)
        if inv_count >= INV_CAP:
            continue

        bot_items.setdefault(bid, []).append((item_idx, tid))
        bot_count[bid] = bot_count.get(bid, 0) + 1
        type_assigned[tid] = type_assigned.get(tid, 0) + 1

    # Phase 2: fill remaining slots
    for cost, bid, item_idx, tid in candidates:
        needed = pickup_list.count(tid)
        if type_assigned.get(tid, 0) >= needed:
            continue
        inv_count = state.bot_inv_count(bid)
        bc = bot_count.get(bid, 0)
        if inv_count + bc >= INV_CAP:
            continue

        bot_items.setdefault(bid, []).append((item_idx, tid))
        bot_count[bid] = bc + 1
        type_assigned[tid] = type_assigned.get(tid, 0) + 1

    # Plan routes
    for bid, items in bot_items.items():
        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        route, cost = best_route((bx, by), items, dist_maps, ms)
        bots[bid].route = route
        bots[bid].route_pos = 0
        bots[bid].phase = 'pickup'

    # Assign preview to truly idle bots
    remaining_idle = [b for b in idle_bots if b not in bot_items]
    if remaining_idle:
        _assign_preview(state, dist_maps, all_orders, bots, remaining_idle, future_types)


def _assign_preview(state, dist_maps, all_orders, bots, idle_bots, future_types):
    """Assign preview/future items to idle bots."""
    ms = state.map_state
    preview = state.get_preview_order()
    if not preview:
        return

    preview_needs = {}
    for tid in preview.needs():
        tid = int(tid)
        preview_needs[tid] = preview_needs.get(tid, 0) + 1

    num_bots = len(state.bot_positions)
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in preview_needs and preview_needs[t] > 0:
                preview_needs[t] -= 1

    # Subtract items being picked by other bots
    for bid in range(num_bots):
        if bots[bid].phase == 'pickup' and bots[bid].route:
            for item_idx, _ in bots[bid].route[bots[bid].route_pos:]:
                tid = int(ms.item_types[item_idx])
                if tid in preview_needs and preview_needs[tid] > 0:
                    preview_needs[tid] -= 1

    for bid in idle_bots[:2]:  # max 2 preview bots
        inv_count = state.bot_inv_count(bid)
        if inv_count >= INV_CAP:
            continue

        bx = int(state.bot_positions[bid, 0])
        by = int(state.bot_positions[bid, 1])
        items = []
        slots = INV_CAP - inv_count

        for tid, cnt in list(preview_needs.items()):
            if cnt <= 0 or len(items) >= slots:
                continue
            # Only pick if type is in future orders (avoid dead inventory)
            if tid not in future_types:
                continue
            for item_idx in find_items_of_type(ms, tid):
                items.append((item_idx, tid))
                preview_needs[tid] -= 1
                break

        if items:
            route, cost = best_route((bx, by), items, dist_maps, ms)
            bots[bid].route = route
            bots[bid].route_pos = 0
            bots[bid].phase = 'pickup'


def decide_actions(state, dist_maps, bots, delivering_bid):
    """Decide actions for all bots."""
    ms = state.map_state
    num_bots = len(state.bot_positions)
    active = state.get_active_order()
    actions = [(ACT_WAIT, -1)] * num_bots

    # Build occupancy for collision avoidance
    occupied = set()
    for bid in range(num_bots):
        occupied.add((int(state.bot_positions[bid, 0]),
                      int(state.bot_positions[bid, 1])))

    # Process delivering bot first (highest priority)
    if delivering_bid >= 0:
        _process_bot(state, delivering_bid, dist_maps, bots, actions, occupied)

    # Then process pickup bots
    for bid in range(num_bots):
        if bid == delivering_bid:
            continue
        if bots[bid].phase == 'pickup':
            _process_bot(state, bid, dist_maps, bots, actions, occupied)

    # Then idle/waiting bots
    for bid in range(num_bots):
        if actions[bid][0] != ACT_WAIT:
            continue
        if bid == delivering_bid:
            continue
        _process_idle(state, bid, dist_maps, bots, actions, occupied, delivering_bid)

    return actions


def _process_bot(state, bid, dist_maps, bots, actions, occupied):
    """Process one bot's action."""
    ms = state.map_state
    bx = int(state.bot_positions[bid, 0])
    by = int(state.bot_positions[bid, 1])
    bot_pos = (bx, by)
    inv = state.bot_inv_list(bid)
    active = state.get_active_order()
    bs = bots[bid]

    # Stall detection
    if bot_pos == bs.last_pos:
        bs.stall_count += 1
    else:
        bs.stall_count = 0
    bs.last_pos = bot_pos

    # Dropoff check
    if (bx == ms.drop_off[0] and by == ms.drop_off[1] and
            inv and active and any(active.needs_type(t) for t in inv)):
        actions[bid] = (ACT_DROPOFF, -1)
        return

    # Deliver phase
    if bs.phase == 'deliver':
        if not inv or not (active and any(active.needs_type(t) for t in inv)):
            bs.phase = 'idle'
            return
        act = _move_toward(bot_pos, ms.drop_off, dist_maps, occupied, ms)
        if act:
            actions[bid] = act
            _update_occupied(occupied, bot_pos, act)
        return

    # Pickup phase
    if bs.phase == 'pickup' and bs.route and bs.route_pos < len(bs.route):
        item_idx, adj = bs.route[bs.route_pos]
        ix = int(ms.item_positions[item_idx, 0])
        iy = int(ms.item_positions[item_idx, 1])

        if abs(bx - ix) + abs(by - iy) == 1 and len(inv) < INV_CAP:
            actions[bid] = (ACT_PICKUP, item_idx)
            bs.route_pos += 1
            if bs.route_pos >= len(bs.route):
                bs.phase = 'wait_deliver'
            return

        act = _move_toward(bot_pos, adj, dist_maps, occupied, ms)
        if act:
            actions[bid] = act
            _update_occupied(occupied, bot_pos, act)
        elif bs.stall_count >= 4:
            # Stalled: try any valid move
            act = _escape(bot_pos, ms.drop_off, dist_maps, occupied, ms)
            if act:
                actions[bid] = act
                _update_occupied(occupied, bot_pos, act)
            bs.stall_count = 0
        return


def _process_idle(state, bid, dist_maps, bots, actions, occupied, delivering_bid):
    """Handle idle/waiting bots."""
    ms = state.map_state
    bx = int(state.bot_positions[bid, 0])
    by = int(state.bot_positions[bid, 1])
    bot_pos = (bx, by)
    bs = bots[bid]

    # If near dropoff and not delivering, move away
    dist_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))
    if dist_to_drop <= 2 and delivering_bid >= 0 and bid != delivering_bid:
        act = _escape(bot_pos, ms.drop_off, dist_maps, occupied, ms)
        if act:
            actions[bid] = act
            _update_occupied(occupied, bot_pos, act)


def _move_toward(pos, target, dist_maps, occupied, ms):
    """Move toward target, avoiding occupied cells."""
    if pos == target:
        return None
    dist = int(get_distance(dist_maps, pos, target))
    if dist <= 0:
        return None

    act = get_first_step(dist_maps, pos, target)
    if act and act > 0:
        dx = [0, 0, 0, -1, 1][act]
        dy = [0, -1, 1, 0, 0][act]
        tc = (pos[0] + dx, pos[1] + dy)
        if tc not in occupied or tc == ms.spawn:
            return (act, -1)

    # Try alternate directions that reduce distance
    bx, by = pos
    for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if (cell == 0 or cell == 3) and (nx, ny) not in occupied:
                d2 = int(get_distance(dist_maps, (nx, ny), target))
                if d2 < dist:
                    return (act_id, -1)
    return None


def _escape(pos, dropoff, dist_maps, occupied, ms):
    """Move away from dropoff."""
    bx, by = pos
    best_act = None
    best_dist = -1

    for dx, dy, act_id in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            cell = ms.grid[ny, nx]
            if (cell == 0 or cell == 3) and (nx, ny) not in occupied:
                d2 = int(get_distance(dist_maps, (nx, ny), dropoff))
                if d2 > best_dist:
                    best_dist = d2
                    best_act = (act_id, -1)
    return best_act


def _update_occupied(occupied, pos, action):
    """Update occupied set after a move action."""
    act_type = action[0]
    if 1 <= act_type <= 4:
        dx = [0, 0, 0, -1, 1][act_type]
        dy = [0, -1, 1, 0, 0][act_type]
        new_pos = (pos[0] + dx, pos[1] + dy)
        occupied.discard(pos)
        occupied.add(new_pos)


# ── Main Solver ───────────────────────────────────────────────────────

def solve(seed, difficulty, verbose=True):
    """Solve using pipeline approach."""
    t0 = time.time()
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"Pipeline solver: {difficulty} seed={seed} bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)

    if verbose:
        print(f"  Distance maps: {len(dist_maps)} cells ({time.time()-t0:.1f}s)")

    bot_state = [Bot() for _ in range(num_bots)]
    action_log = []
    delivering_bid = -1  # which bot currently holds the delivery token
    last_active_id = -1

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        active_id = active.id if active else -1

        # Order changed? Reset state
        if active_id != last_active_id:
            for bid in range(num_bots):
                inv = state.bot_inv_list(bid)
                if active and any(active.needs_type(t) for t in inv):
                    bot_state[bid].phase = 'wait_deliver'
                else:
                    bot_state[bid].phase = 'idle'
                    bot_state[bid].route = []
                    bot_state[bid].route_pos = 0
            delivering_bid = -1
            last_active_id = active_id

        # Select delivering bot: closest bot with active items
        if delivering_bid < 0 or bot_state[delivering_bid].phase not in ('deliver', 'wait_deliver'):
            best_bid = -1
            best_dist = 9999
            for bid in range(num_bots):
                if bot_state[bid].phase == 'wait_deliver':
                    inv = state.bot_inv_list(bid)
                    if active and any(active.needs_type(t) for t in inv):
                        bx = int(state.bot_positions[bid, 0])
                        by = int(state.bot_positions[bid, 1])
                        d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                        if d < best_dist:
                            best_dist = d
                            best_bid = bid

            if best_bid >= 0:
                delivering_bid = best_bid
                bot_state[best_bid].phase = 'deliver'

        # Assign pickups to idle bots
        any_idle = any(bs.phase == 'idle' for bs in bot_state)
        if any_idle:
            assign_pickups(state, dist_maps, all_orders, bot_state, delivering_bid)

        # After delivery, transition to idle
        if delivering_bid >= 0:
            inv = state.bot_inv_list(delivering_bid)
            if not inv or not (active and any(active.needs_type(t) for t in inv)):
                # Just delivered everything (or no matching items)
                bx = int(state.bot_positions[delivering_bid, 0])
                by = int(state.bot_positions[delivering_bid, 1])
                if bx == ms.drop_off[0] and by == ms.drop_off[1]:
                    bot_state[delivering_bid].phase = 'idle'
                    bot_state[delivering_bid].route = []
                    bot_state[delivering_bid].route_pos = 0
                    delivering_bid = -1

        # Decide actions
        actions = decide_actions(state, dist_maps, bot_state, delivering_bid)
        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            dbot = f" D=B{delivering_bid}" if delivering_bid >= 0 else ""
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})'
                f'inv={state.bot_inv_list(bid)}'
                f'[{bot_state[bid].phase[:3]}]'
                for bid in range(num_bots)
            )
            print(f'  R{rnd:3d}: score={state.score:3d} orders={state.orders_completed}{dbot} | {bot_info}')

    if verbose:
        print(f'\nFinal score: {state.score} ({time.time()-t0:.1f}s)')

    return state.score, action_log


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    solve(seed, difficulty)
