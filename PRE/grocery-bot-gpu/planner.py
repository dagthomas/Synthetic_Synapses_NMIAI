"""Coordinated MAPF planner for Grocery Bot.

Combines:
- Centralized item assignment (zero dead inventory)
- Persistent trip planning (multi-item routes with TSP optimization)
- Sequential collision avoidance with alternative routing
- Full order foresight

Different from Zig (per-bot greedy):
- Global simultaneous assignment of all bots
- Full order knowledge (not just active + preview)
- Proactive collision avoidance with parking
"""
import time
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS,
    DX, DY,
)
from pathfinding import precompute_all_distances, get_distance, get_first_step
from action_gen import find_items_of_type, get_active_needed_types, get_preview_needed_types


# Bot states
ST_IDLE = 0
ST_MOVING_TO_ITEM = 1
ST_MOVING_TO_DROPOFF = 2
ST_PARKED = 3


class BotController:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.state = ST_IDLE
        self.trip_items = []      # [(item_idx, adj_cell), ...]
        self.trip_idx = 0         # which item in the trip we're going to
        self.target = None        # (x, y) current movement target
        self.item_to_pick = -1    # item_idx to pick when adjacent
        self.stuck_count = 0      # rounds stuck without progress
        self.last_pos = None
        self.park_target = None   # where to park when idle

    def is_busy(self):
        return self.state in (ST_MOVING_TO_ITEM, ST_MOVING_TO_DROPOFF)

    def assign_trip(self, items_with_cells):
        """Assign a pickup trip: list of (item_idx, adj_cell)."""
        self.trip_items = list(items_with_cells)
        self.trip_idx = 0
        if self.trip_items:
            item_idx, adj_cell = self.trip_items[0]
            self.target = adj_cell
            self.item_to_pick = item_idx
            self.state = ST_MOVING_TO_ITEM
        self.stuck_count = 0

    def assign_deliver(self, dropoff):
        """Go deliver at dropoff."""
        self.trip_items = []
        self.trip_idx = 0
        self.target = dropoff
        self.state = ST_MOVING_TO_DROPOFF
        self.stuck_count = 0

    def assign_park(self, park_pos):
        """Park at the given position (None = just move away from dropoff)."""
        self.state = ST_PARKED
        self.park_target = park_pos
        self.target = park_pos
        self.trip_items = []
        self.trip_idx = 0
        self.item_to_pick = -1
        self.stuck_count = 0

    def set_idle(self):
        self.state = ST_IDLE
        self.trip_items = []
        self.trip_idx = 0
        self.target = None
        self.item_to_pick = -1
        self.stuck_count = 0


def find_best_adj_cell(dist_maps, bot_pos, item_idx, ms):
    """Find the nearest walkable cell adjacent to an item."""
    adj = ms.item_adjacencies.get(item_idx, [])
    best = None
    best_d = 9999
    for (cx, cy) in adj:
        d = int(get_distance(dist_maps, bot_pos, (cx, cy)))
        if d < best_d:
            best_d = d
            best = (cx, cy)
    return best, best_d


def optimize_trip_order(bot_pos, items_with_cells, dropoff, dist_maps):
    """Find the best order to visit items (mini-TSP).

    items_with_cells: list of (item_idx, adj_cell)
    Returns reordered list.
    """
    n = len(items_with_cells)
    if n <= 1:
        return items_with_cells

    best_cost = 999999
    best_order = items_with_cells

    for perm in permutations(range(n)):
        cost = 0
        pos = bot_pos
        for i in perm:
            _, cell = items_with_cells[i]
            cost += int(get_distance(dist_maps, pos, cell))
            pos = cell
        cost += int(get_distance(dist_maps, pos, dropoff))

        if cost < best_cost:
            best_cost = cost
            best_order = [items_with_cells[i] for i in perm]

    return best_order


def compute_remaining_needs(state, controllers, order, ms):
    """Compute what an order still needs after accounting for inventories and in-flight pickups."""
    num_bots = len(state.bot_positions)
    needs = {}
    for tid in order.needs():
        needs[tid] = needs.get(tid, 0) + 1

    # Subtract ALL bots' inventories that match this order
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in needs and needs[t] > 0:
                needs[t] -= 1

    # Subtract items already assigned to busy bots
    for bc in controllers:
        if bc.is_busy() and bc.state == ST_MOVING_TO_ITEM:
            for item_idx, _ in bc.trip_items[bc.trip_idx:]:
                tid = int(ms.item_types[item_idx])
                if tid in needs and needs[tid] > 0:
                    needs[tid] -= 1

    return needs


def assign_items_globally(state, dist_maps, all_orders, controllers, max_active_bots=None):
    """Assign items from active order to idle bots using Hungarian matching.

    Returns dict: bot_id -> list of (item_idx, adj_cell)
    Only assigns to bots that are IDLE and have inventory space.
    Ensures no over-collection (items needed - items in all bots' inventories - items assigned).
    """
    ms = state.map_state
    active = state.get_active_order()
    preview = state.get_preview_order()
    num_bots = len(state.bot_positions)

    if not active:
        return {}

    active_needs = compute_remaining_needs(state, controllers, active, ms)

    # What still needs to be picked up
    remaining = []
    for tid, count in active_needs.items():
        for _ in range(count):
            remaining.append(tid)

    if not remaining:
        # Active order is fully covered! Try preview
        if preview:
            preview_needs = compute_remaining_needs(state, controllers, preview, ms)
            remaining = []
            for tid, count in preview_needs.items():
                for _ in range(count):
                    remaining.append(tid)

    if not remaining:
        return {}

    # Determine how many bots should be active
    if max_active_bots is None:
        max_active_bots = num_bots

    # Count currently active bots
    active_count = sum(1 for bc in controllers if bc.is_busy())

    # Available bots: idle or parked with inventory space
    available = []
    for bc in controllers:
        if bc.state in (ST_IDLE, ST_PARKED):
            inv_count = state.bot_inv_count(bc.bot_id)
            if inv_count < INV_CAP and active_count < max_active_bots:
                available.append(bc.bot_id)
                active_count += 1

    if not available:
        return {}

    use_round_trip = num_bots <= 1

    # Precompute best (item_idx, adj_cell, cost) for each (bot, need_index)
    # need_item_options[ni] = list of (item_idx, adj_cell) per need
    need_item_cache = {}  # (bid, ni) -> (cost, item_idx, adj_cell)
    for ni, type_id in enumerate(remaining):
        items_of_type = find_items_of_type(ms, type_id)
        for bid in available:
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            best_cost = 9999
            best_item = None
            best_cell = None
            for item_idx in items_of_type:
                adj_cell, d_to_item = find_best_adj_cell(dist_maps, (bx, by), item_idx, ms)
                if adj_cell and d_to_item < 9999:
                    if use_round_trip:
                        d_to_drop = int(get_distance(dist_maps, adj_cell, ms.drop_off))
                        cost = d_to_item + d_to_drop
                    else:
                        cost = d_to_item
                    if cost < best_cost:
                        best_cost = cost
                        best_item = item_idx
                        best_cell = adj_cell
            if best_item is not None:
                need_item_cache[(bid, ni)] = (best_cost, best_item, best_cell)

    # Try Hungarian assignment for first item per bot
    n_bots = len(available)
    n_needs = len(remaining)

    if n_bots >= 2 and n_needs >= 2:
        # Build cost matrix: rows=bots, cols=needs
        BIG = 100000
        cost_matrix = np.full((n_bots, n_needs), BIG, dtype=np.float64)
        item_info = {}  # (row, col) -> (item_idx, adj_cell)

        for row, bid in enumerate(available):
            for col in range(n_needs):
                key = (bid, col)
                if key in need_item_cache:
                    c, item_idx, adj_cell = need_item_cache[key]
                    cost_matrix[row, col] = c
                    item_info[(row, col)] = (item_idx, adj_cell)

        # Solve optimal matching
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assignments = {bid: [] for bid in available}
            bot_slots = {}
            for bid in available:
                bot_slots[bid] = INV_CAP - state.bot_inv_count(bid)

            assigned_needs = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] >= BIG:
                    continue
                bid = available[r]
                if len(assignments[bid]) >= bot_slots[bid]:
                    continue
                if (r, c) in item_info:
                    assignments[bid].append(item_info[(r, c)])
                    assigned_needs.add(c)

            # Fill remaining slots: bots with space get additional nearby items
            for bid in available:
                if len(assignments[bid]) >= bot_slots[bid]:
                    continue
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                # Use location of last assigned item as starting point
                if assignments[bid]:
                    _, last_cell = assignments[bid][-1]
                    pos = last_cell
                else:
                    pos = (bx, by)
                # Find closest unassigned needs
                extras = []
                for ni in range(n_needs):
                    if ni in assigned_needs:
                        continue
                    key = (bid, ni)
                    if key in need_item_cache:
                        c, item_idx, adj_cell = need_item_cache[key]
                        extras.append((c, ni, item_idx, adj_cell))
                extras.sort()
                for _, ni, item_idx, adj_cell in extras:
                    if ni in assigned_needs:
                        continue
                    if len(assignments[bid]) >= bot_slots[bid]:
                        break
                    assignments[bid].append((item_idx, adj_cell))
                    assigned_needs.add(ni)

            result = {bid: items for bid, items in assignments.items() if items}
            if result:
                return result
        except Exception:
            pass  # Fall through to greedy

    # Greedy fallback (also used for single-bot or degenerate cases)
    candidates = []
    for ni, type_id in enumerate(remaining):
        for bid in available:
            key = (bid, ni)
            if key in need_item_cache:
                cost, item_idx, adj_cell = need_item_cache[key]
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

        assignments[bid].append((item_idx, adj_cell))
        assigned_needs.add(ni)

        if len(assigned_needs) == len(remaining):
            break

    return {bid: items for bid, items in assignments.items() if items}


def pipeline_assign(state, dist_maps, all_orders, controllers, max_active_bots=None, pipeline_depth=3):
    """Assign items looking ahead pipeline_depth orders with full foresight.

    Unlike assign_items_globally which only looks at active+preview,
    this function assigns idle bots to future order items proactively.
    Returns dict: bot_id -> list of (item_idx, adj_cell)
    """
    ms = state.map_state
    active = state.get_active_order()
    num_bots = len(state.bot_positions)

    if not active:
        return {}

    # First: assign active order items (same as assign_items_globally)
    active_assignments = assign_items_globally(state, dist_maps, all_orders, controllers, max_active_bots)

    # Only pipeline future items when active order is truly covered:
    # All needed items must be IN inventories (not just assigned/in-flight),
    # AND no bot should still be picking items
    any_picking = any(bc.state == ST_MOVING_TO_ITEM for bc in controllers)
    if any_picking:
        return active_assignments

    # Check active order needs from inventories only (not in-flight)
    inv_needs = {}
    for tid in active.needs():
        inv_needs[tid] = inv_needs.get(tid, 0) + 1
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in inv_needs and inv_needs[t] > 0:
                inv_needs[t] -= 1
    inv_remaining = sum(max(0, v) for v in inv_needs.values())
    if inv_remaining > 0:
        return active_assignments

    # Find idle bots NOT assigned by active assignment
    assigned_bots = set(active_assignments.keys())
    idle_bots = []
    for bc in controllers:
        if bc.bot_id not in assigned_bots and bc.state in (ST_IDLE, ST_PARKED):
            inv_count = state.bot_inv_count(bc.bot_id)
            if inv_count < INV_CAP:
                idle_bots.append(bc.bot_id)

    if not idle_bots:
        return active_assignments

    # Look ahead at future orders
    current_idx = state.active_idx
    future_needs = {}  # type_id -> count needed across future orders

    # Collect needs from orders[active_idx+1 : active_idx+pipeline_depth]
    # (active_idx+0 is current active, already handled above)
    for oi in range(current_idx + 1, min(current_idx + pipeline_depth, len(all_orders))):
        if oi >= len(all_orders):
            break
        order = all_orders[oi]
        for tid in order.required:
            tid = int(tid)
            future_needs[tid] = future_needs.get(tid, 0) + 1

    if not future_needs:
        return active_assignments

    # Subtract items already in bots' inventories that match future orders
    for bid in range(num_bots):
        for t in state.bot_inv_list(bid):
            if t in future_needs and future_needs[t] > 0:
                future_needs[t] -= 1

    # Subtract items assigned to busy bots
    for bc in controllers:
        if bc.is_busy() and bc.state == ST_MOVING_TO_ITEM:
            for item_idx, _ in bc.trip_items[bc.trip_idx:]:
                tid = int(ms.item_types[item_idx])
                if tid in future_needs and future_needs[tid] > 0:
                    future_needs[tid] -= 1

    # Subtract items from active_assignments
    for bid, items in active_assignments.items():
        for item_idx, _ in items:
            tid = int(ms.item_types[item_idx])
            if tid in future_needs and future_needs[tid] > 0:
                future_needs[tid] -= 1

    # Build remaining future needs
    future_remaining = []
    for tid, count in future_needs.items():
        for _ in range(max(0, count)):
            future_remaining.append(tid)

    if not future_remaining:
        return active_assignments

    # Assign future items to idle bots using greedy
    use_round_trip = num_bots <= 1
    candidates = []
    for ni, type_id in enumerate(future_remaining):
        for item_idx in find_items_of_type(ms, type_id):
            for bid in idle_bots:
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

    pipeline_assignments = {}
    assigned_needs = set()
    bot_slots = {}
    for bid in idle_bots:
        inv_count = state.bot_inv_count(bid)
        bot_slots[bid] = INV_CAP - inv_count
        pipeline_assignments[bid] = []

    for d, ni, bid, item_idx, adj_cell in candidates:
        if ni in assigned_needs:
            continue
        if len(pipeline_assignments.get(bid, [])) >= bot_slots.get(bid, 0):
            continue
        pipeline_assignments[bid].append((item_idx, adj_cell))
        assigned_needs.add(ni)
        if len(assigned_needs) == len(future_remaining):
            break

    # Merge active + pipeline assignments
    result = dict(active_assignments)
    for bid, items in pipeline_assignments.items():
        if items:
            result[bid] = items
    return result


def bfs_first_step_avoiding(start, goal, ms, occupied):
    """BFS from start to goal, treating occupied cells as walls.

    Returns first action ID (1-4) or 0 if no path.
    Much better than simple direction checking when corridors are congested.
    """
    from collections import deque

    if start == goal:
        return 0

    queue = deque([(start, 0)])  # (pos, first_action)
    visited = {start}

    while queue:
        (x, y), first_act = queue.popleft()

        for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            dx, dy = DX[act_id], DY[act_id]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < ms.width and 0 <= ny < ms.height):
                continue
            cell = ms.grid[ny, nx]
            if cell != 0 and cell != 3:
                continue
            npos = (nx, ny)
            if npos in visited:
                continue
            if npos in occupied and npos != ms.spawn:
                continue

            fa = first_act if first_act != 0 else act_id
            if npos == goal:
                return fa
            visited.add(npos)
            queue.append((npos, fa))

    return 0  # No path found


def plan_path_steps(start, target, dist_maps, ms, steps=3):
    """Plan up to `steps` positions along the optimal path from start to target.

    Returns list of (x, y) positions for each step (NOT including start).
    """
    path = []
    pos = start
    for _ in range(steps):
        if pos == target:
            path.append(pos)  # stay at target
            continue
        act = get_first_step(dist_maps, pos, target)
        if act <= 0:
            path.append(pos)  # can't move
            continue
        dx, dy = DX[act], DY[act]
        pos = (pos[0] + dx, pos[1] + dy)
        path.append(pos)
    return path


def reservation_move(bot_pos, target, dist_maps, ms, reservation_table, time_step=0):
    """Move one step toward target using a reservation table for multi-step lookahead.

    reservation_table: dict of (x, y, t) -> bot_id that has reserved that cell at time t
    time_step: current time step (0 = this round)

    Returns (action, reserved_path) where reserved_path is list of (x, y) for steps 0..2
    """
    if bot_pos == target:
        # At target, reserve staying here
        path = [bot_pos] * 3
        return (ACT_WAIT, -1), path

    # Try optimal direction first
    act = get_first_step(dist_maps, bot_pos, target)
    if act > 0:
        dx, dy = DX[act], DY[act]
        new_pos = (bot_pos[0] + dx, bot_pos[1] + dy)

        # Check reservation table for next 3 steps
        if (new_pos[0], new_pos[1], time_step) not in reservation_table or new_pos == ms.spawn:
            # Plan ahead from new_pos
            future = plan_path_steps(new_pos, target, dist_maps, ms, steps=2)
            # Check if future path is clear
            conflict = False
            for i, fpos in enumerate(future):
                if (fpos[0], fpos[1], time_step + 1 + i) in reservation_table and fpos != ms.spawn:
                    conflict = True
                    break

            if not conflict:
                path = [new_pos] + future
                return (act, -1), path

    # Try alternatives sorted by distance
    target_dist = int(get_distance(dist_maps, bot_pos, target))
    alternatives = []

    for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        dx, dy = DX[act_id], DY[act_id]
        nx, ny = bot_pos[0] + dx, bot_pos[1] + dy
        if not (0 <= nx < ms.width and 0 <= ny < ms.height):
            continue
        cell = ms.grid[ny, nx]
        if cell != 0 and cell != 3:
            continue
        if (nx, ny, time_step) in reservation_table and (nx, ny) != ms.spawn:
            continue
        d = int(get_distance(dist_maps, (nx, ny), target))
        if d < 9999 and d <= target_dist + 2:
            alternatives.append((d, act_id, (nx, ny)))

    alternatives.sort()
    for d, act_id, new_pos in alternatives:
        future = plan_path_steps(new_pos, target, dist_maps, ms, steps=2)
        conflict = False
        for i, fpos in enumerate(future):
            if (fpos[0], fpos[1], time_step + 1 + i) in reservation_table and fpos != ms.spawn:
                conflict = True
                break
        if not conflict:
            path = [new_pos] + future
            return (act_id, -1), path

    # Fallback: BFS avoiding reserved cells at t=0
    occupied_t0 = {(x, y) for (x, y, t) in reservation_table if t == time_step}
    bfs_act = bfs_first_step_avoiding(bot_pos, target, ms, occupied_t0)
    if bfs_act > 0:
        dx, dy = DX[bfs_act], DY[bfs_act]
        new_pos = (bot_pos[0] + dx, bot_pos[1] + dy)
        path = [new_pos] + [new_pos] * 2  # approximate future
        return (bfs_act, -1), path

    # Wait
    path = [bot_pos] * 3
    return (ACT_WAIT, -1), path


def move_toward_avoiding(bot_pos, target, dist_maps, ms, occupied, bot_id=None):
    """Move one step toward target, avoiding occupied cells.

    Uses strategies in order:
    1. Try optimal BFS first-step, check if clear
    2. Try alternative directions within +2 distance of optimal
    3. BFS fallback: full pathfinding around occupied cells
    4. Wait only as last resort
    """
    if bot_pos == target:
        return (ACT_WAIT, -1)

    # Strategy 1: optimal direction if clear
    act = get_first_step(dist_maps, bot_pos, target)
    if act > 0:
        dx, dy = DX[act], DY[act]
        new_pos = (bot_pos[0] + dx, bot_pos[1] + dy)
        if new_pos not in occupied or new_pos == ms.spawn:
            return (act, -1)

    # Strategy 2: try alternatives that reduce distance to target
    target_dist = int(get_distance(dist_maps, bot_pos, target))
    alternatives = []

    for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        dx, dy = DX[act_id], DY[act_id]
        nx, ny = bot_pos[0] + dx, bot_pos[1] + dy
        if not (0 <= nx < ms.width and 0 <= ny < ms.height):
            continue
        cell = ms.grid[ny, nx]
        if cell != 0 and cell != 3:
            continue
        if (nx, ny) in occupied and (nx, ny) != ms.spawn:
            continue
        d = int(get_distance(dist_maps, (nx, ny), target))
        if d < 9999:
            alternatives.append((d, act_id))

    alternatives.sort()
    if alternatives:
        d, act_id = alternatives[0]
        if d <= target_dist + 2:  # allow slight detour
            return (act_id, -1)

    # Strategy 3: BFS detour around occupied cells
    bfs_act = bfs_first_step_avoiding(bot_pos, target, ms, occupied)
    if bfs_act > 0:
        return (bfs_act, -1)

    return (ACT_WAIT, -1)


def move_away_from(bot_pos, avoid_pos, dist_maps, ms, occupied):
    """Move away from avoid_pos, preferring cells farther from it."""
    avoid_dist = int(get_distance(dist_maps, bot_pos, avoid_pos))
    best_act = ACT_WAIT
    best_d = avoid_dist

    for act_id in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        dx, dy = DX[act_id], DY[act_id]
        nx, ny = bot_pos[0] + dx, bot_pos[1] + dy
        if not (0 <= nx < ms.width and 0 <= ny < ms.height):
            continue
        cell = ms.grid[ny, nx]
        if cell != 0 and cell != 3:
            continue
        if (nx, ny) in occupied and (nx, ny) != ms.spawn:
            continue
        d = int(get_distance(dist_maps, (nx, ny), avoid_pos))
        if d > best_d:
            best_d = d
            best_act = act_id

    return (best_act, -1) if best_act != ACT_WAIT else (ACT_WAIT, -1)


def find_parking_spots(ms, dist_maps, num_spots=10):
    """Find good parking spots: walkable cells far from dropoff and corridors."""
    spots = []
    mid_y = ms.height // 2

    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] != 0:
                continue
            d = int(get_distance(dist_maps, (x, y), ms.drop_off))
            if d < 3:
                continue
            # Prefer cells in aisles (away from horizontal corridors)
            corridor_penalty = 0
            if y == ms.height - 2 or y == mid_y:
                corridor_penalty = 5
            spots.append((-(d - corridor_penalty), x, y))

    spots.sort()
    return [(x, y) for _, x, y in spots[:num_spots]]


def solve(seed=None, difficulty=None, verbose=True, max_active_bots=None, game_factory=None,
          pipeline_depth=0):
    """Solve using coordinated MAPF planning."""
    t0 = time.time()
    if game_factory:
        state, all_orders = game_factory()
    else:
        state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    num_bots = len(state.bot_positions)

    if verbose:
        print(f"MAPF Planner: bots={num_bots} map={ms.width}x{ms.height}")

    dist_maps = precompute_all_distances(ms)
    parking_spots = find_parking_spots(ms, dist_maps, num_spots=num_bots * 2)

    if verbose:
        print(f"  Distance maps: {len(dist_maps)} cells ({time.time()-t0:.1f}s)")

    # Default max active bots based on difficulty
    # Conservative defaults to avoid deadlock; multi_solve tries all values
    if max_active_bots is None:
        max_active_bots = min(num_bots, 2)

    # Initialize bot controllers
    controllers = [BotController(bid) for bid in range(num_bots)]

    action_log = []
    last_orders_completed = 0
    last_active_order_id = -1

    for rnd in range(MAX_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()
        active_id = active.id if active else -1

        # Check if order changed (completed or new active)
        order_changed = False
        if state.orders_completed > last_orders_completed:
            order_changed = True
            last_orders_completed = state.orders_completed
        if active_id != last_active_order_id:
            order_changed = True
            last_active_order_id = active_id

        # If order changed, reset all picking bots (simple & reliable)
        if order_changed:
            for bc in controllers:
                if bc.state == ST_MOVING_TO_ITEM:
                    bc.set_idle()

        # Check for bots that finished their trips
        for bc in controllers:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

            if bc.state == ST_MOVING_TO_DROPOFF and bot_pos == ms.drop_off:
                # Will deliver this round, then become idle
                pass  # handled in action generation

            if bc.state == ST_MOVING_TO_ITEM and bc.target and bot_pos == bc.target:
                # At pickup cell - will pick this round, then advance to next item
                pass  # handled in action generation

            # Detect stuck bots
            if bc.last_pos == bot_pos and bc.is_busy():
                bc.stuck_count += 1
            else:
                bc.stuck_count = 0
            bc.last_pos = bot_pos

            # If stuck for too long, try replanning
            if bc.stuck_count > 8:
                bc.set_idle()

        # Assign items to idle bots
        # Note: pipeline_assign was too aggressive (order_changed resets picking bots,
        # creating dead inventory). Pipeline depth only affects auto-delivery staging.
        assignments = assign_items_globally(
            state, dist_maps, all_orders, controllers, max_active_bots
        )
        for bid, items in assignments.items():
            bc = controllers[bid]
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

            # Optimize trip order (mini-TSP)
            optimized = optimize_trip_order(bot_pos, items, ms.drop_off, dist_maps)
            bc.assign_trip(optimized)

        # Check if any idle bot should deliver (has items matching active order)
        for bc in controllers:
            if bc.state in (ST_IDLE, ST_PARKED):
                bid = bc.bot_id
                inv = state.bot_inv_list(bid)
                if inv and active and any(active.needs_type(t) for t in inv):
                    bc.assign_deliver(ms.drop_off)

        # AUTO-DELIVERY STAGING: route bots with preview/future items to dropoff
        # Compute how many active items are still uncovered
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
            # Active order fully covered - stage bots with preview/future items
            # Collect needs from next few orders for staging check
            future_types = set()
            if preview:
                for tid in preview.needs():
                    future_types.add(tid)
            # Also check further-ahead orders with full foresight
            if pipeline_depth > 0:
                for oi in range(state.active_idx + 2,
                                min(state.active_idx + pipeline_depth + 1, len(all_orders))):
                    for tid in all_orders[oi].required:
                        future_types.add(int(tid))

            for bc in controllers:
                if bc.state in (ST_IDLE, ST_PARKED):
                    bid = bc.bot_id
                    inv = state.bot_inv_list(bid)
                    if inv and any(t in future_types for t in inv):
                        bc.assign_deliver(ms.drop_off)
        elif active_uncovered <= 2 and preview:
            # Active almost done - stage bots with preview items near dropoff
            for bc in controllers:
                if bc.state in (ST_IDLE, ST_PARKED):
                    bid = bc.bot_id
                    inv = state.bot_inv_list(bid)
                    bx = int(state.bot_positions[bid, 0])
                    by = int(state.bot_positions[bid, 1])
                    d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                    if inv and d <= 4 and any(preview.needs_type(t) for t in inv):
                        bc.assign_deliver(ms.drop_off)

        # Park idle bots that are near dropoff or blocking delivery
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

                # Don't park bots with preview items near dropoff - they're staging
                if inv and preview and any(preview.needs_type(t) for t in inv):
                    continue

                # Check if blocking a delivering bot
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

        # Generate actions: process bots in priority order
        # Priority: delivering > picking > parking > idle
        def bot_priority(bc):
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])

            if bc.state == ST_MOVING_TO_DROPOFF:
                if (bx, by) == ms.drop_off:
                    return (0, 0)  # At dropoff, deliver now
                d = int(get_distance(dist_maps, (bx, by), ms.drop_off))
                return (1, d)  # Delivering, closer = higher priority
            if bc.state == ST_MOVING_TO_ITEM:
                return (2, 0)
            if bc.state == ST_PARKED:
                return (3, 0)
            return (4, 0)  # Idle

        sorted_bots = sorted(controllers, key=bot_priority)

        # Dropoff traffic management: only allow 1 bot at a time to approach
        # Count bots at or adjacent to dropoff
        dropoff_arrivals = 0  # how many bots are actively heading to dropoff
        max_dropoff_arrivals = 1 if num_bots >= 3 else num_bots
        dropoff_wait_dist = 3  # bots wait this far from dropoff until their turn

        actions = [(ACT_WAIT, -1)] * num_bots

        # Reservation table for multi-step lookahead collision avoidance
        reservation_table = {}  # (x, y, t) -> bot_id
        committed_positions = {}  # bot_id -> new_pos (immediate step)

        def reserve_bot(bid, new_pos, future_path=None):
            """Reserve a bot's position and optional future path."""
            committed_positions[bid] = new_pos
            reservation_table[(new_pos[0], new_pos[1], 0)] = bid
            if future_path:
                for t, fpos in enumerate(future_path):
                    reservation_table[(fpos[0], fpos[1], t + 1)] = bid
            else:
                # Reserve staying at same position for 2 more steps
                for t in range(1, 3):
                    reservation_table[(new_pos[0], new_pos[1], t)] = bid

        for bc in sorted_bots:
            bid = bc.bot_id
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)

            # Compute "occupied after committed moves"
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
                        reserve_bot(bid, bot_pos)
                        bc.set_idle()
                        continue
                    elif inv and preview and any(preview.needs_type(t) for t in inv):
                        # Has preview items at dropoff for auto-delivery.
                        # But yield if an active delivery bot needs the cell.
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
                                reserve_bot(bid, (bx + dx, by + dy))
                            else:
                                reserve_bot(bid, bot_pos)
                        else:
                            actions[bid] = (ACT_WAIT, -1)
                            reserve_bot(bid, bot_pos)
                        continue
                    else:
                        # No useful items - set idle
                        bc.set_idle()

                if bc.state == ST_MOVING_TO_DROPOFF:
                    d_to_drop = int(get_distance(dist_maps, bot_pos, ms.drop_off))

                    # Dropoff traffic: if too many bots already approaching, wait at distance
                    if d_to_drop > 1 and dropoff_arrivals >= max_dropoff_arrivals and d_to_drop <= dropoff_wait_dist:
                        # Wait in place (let the closer bot deliver first)
                        actions[bid] = (ACT_WAIT, -1)
                        reserve_bot(bid, bot_pos)
                        continue

                    act, path = reservation_move(bot_pos, ms.drop_off, dist_maps, ms, reservation_table)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        new_pos = (bx + dx, by + dy)
                        reserve_bot(bid, new_pos, path[1:] if len(path) > 1 else None)
                    else:
                        reserve_bot(bid, bot_pos)

                    if d_to_drop <= dropoff_wait_dist:
                        dropoff_arrivals += 1
                    continue

            if bc.state == ST_MOVING_TO_ITEM:
                if bot_pos == bc.target and bc.item_to_pick >= 0:
                    # At pickup cell - pick up the item
                    actions[bid] = (ACT_PICKUP, bc.item_to_pick)
                    reserve_bot(bid, bot_pos)

                    # Advance to next item or switch to delivery
                    bc.trip_idx += 1
                    if bc.trip_idx < len(bc.trip_items):
                        next_item, next_cell = bc.trip_items[bc.trip_idx]
                        bc.target = next_cell
                        bc.item_to_pick = next_item
                    else:
                        # All items picked, go deliver
                        bc.assign_deliver(ms.drop_off)
                    continue
                else:
                    # Move toward pickup cell
                    act, path = reservation_move(bot_pos, bc.target, dist_maps, ms, reservation_table)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        new_pos = (bx + dx, by + dy)
                        reserve_bot(bid, new_pos, path[1:] if len(path) > 1 else None)
                    else:
                        reserve_bot(bid, bot_pos)
                    continue

            if bc.state == ST_PARKED:
                if bc.park_target is None:
                    # No specific target - just move away from dropoff
                    act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        new_pos = (bx + dx, by + dy)
                        reserve_bot(bid, new_pos)
                        d_new = int(get_distance(dist_maps, new_pos, ms.drop_off))
                        if d_new > 4:
                            bc.set_idle()  # Far enough, go back to idle
                    else:
                        reserve_bot(bid, bot_pos)
                elif bot_pos != bc.park_target:
                    act = move_toward_avoiding(bot_pos, bc.park_target, dist_maps, ms, occ_after)
                    actions[bid] = act
                    if act[0] != ACT_WAIT:
                        dx, dy = DX[act[0]], DY[act[0]]
                        reserve_bot(bid, (bx + dx, by + dy))
                    else:
                        reserve_bot(bid, bot_pos)
                else:
                    # At park spot - stay unless someone else needs this cell
                    any_heading_here = any(
                        bc2.bot_id != bid and bc2.target == bot_pos
                        for bc2 in controllers
                    )
                    if any_heading_here:
                        act = move_away_from(bot_pos, ms.drop_off, dist_maps, ms, occ_after)
                        actions[bid] = act
                        if act[0] != ACT_WAIT:
                            dx, dy = DX[act[0]], DY[act[0]]
                            reserve_bot(bid, (bx + dx, by + dy))
                        else:
                            reserve_bot(bid, bot_pos)
                    else:
                        reserve_bot(bid, bot_pos)
                continue

            # ST_IDLE: wait
            reserve_bot(bid, bot_pos)

        action_log.append(actions)
        step(state, actions, all_orders)

        if verbose and (rnd < 10 or rnd % 50 == 0 or rnd == MAX_ROUNDS - 1):
            bot_info = ' | '.join(
                f'B{bid}@({state.bot_positions[bid,0]},{state.bot_positions[bid,1]})'
                f'[{controllers[bid].state}]inv={state.bot_inv_list(bid)}'
                for bid in range(num_bots)
            )
            print(f'  R{rnd:3d}: score={state.score:3d} ord={state.orders_completed} | {bot_info}')

    if verbose:
        print(f'\nFinal score: {state.score} ({time.time()-t0:.1f}s)')

    return state.score, action_log


def solve_hybrid(seed, difficulty, verbose=True, beam_width=10):
    """Hybrid solver: beam search for 1-bot, MAPF planner for multi-bot."""
    from configs import CONFIGS
    cfg = CONFIGS[difficulty]

    if cfg['bots'] == 1:
        # Use beam search for single bot (much better)
        from beam_search import beam_search
        score, actions, _ = beam_search(seed, difficulty, beam_width=beam_width,
                                         max_per_bot=4, verbose=verbose)
        return score, actions
    else:
        return solve(seed, difficulty, verbose=verbose)


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    hybrid = '--hybrid' in sys.argv

    if hybrid:
        score, actions = solve_hybrid(seed, difficulty)
    else:
        score, actions = solve(seed, difficulty)
