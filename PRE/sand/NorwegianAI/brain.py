"""Bot decision-making with DistanceMatrix, batch-aware selection, and route following.

Improvements:
1. DistanceMatrix for O(1) exact BFS distances (replaces Manhattan heuristic)
2. Batch-aware item selection: enumerates up to 3-item batches scored by TSP route
3. Route-following state machine: computed route followed step-by-step
4. Preview pre-picking: fill remaining slots with next order's items
5. Background planning: pre-compute preview order route while delivering active order
"""

from itertools import combinations, permutations
from pathfinding import build_blocked_set, bfs, path_to_action
from distance import DistanceMatrix


# Tunable parameters (overridable by param_search.py)
_PARAMS = {}

# Module-level state (reset on round 0)
_dm = None          # DistanceMatrix instance
_bot_routes = {}    # bot_id -> {"targets": [...], "step": int, "order_id": str}
_multi = {"order_id": None, "active_assignments": {}, "preview_assignments": {}, "last_delivered_count": 0, "bot_aisle": {}, "aisle_xs": [], "inv_baseline": {}}
_bot_history = {}   # bot_id -> list of recent positions (anti-oscillation)
_expert = {}        # Expert mode state (reset on round 0)

def get_needed_items(order):
    """Return {item_type: count_still_needed} for an order."""
    needed = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def items_still_needed_from_map(order, all_bots):
    """Items still needed from shelves (not delivered, not in any inventory)."""
    needed = get_needed_items(order)
    for bot in all_bots:
        for item_type in bot["inventory"]:
            if item_type in needed and needed[item_type] > 0:
                needed[item_type] -= 1
    return {k: v for k, v in needed.items() if v > 0}


def _get_pipeline_needs(state, all_bots):
    """Get items to pre-pick for future orders beyond preview (deep pipeline).

    Uses _multi["future_orders"] (from scout data) to know what future orders need.
    Returns {item_type: count} for items needed by orders N+2, N+3, ...
    Preview order (N+1) is handled separately by the existing preview picking logic.
    Returns None if no future orders known or no items needed.
    """
    future_orders = _multi.get("future_orders", [])
    if not future_orders:
        return None

    orders = state["orders"]
    active = next((o for o in orders if o.get("status") == "active"), None)
    if not active:
        return None

    # Find active order's index in future_orders
    active_idx = None
    for i, fo in enumerate(future_orders):
        if fo["id"] == active["id"]:
            active_idx = i
            break
    if active_idx is None:
        return None

    # Build pool of items in all inventories
    all_held = []
    for bot in all_bots:
        all_held.extend(bot["inventory"])

    # Remove items committed to active order (will be delivered soon)
    active_needed = get_needed_items(active)
    for itype, count in active_needed.items():
        for _ in range(count):
            if itype in all_held:
                all_held.remove(itype)

    # Remove items committed to preview order (N+1)
    preview = next((o for o in orders if o.get("status") == "preview"), None)
    if preview:
        preview_needed = get_needed_items(preview)
        for itype, count in preview_needed.items():
            for _ in range(count):
                if itype in all_held:
                    all_held.remove(itype)

    # Available pool: items not committed to active/preview
    available = list(all_held)

    # Look at future orders N+2 through N+5
    pipeline_needs = {}
    for offset in range(2, 6):
        order_idx = active_idx + offset
        if order_idx >= len(future_orders):
            break

        fo = future_orders[order_idx]
        fo_needs = {}
        for item in fo["items_required"]:
            fo_needs[item] = fo_needs.get(item, 0) + 1

        # Subtract available items from pool
        for itype in list(fo_needs.keys()):
            while fo_needs.get(itype, 0) > 0 and itype in available:
                fo_needs[itype] -= 1
                available.remove(itype)

        fo_needs = {k: v for k, v in fo_needs.items() if v > 0}
        for itype, count in fo_needs.items():
            pipeline_needs[itype] = pipeline_needs.get(itype, 0) + count

    return pipeline_needs if pipeline_needs else None


def _pos_to_action(from_pos, to_pos):
    """Convert a move from one cell to adjacent cell into action string."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1: return "move_right"
    if dx == -1: return "move_left"
    if dy == 1: return "move_down"
    if dy == -1: return "move_up"
    return "wait"


def _find_best_batch(bot_pos, items, needed_map, claimed_items, drop_off, max_slots):
    """Find best batch of items to pick up, scored by TSP route cost.

    Allows revisiting the same shelf (items restock). Adds re-approach cost of 2
    for each revisit to the same shelf within a batch.
    Returns list of item objects, or empty list.
    """
    global _dm
    if not _dm or not needed_map or max_slots <= 0:
        return []

    # Collect candidate items with virtual duplicates for shelf revisiting
    candidates = []  # (cost, item, copy_idx)
    for item in items:
        if item["id"] in claimed_items:
            continue
        itype = item["type"]
        if needed_map.get(itype, 0) <= 0:
            continue
        shelf = (item["position"][0], item["position"][1])
        cost = _dm.trip_cost(bot_pos, shelf)
        if cost >= 999:
            continue
        # Add original + virtual copies for revisits (up to needed count)
        n_copies = min(needed_map.get(itype, 1), max_slots)
        for copy_i in range(n_copies):
            candidates.append((cost, item, copy_i))  # revisit is free (no movement needed)

    # Sort by individual trip cost for pruning
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:12]

    if not candidates:
        return []

    batch_size = min(max_slots, 3, len(candidates))
    best_batch = None
    best_score = float("inf")

    # When all needed items fit in available slots, force picking all of them
    # (picking fewer means an extra trip for the rest, always worse).
    # When there are more items than slots, use cost/item scoring freely.
    total_needed = sum(needed_map.values())
    min_size = min(total_needed, batch_size) if total_needed <= max_slots else 1

    for size in range(min_size, batch_size + 1):
        for combo in combinations(range(len(candidates)), size):
            # Check type constraints and no duplicate copies from same shelf
            batch_types = {}
            shelf_copies = {}  # shelf_pos -> max copy_idx used
            valid = True
            for idx in combo:
                _, item, copy_i = candidates[idx]
                itype = item["type"]
                batch_types[itype] = batch_types.get(itype, 0) + 1
                if batch_types[itype] > needed_map.get(itype, 0):
                    valid = False
                    break
                shelf = (item["position"][0], item["position"][1])
                key = (shelf, copy_i)
                if key in shelf_copies:
                    valid = False  # Can't use same copy twice
                    break
                shelf_copies[key] = True
            if not valid:
                continue

            batch_items = [candidates[idx][1] for idx in combo]
            shelves = [(it["position"][0], it["position"][1]) for it in batch_items]

            # TSP with re-approach cost for revisits to same shelf
            cost = _tsp_cost_revisit(bot_pos, shelves, drop_off)
            score = cost / size
            if score < best_score or (score == best_score and size > len(best_batch or [])):
                best_score = score
                best_batch = batch_items

    return best_batch or []


def _find_best_batch_combined(bot_pos, items, active_needs, combined_needs,
                              claimed_items, drop_off, max_slots):
    """Find best batch combining active items + preview items.

    Active items must be satisfied first; remaining slots filled with preview.
    Scored purely by cost/size — the marginal cost check in Phase 4 gates acceptance.
    """
    global _dm
    if not _dm or not combined_needs or max_slots <= 0:
        return []

    # Collect candidates with virtual duplicates for shelf revisiting
    candidates = []
    for item in items:
        if item["id"] in claimed_items:
            continue
        itype = item["type"]
        if combined_needs.get(itype, 0) <= 0:
            continue
        shelf = (item["position"][0], item["position"][1])
        cost = _dm.trip_cost(bot_pos, shelf)
        if cost >= 999:
            continue
        is_active = active_needs.get(itype, 0) > 0
        n_copies = min(combined_needs.get(itype, 1), max_slots)
        for copy_i in range(n_copies):
            candidates.append((cost, item, is_active, copy_i))

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:12]

    if not candidates:
        return []

    batch_size = min(max_slots, 3, len(candidates))
    best_batch = None
    best_score = float("inf")
    active_count_needed = sum(active_needs.values())

    for size in range(1, batch_size + 1):
        for combo in combinations(range(len(candidates)), size):
            batch_types = {}
            shelf_copies = {}
            active_in_batch = 0
            valid = True
            for idx in combo:
                _, item, _, copy_i = candidates[idx]
                itype = item["type"]
                batch_types[itype] = batch_types.get(itype, 0) + 1
                if batch_types[itype] > combined_needs.get(itype, 0):
                    valid = False
                    break
                shelf = (item["position"][0], item["position"][1])
                key = (shelf, copy_i)
                if key in shelf_copies:
                    valid = False
                    break
                shelf_copies[key] = True
                if batch_types[itype] <= active_needs.get(itype, 0):
                    active_in_batch += 1
            if not valid:
                continue

            min_active = min(active_count_needed, size)
            if active_in_batch < min_active:
                continue

            batch_items = [candidates[idx][1] for idx in combo]
            shelves = [(it["position"][0], it["position"][1]) for it in batch_items]
            cost = _tsp_cost_revisit(bot_pos, shelves, drop_off)
            score = cost / size
            if score < best_score or (score == best_score and size > len(best_batch or [])):
                best_score = score
                best_batch = batch_items

    return best_batch or []


def _tsp_cost_revisit(start, shelves, drop_off):
    """TSP cost with re-approach handling for duplicate shelves.

    When the same shelf appears multiple times, adds 2 rounds for re-approach
    (move away 1 step + come back 1 step).
    """
    global _dm
    if len(shelves) == 0:
        return 0
    if len(shelves) == 1:
        adj, d_to = _dm.best_adjacent(start, shelves[0])
        if adj is None:
            return 999
        d_back = _dm.dist_to_dropoff(adj)
        return d_to + 1 + d_back

    best = float("inf")
    for perm in permutations(range(len(shelves))):
        cost = 0
        pos = start
        prev_shelf = None
        for i, idx in enumerate(perm):
            shelf = shelves[idx]
            if shelf == prev_shelf:
                cost += 1  # pickup only
            else:
                is_last = (i == len(perm) - 1)
                if is_last:
                    # For last shelf, try ALL adjacent cells and pick the one
                    # minimizing travel_to + pickup + return_to_dropoff
                    best_last = float("inf")
                    for adj_cell in _dm.adjacent_cells(shelf):
                        d_to = _dm.dist(pos, adj_cell)
                        d_back = _dm.dist(adj_cell, drop_off)
                        total = d_to + 1 + d_back
                        if total < best_last:
                            best_last = total
                    if best_last >= 999:
                        cost = float("inf")
                        break
                    cost += best_last
                    pos = None  # not needed after last
                else:
                    adj, d = _dm.best_adjacent(pos, shelf)
                    if adj is None:
                        cost = float("inf")
                        break
                    cost += d + 1
                    pos = adj
            prev_shelf = shelf
        if cost < float("inf") and pos is not None:
            cost += _dm.dist(pos, drop_off)
        if cost < best:
            best = cost
    return best


def _eval_trip_cost_for_needs(items, needs, start_pos, drop_off):
    """Evaluate cost of a single trip picking ALL items matching 'needs'.

    Uses the closest shelf for each item type and computes TSP cost.
    """
    global _dm
    total_needed = sum(needs.values())
    if total_needed == 0:
        return 0

    # Find the best shelf for each needed type (closest to start_pos)
    best_shelves = {}  # type -> item
    for item in items:
        itype = item["type"]
        if needs.get(itype, 0) <= 0:
            continue
        shelf = (item["position"][0], item["position"][1])
        cost = _dm.trip_cost(start_pos, shelf)
        if cost >= 999:
            continue
        if itype not in best_shelves or cost < best_shelves[itype][1]:
            best_shelves[itype] = (item, cost)

    # Build shelf list: each needed type × count (with revisits for duplicates)
    shelves = []
    for itype, count in needs.items():
        if itype not in best_shelves:
            return 999
        item_obj = best_shelves[itype][0]
        shelf_pos = (item_obj["position"][0], item_obj["position"][1])
        for _ in range(count):
            shelves.append(shelf_pos)

    return _tsp_cost_revisit(start_pos, shelves, drop_off)


def _plan_order_trips(needed_map, items, bot_pos, drop_off, max_capacity=3):
    """Plan optimal trip split for a multi-item order.

    For orders fitting in one trip (≤3 items): returns [needed_map].
    For 4-item orders: evaluates all {3+1} and {2+2} splits and returns
    the split minimizing total trip cost across both trips.

    Trip 1 is evaluated from bot_pos, Trip 2+ from drop_off.
    Returns list of need-maps, one per trip.
    """
    total = sum(needed_map.values())
    if total <= max_capacity:
        return [needed_map]

    # Build flat item list: [type, type, type, type]
    items_list = []
    for itype, count in needed_map.items():
        items_list.extend([itype] * count)

    best_split = None
    best_total = float("inf")
    seen = set()

    # --- {3+1} splits: leave one item for trip 2 ---
    for i in range(len(items_list)):
        trip1 = items_list[:i] + items_list[i + 1:]
        trip2 = [items_list[i]]

        key = (tuple(sorted(trip1)), tuple(sorted(trip2)))
        if key in seen:
            continue
        seen.add(key)

        t1_needs = {}
        for t in trip1:
            t1_needs[t] = t1_needs.get(t, 0) + 1
        t2_needs = {trip2[0]: 1}

        t1_cost = _eval_trip_cost_for_needs(items, t1_needs, bot_pos, drop_off)
        t2_cost = _eval_trip_cost_for_needs(items, t2_needs, drop_off, drop_off)
        total_cost = t1_cost + t2_cost

        if total_cost < best_total:
            best_total = total_cost
            best_split = [t1_needs, t2_needs]

    # --- {2+2} splits ---
    for combo in combinations(range(len(items_list)), 2):
        trip1 = [items_list[j] for j in combo]
        trip2 = [items_list[j] for j in range(len(items_list)) if j not in combo]

        key = (tuple(sorted(trip1)), tuple(sorted(trip2)))
        rev_key = (tuple(sorted(trip2)), tuple(sorted(trip1)))
        if key in seen or rev_key in seen:
            continue
        seen.add(key)

        t1_needs = {}
        for t in trip1:
            t1_needs[t] = t1_needs.get(t, 0) + 1
        t2_needs = {}
        for t in trip2:
            t2_needs[t] = t2_needs.get(t, 0) + 1

        # For 2+2, evaluate both orderings (bot_pos→trip1 then dropoff→trip2,
        # and bot_pos→trip2 then dropoff→trip1) since bot_pos might favor one
        t1_cost_a = _eval_trip_cost_for_needs(items, t1_needs, bot_pos, drop_off)
        t2_cost_a = _eval_trip_cost_for_needs(items, t2_needs, drop_off, drop_off)
        total_a = t1_cost_a + t2_cost_a

        t1_cost_b = _eval_trip_cost_for_needs(items, t2_needs, bot_pos, drop_off)
        t2_cost_b = _eval_trip_cost_for_needs(items, t1_needs, drop_off, drop_off)
        total_b = t1_cost_b + t2_cost_b

        if total_a <= total_b and total_a < best_total:
            best_total = total_a
            best_split = [t1_needs, t2_needs]
        elif total_b < total_a and total_b < best_total:
            best_total = total_b
            best_split = [t2_needs, t1_needs]

    return best_split or [needed_map]


def _build_route(bot_pos, batch_items, drop_off):
    """Build a route with TSP-optimized pickup order, handling shelf revisits.

    Returns list of route steps:
      [{"type": "pickup", "shelf": pos, "adj": pos, "item": obj}, ...]
      + [{"type": "deliver", "target": drop_off}]
    """
    global _dm
    if not batch_items:
        return [{"type": "deliver", "target": drop_off}]

    shelves = [(it["position"][0], it["position"][1]) for it in batch_items]

    # Find optimal pickup order via TSP (with revisit handling)
    best_perm = list(range(len(shelves)))
    best_cost = float("inf")

    for perm in permutations(range(len(shelves))):
        cost = 0
        pos = bot_pos
        prev_shelf = None
        for idx in perm:
            shelf = shelves[idx]
            if shelf == prev_shelf:
                cost += 1  # just pickup, no movement needed
            else:
                adj, d = _dm.best_adjacent(pos, shelf)
                if adj is None:
                    cost = float("inf")
                    break
                cost += d + 1
                pos = adj
            prev_shelf = shelf
        if cost < float("inf"):
            cost += _dm.dist(pos, drop_off)
        if cost < best_cost:
            best_cost = cost
            best_perm = list(perm)

    route = []
    prev_shelf = None
    for i, idx in enumerate(best_perm):
        shelf = shelves[idx]
        is_last = (i == len(best_perm) - 1)
        if shelf == prev_shelf and route:
            route.append({
                "type": "pickup",
                "shelf": shelf,
                "adj": route[-1]["adj"],
                "item": batch_items[idx],
            })
        elif is_last:
            # For last shelf, pick adjacent cell minimizing total (approach + return)
            prev_pos = bot_pos if not route else route[-1]["adj"]
            best_adj = None
            best_total = float("inf")
            for adj_cell in _dm.adjacent_cells(shelf):
                d_to = _dm.dist(prev_pos, adj_cell)
                d_back = _dm.dist(adj_cell, drop_off)
                total = d_to + d_back
                if total < best_total:
                    best_total = total
                    best_adj = adj_cell
            if best_adj is None:
                best_adj, _ = _dm.best_adjacent(prev_pos, shelf)
            route.append({
                "type": "pickup",
                "shelf": shelf,
                "adj": best_adj,
                "item": batch_items[idx],
            })
        else:
            prev_pos = bot_pos if not route else route[-1]["adj"]
            adj, _ = _dm.best_adjacent(prev_pos, shelf)
            route.append({
                "type": "pickup",
                "shelf": shelf,
                "adj": adj,
                "item": batch_items[idx],
            })
        prev_shelf = shelf

    route.append({"type": "deliver", "target": drop_off})
    return route



def _detect_aisles(items, state=None):
    """Detect aisle x-coordinates from shelf positions and walls.

    An aisle is a walkable column between shelf pairs (shelf at x, aisle at x+1, shelf at x+2).
    Verifies the aisle cell is not a wall at the shelf row.
    Returns sorted list of aisle x-coordinates.
    """
    shelf_xs = sorted(set(it["position"][0] for it in items))
    shelf_set = set(shelf_xs)

    wall_set = set()
    if state:
        for w in state["grid"]["walls"]:
            wall_set.add((w[0], w[1]))

    # Map each shelf x to a representative y for walkability check
    shelf_ys = {}
    for it in items:
        x = it["position"][0]
        if x not in shelf_ys:
            shelf_ys[x] = it["position"][1]

    aisle_xs = []
    for x in shelf_xs:
        if (x + 2) not in shelf_set:
            continue
        aisle_x = x + 1
        y = shelf_ys[x]
        if (aisle_x, y) not in wall_set:
            aisle_xs.append(aisle_x)

    return sorted(set(aisle_xs))


def _filter_items_for_aisle(items, aisle_x):
    """Return only items on shelves adjacent to this aisle (x-1 and x+1)."""
    return [it for it in items if abs(it["position"][0] - aisle_x) <= 1]


def _find_cheapest_shelf(pos, item_type, items):
    """Find the cheapest shelf for item_type from pos. Returns (item_obj, cost)."""
    global _dm
    best_item = None
    best_cost = float("inf")
    for item in items:
        if item["type"] == item_type:
            shelf = tuple(item["position"])
            cost = _dm.trip_cost(pos, shelf)
            if cost < best_cost:
                best_cost = cost
                best_item = item
    return best_item, best_cost


def _clean_stale_assignments(bots):
    """Remove assignment items that bots have already picked up.

    Compares current inventory against baseline (snapshot at assignment time).
    New items in inventory beyond baseline are matched against assignments and removed.
    """
    global _multi
    for bot in bots:
        bid = bot["id"]
        baseline = list(_multi.get("inv_baseline", {}).get(bid, []))
        current = list(bot["inventory"])

        # Compute newly picked up items (in current but not in baseline)
        new_items = list(current)
        for item in baseline:
            if item in new_items:
                new_items.remove(item)

        # Remove matched items from active assignments first
        remaining_new = list(new_items)
        active = _multi["active_assignments"].get(bid, [])
        new_active = list(active)
        for item_type in remaining_new[:]:
            if item_type in new_active:
                new_active.remove(item_type)
                remaining_new.remove(item_type)
        _multi["active_assignments"][bid] = new_active

        # Then from preview assignments
        preview = _multi["preview_assignments"].get(bid, [])
        new_preview = list(preview)
        for item_type in remaining_new[:]:
            if item_type in new_preview:
                new_preview.remove(item_type)
                remaining_new.remove(item_type)
        _multi["preview_assignments"][bid] = new_preview


def _should_redistribute(active, bots):
    """Check if items should be redistributed across bots."""
    global _multi
    if active["id"] != _multi["order_id"]:
        return True  # New order activated
    if len(active["items_delivered"]) != _multi["last_delivered_count"]:
        # Delivery happened — only redistribute if there are uncovered items
        needed = get_needed_items(active)
        for bot in bots:
            for itype in bot["inventory"]:
                if itype in needed and needed[itype] > 0:
                    needed[itype] -= 1
        needed = {k: v for k, v in needed.items() if v > 0}
        if not needed:
            _multi["last_delivered_count"] = len(active["items_delivered"])
            return False
        covered = dict(needed)
        for bid_key, asgn in _multi["active_assignments"].items():
            for itype in asgn:
                if itype in covered and covered[itype] > 0:
                    covered[itype] -= 1
        if any(v > 0 for v in covered.values()):
            return True
        _multi["last_delivered_count"] = len(active["items_delivered"])
        return False

    return False


def _execute_optimized_plan(state, plan):
    """Execute pre-computed optimized plan with desync correction."""
    rnd = state["round"]
    actions = []
    for bot in state["bots"]:
        bot_id = bot["id"]
        bot_actions = plan["bot_actions"].get(str(bot_id), [])
        expected_positions = plan["bot_positions"].get(str(bot_id), [])

        if rnd < len(bot_actions):
            action = dict(bot_actions[rnd])
            # Check desync
            actual_pos = tuple(bot["position"])
            expected_pos = tuple(expected_positions[rnd]) if rnd < len(expected_positions) else None

            if expected_pos and actual_pos != expected_pos:
                dist = abs(actual_pos[0] - expected_pos[0]) + abs(actual_pos[1] - expected_pos[1])
                if dist >= 3:
                    # Severe desync: fallback to reactive brain
                    return decide_actions(state, game_plan=None)
                # Mild desync: navigate toward next planned position
                if rnd + 1 < len(expected_positions):
                    goal = tuple(expected_positions[rnd + 1])
                else:
                    goal = expected_pos
                # Simple BFS step toward goal
                from pathfinding import build_blocked_set
                blocked = build_blocked_set(state, exclude_bot_id=bot_id)
                if _dm:
                    ns = _dm.next_step(actual_pos, goal)
                    if ns and ns != actual_pos and ns not in blocked:
                        action = {"action": _pos_to_action(actual_pos, ns)}
                    else:
                        action = {"action": "wait"}
                else:
                    action = {"action": "wait"}

            action["bot"] = bot_id
        else:
            action = {"bot": bot_id, "action": "wait"}
        actions.append(action)
    return actions


# ============================================================
# Hard mode V2: dedicated aisle assignments
# ============================================================

_hard = {
    "homes": {},         # bot_id -> home aisle x
    "home_shelves": {},  # bot_id -> [(shelf_x, shelf_y), ...]
    "order_id": None,    # last active order id
    "initialized": False,
}


def _init_hard_assignments(state):
    """Assign each bot to a home aisle based on shelf layout."""
    global _hard
    items = state["items"]
    bots = state["bots"]
    drop_off = tuple(state["drop_off"])

    # Detect aisle x-coordinates (walkable corridors between shelf columns)
    shelf_xs = sorted(set(item["position"][0] for item in items))
    # Aisles are gaps between adjacent shelf columns
    aisle_xs = []
    for i in range(len(shelf_xs) - 1):
        if shelf_xs[i + 1] - shelf_xs[i] == 2:
            aisle_xs.append(shelf_xs[i] + 1)

    # Assign bots to aisles: spread across, double up on closest to dropoff
    # Sort aisles by distance to dropoff (closest first)
    aisle_xs.sort(key=lambda x: abs(x - drop_off[0]))
    n_bots = len(bots)
    bot_ids = sorted(b["id"] for b in bots)

    homes = {}
    if n_bots <= len(aisle_xs):
        for i, bid in enumerate(bot_ids):
            homes[bid] = aisle_xs[i]
    else:
        # More bots than aisles: distribute evenly, extra bots to closest aisles
        for i, bid in enumerate(bot_ids):
            homes[bid] = aisle_xs[i % len(aisle_xs)]

    # Map each bot to shelves accessible from their aisle (x-1 and x+1)
    home_shelves = {}
    for bid, ax in homes.items():
        shelves = []
        for item in items:
            sx = item["position"][0]
            if sx == ax - 1 or sx == ax + 1:
                shelves.append(tuple(item["position"]))
        home_shelves[bid] = list(set(shelves))  # deduplicate

    _hard["homes"] = homes
    _hard["home_shelves"] = home_shelves
    _hard["aisle_xs"] = aisle_xs
    _hard["order_id"] = None
    _hard["initialized"] = True


def _find_item_on_home_shelves(bid, items, needed_type):
    """Find an item of needed_type on bot's home aisle shelves."""
    home_shelves = _hard["home_shelves"].get(bid, [])
    home_x = _hard["homes"].get(bid)
    if not home_x:
        return None

    for item in items:
        if item["type"] == needed_type:
            sx = item["position"][0]
            if sx == home_x - 1 or sx == home_x + 1:
                return item
    return None


def _find_any_item_on_home(bid, items, needed_types, exclude_types=None):
    """Find any item matching needed_types on bot's home aisle. Returns cheapest."""
    global _dm
    home_x = _hard["homes"].get(bid)
    if not home_x:
        return None

    best = None
    best_cost = float("inf")
    for item in items:
        if item["type"] not in needed_types:
            continue
        if exclude_types and item["type"] in exclude_types:
            continue
        sx = item["position"][0]
        if sx == home_x - 1 or sx == home_x + 1:
            shelf = tuple(item["position"])
            cost = _dm.trip_cost((home_x, 6), shelf) if _dm else 0  # rough cost from mid-aisle
            if cost < best_cost:
                best_cost = cost
                best = item
    return best


def _decide_actions_hard_v2(state):
    """Hard mode: each bot owns a home aisle and only picks from there.

    Strategy:
    - 5 bots spread across 4 aisles (closest aisle gets 2 bots)
    - Each bot keeps inventory full with items from their home aisle
    - Priority: active order items > preview items > any item from home aisle
    - Only bots holding active order items deliver to dropoff
    - After delivery, return to home aisle and refill
    """
    global _dm, _hard

    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = tuple(state["drop_off"])

    if not _hard["initialized"]:
        _init_hard_assignments(state)

    active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
    if not active:
        return [{"bot": b["id"], "action": "wait"} for b in bots]

    preview = next((o for o in orders if o.get("status") == "preview" and not o["complete"]), None)

    active_needed = get_needed_items(active)
    preview_needed = get_needed_items(preview) if preview else {}

    # Subtract items already in bot inventories from needed counts
    active_remaining = dict(active_needed)
    preview_remaining = dict(preview_needed)
    for bot in bots:
        for itype in bot["inventory"]:
            if itype in active_remaining and active_remaining[itype] > 0:
                active_remaining[itype] -= 1
            elif itype in preview_remaining and preview_remaining[itype] > 0:
                preview_remaining[itype] -= 1
    active_remaining = {k: v for k, v in active_remaining.items() if v > 0}
    preview_remaining = {k: v for k, v in preview_remaining.items() if v > 0}

    # Update position history for anti-oscillation
    for bot in bots:
        bid = bot["id"]
        pos = tuple(bot["position"])
        if bid not in _bot_history:
            _bot_history[bid] = []
        hist = _bot_history[bid]
        hist.append(pos)
        if len(hist) > 8:
            hist.pop(0)

    # Identify which bots have active order items to deliver
    delivering_bots = set()
    for bot in bots:
        if any(active_needed.get(it, 0) > 0 for it in bot["inventory"]):
            delivering_bots.add(bot["id"])

    # Process bots: delivering bots first for lock priority
    locked = set(tuple(b["position"]) for b in bots)
    actions = []

    # Sort: delivering bots first (by distance to dropoff), then others by ID
    sorted_bots = sorted(bots, key=lambda b: (
        0 if b["id"] in delivering_bots else 1,
        _dm.dist(tuple(b["position"]), drop_off) if _dm and b["id"] in delivering_bots else b["id"]
    ))

    # Track which active items are being picked this round (prevent duplicates)
    picking_this_round = []

    for bot in sorted_bots:
        bid = bot["id"]
        pos = tuple(bot["position"])
        inv = bot["inventory"]
        home_x = _hard["homes"].get(bid, 4)
        locked.discard(pos)

        has_active_items = any(active_needed.get(it, 0) > 0 for it in inv)

        # PHASE 1: At dropoff with useful items → drop_off
        if pos == drop_off and has_active_items:
            action = {"bot": bid, "action": "drop_off"}

        # PHASE 2: Has active items + full or no more to pick → deliver
        elif has_active_items and (len(inv) >= 3 or not active_remaining):
            action = _navigate_locked(bid, pos, drop_off, state, locked, urgent=True)

        # PHASE 3: Has active items + items still needed → pick more if close, else deliver
        elif has_active_items and active_remaining:
            # Check if any needed item is on home aisle and close
            found_close = None
            for atype, cnt in active_remaining.items():
                if cnt <= 0 or atype in [p for p in picking_this_round]:
                    continue
                item = _find_item_on_home_shelves(bid, items, atype)
                if item:
                    shelf = tuple(item["position"])
                    adj, _ = _dm.best_adjacent(pos, shelf)
                    if adj and _dm.dist(pos, adj) <= 3:  # close enough to pick first
                        found_close = (item, adj)
                        break

            if found_close and len(inv) < 3:
                item, adj = found_close
                if pos == adj or (abs(pos[0] - item["position"][0]) + abs(pos[1] - item["position"][1]) == 1):
                    picking_this_round.append(item["type"])
                    action = {"bot": bid, "action": "pick_up", "item_id": item["id"]}
                else:
                    action = _navigate_locked(bid, pos, adj, state, locked)
            else:
                action = _navigate_locked(bid, pos, drop_off, state, locked, urgent=True)

        # PHASE 4: Not delivering, inventory not full → pick items
        elif len(inv) < 3:
            action = _hard_pick_item(bid, pos, items, active_remaining, preview_remaining,
                                     picking_this_round, state, locked)

        # PHASE 5: Full inventory, no active items → park at home aisle
        else:
            park_pos = (home_x, 1)  # top of home aisle
            if pos != park_pos:
                action = _navigate_locked(bid, pos, park_pos, state, locked)
            else:
                action = {"bot": bid, "action": "wait"}

        actions.append(action)

        # Lock target position
        act_str = action["action"]
        if act_str.startswith("move_"):
            dx, dy = {"move_right": (1, 0), "move_left": (-1, 0),
                      "move_down": (0, 1), "move_up": (0, -1)}[act_str]
            locked.add((pos[0] + dx, pos[1] + dy))
        else:
            locked.add(pos)

    # Swap and same-dest conflict resolution (reuse from multi)
    _MOVE_DELTA = {"move_right": (1, 0), "move_left": (-1, 0),
                   "move_down": (0, 1), "move_up": (0, -1)}
    bot_pos = {b["id"]: tuple(b["position"]) for b in bots}
    bot_dest = {}
    for a in actions:
        bid = a["bot"]
        act = a["action"]
        if act in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[act]
            bot_dest[bid] = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
        else:
            bot_dest[bid] = bot_pos[bid]

    action_map = {a["bot"]: a for a in actions}
    for i, a1 in enumerate(actions):
        for a2 in actions[i+1:]:
            b1, b2 = a1["bot"], a2["bot"]
            if bot_dest[b1] == bot_pos[b2] and bot_dest[b2] == bot_pos[b1]:
                waiter = max(b1, b2)
                action_map[waiter]["action"] = "wait"
                action_map[waiter].pop("item_id", None)

    dest_to_bots = {}
    for a in actions:
        bid = a["bot"]
        if a["action"] in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[a["action"]]
            dest = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
            dest_to_bots.setdefault(dest, []).append(bid)
    for dest, bids in dest_to_bots.items():
        if len(bids) > 1:
            bids.sort()
            for bid in bids[1:]:
                action_map[bid]["action"] = "wait"
                action_map[bid].pop("item_id", None)

    return actions


def _hard_pick_item(bid, pos, items, active_remaining, preview_remaining,
                    picking_this_round, state, locked):
    """Pick the best item from bot's home aisle. Priority: active > preview > speculative."""
    global _dm

    # 1. Try active order items from home aisle
    for atype, cnt in active_remaining.items():
        if cnt <= 0:
            continue
        # Don't double-pick same type this round
        pick_count = picking_this_round.count(atype)
        if pick_count >= cnt:
            continue
        item = _find_item_on_home_shelves(bid, items, atype)
        if item:
            shelf = tuple(item["position"])
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                picking_this_round.append(item["type"])
                return {"bot": bid, "action": "pick_up", "item_id": item["id"]}
            adj, _ = _dm.best_adjacent(pos, shelf)
            if adj:
                return _navigate_locked(bid, pos, adj, state, locked)

    # 2. Try preview order items from home aisle
    for ptype, cnt in preview_remaining.items():
        if cnt <= 0:
            continue
        pick_count = picking_this_round.count(ptype)
        if pick_count >= cnt:
            continue
        item = _find_item_on_home_shelves(bid, items, ptype)
        if item:
            shelf = tuple(item["position"])
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                picking_this_round.append(item["type"])
                return {"bot": bid, "action": "pick_up", "item_id": item["id"]}
            adj, _ = _dm.best_adjacent(pos, shelf)
            if adj:
                return _navigate_locked(bid, pos, adj, state, locked)

    # 3. Speculative: pick ANY item from home aisle not already held by any bot
    held_types = set()
    for b in state["bots"]:
        for it in b["inventory"]:
            held_types.add(it)
    for pt in picking_this_round:
        held_types.add(pt)

    home_x = _hard["homes"].get(bid)
    best_item = None
    best_cost = float("inf")
    for item in items:
        if item["type"] in held_types:
            continue
        sx = item["position"][0]
        if sx != home_x - 1 and sx != home_x + 1:
            continue
        shelf = tuple(item["position"])
        cost = _dm.dist(pos, _dm.best_adjacent(pos, shelf)[0]) if _dm else 99
        if cost < best_cost:
            best_cost = cost
            best_item = item

    if best_item:
        shelf = tuple(best_item["position"])
        if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
            picking_this_round.append(best_item["type"])
            return {"bot": bid, "action": "pick_up", "item_id": best_item["id"]}
        adj, _ = _dm.best_adjacent(pos, shelf)
        if adj:
            return _navigate_locked(bid, pos, adj, state, locked)

    # 4. Nothing to pick on home aisle → pick active items from ANY aisle
    for atype, cnt in active_remaining.items():
        if cnt <= 0:
            continue
        pick_count = picking_this_round.count(atype)
        if pick_count >= cnt:
            continue
        best_item = None
        best_cost = float("inf")
        for item in items:
            if item["type"] == atype:
                shelf = tuple(item["position"])
                cost = _dm.trip_cost(pos, shelf)
                if cost < best_cost:
                    best_cost = cost
                    best_item = item
        if best_item:
            shelf = tuple(best_item["position"])
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                picking_this_round.append(best_item["type"])
                return {"bot": bid, "action": "pick_up", "item_id": best_item["id"]}
            adj, _ = _dm.best_adjacent(pos, shelf)
            if adj:
                return _navigate_locked(bid, pos, adj, state, locked)

    # Nothing to do → park at home
    home_x = _hard["homes"].get(bid, pos[0])
    park_pos = (home_x, 1)
    if pos != park_pos:
        return _navigate_locked(bid, pos, park_pos, state, locked)
    return {"bot": bid, "action": "wait"}


# ============================================================
# Expert mode: multi-bot with expert-specific tuning
# ============================================================

def _decide_actions_expert(state):
    """Expert mode: multi-bot logic with tuned parameters for 28x18 grid."""
    # Set expert-specific defaults (larger grid needs wider detours)
    saved = {}
    expert_defaults = {"max_detour": 14}
    for k, v in expert_defaults.items():
        if k not in _PARAMS:
            saved[k] = None
            _PARAMS[k] = v

    actions = _decide_actions_multi(state)

    # Restore (don't persist expert defaults if not explicitly set)
    for k, v in saved.items():
        if v is None:
            _PARAMS.pop(k, None)

    return actions


def decide_actions(state, game_plan=None):
    """Decide actions for all bots. Dispatches to single or multi-bot logic."""
    global _dm, _bot_routes, _multi

    # Optimized plan: pre-computed per-bot actions
    if game_plan and isinstance(game_plan, dict) and game_plan.get("type") == "optimized":
        if state["round"] == 0:
            _dm = DistanceMatrix(state)
        return _execute_optimized_plan(state, game_plan)

    if state["round"] == 0:
        _dm = DistanceMatrix(state)
        _bot_routes = {}
        _multi = {"order_id": None, "active_assignments": {}, "preview_assignments": {}, "last_delivered_count": 0, "bot_aisle": {}, "aisle_xs": [], "inv_baseline": {}, "future_orders": []}
        _hard["initialized"] = False
        _bot_history.clear()
        if game_plan and isinstance(game_plan, list):
            _multi["future_orders"] = game_plan

    if len(state["bots"]) >= 10:
        actions = _decide_actions_expert(state)
    elif len(state["bots"]) == 1:
        actions = _decide_actions_single(state)
    else:
        actions = _decide_actions_multi(state)

    # Apply per-round overrides from perturbation search
    overrides = _PARAMS.get("overrides")
    if overrides:
        rnd = state["round"]
        for a in actions:
            key = f"{rnd},{a['bot']}"
            if key in overrides:
                a["action"] = overrides[key]
                a.pop("item_id", None)

    return actions


def _decide_actions_single(state):
    """Single-bot logic (easy mode). One bot handles everything."""
    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = tuple(state["drop_off"])

    active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
    preview = next((o for o in orders if o.get("status") == "preview" and not o["complete"]), None)

    active_needed = get_needed_items(active) if active else {}
    active_from_map = items_still_needed_from_map(active, bots) if active else {}
    preview_needed = get_needed_items(preview) if preview else {}

    claimed_items = set()
    action = _decide_single_bot(
        bots[0], bots, items, active, active_needed, active_from_map,
        preview, preview_needed, drop_off, state, claimed_items,
        multi_bot=False,
    )
    return [action]


def _decide_actions_multi(state):
    """Multi-bot: distribute order items across bots, lock-based collision avoidance."""
    global _dm, _multi

    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = tuple(state["drop_off"])

    active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
    if not active:
        return [{"bot": b["id"], "action": "wait"} for b in bots]

    preview = next((o for o in orders if o.get("status") == "preview" and not o["complete"]), None)

    # Compute bot-to-aisle mapping once per game
    if not _multi["bot_aisle"]:
        aisle_xs = _detect_aisles(items, state)
        _multi["aisle_xs"] = aisle_xs
        if aisle_xs:
            for i, bot in enumerate(bots):
                _multi["bot_aisle"][bot["id"]] = aisle_xs[i % len(aisle_xs)]

    # Clean stale assignments (items picked up since last round)
    _clean_stale_assignments(bots)

    # (Re)distribute when order changes OR partial delivery happened
    if _should_redistribute(active, bots):
        _distribute_items(bots, active, preview, items, drop_off)

    # Track preview picks within this round to coordinate between bots
    _multi["preview_picks_round"] = []

    # Update position history for anti-oscillation (all multi-bot modes)
    if len(bots) >= 2:
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            if bid not in _bot_history:
                _bot_history[bid] = []
            hist = _bot_history[bid]
            hist.append(pos)
            if len(hist) > 8:
                hist.pop(0)

    # Delivery queue: limit how many bots head to dropoff simultaneously
    max_delivering = _PARAMS.get("max_delivering", len(bots))
    if max_delivering < len(bots):
        active_needed = get_needed_items(active) if active else {}
        delivery_candidates = []
        for b in bots:
            b_useful = any(active_needed.get(it, 0) > 0 for it in b["inventory"])
            b_assigned = _multi["active_assignments"].get(b["id"], [])
            if b_useful and (len(b["inventory"]) >= 3 or not b_assigned):
                d = _dm.dist(tuple(b["position"]), drop_off) if _dm else 999
                delivery_candidates.append((d, b["id"]))
        delivery_candidates.sort()
        _multi["delivering_bids"] = {bid for _, bid in delivery_candidates[:max_delivering]}
    else:
        _multi["delivering_bids"] = None  # no limit

    # Lock all current bot positions; release each bot's cell on its turn
    locked = set(tuple(b["position"]) for b in bots)
    actions = []

    bot_order = _PARAMS.get("bot_order")
    if bot_order:
        order_map = {bid: i for i, bid in enumerate(bot_order)}
        sorted_bots = sorted(bots, key=lambda b: order_map.get(b["id"], b["id"]))
    else:
        sorted_bots = sorted(bots, key=lambda b: b["id"])
    for bot in sorted_bots:
        pos = tuple(bot["position"])
        locked.discard(pos)  # free our own cell

        action = _bot_step_multi(bot, bots, items, active, drop_off, state, locked)
        actions.append(action)

        # Lock the cell we'll occupy after this action
        act_str = action["action"]
        if act_str.startswith("move_"):
            dx, dy = {"move_right": (1, 0), "move_left": (-1, 0),
                      "move_down": (0, 1), "move_up": (0, -1)}[act_str]
            locked.add((pos[0] + dx, pos[1] + dy))
        else:
            locked.add(pos)

    # --- Simultaneous-safe conflict resolution ---
    # Fix swap conflicts: if bot A→B and bot B→A, the server (simultaneous)
    # blocks both. Make the higher-ID bot wait instead.
    _MOVE_DELTA = {"move_right": (1, 0), "move_left": (-1, 0),
                   "move_down": (0, 1), "move_up": (0, -1)}
    bot_pos = {b["id"]: tuple(b["position"]) for b in bots}
    bot_dest = {}
    for a in actions:
        bid = a["bot"]
        act = a["action"]
        if act in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[act]
            bot_dest[bid] = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
        else:
            bot_dest[bid] = bot_pos[bid]

    # Detect swaps: bot i moves to bot j's position AND bot j moves to bot i's position
    action_map = {a["bot"]: a for a in actions}
    for i, a1 in enumerate(actions):
        for a2 in actions[i+1:]:
            b1, b2 = a1["bot"], a2["bot"]
            if bot_dest[b1] == bot_pos[b2] and bot_dest[b2] == bot_pos[b1]:
                # Swap detected — higher-ID bot waits
                waiter = max(b1, b2)
                action_map[waiter]["action"] = "wait"
                action_map[waiter].pop("item_id", None)

    # Detect same-destination conflicts (two bots trying to move to same cell)
    dest_to_bots = {}
    for a in actions:
        bid = a["bot"]
        if a["action"] in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[a["action"]]
            dest = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
            dest_to_bots.setdefault(dest, []).append(bid)
    for dest, bids in dest_to_bots.items():
        if len(bids) > 1:
            # Multiple bots heading to same cell — keep lowest ID, others wait
            bids.sort()
            for bid in bids[1:]:
                action_map[bid]["action"] = "wait"
                action_map[bid].pop("item_id", None)

    return actions


def _distribute_items(bots, active, preview, items, drop_off):
    """Assign active + preview items to bots.

    Active items: min-makespan enumeration (same as before).
    Preview items: greedy assignment to bots with remaining capacity, cheapest first.
    """
    global _dm, _multi
    from itertools import product

    bot_ids = [b["id"] for b in bots]
    bot_map = {b["id"]: b for b in bots}

    # --- Active items: compute what still needs picking from shelves ---
    needed = get_needed_items(active)
    for bot in bots:
        for itype in bot["inventory"]:
            if itype in needed and needed[itype] > 0:
                needed[itype] -= 1
    needed = {k: v for k, v in needed.items() if v > 0}

    item_list = []
    for itype, count in needed.items():
        item_list.extend([itype] * count)

    # --- Min-makespan distribution of active items ---
    active_assignment = {bid: [] for bid in bot_ids}

    if item_list:
        def _estimate_bot_time(bid, assigned_types):
            """Estimate rounds for bot to pick assigned items and deliver."""
            if not assigned_types:
                return 0
            bot = bot_map[bid]
            bpos = tuple(bot["position"])
            inv = bot["inventory"]
            has_useful = any(needed.get(it, 0) > 0 for it in inv)

            shelves = []
            for itype in assigned_types:
                best_shelf = None
                best_cost = float("inf")
                for item in items:
                    if item["type"] == itype:
                        shelf = tuple(item["position"])
                        trip = _dm.trip_cost(bpos, shelf)
                        if trip < best_cost:
                            best_cost = trip
                            best_shelf = shelf
                if best_shelf:
                    shelves.append(best_shelf)

            if len(shelves) != len(assigned_types):
                return 999

            if has_useful:
                deliver_time = _dm.dist(bpos, drop_off) + 1
                pick_time = _tsp_cost_revisit(drop_off, shelves, drop_off)
                return deliver_time + pick_time
            else:
                return _tsp_cost_revisit(bpos, shelves, drop_off)

        n_items = len(item_list)
        n_bots = len(bots)

        if n_bots > 5:
            # Greedy min-makespan for many bots (expert mode: 10 bots, 4-6 items)
            # O(n_items^2 * n_bots) instead of O(n_bots^n_items)
            assignment = {bid: [] for bid in bot_ids}
            remaining = list(range(n_items))

            for _ in range(n_items):
                best_choice = None  # (item_idx, bid, resulting_makespan, trip_cost)
                for ii in remaining:
                    itype = item_list[ii]
                    for bid in bot_ids:
                        if len(bot_map[bid]["inventory"]) + len(assignment[bid]) >= 3:
                            continue
                        # Compute makespan if we assign this item to this bot
                        trial = assignment[bid] + [itype]
                        bot_time = _estimate_bot_time(bid, trial)
                        if bot_time >= 999:
                            continue
                        # Makespan = max over all bots
                        ms = bot_time
                        for other_bid in bot_ids:
                            if other_bid == bid:
                                continue
                            t = _estimate_bot_time(other_bid, assignment[other_bid])
                            if t > ms:
                                ms = t
                        # Also account for bots with useful inventory but no assignment
                        for bot in bots:
                            obid = bot["id"]
                            has_assign = (obid == bid and trial) or (obid != bid and assignment[obid])
                            if not has_assign and any(needed.get(it, 0) > 0 for it in bot["inventory"]):
                                dt = _dm.dist(tuple(bot["position"]), drop_off) + 1
                                if dt > ms:
                                    ms = dt
                        if best_choice is None or ms < best_choice[2] or (ms == best_choice[2] and bot_time < best_choice[3]):
                            best_choice = (ii, bid, ms, bot_time)

                if best_choice is None:
                    break
                ii, bid, _, _ = best_choice
                assignment[bid].append(item_list[ii])
                remaining.remove(ii)

            if not remaining:
                active_assignment = assignment
        else:
            # Exhaustive min-makespan for <=5 bots (easy/medium/hard)
            best_assignment = None
            best_makespan = float("inf")
            seen = set()

            for combo in product(range(n_bots), repeat=n_items):
                assignment = {bid: [] for bid in bot_ids}
                valid = True
                for i, bi in enumerate(combo):
                    bid = bot_ids[bi]
                    if len(bot_map[bid]["inventory"]) + len(assignment[bid]) >= 3:
                        valid = False
                        break
                    assignment[bid].append(item_list[i])
                if not valid:
                    continue

                key = tuple(tuple(sorted(assignment[bid])) for bid in bot_ids)
                if key in seen:
                    continue
                seen.add(key)

                makespan = 0
                for bid in bot_ids:
                    t = _estimate_bot_time(bid, assignment[bid])
                    if t >= 999:
                        makespan = 999
                        break
                    makespan = max(makespan, t)

                for bot in bots:
                    bid = bot["id"]
                    if not assignment[bid] and any(needed.get(it, 0) > 0 for it in bot["inventory"]):
                        makespan = max(makespan, _dm.dist(tuple(bot["position"]), drop_off) + 1)

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_assignment = assignment

            if best_assignment:
                active_assignment = best_assignment

    # --- Preview items: combined-trip assignment ---
    # Assign preview items to bots with active assignments ONLY when marginal cost is low.
    # This lets bots pick up preview items on the same trip as active items.
    # Remaining preview items are left for dynamic idle picking.
    preview_assignment = {bid: [] for bid in bot_ids}

    if preview:
        prev_needed = get_needed_items(preview)
        for bot in bots:
            for itype in bot["inventory"]:
                if itype in prev_needed and prev_needed[itype] > 0:
                    prev_needed[itype] -= 1
        prev_needed = {k: v for k, v in prev_needed.items() if v > 0}

        if prev_needed:
            # For each bot with active assignments and free slots, try adding preview items
            for bid in bot_ids:
                if not active_assignment[bid]:
                    continue
                bot = bot_map[bid]
                bpos = tuple(bot["position"])
                free_slots = 3 - len(bot["inventory"]) - len(active_assignment[bid])
                if free_slots <= 0:
                    continue

                # Build active shelves for TSP cost computation
                active_shelves = []
                for itype in active_assignment[bid]:
                    item_obj, _ = _find_cheapest_shelf(bpos, itype, items)
                    if item_obj:
                        active_shelves.append(tuple(item_obj["position"]))
                active_cost = _tsp_cost_revisit(bpos, active_shelves, drop_off)

                # Try adding preview items with low marginal cost
                for ptype in list(prev_needed.keys()):
                    if prev_needed.get(ptype, 0) <= 0 or free_slots <= 0:
                        continue
                    pitem, _ = _find_cheapest_shelf(bpos, ptype, items)
                    if not pitem:
                        continue
                    preview_shelf = tuple(pitem["position"])
                    combined_shelves = active_shelves + [preview_shelf]
                    combined_cost = _tsp_cost_revisit(bpos, combined_shelves, drop_off)
                    marginal_cost = combined_cost - active_cost
                    if marginal_cost <= _PARAMS.get("marginal_cost", 3):
                        preview_assignment[bid].append(ptype)
                        prev_needed[ptype] -= 1
                        free_slots -= 1
                        active_shelves = combined_shelves
                        active_cost = combined_cost

    # Store state
    _multi["order_id"] = active["id"]
    _multi["active_assignments"] = active_assignment
    _multi["preview_assignments"] = preview_assignment
    _multi["last_delivered_count"] = len(active["items_delivered"])
    _multi["inv_baseline"] = {b["id"]: list(b["inventory"]) for b in bots}


def _bot_step_multi(bot, all_bots, items, active, drop_off, state, locked):
    """Decide one bot's action in multi-bot mode.

    Priority: clear dropoff > deliver > pick up assigned items > preview > park.
    """
    global _dm, _multi

    bid = bot["id"]
    pos = tuple(bot["position"])
    inventory = bot["inventory"]
    assigned = _multi["active_assignments"].get(bid, [])
    preview_assigned = _multi["preview_assignments"].get(bid, [])

    active_needed = get_needed_items(active) if active else {}
    has_useful = any(active_needed.get(it, 0) > 0 for it in inventory)

    # --- Expert: yield x=1 corridor to delivering bots ---
    # If idle bot on x=1 (y 10-15) and a delivering bot also on x=1 heading south, move east
    if len(all_bots) >= 10 and not has_useful and pos[0] == 1 and 10 <= pos[1] <= 15:
        for other in all_bots:
            if other["id"] == bid:
                continue
            opos = tuple(other["position"])
            o_useful = any(active_needed.get(it, 0) > 0 for it in other["inventory"])
            # Delivering bot on x=1, north of us or same row, heading toward dropoff
            if o_useful and opos[0] <= 2 and opos[1] <= pos[1]:
                # Yield: move east off x=1
                nxt = (pos[0] + 1, pos[1])
                if _dm and _dm.dist(pos, nxt) == 1 and nxt not in locked:
                    return {"bot": bid, "action": "move_right"}
                break

    # --- Clear dropoff area when carrying useless items ---
    # Skip clearing if items match preview order (stay near dropoff for fast next delivery)
    if not has_useful and inventory and (pos == drop_off or pos[1] == drop_off[1]):
        preview_order = next(
            (o for o in state["orders"] if o.get("status") == "preview" and not o["complete"]),
            None,
        )
        has_preview_useful = False
        if preview_order:
            pn = get_needed_items(preview_order)
            has_preview_useful = any(pn.get(it, 0) > 0 for it in inventory)
        if not has_preview_useful:
            # Expert at x<=2: prefer clearing EAST to avoid x=1 dead-end corridor
            if len(all_bots) >= 10 and pos[0] <= 2:
                clear_dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
            else:
                clear_dirs = [(0, -1), (1, 0), (-1, 0), (0, 1)]
            for dx, dy in clear_dirs:
                nxt = (pos[0] + dx, pos[1] + dy)
                if _dm and _dm.dist(pos, nxt) == 1 and nxt not in locked:
                    return {"bot": bid, "action": _pos_to_action(pos, nxt)}
            return {"bot": bid, "action": "wait"}

    # --- At dropoff with useful items → deliver ---
    if has_useful and pos == drop_off:
        return {"bot": bid, "action": "drop_off"}

    # --- Useful items + (full inventory OR no active assigned) → go deliver ---
    # Delivery queue: defer non-priority bots from delivering
    delivering_bids = _multi.get("delivering_bids")
    delivery_deferred = (delivering_bids is not None and bid not in delivering_bids
                         and has_useful and not assigned and len(inventory) < 3)
    if not delivery_deferred and has_useful and (len(inventory) >= 3 or not assigned):
        # On-route preview pick: grab a preview item if it's on/near the delivery path
        if len(inventory) < 3:
            preview = next(
                (o for o in state["orders"] if o.get("status") == "preview" and not o["complete"]),
                None,
            )
            if preview:
                prev_needed = get_needed_items(preview)
                for b in all_bots:
                    for it in b["inventory"]:
                        if it in prev_needed and prev_needed[it] > 0:
                            prev_needed[it] -= 1
                for pt in _multi.get("preview_picks_round", []):
                    if pt in prev_needed and prev_needed[pt] > 0:
                        prev_needed[pt] -= 1
                # Check adjacent first (free pickup)
                for item in items:
                    if item["type"] in prev_needed and prev_needed.get(item["type"], 0) > 0:
                        shelf = tuple(item["position"])
                        if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                            _multi.setdefault("preview_picks_round", []).append(item["type"])
                            return {"bot": bid, "action": "pick_up", "item_id": item["id"]}
        # Expert: unlimited detour for delivery (10 bots clog y=1 corridor)
        return _navigate_locked(bid, pos, drop_off, state, locked,
                                urgent=(len(all_bots) >= 10))

    # --- Pick up next assigned active item ---
    if assigned and len(inventory) < 3:
        # Opportunistic: if adjacent to ANY assigned item (active or preview), pick it up
        for assigned_list in [assigned, preview_assigned]:
            for ai, atype in enumerate(assigned_list):
                for item in items:
                    if item["type"] == atype:
                        shelf = tuple(item["position"])
                        if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                            assigned_list.pop(ai)
                            return {"bot": bid, "action": "pick_up", "item_id": item["id"]}

        # Navigate to cheapest ACTIVE assigned item (don't detour for preview)
        target_type = assigned[0]
        best_item = None
        best_cost = float("inf")
        for item in items:
            if item["type"] == target_type:
                shelf = tuple(item["position"])
                trip = _dm.trip_cost(pos, shelf)
                if trip < best_cost:
                    best_cost = trip
                    best_item = item

        if best_item:
            shelf = tuple(best_item["position"])
            adj, _ = _dm.best_adjacent(pos, shelf)
            if adj:
                return _navigate_locked(bid, pos, adj, state, locked)

    # --- Preview pre-picking: idle bots (or delivery-deferred bots) pick next order's items ---
    if (not assigned and not has_useful and len(inventory) < 3) or delivery_deferred:
        # First try: items for the visible preview order
        preview = next(
            (o for o in state["orders"] if o.get("status") == "preview" and not o["complete"]),
            None,
        )

        pick_needed = None
        pick_is_speculative = False
        if preview:
            preview_needed = get_needed_items(preview)
            for b in all_bots:
                for itype in b["inventory"]:
                    if itype in preview_needed and preview_needed[itype] > 0:
                        preview_needed[itype] -= 1
            for picked_type in _multi.get("preview_picks_round", []):
                if picked_type in preview_needed and preview_needed[picked_type] > 0:
                    preview_needed[picked_type] -= 1
            preview_needed = {k: v for k, v in preview_needed.items() if v > 0}
            if preview_needed:
                pick_needed = preview_needed

        # Fallback: speculative picking of cheap items (5+ bots only)
        if pick_needed is None and len(all_bots) >= 5:
            held_types = set()
            for b in all_bots:
                for itype in b["inventory"]:
                    held_types.add(itype)
            for picked_type in _multi.get("preview_picks_round", []):
                held_types.add(picked_type)
            # Expert: lower threshold — only pick cheap items to keep slots free for active orders
            spec_threshold = _PARAMS.get("spec_threshold", 35 if len(all_bots) >= 10 else 15)
            spec_needed = {}
            for item in items:
                itype = item["type"]
                if itype not in held_types:
                    shelf = tuple(item["position"])
                    cost = _dm.trip_cost(drop_off, shelf)
                    if cost <= spec_threshold and (itype not in spec_needed or cost < spec_needed[itype]):
                        spec_needed[itype] = cost
            if spec_needed:
                pick_needed = {k: 1 for k in spec_needed}
                pick_is_speculative = True

        if pick_needed:
            best_item = None
            best_cost = float("inf")
            home_x = _multi["bot_aisle"].get(bid)
            for item in items:
                if item["type"] not in pick_needed:
                    continue
                shelf = tuple(item["position"])
                cost = _dm.trip_cost(pos, shelf)
                # For speculative picks only: prefer home aisle to spread bots (avoid convergence deadlocks)
                home_penalty = _PARAMS.get("home_aisle_penalty", 0)
                if pick_is_speculative and home_penalty and home_x and len(all_bots) >= 5:
                    shelf_x = shelf[0]
                    on_home = (shelf_x == home_x - 1 or shelf_x == home_x + 1)
                    if not on_home:
                        cost += home_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_item = item

            if best_item:
                shelf = tuple(best_item["position"])
                if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                    _multi.setdefault("preview_picks_round", []).append(best_item["type"])
                    return {"bot": bid, "action": "pick_up", "item_id": best_item["id"]}
                adj, _ = _dm.best_adjacent(pos, shelf)
                if adj:
                    return _navigate_locked(bid, pos, adj, state, locked)

    # Default: park out of delivery paths
    home_x = _multi["bot_aisle"].get(bid, pos[0])
    # Expert (10+ bots, 28x18 grid): park at y=9 (middle corridor, closer to delivery)
    # Others: park at y=1 (top corridor, out of the way)
    park_y = 9 if len(all_bots) >= 10 else 1
    park_pos = (home_x, park_y)

    if pos != park_pos:
        return _navigate_locked(bid, pos, park_pos, state, locked)

    return {"bot": bid, "action": "wait"}


def _navigate_locked(bid, pos, target, state, locked, urgent=False):
    """Navigate toward target, treating locked cells as impassable.

    Uses DM gradient descent first; falls back to BFS around locked cells.
    The locked set has correct updated positions (not stale bot positions).
    If urgent=True, accept any detour length (delivering bot must reach target).
    """
    global _dm

    if pos == target:
        return {"bot": bid, "action": "wait"}

    n_bots = len(state.get("bots", []))

    # DM gradient descent — skip if next cell is locked
    if _dm:
        nxt = _dm.next_step(pos, target)
        if nxt and nxt != pos and nxt not in locked:
            return {"bot": bid, "action": _pos_to_action(pos, nxt)}

    # BFS with walls + shelves + locked (NOT build_blocked_set which has stale bot positions)
    blocked = set()
    for w in state["grid"]["walls"]:
        blocked.add((w[0], w[1]))
    for item in state["items"]:
        blocked.add((item["position"][0], item["position"][1]))
    blocked.update(locked)

    path = bfs(pos, target, blocked, state["grid"]["width"], state["grid"]["height"])
    if path and len(path) >= 2:
        direct = _dm.dist(pos, target) if _dm else len(path) - 1
        # Only take BFS detour if it's not wildly longer than direct path.
        # Massive detours (through y=1) waste 20+ rounds — better to wait 1-2 rounds.
        # Expert (28x18): allow wider detours since grid is bigger and more congested.
        # Urgent (delivering): accept any detour — must reach dropoff.
        if urgent:
            max_detour = 999
        else:
            max_detour = _PARAMS.get("max_detour", 10 if len(state.get("bots", [])) >= 10 else 6)
        if len(path) - 1 <= direct + max_detour:
            return {"bot": bid, "action": path_to_action(pos, path[1])}

    # Deadlock breaker: prefer closer moves, then sideways, then away (prevents permanent stall)
    # Anti-oscillation: penalize recently visited cells for expert mode.
    if _dm:
        cur_dist = _dm.dist(pos, target)
        hist = _bot_history.get(bid, [])
        recent = set(hist)
        # Detect if bot is stuck in a cycle (oscillating between 1-2 cells for 8 rounds)
        stuck_in_cycle = (n_bots >= 2 and len(hist) >= 8
                          and len(set(hist[-8:])) <= 2)
        options = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (pos[0] + dx, pos[1] + dy)
            if _dm.dist(pos, nxt) == 1 and nxt not in locked:
                d = _dm.dist(nxt, target)
                # Priority: closer=0, same=1, away=2
                priority = 0 if d < cur_dist else (1 if d == cur_dist else 2)
                # Anti-oscillation: only activate when stuck in a cycle
                if stuck_in_cycle and nxt in recent:
                    priority += 3
                elif n_bots >= 10 and nxt in recent:
                    priority += 3
                options.append((priority, d, nxt))
        if options:
            options.sort()
            best = options[0]
            if best[0] <= 1:  # closer or sideways (not recently visited)
                return {"bot": bid, "action": _pos_to_action(pos, best[2])}
            # All normal options move away — wait some of the time to prevent oscillation
            # Expert: always move (10 bots in narrow aisles → permanent deadlocks if waiting)
            # Others: move 50% of the time
            n_bots = len(state.get("bots", []))
            if n_bots >= 10 or state.get("round", 0) % 2 == 0:
                return {"bot": bid, "action": _pos_to_action(pos, best[2])}

    return {"bot": bid, "action": "wait"}


# ============================================================




# ============================================================

def _decide_single_bot(bot, all_bots, items, active_order, active_needed,
                       active_from_map, preview_order, preview_needed,
                       drop_off, state, claimed_items, multi_bot=False):
    """Decide action for a single bot using route-following state machine."""
    global _dm, _bot_routes

    bid = bot["id"]
    pos = (bot["position"][0], bot["position"][1])
    inventory = bot["inventory"]

    if not active_order:
        return {"bot": bid, "action": "wait"}

    has_useful = any(active_needed.get(item, 0) > 0 for item in inventory)
    all_active_covered = len(active_from_map) == 0

    # --- PHASE 0 (multi-bot): Clear the dropoff and delivery corridor ---
    # Bots with no useful items must not block the dropoff or approach path.
    if multi_bot and not has_useful and inventory:
        drop_y = drop_off[1]
        if pos == drop_off or pos[1] == drop_y:
            # Move up off the delivery row to make room
            _bot_routes.pop(bid, None)
            nxt = (pos[0], pos[1] - 1)
            if _dm and _dm.dist(pos, nxt) == 1:
                occupied = any(tuple(b["position"]) == nxt for b in all_bots if b["id"] != bid)
                if not occupied:
                    return {"bot": bid, "action": "move_up"}
            # Try moving right instead if up is blocked
            nxt = (pos[0] + 1, pos[1])
            if _dm and _dm.dist(pos, nxt) == 1:
                occupied = any(tuple(b["position"]) == nxt for b in all_bots if b["id"] != bid)
                if not occupied:
                    return {"bot": bid, "action": "move_right"}

    # --- PHASE 1: Drop off if at dropoff with useful items ---
    if has_useful and pos == drop_off:
        _bot_routes.pop(bid, None)
        return {"bot": bid, "action": "drop_off"}

    # --- PHASE 2: Full inventory with useful items → deliver ---
    if has_useful and len(inventory) >= 3:
        _bot_routes.pop(bid, None)
        return _navigate_dm(bid, pos, drop_off, state)

    # --- Invalidate route if order changed ---
    active_id = active_order["id"] if active_order else None
    route_info = _bot_routes.get(bid)
    if route_info and route_info.get("order_id") != active_id:
        _bot_routes.pop(bid, None)
        route_info = None

    # --- PHASE 3: Follow existing route ---
    if route_info and route_info["steps"]:
        step = route_info["steps"][0]

        if step["type"] == "pickup":
            shelf = step["shelf"]
            adj = step["adj"]
            item_obj = step["item"]

            # Adjacent to shelf → pick up
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                route_info["steps"].pop(0)
                claimed_items.add(item_obj["id"])
                return {"bot": bid, "action": "pick_up", "item_id": item_obj["id"]}

            # Navigate to adj cell
            return _navigate_dm(bid, pos, adj, state)

        elif step["type"] == "deliver":
            if pos == drop_off:
                route_info["steps"].pop(0)
                _bot_routes.pop(bid, None)
                return {"bot": bid, "action": "drop_off"}
            return _navigate_dm(bid, pos, drop_off, state)

    # --- PHASE 4: Compute new batch & route ---
    free_slots = 3 - len(inventory)

    batch = []
    if not all_active_covered and free_slots > 0:
        trip_needs = active_from_map
        total_active_needed = sum(active_from_map.values())

        # Use CVRP planning for orders needing more items than capacity
        if total_active_needed > free_slots:
            trips = _plan_order_trips(active_from_map, items, pos, drop_off)
            trip_needs = trips[0]

        # Get batch for this trip's needs
        active_batch = _find_best_batch(pos, items, trip_needs, claimed_items, drop_off, free_slots)

        # Try to fill remaining slots with preview items (only if detour is cheap)
        combined_batch = active_batch
        if active_batch and preview_needed and len(active_batch) < free_slots:
            combined = dict(trip_needs)
            for k, v in preview_needed.items():
                combined[k] = combined.get(k, 0) + v
            candidate = _find_best_batch_combined(
                pos, items, trip_needs, combined, claimed_items, drop_off, free_slots)
            if candidate and len(candidate) > len(active_batch):
                # Check marginal cost: reject if preview detour is too expensive
                active_shelves = [(it["position"][0], it["position"][1]) for it in active_batch]
                combined_shelves = [(it["position"][0], it["position"][1]) for it in candidate]
                active_cost = _tsp_cost_revisit(pos, active_shelves, drop_off)
                combined_cost = _tsp_cost_revisit(pos, combined_shelves, drop_off)
                marginal_cost = combined_cost - active_cost
                n_preview = len(candidate) - len(active_batch)
                if marginal_cost <= 6 * n_preview:
                    combined_batch = candidate
        batch = combined_batch

    elif all_active_covered and preview_needed and free_slots > 0:
        # All active covered, just pick preview items
        batch = _find_best_batch(pos, items, preview_needed, claimed_items, drop_off, free_slots)

    if batch:
        # Build and store route
        route = _build_route(pos, batch, drop_off)
        _bot_routes[bid] = {"steps": route, "order_id": active_id}
        # Claim batch items
        for it in batch:
            claimed_items.add(it["id"])
            itype = it["type"]
            if itype in active_from_map and active_from_map[itype] > 0:
                active_from_map[itype] -= 1
                if active_from_map[itype] <= 0:
                    del active_from_map[itype]
            elif itype in preview_needed and preview_needed[itype] > 0:
                preview_needed[itype] -= 1
                if preview_needed[itype] <= 0:
                    del preview_needed[itype]

        # Execute first step of the new route
        return _decide_single_bot(bot, all_bots, items, active_order, active_needed,
                                  active_from_map, preview_order if preview_order else None,
                                  preview_needed, drop_off, state, claimed_items,
                                  multi_bot=multi_bot)

    # --- PHASE 5: Go deliver if we have useful items ---
    if has_useful:
        return _navigate_dm(bid, pos, drop_off, state)

    # --- PHASE 6: Full inventory with no useful items → go to drop-off ---
    # Single-bot: go to dropoff for cascade when active order completes.
    # Multi-bot: stay put — going to dropoff blocks other bots from delivering.
    if len(inventory) >= 3 and pos != drop_off and not multi_bot:
        return _navigate_dm(bid, pos, drop_off, state)

    # --- PHASE 7: Nothing to do ---
    return {"bot": bid, "action": "wait"}


def _navigate_dm(bid, pos, target, state):
    """Navigate using DistanceMatrix gradient descent, with BFS fallback.

    Uses pre-computed distances for O(1) next-step, but falls back to BFS
    when the ideal next cell is blocked by another bot (collision avoidance).
    """
    global _dm
    if _dm:
        nxt = _dm.next_step(pos, target)
        if nxt and nxt != pos:
            # Check if next cell is occupied by another bot
            occupied = False
            for b in state["bots"]:
                if b["id"] != bid and tuple(b["position"]) == nxt:
                    occupied = True
                    break
            if not occupied:
                return {"bot": bid, "action": _pos_to_action(pos, nxt)}

    # Fallback: BFS with dynamic obstacles (other bots)
    blocked = build_blocked_set(state, exclude_bot_id=bid)
    width = state["grid"]["width"]
    height = state["grid"]["height"]
    path = bfs(pos, target, blocked, width, height)
    if path and len(path) >= 2:
        return {"bot": bid, "action": path_to_action(pos, path[1])}

    # Deadlock breaker: step to any reachable adjacent cell not blocked by a bot.
    if _dm:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (pos[0] + dx, pos[1] + dy)
            if _dm.dist(pos, nxt) == 1 and nxt not in blocked:
                return {"bot": bid, "action": _pos_to_action(pos, nxt)}

    return {"bot": bid, "action": "wait"}
