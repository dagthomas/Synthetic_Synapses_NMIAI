"""TSP route optimizer and full game planner.

- TSP brute-force for pickup order (max 6 permutations for 3 items)
- CVRP batch splitter for 4+ item orders
- Full game planner using scouted order sequence
"""

import json
import os
from itertools import combinations, permutations

from distance import DistanceMatrix


def optimal_pickup_route(bot_pos, item_shelves, drop_off, dist_matrix):
    """Find optimal order to pick up items, minimizing total travel.

    Args:
        bot_pos: Current bot position (x, y)
        item_shelves: List of shelf positions [(x, y), ...] to visit
        drop_off: Drop-off position
        dist_matrix: Pre-computed DistanceMatrix

    Returns:
        (ordered_shelves, total_cost, route_positions)
        route_positions includes the adjacent cells the bot actually walks to.
    """
    if not item_shelves:
        return [], 0, []

    if len(item_shelves) == 1:
        shelf = item_shelves[0]
        adj, d_to = dist_matrix.best_adjacent(bot_pos, shelf)
        d_back = dist_matrix.dist_to_dropoff(adj) if adj else 999
        cost = d_to + 1 + d_back  # +1 for pickup
        return [shelf], cost, [adj]

    best_order = None
    best_cost = float("inf")
    best_adjs = None

    for perm in permutations(range(len(item_shelves))):
        cost = 0
        pos = bot_pos
        adjs = []

        for idx in perm:
            shelf = item_shelves[idx]
            adj, d = dist_matrix.best_adjacent(pos, shelf)
            if adj is None:
                cost = float("inf")
                break
            cost += d + 1  # travel + pickup
            adjs.append(adj)
            pos = adj

        # Add return to drop-off
        if cost < float("inf"):
            cost += dist_matrix.dist(pos, drop_off)

        if cost < best_cost:
            best_cost = cost
            best_order = [item_shelves[perm[i]] for i in range(len(perm))]
            best_adjs = adjs

    return best_order or [], best_cost, best_adjs or []


def plan_order_trips(bot_pos, item_shelves, drop_off, dist_matrix, capacity=3):
    """Plan optimal trip batching for an order.

    Splits items into trips of at most `capacity`, minimizing total rounds.

    Returns list of trips: [(ordered_shelves, total_cost, route_adjs), ...]
    """
    if len(item_shelves) <= capacity:
        route = optimal_pickup_route(bot_pos, item_shelves, drop_off, dist_matrix)
        return [route]

    best_plan = None
    best_total = float("inf")

    # Try all ways to split into first batch (capacity) + remaining
    for first_indices in combinations(range(len(item_shelves)), capacity):
        second_indices = [i for i in range(len(item_shelves)) if i not in first_indices]

        first_items = [item_shelves[i] for i in first_indices]
        second_items = [item_shelves[i] for i in second_indices]

        trip1 = optimal_pickup_route(bot_pos, first_items, drop_off, dist_matrix)
        trip2 = optimal_pickup_route(drop_off, second_items, drop_off, dist_matrix)

        total = trip1[1] + trip2[1]
        if total < best_total:
            best_total = total
            best_plan = [trip1, trip2]

    return best_plan or []


# --- Scout data extraction ---


def extract_orders_from_recording(recording_path):
    """Extract the full order sequence from a scout recording.

    Returns list of orders in sequence: [{"id": ..., "items_required": [...]}, ...]
    """
    with open(recording_path) as f:
        data = json.load(f)

    seen_ids = set()
    order_sequence = []

    for rnd in data["rounds"]:
        state = rnd["state"]
        for order in state["orders"]:
            oid = order["id"]
            if oid not in seen_ids:
                seen_ids.add(oid)
                order_sequence.append({
                    "id": oid,
                    "items_required": order["items_required"],
                })

    return order_sequence


def extract_initial_items(recording_path):
    """Extract all items from round 0 of a recording."""
    with open(recording_path) as f:
        data = json.load(f)

    return data["rounds"][0]["state"]["items"]


# --- Full game planner ---


def build_game_plan(dist_matrix, items, order_sequence, drop_off, capacity=3):
    """Build a complete game plan: which items to pick for each order and in what route.

    Args:
        dist_matrix: Pre-computed DistanceMatrix
        items: List of all items on the map at game start
        order_sequence: Full order sequence from scout
        drop_off: Drop-off position
        capacity: Bot inventory capacity

    Returns:
        List of order plans: [
            {
                "order_id": str,
                "assigned_items": [{"id": ..., "type": ..., "position": ...}, ...],
                "trips": [(ordered_shelves, cost, route_adjs), ...]
            },
            ...
        ]
    """
    # Items are infinite (shelves restock). Index by type for fast lookup.
    items_by_type = {}
    for it in items:
        items_by_type.setdefault(it["type"], []).append(it)

    # Pre-sort each type by trip cost from drop-off (cheapest first)
    for item_type in items_by_type:
        items_by_type[item_type].sort(
            key=lambda it: dist_matrix.trip_cost(
                drop_off, (it["position"][0], it["position"][1])
            )
        )

    plan = []
    bot_pos = drop_off  # Assume bot starts/returns to drop-off between orders

    for order in order_sequence:
        needed = {}
        for item_type in order["items_required"]:
            needed[item_type] = needed.get(item_type, 0) + 1

        # Assign best items (cheapest trip cost). Items never deplete.
        assigned = []
        for item_type, count in needed.items():
            candidates = items_by_type.get(item_type, [])
            for it in candidates[:count]:
                assigned.append(it)

        if len(assigned) != sum(needed.values()):
            break

        # Plan pickup routes with TSP optimization
        shelves = [(it["position"][0], it["position"][1]) for it in assigned]
        trips = plan_order_trips(bot_pos, shelves, drop_off, dist_matrix, capacity)

        plan.append({
            "order_id": order["id"],
            "assigned_items": assigned,
            "trips": trips,
        })

        bot_pos = drop_off  # Bot returns to drop-off after each order

    return plan


def save_game_plan(plan, difficulty):
    """Save a computed game plan to the simulation folder."""
    folder = os.path.join(os.path.dirname(__file__), "simulation", difficulty)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "plan.json")

    # Convert tuples to lists for JSON
    serializable = []
    for order_plan in plan:
        trips = []
        for shelves, cost, adjs in order_plan["trips"]:
            trips.append({
                "shelves": [list(s) for s in shelves],
                "cost": cost,
                "adjacent_cells": [list(a) for a in adjs],
            })
        serializable.append({
            "order_id": order_plan["order_id"],
            "assigned_items": order_plan["assigned_items"],
            "trips": trips,
        })

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Plan saved: {filepath}")
    print(f"  Orders planned: {len(plan)}")
    total_cost = sum(t[1] for op in plan for t in op["trips"])
    print(f"  Estimated total rounds: {total_cost}")
    return filepath


def load_game_plan(difficulty):
    """Load a saved game plan."""
    filepath = os.path.join(os.path.dirname(__file__), "simulation", difficulty, "plan.json")
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        data = json.load(f)

    # Convert lists back to tuples
    plan = []
    for order_plan in data:
        trips = []
        for trip in order_plan["trips"]:
            shelves = [tuple(s) for s in trip["shelves"]]
            adjs = [tuple(a) for a in trip["adjacent_cells"]]
            trips.append((shelves, trip["cost"], adjs))
        plan.append({
            "order_id": order_plan["order_id"],
            "assigned_items": order_plan["assigned_items"],
            "trips": trips,
        })

    return plan
