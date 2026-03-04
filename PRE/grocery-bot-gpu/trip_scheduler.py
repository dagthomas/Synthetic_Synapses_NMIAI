"""Trip-level joint scheduler for multi-bot Expert gameplay.

Produces a warm-start combined_actions[300][num_bots] for GPU refinement
by assigning orders to bots at the trip granularity and converting to
BFS-based action sequences.

Algorithm:
  1. Greedy init: assign orders to bots round-robin, generate trips greedily.
  2. ILS: swap order assignments between bots; keep if score improves.
  3. to_init_actions: convert schedule to BFS path actions.

The output may have soft collisions — GPU refinement fixes those.

Usage (standalone test):
    python trip_scheduler.py expert 42
"""
import sys
import time
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from game_engine import (
    init_game, MAX_ROUNDS, INV_CAP,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, DX, DY,
)
from precompute import PrecomputedTables
from configs import CONFIGS


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ScheduledTrip:
    """One pickup-and-deliver trip for a bot."""
    bot_id: int
    order_idx: int
    items: list          # list of item_idx (into ms.items)
    est_cost: int        # estimated rounds for this trip
    dropoff_round: int   # estimated round when delivery completes


class DropoffSlotReservation:
    """Tracks reserved delivery rounds to stagger bot arrivals."""

    def __init__(self):
        self._reserved = {}  # round → bot_id

    def next_free_slot(self, earliest_round):
        """Return earliest free drop-off round >= earliest_round."""
        r = earliest_round
        while r in self._reserved:
            r += 1
        return r

    def reserve(self, round_num, bot_id):
        self._reserved[round_num] = bot_id


@dataclass
class BotSchedule:
    """A bot's planned sequence of trips."""
    bot_id: int
    trips: list = field(default_factory=list)    # list of ScheduledTrip
    current_round: int = 0                        # next available round


# ── Trip cost estimation ──────────────────────────────────────────────────────

def _estimate_trip_cost(start_pos, item_indices, ms, tables):
    """Estimate rounds for: start → item_1_adj → ... → item_k_adj → dropoff.

    Uses O(1) BFS distance lookups from PrecomputedTables.
    Adds +1 per pickup action.

    Returns (est_cost, item_adj_sequence) where item_adj_sequence is the
    list of adj cells used (one per item, in greedy-nearest order).
    """
    pos = start_pos
    total_cost = 0
    adj_sequence = []

    for item_idx in item_indices:
        result = tables.get_nearest_item_cell(pos, item_idx, ms)
        if result is None:
            return 9999, []
        adj = (result[0], result[1])
        total_cost += result[2] + 1  # walk + pickup
        adj_sequence.append(adj)
        pos = adj

    # Walk from last adj to dropoff
    total_cost += tables.get_distance(pos, ms.drop_off) + 1  # walk + dropoff action
    return total_cost, adj_sequence


def _select_greedy_items(start_pos, needed_types_counter, ms, tables, max_items=INV_CAP):
    """Select up to max_items items for one trip using nearest-neighbor heuristic.

    Returns list of item_idx (greedily chosen to minimize total trip cost).
    """
    remaining = Counter(needed_types_counter)
    trip_items = []
    pos = start_pos

    for _ in range(max_items):
        if not remaining:
            break

        best_item = None
        best_adj = None
        best_dist = 9999

        for item_idx in range(ms.num_items):
            type_id = int(ms.item_types[item_idx])
            if remaining.get(type_id, 0) <= 0:
                continue

            result = tables.get_nearest_item_cell(pos, item_idx, ms)
            if result and result[2] < best_dist:
                best_dist = result[2]
                best_item = item_idx
                best_adj = (result[0], result[1])

        if best_item is None:
            break

        trip_items.append(best_item)
        type_id = int(ms.item_types[best_item])
        remaining[type_id] -= 1
        if remaining[type_id] <= 0:
            del remaining[type_id]
        pos = best_adj

    return trip_items


def _plan_order_trips(order, start_pos_initial, ms, tables, start_round):
    """Plan all trips needed to complete an order.

    Returns list of ScheduledTrip and the round after the last delivery.
    """
    needed = Counter(int(t) for t in order.required)
    pos = start_pos_initial
    current_round = start_round
    trips = []

    while needed and current_round < MAX_ROUNDS:
        items = _select_greedy_items(pos, needed, ms, tables)
        if not items:
            break

        cost, _ = _estimate_trip_cost(pos, items, ms, tables)
        dropoff_round = current_round + cost

        trips.append(ScheduledTrip(
            bot_id=-1,  # filled by caller
            order_idx=order.id,
            items=items,
            est_cost=cost,
            dropoff_round=dropoff_round,
        ))

        # Update remaining needs
        for item_idx in items:
            type_id = int(ms.item_types[item_idx])
            needed[type_id] -= 1
            if needed[type_id] <= 0:
                del needed[type_id]

        # Bot is now at dropoff; next trip starts from there
        pos = ms.drop_off
        current_round = dropoff_round + 1

    return trips, current_round


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TripScheduler:
    """Assign orders to bots at trip granularity with ILS improvement."""

    def __init__(self, ms, all_orders, tables, num_bots, difficulty):
        self.ms = ms
        self.all_orders = all_orders
        self.tables = tables
        self.num_bots = num_bots
        self.difficulty = difficulty

    def run(self, time_budget_s=5.0):
        """Run greedy init + ILS. Returns list of BotSchedule."""
        schedules = self._greedy_init()
        score = self._evaluate(schedules)
        print(f"  [trip_sched] Greedy init score estimate: {score:.0f}", file=sys.stderr)

        t0 = time.time()
        improved = 0
        iters = 0

        rng = random.Random(7)
        num_orders = len(self.all_orders)

        while (time.time() - t0) < time_budget_s:
            iters += 1
            # Pick two distinct bots
            b1 = rng.randrange(self.num_bots)
            b2 = rng.randrange(self.num_bots)
            if b1 == b2:
                continue

            sched1 = schedules[b1]
            sched2 = schedules[b2]

            # Try swapping the last order from b1 to b2 (or vice versa)
            # Use b1→b2 direction (swap b1's last order to b2)
            if not sched1.trips:
                continue

            # Find order indices assigned to b1 by grouping trips
            order_groups = _group_trips_by_order(sched1.trips)
            if not order_groups:
                continue

            # Pick a random order group from b1 to move to b2
            pick_idx = rng.randrange(len(order_groups))
            order_trips_to_move = order_groups[pick_idx]
            order_idx = order_trips_to_move[0].order_idx

            # Build candidate schedules
            new_sched1_trips = [t for t in sched1.trips if t.order_idx != order_idx]
            new_sched2_trips = list(sched2.trips)

            # Rebuild bot1 schedule without this order
            new_s1 = self._rebuild_schedule(b1, new_sched1_trips)
            # Rebuild bot2 schedule with this order appended
            new_s2 = self._rebuild_schedule_with_order(b2, new_sched2_trips, order_idx)

            new_schedules = list(schedules)
            new_schedules[b1] = new_s1
            new_schedules[b2] = new_s2

            new_score = self._evaluate(new_schedules)
            if new_score > score:
                schedules = new_schedules
                score = new_score
                improved += 1

        elapsed = time.time() - t0
        print(f"  [trip_sched] ILS done: {iters} iters, {improved} improvements, "
              f"score={score:.0f}, {elapsed:.1f}s", file=sys.stderr)

        return schedules

    def _greedy_init(self):
        """Round-robin order assignment. Bot i gets orders i, i+N, i+2N, ..."""
        schedules = [BotSchedule(bot_id=b) for b in range(self.num_bots)]

        for order_idx, order in enumerate(self.all_orders):
            bot_id = order_idx % self.num_bots
            sched = schedules[bot_id]

            start_pos = (self.ms.drop_off if sched.trips
                         else self.ms.spawn)
            trips, next_round = _plan_order_trips(
                order, start_pos, self.ms, self.tables, sched.current_round)

            for trip in trips:
                trip.bot_id = bot_id
            sched.trips.extend(trips)
            sched.current_round = next_round

        return schedules

    def _evaluate(self, schedules):
        """Estimate total score from schedules.

        Score = items_delivered × 1 + orders_completed × 5, capped at round 300.
        """
        total = 0.0
        for sched in schedules:
            order_items = {}   # order_idx → items delivered count
            order_needed = {}  # order_idx → total items needed

            for trip in sched.trips:
                if trip.dropoff_round >= MAX_ROUNDS:
                    continue
                oidx = trip.order_idx
                if 0 <= oidx < len(self.all_orders):
                    order_needed.setdefault(oidx, len(self.all_orders[oidx].required))
                    order_items[oidx] = order_items.get(oidx, 0) + len(trip.items)
                    total += len(trip.items)  # +1 per item

            for oidx, delivered in order_items.items():
                needed = order_needed.get(oidx, 0)
                if delivered >= needed > 0:
                    total += 5  # +5 for completed order

        return total

    def _rebuild_schedule(self, bot_id, remaining_trips):
        """Rebuild a BotSchedule from a subset of trips, recomputing timing."""
        sched = BotSchedule(bot_id=bot_id)
        pos = self.ms.spawn
        current_round = 0

        # Group by order to preserve sequential delivery within an order
        order_groups = _group_trips_by_order(remaining_trips)
        for group in order_groups:
            for trip in group:
                cost, _ = _estimate_trip_cost(pos, trip.items, self.ms, self.tables)
                trip.dropoff_round = current_round + cost
                trip.est_cost = cost
                trip.bot_id = bot_id
                sched.trips.append(trip)
                pos = self.ms.drop_off
                current_round = trip.dropoff_round + 1

        sched.current_round = current_round
        return sched

    def _rebuild_schedule_with_order(self, bot_id, existing_trips, new_order_idx):
        """Append a new order to a bot's existing schedule."""
        # Rebuild existing
        sched = self._rebuild_schedule(bot_id, existing_trips)
        if new_order_idx >= len(self.all_orders):
            return sched

        order = self.all_orders[new_order_idx]
        start_pos = self.ms.drop_off if sched.trips else self.ms.spawn
        trips, next_round = _plan_order_trips(
            order, start_pos, self.ms, self.tables, sched.current_round)
        for trip in trips:
            trip.bot_id = bot_id
        sched.trips.extend(trips)
        sched.current_round = next_round
        return sched

    def to_init_actions(self, schedules):
        """Convert bot schedules to combined_actions[300][num_bots].

        Generates BFS path actions for each trip. Pads remaining rounds with WAIT.
        Soft collisions are left for GPU refinement to fix.

        Returns:
            List of 300 round-action lists, each is [(act, item_idx)] × num_bots.
        """
        ms = self.ms
        tables = self.tables
        num_bots = len(schedules)

        # Build action list per bot
        bot_acts = []
        for sched in schedules:
            acts = _generate_bot_actions(sched, ms, tables)
            bot_acts.append(acts)

        # Combine into per-round format
        combined = []
        for r in range(MAX_ROUNDS):
            round_acts = []
            for b in range(num_bots):
                if r < len(bot_acts[b]):
                    round_acts.append(bot_acts[b][r])
                else:
                    round_acts.append((ACT_WAIT, -1))
            combined.append(round_acts)

        return combined


# ── Action generation ─────────────────────────────────────────────────────────

def _group_trips_by_order(trips):
    """Group consecutive trips by order_idx. Returns list of groups."""
    if not trips:
        return []
    groups = []
    current_group = [trips[0]]
    for trip in trips[1:]:
        if trip.order_idx == current_group[0].order_idx:
            current_group.append(trip)
        else:
            groups.append(current_group)
            current_group = [trip]
    groups.append(current_group)
    return groups


def _navigate(pos, target, tables, acts):
    """Append move actions from pos → target. Returns final position."""
    cur = pos
    steps = 0
    max_steps = 1000  # safety limit
    while cur != target and len(acts) < MAX_ROUNDS and steps < max_steps:
        act = tables.get_first_step(cur, target)
        if act == 0:
            break
        acts.append((act, -1))
        cur = (cur[0] + DX[act], cur[1] + DY[act])
        steps += 1
    return cur


def _generate_bot_actions(sched, ms, tables):
    """Generate action sequence for one bot's full schedule.

    Returns list of (action_type, item_idx) of length <= MAX_ROUNDS.
    """
    acts = []
    pos = ms.spawn

    for trip in sched.trips:
        if len(acts) >= MAX_ROUNDS:
            break

        # Navigate to and pickup each item in order
        for item_idx in trip.items:
            if len(acts) >= MAX_ROUNDS:
                break

            # Find best adj cell (nearest to current pos)
            result = tables.get_nearest_item_cell(pos, item_idx, ms)
            if result is None:
                continue

            target = (result[0], result[1])
            pos = _navigate(pos, target, tables, acts)

            if len(acts) < MAX_ROUNDS:
                acts.append((ACT_PICKUP, item_idx))
                # pos stays at adj cell (pickup doesn't move the bot)

        # Navigate to dropoff and deliver
        if len(acts) < MAX_ROUNDS:
            pos = _navigate(pos, ms.drop_off, tables, acts)

        if len(acts) < MAX_ROUNDS:
            acts.append((ACT_DROPOFF, -1))
            pos = ms.drop_off

    return acts


# ── CLI standalone test ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trip scheduler standalone test')
    parser.add_argument('difficulty', default='expert', nargs='?',
                        choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('seed', type=int, default=42, nargs='?')
    parser.add_argument('--time', type=float, default=5.0,
                        help='ILS time budget in seconds')
    args = parser.parse_args()

    from game_engine import build_map, generate_all_orders, step as cpu_step, GameState
    import numpy as np

    print(f"=== TripScheduler: {args.difficulty} seed={args.seed} ===")

    ms = build_map(args.difficulty)
    all_orders = generate_all_orders(args.seed, ms, args.difficulty)
    cfg = CONFIGS[args.difficulty]
    num_bots = cfg['bots']

    print(f"  Map: {ms.width}x{ms.height}, bots={num_bots}, orders={len(all_orders)}")

    t0 = time.time()
    tables = PrecomputedTables.get(ms)
    print(f"  Tables: {tables.n_cells} cells, {(time.time()-t0)*1000:.0f}ms")

    scheduler = TripScheduler(ms, all_orders, tables, num_bots, args.difficulty)
    schedules = scheduler.run(time_budget_s=args.time)
    combined_actions = scheduler.to_init_actions(schedules)

    # CPU verify
    from game_engine import init_game
    gs, _ = init_game(args.seed, args.difficulty)
    for r in range(MAX_ROUNDS):
        gs.round = r
        cpu_step(gs, combined_actions[r], all_orders)

    print(f"\nFinal score: {gs.score}")
    print(f"Orders completed: {gs.orders_completed}")
    print(f"Items delivered: {gs.items_delivered}")
