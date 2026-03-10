"""Temporal task scheduler (JSSP-inspired) for nightmare mode.

Pre-computes a complete delivery schedule over 500 rounds instead of
re-allocating tasks every round. Models order-to-bot assignment as
a Job Shop Scheduling problem.

Key features:
1. Min-cost matching for initial item-to-bot assignment (Hungarian-inspired greedy)
2. Temporal deconfliction via estimated travel times
3. Chain-aware scheduling: stages bots at dropoff before trigger arrives
4. Online adaptation: replan if bot falls behind schedule

Usage:
    from nightmare_scheduler import NightmareScheduler
    scheduler = NightmareScheduler(ms, tables, all_orders, drop_zones)
    schedule = scheduler.build_schedule(num_bots=20, num_rounds=500)
    goal = schedule.get_goal(bid=5, rnd=42)

Integration:
    Replaces NightmareTaskAlloc.allocate() as the task assignment source.
    V4 solver calls scheduler instead of greedy allocation.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from game_engine import MapState, Order, INV_CAP
from precompute import PrecomputedTables


@dataclass
class Trip:
    """A single pickup-deliver trip for one bot."""
    bot_id: int
    items: list[int]           # item type IDs to pick up
    pickup_cells: list[tuple[int, int]]  # where to pick each item (adjacent cell)
    pickup_indices: list[int]  # item index in MapState.items
    dropoff_zone: tuple[int, int]        # which of 3 dropoffs
    est_start: int             # estimated start round
    est_end: int               # estimated delivery round
    order_id: int              # which order this trip serves
    is_chain_stage: bool = False  # True if staging for chain reaction
    completed: bool = False


@dataclass
class Schedule:
    """Complete delivery schedule for all bots over 500 rounds."""
    trips: dict[int, list[Trip]] = field(default_factory=dict)  # bot_id -> ordered trips
    _current_trip_idx: dict[int, int] = field(default_factory=dict)  # bot_id -> current trip index

    def get_current_trip(self, bid: int) -> Trip | None:
        """Get the current (in-progress) trip for a bot."""
        idx = self._current_trip_idx.get(bid, 0)
        trips = self.trips.get(bid, [])
        while idx < len(trips):
            if not trips[idx].completed:
                return trips[idx]
            idx += 1
        self._current_trip_idx[bid] = idx
        return None

    def advance_trip(self, bid: int):
        """Mark current trip as completed, advance to next."""
        idx = self._current_trip_idx.get(bid, 0)
        trips = self.trips.get(bid, [])
        if idx < len(trips):
            trips[idx].completed = True
            self._current_trip_idx[bid] = idx + 1

    def total_trips(self) -> int:
        return sum(len(t) for t in self.trips.values())

    def trips_for_order(self, order_id: int) -> list[Trip]:
        """Get all trips serving a specific order."""
        result = []
        for trips in self.trips.values():
            for trip in trips:
                if trip.order_id == order_id:
                    result.append(trip)
        return result


class NightmareScheduler:
    """Pre-compute a complete delivery schedule over 500 rounds.

    Uses min-cost greedy assignment with temporal deconfliction.
    """

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 all_orders: list[Order], drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.all_orders = all_orders
        self.drop_zones = drop_zones
        self.drop_set = set(tuple(dz) for dz in drop_zones)

        # type -> [(item_idx, adj_cells)]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for i in range(ms.num_items):
            tid = int(ms.item_types[i])
            adj = ms.item_adjacencies.get(i, [])
            if adj:
                self.type_items.setdefault(tid, []).append((i, adj))

        # Zone classification for bots
        sorted_zones = sorted(drop_zones, key=lambda z: z[0])
        self.zone_dropoff = {i: z for i, z in enumerate(sorted_zones)}
        n_zones = len(sorted_zones)
        self._zone_thresholds = []
        if n_zones > 1:
            for i in range(n_zones - 1):
                mid = (sorted_zones[i][0] + sorted_zones[i + 1][0]) // 2
                self._zone_thresholds.append(mid)

    def _classify_zone(self, x: int) -> int:
        """Classify an x-coordinate to a zone index."""
        for i, thresh in enumerate(self._zone_thresholds):
            if x <= thresh:
                return i
        return len(self._zone_thresholds)

    def build_schedule(self, num_bots: int = 20, num_rounds: int = 500,
                       spawn: tuple[int, int] | None = None) -> Schedule:
        """Build a complete delivery schedule.

        Assigns bots to trips (pickup runs) covering all known orders.
        """
        if spawn is None:
            spawn = self.ms.spawn

        schedule = Schedule()
        for bid in range(num_bots):
            schedule.trips[bid] = []
            schedule._current_trip_idx[bid] = 0

        # Assign bots to zones (roughly equal split)
        zone_bots: dict[int, list[int]] = defaultdict(list)
        n_zones = len(self.drop_zones)
        for bid in range(num_bots):
            zone_idx = bid % n_zones
            zone_bots[zone_idx].append(bid)

        # Process orders in sequence
        bot_available_round: dict[int, int] = {b: 0 for b in range(num_bots)}
        bot_position: dict[int, tuple[int, int]] = {b: spawn for b in range(num_bots)}

        for order_idx, order in enumerate(self.all_orders):
            if order_idx >= 80:  # Don't over-plan beyond reasonable horizon
                break

            needs = order.needs()
            if not needs:
                continue

            # Group items into trips (max INV_CAP=3 per trip)
            trips_needed = self._plan_trips_for_order(
                order_idx, needs, bot_available_round, bot_position,
                zone_bots, num_rounds)

            for trip in trips_needed:
                bid = trip.bot_id
                schedule.trips[bid].append(trip)
                bot_available_round[bid] = trip.est_end + 1
                bot_position[bid] = trip.dropoff_zone

        # Add chain staging trips for preview items
        self._add_chain_staging(schedule, bot_available_round, bot_position,
                                zone_bots, num_rounds)

        return schedule

    def _plan_trips_for_order(self, order_idx: int, needs: list[int],
                              bot_available: dict[int, int],
                              bot_position: dict[int, tuple[int, int]],
                              zone_bots: dict[int, list[int]],
                              num_rounds: int) -> list[Trip]:
        """Plan trips to fulfill one order's needs."""
        trips = []
        remaining = list(needs)

        while remaining:
            # Take up to INV_CAP items for this trip
            batch = remaining[:INV_CAP]
            remaining = remaining[INV_CAP:]

            # Find the best bot for this batch
            best_bot = None
            best_cost = float('inf')
            best_trip = None

            for bid, avail_rnd in bot_available.items():
                if avail_rnd >= num_rounds - 20:
                    continue  # Not enough time

                pos = bot_position[bid]
                trip = self._build_trip(bid, batch, pos, avail_rnd, order_idx)
                if trip is None:
                    continue

                # Cost = total estimated time
                cost = trip.est_end - avail_rnd
                # Penalize bots that are far away
                cost += max(0, avail_rnd - 5) * 0.1  # Small penalty for late starts

                if cost < best_cost:
                    best_cost = cost
                    best_bot = bid
                    best_trip = trip

            if best_trip is not None:
                trips.append(best_trip)
            else:
                # Can't plan this batch — skip
                break

        return trips

    def _build_trip(self, bid: int, item_types: list[int],
                    start_pos: tuple[int, int], start_round: int,
                    order_idx: int) -> Trip | None:
        """Build a single trip: plan pickup sequence + delivery."""
        pickup_cells = []
        pickup_indices = []
        current_pos = start_pos
        total_time = 0

        for tid in item_types:
            # Find nearest item of this type
            best_adj = None
            best_idx = None
            best_d = 9999

            for item_idx, adj_cells in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(current_pos, adj)
                    if d < best_d:
                        best_d = d
                        best_adj = adj
                        best_idx = item_idx

            if best_adj is None:
                return None  # Can't find item

            pickup_cells.append(best_adj)
            pickup_indices.append(best_idx)
            total_time += best_d + 1  # +1 for pickup action
            current_pos = best_adj

        # Travel to nearest dropoff
        best_dz = None
        best_dd = 9999
        for dz in self.drop_zones:
            dd = self.tables.get_distance(current_pos, tuple(dz))
            if dd < best_dd:
                best_dd = dd
                best_dz = tuple(dz)

        total_time += best_dd + 1  # +1 for dropoff action

        return Trip(
            bot_id=bid,
            items=list(item_types),
            pickup_cells=pickup_cells,
            pickup_indices=pickup_indices,
            dropoff_zone=best_dz or self.drop_zones[0],
            est_start=start_round,
            est_end=start_round + total_time,
            order_id=order_idx,
        )

    def _add_chain_staging(self, schedule: Schedule,
                           bot_available: dict[int, int],
                           bot_position: dict[int, tuple[int, int]],
                           zone_bots: dict[int, list[int]],
                           num_rounds: int):
        """Add staging trips for chain reaction setup.

        For each upcoming order pair (active + preview), identify items
        that should be pre-staged at dropoff for auto-delivery chain.
        """
        # Look at orders in pairs: when order N completes,
        # order N+1 becomes active. Bots at dropoff auto-deliver.
        for i in range(0, min(len(self.all_orders) - 1, 40), 2):
            order_a = self.all_orders[i]
            order_b = self.all_orders[i + 1] if i + 1 < len(self.all_orders) else None
            if order_b is None:
                continue

            # Items needed by order_b that could be staged
            stage_needs = order_b.needs()[:INV_CAP]
            if not stage_needs:
                continue

            # Find an available bot to stage
            best_bot = None
            best_avail = num_rounds

            for bid, avail_rnd in bot_available.items():
                if avail_rnd < best_avail and avail_rnd < num_rounds - 30:
                    best_avail = avail_rnd
                    best_bot = bid

            if best_bot is None:
                continue

            pos = bot_position[best_bot]
            trip = self._build_trip(best_bot, stage_needs, pos,
                                    best_avail, i + 1)
            if trip is not None:
                trip.is_chain_stage = True
                schedule.trips[best_bot].append(trip)
                bot_available[best_bot] = trip.est_end + 1
                bot_position[best_bot] = trip.dropoff_zone

    def adapt(self, schedule: Schedule, bid: int, current_pos: tuple[int, int],
              current_round: int, actual_inv: list[int]) -> tuple[int, int] | None:
        """Online adaptation: compute current goal for a bot.

        Returns (goal_x, goal_y) or None if no active trip.
        """
        trip = schedule.get_current_trip(bid)
        if trip is None:
            return None

        # Check if we've completed pickups
        items_held = len(actual_inv)

        if items_held >= len(trip.items):
            # All items picked up — go to dropoff
            return trip.dropoff_zone

        # Need more pickups
        next_pickup_idx = items_held
        if next_pickup_idx < len(trip.pickup_cells):
            return trip.pickup_cells[next_pickup_idx]

        # Fallback: go to dropoff
        return trip.dropoff_zone

    def replan_bot(self, schedule: Schedule, bid: int,
                   current_pos: tuple[int, int], current_round: int,
                   actual_inv: list[int], active_order: Order | None,
                   preview_order: Order | None) -> Trip | None:
        """Replan a bot's remaining trips if it fell behind schedule.

        Returns new trip or None if current plan is still valid.
        """
        trip = schedule.get_current_trip(bid)
        if trip is None:
            # No current trip — create a new one based on active needs
            if active_order:
                needs = active_order.needs()[:INV_CAP - len(actual_inv)]
                if needs:
                    new_trip = self._build_trip(
                        bid, needs, current_pos, current_round,
                        trip.order_id if trip else 0)
                    if new_trip:
                        schedule.trips[bid].append(new_trip)
                        return new_trip
            return None

        # Check if significantly behind schedule
        delay = current_round - trip.est_end
        if delay > 10:
            # Too far behind — rebuild trip from current position
            remaining_items = trip.items[len(actual_inv):]
            if remaining_items:
                new_trip = self._build_trip(
                    bid, remaining_items, current_pos, current_round,
                    trip.order_id)
                if new_trip:
                    trip.completed = True  # Abandon old trip
                    schedule.trips[bid].append(new_trip)
                    return new_trip

        return None


class SchedulerAllocator:
    """Adapter that wraps NightmareScheduler to provide allocate()-compatible interface.

    Used as a drop-in replacement for NightmareTaskAlloc in the V4 solver.
    """

    def __init__(self, scheduler: NightmareScheduler, schedule: Schedule):
        self.scheduler = scheduler
        self.schedule = schedule

    def get_goals(self, bot_positions: dict[int, tuple[int, int]],
                  bot_inventories: dict[int, list[int]],
                  active_order: Order | None,
                  preview_order: Order | None,
                  round_num: int) -> tuple[dict[int, tuple[int, int]], dict[int, str]]:
        """Get goals and goal types from the schedule.

        Returns (goals, goal_types) dicts.
        """
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}

        for bid, pos in bot_positions.items():
            inv = bot_inventories.get(bid, [])

            # Try scheduled goal first
            goal = self.scheduler.adapt(
                self.schedule, bid, pos, round_num, inv)

            if goal is not None:
                goals[bid] = goal
                trip = self.schedule.get_current_trip(bid)
                if trip:
                    items_held = len(inv)
                    if items_held >= len(trip.items):
                        if trip.is_chain_stage:
                            goal_types[bid] = 'stage'
                        else:
                            goal_types[bid] = 'deliver'
                    else:
                        goal_types[bid] = 'pickup_active'
                else:
                    goal_types[bid] = 'idle'
            else:
                # No scheduled trip — use reactive fallback
                if inv and active_order:
                    has_active = any(active_order.needs_type(t) for t in inv)
                    if has_active:
                        # Deliver
                        best_dz = min(self.scheduler.drop_zones,
                                      key=lambda dz: self.scheduler.tables.get_distance(pos, tuple(dz)))
                        goals[bid] = tuple(best_dz)
                        goal_types[bid] = 'deliver'
                        continue

                # Park
                goal_types[bid] = 'idle'

        return goals, goal_types

    def check_completions(self, bot_inventories: dict[int, list[int]],
                          active_order: Order | None):
        """Check if any bot has completed its trip (delivered all items)."""
        for bid in list(bot_inventories.keys()):
            trip = self.schedule.get_current_trip(bid)
            if trip and not trip.completed:
                # If bot is empty and trip expected delivery, mark complete
                if not bot_inventories.get(bid) and trip.est_start < 9999:
                    self.schedule.advance_trip(bid)
