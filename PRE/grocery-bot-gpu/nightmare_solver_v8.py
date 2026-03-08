#!/usr/bin/env python3
"""Nightmare solver V8: Trip-committed with multi-item pickup batching.

Key ideas:
1. Bots commit to multi-step trips (pickup 1-3 items → deliver to dropoff)
2. Trip planning considers item clustering and proximity
3. Bots DON'T replan mid-trip (eliminates oscillation)
4. PIBT pathfinding with committed goals
5. When a trip completes, immediately assign new trip

V6 bottleneck: 61 rounds/trip avg, 1.65 items/trip. Target: 30 rounds/trip, 2.5 items/trip.
"""
from __future__ import annotations
import sys, time
import numpy as np
from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

sys.stdout.reconfigure(encoding='utf-8')

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class Trip:
    """A multi-step trip: pick up items, then deliver to dropoff."""
    __slots__ = ['pickups', 'dropoff', 'phase', 'pickup_idx', 'trip_type']

    def __init__(self, pickups, dropoff, trip_type='active'):
        """
        pickups: list of (item_idx, adj_cell) - items to pick up in order
        dropoff: (x, y) - dropoff zone to deliver to
        trip_type: 'active', 'preview', 'future'
        """
        self.pickups = pickups  # [(item_idx, (x,y)), ...]
        self.dropoff = dropoff  # (x,y)
        self.phase = 'pickup'   # 'pickup' or 'deliver'
        self.pickup_idx = 0     # which pickup we're working on
        self.trip_type = trip_type

    @property
    def current_goal(self):
        if self.phase == 'pickup' and self.pickup_idx < len(self.pickups):
            return self.pickups[self.pickup_idx][1]  # adj_cell
        return self.dropoff

    @property
    def current_pickup_item(self):
        if self.phase == 'pickup' and self.pickup_idx < len(self.pickups):
            return self.pickups[self.pickup_idx][0]  # item_idx
        return None

    def advance_pickup(self):
        """Move to next pickup or switch to deliver phase."""
        self.pickup_idx += 1
        if self.pickup_idx >= len(self.pickups):
            self.phase = 'deliver'

    @property
    def is_complete(self):
        return self.phase == 'deliver' and False  # never auto-complete; dropoff action completes

    def __repr__(self):
        return f"Trip({self.trip_type}, pickups={len(self.pickups)}, phase={self.phase}, idx={self.pickup_idx})"


class TripPlanner:
    """Plans trips for bots based on current order state."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)

        # Build type→items lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best, best_d

    def plan_trip(self, bot_pos, bot_inv, needed_types, claimed_items,
                  max_pickups=None, type_assigned=None):
        """Plan a trip to pick up 1-3 needed items and deliver.

        Returns Trip or None.
        needed_types: dict type_id → count still needed
        claimed_items: set of item_idx already claimed
        type_assigned: dict type_id → count already assigned (for over-assignment prevention)
        """
        if max_pickups is None:
            free_slots = INV_CAP - len(bot_inv)
            max_pickups = free_slots

        if max_pickups <= 0:
            return None

        ta = type_assigned or {}
        bot_types = set(bot_inv)

        # Find best first item
        best_items = []  # (cost, item_idx, adj_cell, type_id)
        for tid, need_count in needed_types.items():
            if need_count <= 0:
                continue
            assigned = ta.get(tid, 0)
            if assigned >= need_count + 1:  # allow slight over-assignment
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in claimed_items:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    # Also consider return distance to nearest dropoff
                    _, drop_d = self._nearest_drop(adj)
                    cost = d + drop_d * 0.5
                    best_items.append((cost, d, item_idx, adj, tid))

        if not best_items:
            return None

        best_items.sort()

        # Pick best first item
        _, _, first_item, first_adj, first_tid = best_items[0]
        pickups = [(first_item, first_adj)]
        used_items = {first_item}
        used_types = {first_tid: 1}
        current_pos = first_adj

        # Try to add more pickups nearby (batching)
        for extra in range(max_pickups - 1):
            best_extra = None
            best_extra_cost = 999

            for tid, need_count in needed_types.items():
                assigned = ta.get(tid, 0) + used_types.get(tid, 0)
                if assigned >= need_count + 1:
                    continue
                for item_idx, adj_cells in self.type_items.get(tid, []):
                    if item_idx in claimed_items or item_idx in used_items:
                        continue
                    for adj in adj_cells:
                        d = self.tables.get_distance(current_pos, adj)
                        # Only add if close to current path (< 8 extra steps)
                        if d < 8:
                            _, drop_d = self._nearest_drop(adj)
                            cost = d + drop_d * 0.3
                            if cost < best_extra_cost:
                                best_extra_cost = cost
                                best_extra = (item_idx, adj, tid)

            if best_extra is None:
                break

            item_idx, adj, tid = best_extra
            pickups.append((item_idx, adj))
            used_items.add(item_idx)
            used_types[tid] = used_types.get(tid, 0) + 1
            current_pos = adj

        # Determine dropoff: nearest to last pickup
        dropoff, _ = self._nearest_drop(current_pos)

        return Trip(pickups, dropoff, trip_type='active')


class NightmareSolverV8:
    """V8: Trip-committed solver."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)
        self.trip_planner = TripPlanner(ms, tables, self.drop_zones)

        self.future_orders = future_orders or []

        # Per-bot state
        self.bot_trips: dict[int, Trip | None] = {}
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Item claim tracking (persists across rounds within a trip)
        self.claimed_items: set[int] = set()
        self.type_assigned: dict[int, int] = {}

        # Order tracking
        self._last_active_oid = -1
        self._last_preview_oid = -1

        # Pos→items lookup
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            for adj in ms.item_adjacencies.get(idx, []):
                if adj not in self.pos_to_items:
                    self.pos_to_items[adj] = []
                self.pos_to_items[adj].append((idx, tid))

    def _reset_claims(self):
        """Reset item claims when order changes."""
        self.claimed_items.clear()
        self.type_assigned.clear()
        # Re-claim items from existing trips
        for bid, trip in self.bot_trips.items():
            if trip and trip.phase == 'pickup':
                for i in range(trip.pickup_idx, len(trip.pickups)):
                    item_idx = trip.pickups[i][0]
                    self.claimed_items.add(item_idx)
                    tid = int(self.ms.item_types[item_idx])
                    self.type_assigned[tid] = self.type_assigned.get(tid, 0) + 1

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        ms = self.ms
        num_bots = len(state.bot_positions)

        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))

        # Stall tracking
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Detect order changes → reset claims and invalidate trips
        active_oid = active_order.id if active_order else -1
        preview_oid = preview_order.id if preview_order else -1
        if active_oid != self._last_active_oid:
            self._last_active_oid = active_oid
            # Active order changed — invalidate all active trips
            for bid in list(self.bot_trips.keys()):
                trip = self.bot_trips[bid]
                if trip and trip.trip_type == 'active':
                    # Trip is now potentially invalid — check if items still needed
                    pass  # We'll re-validate below
            self._reset_claims()

        if preview_oid != self._last_preview_oid:
            self._last_preview_oid = preview_oid
            # Preview changed — invalidate preview trips
            for bid in list(self.bot_trips.keys()):
                trip = self.bot_trips[bid]
                if trip and trip.trip_type == 'preview':
                    # Invalidate preview trips; items may now be active
                    self.bot_trips[bid] = None
            self._reset_claims()

        # Compute needs
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Carrying analysis
        carrying_active = {}
        carrying_preview = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1
                elif t in preview_needs:
                    carrying_preview[t] = carrying_preview.get(t, 0) + 1

        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_short = {}
        for t, need in preview_needs.items():
            s = need - carrying_preview.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Validate existing trips and detect completed pickups
        for bid in range(num_bots):
            trip = self.bot_trips.get(bid)
            if trip is None:
                continue

            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # If trip is deliver phase and bot has nothing to deliver → trip done
            if trip.phase == 'deliver':
                has_useful = False
                for t in inv:
                    if t in active_needs or t in preview_needs:
                        has_useful = True
                        break
                if not has_useful and not inv:
                    self.bot_trips[bid] = None
                    continue

            # If pickup phase: check if the target item's type is still needed
            if trip.phase == 'pickup':
                item_idx = trip.current_pickup_item
                if item_idx is not None:
                    tid = int(ms.item_types[item_idx])
                    if trip.trip_type == 'active':
                        still_needed = tid in active_needs
                    else:
                        still_needed = tid in preview_needs
                    if not still_needed:
                        # Skip this pickup, advance to next
                        trip.advance_pickup()

        # Assign trips to bots without trips
        # Sort bots by proximity to items for better assignment
        idle_bots = []
        delivering_bots = []
        for bid in range(num_bots):
            trip = self.bot_trips.get(bid)
            inv = bot_inventories[bid]

            if trip is not None:
                if trip.phase == 'deliver':
                    delivering_bots.append(bid)
                continue

            # Bot has no trip — classify
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)

            if has_active:
                # Create deliver trip immediately
                dz_best = self.drop_zones[0]
                dz_d = self.tables.get_distance(bot_positions[bid], dz_best)
                for dz in self.drop_zones[1:]:
                    d = self.tables.get_distance(bot_positions[bid], dz)
                    if d < dz_d:
                        dz_d = d
                        dz_best = dz
                self.bot_trips[bid] = Trip([], dz_best, trip_type='active')
                self.bot_trips[bid].phase = 'deliver'
                delivering_bots.append(bid)
            elif has_preview and not active_short:
                # Stage at dropoff
                dz_best = self.drop_zones[0]
                dz_d = self.tables.get_distance(bot_positions[bid], dz_best)
                for dz in self.drop_zones[1:]:
                    d = self.tables.get_distance(bot_positions[bid], dz)
                    if d < dz_d:
                        dz_d = d
                        dz_best = dz
                self.bot_trips[bid] = Trip([], dz_best, trip_type='preview')
                self.bot_trips[bid].phase = 'deliver'
                delivering_bots.append(bid)
            elif len(inv) >= INV_CAP:
                # Full inventory, dead items — park
                idle_bots.append(bid)
            else:
                idle_bots.append(bid)

        # Sort idle bots by who can most quickly reach needed items
        def _idle_sort_key(bid):
            pos = bot_positions[bid]
            if active_short:
                min_d = 9999
                for tid in active_short:
                    for item_idx, adj_cells in self.trip_planner.type_items.get(tid, []):
                        if item_idx in self.claimed_items:
                            continue
                        for adj in adj_cells:
                            d = self.tables.get_distance(pos, adj)
                            min_d = min(min_d, d)
                return min_d
            return 9999

        idle_bots.sort(key=_idle_sort_key)

        # Assign pickup trips to idle bots
        for bid in idle_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            if free_slots <= 0:
                continue  # Can't pick up anything

            # Try active pickup trip
            if active_short:
                trip = self.trip_planner.plan_trip(
                    pos, inv, active_short, self.claimed_items,
                    max_pickups=free_slots, type_assigned=self.type_assigned)
                if trip:
                    trip.trip_type = 'active'
                    self.bot_trips[bid] = trip
                    for item_idx, _ in trip.pickups:
                        self.claimed_items.add(item_idx)
                        tid = int(ms.item_types[item_idx])
                        self.type_assigned[tid] = self.type_assigned.get(tid, 0) + 1
                    continue

            # Try preview pickup trip
            remaining_active = sum(max(0, s - self.type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active == 0 and preview_short:
                trip = self.trip_planner.plan_trip(
                    pos, inv, preview_short, self.claimed_items,
                    max_pickups=free_slots, type_assigned=self.type_assigned)
                if trip:
                    trip.trip_type = 'preview'
                    self.bot_trips[bid] = trip
                    for item_idx, _ in trip.pickups:
                        self.claimed_items.add(item_idx)
                        tid = int(ms.item_types[item_idx])
                        self.type_assigned[tid] = self.type_assigned.get(tid, 0) + 1
                    continue

        # Build goals for pathfinder
        goals = {}
        goal_types = {}
        pickup_targets = {}

        for bid in range(num_bots):
            trip = self.bot_trips.get(bid)
            if trip is None:
                # Park in corridor
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            goal = trip.current_goal
            goals[bid] = goal

            if trip.phase == 'pickup':
                goal_types[bid] = 'pickup'
                pickup_targets[bid] = trip.current_pickup_item
            else:
                goal_types[bid] = 'deliver'

        # Urgency order: deliver > pickup > park
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'pickup':
                return (2, dist)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # Pathfind
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Generate actions
        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            trip = self.bot_trips.get(bid)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 5:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                # Cancel trip if stalled too long
                if self.stall_counts.get(bid, 0) >= 10:
                    self.bot_trips[bid] = None
                continue

            if trip is None:
                actions[bid] = (path_actions.get(bid, ACT_WAIT), -1)
                continue

            # At dropoff: deliver
            if trip.phase == 'deliver' and pos in self.drop_set:
                inv = bot_inventories[bid]
                if inv:
                    actions[bid] = (ACT_DROPOFF, -1)
                    # After dropoff, trip is complete
                    self.bot_trips[bid] = None
                    continue
                else:
                    self.bot_trips[bid] = None
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # At pickup target: pick up
            if trip.phase == 'pickup' and pos == trip.current_goal:
                item_idx = trip.current_pickup_item
                if item_idx is not None:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    trip.advance_pickup()
                    continue

            # Opportunistic adjacent pickup (active items)
            if len(bot_inventories[bid]) < INV_CAP:
                opp = self._check_adjacent_pickup(bid, pos, active_needs, preview_needs,
                                                   bot_inventories[bid], active_short)
                if opp is not None:
                    actions[bid] = opp
                    # If we picked up our trip target opportunistically, advance
                    if trip.phase == 'pickup':
                        item_idx = trip.current_pickup_item
                        if opp[1] == item_idx:
                            trip.advance_pickup()
                    continue

            # Move toward goal
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_pickup(self, bid, pos, active_needs, preview_needs,
                                bot_inv, active_short):
        """Pick up adjacent item if it's needed."""
        adjacent_items = self.pos_to_items.get(pos, [])
        if not adjacent_items:
            return None
        bot_types = set(bot_inv)
        total_short = sum(active_short.values())
        for item_idx, tid in adjacent_items:
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
                return (ACT_PICKUP, item_idx)
            elif total_short == 0 and tid in preview_needs:
                if tid in bot_types:
                    continue
                return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid, pos, rnd):
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV8(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            step(state, actions, all_orders)

            if verbose and (rnd < 5 or rnd % 50 == 0):
                active = state.get_active_order()
                trips_active = sum(1 for t in solver.bot_trips.values() if t is not None)
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + f" Trips={trips_active}")

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Time={elapsed:.1f}s")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare solver V8')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = NightmareSolverV8.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f"Seed {seed}: {score}")

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()
