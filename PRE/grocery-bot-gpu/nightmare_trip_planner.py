"""Nightmare Trip Planner: Commit bots to complete pickup→deliver trips.

Key difference from V6: bots are COMMITTED to multi-round trips instead of
replanning every round. This eliminates flip-flopping and enables pipeline.

Architecture:
- Trip = (bot, target_item, pickup_pos, dropoff_pos, phase)
- Phases: TRAVEL_TO_PICKUP → PICKUP → TRAVEL_TO_DROPOFF → DROPOFF → DONE
- Bots follow committed trips until completion
- New trips assigned when bots become free
- Cascade staging: dedicated trips that end with WAIT at dropoff (no dropoff action)
"""
from __future__ import annotations

import time
import copy
import sys
from collections import defaultdict

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class Trip:
    """A committed bot trip: pickup → deliver."""
    __slots__ = ['bot_id', 'item_idx', 'item_type', 'pickup_pos', 'dropoff',
                 'phase', 'is_staging', 'target_order_idx']

    TRAVEL_PICKUP = 0
    PICKUP = 1
    TRAVEL_DROPOFF = 2
    DROPOFF = 3
    WAIT_AT_DROPOFF = 4  # staging: wait for cascade
    DONE = 5

    def __init__(self, bot_id, item_idx, item_type, pickup_pos, dropoff, is_staging=False):
        self.bot_id = bot_id
        self.item_idx = item_idx
        self.item_type = item_type
        self.pickup_pos = pickup_pos
        self.dropoff = dropoff
        self.phase = self.TRAVEL_PICKUP
        self.is_staging = is_staging
        self.target_order_idx = -1


class NightmareTripPlanner:
    """Trip-committed solver for nightmare mode."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']
        self.future_orders = future_orders or []

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # Type → [(item_idx, adj_positions, zone)]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            ix = int(ms.item_positions[idx, 0])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        # Bot zone assignments
        self.bot_zone = {}
        for bid in range(20):
            self.bot_zone[bid] = 0 if bid < 7 else (1 if bid < 14 else 2)

        # Active trips per bot
        self.trips: dict[int, Trip] = {}
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._last_order_id = -1

        # Corridor parking cells
        self._corridor_cells = []
        for cy in [1, ms.height // 2, ms.height - 3]:
            for cx in range(ms.width):
                cell = (cx, cy)
                if cell in tables.pos_to_idx and cell not in self.drop_set:
                    if not any(tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                        self._corridor_cells.append(cell)

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _balanced_dropoff(self, pos, loads):
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 5
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _find_item(self, pos, needed_types, claimed_items, type_counts=None):
        """Find best item to pick up: nearest (cost = d_to + d_drop * 0.8)."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        best_type = -1

        for tid in needed_types:
            if type_counts and type_counts.get(tid, 0) >= needed_types[tid]:
                continue
            for item_idx, adj_cells, zone in self.type_items.get(tid, []):
                if item_idx in claimed_items:
                    continue
                for adj in adj_cells:
                    d_to = self.tables.get_distance(pos, adj)
                    d_drop = min(self.tables.get_distance(adj, dz) for dz in self.drop_zones)
                    cost = d_to + d_drop * 0.8
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
                        best_type = tid
        return best_idx, best_adj, best_type

    def _assign_trips(self, bot_positions, bot_inventories,
                      active_order, preview_order, round_num):
        """Assign new trips to free bots."""
        # Which bots have active trips?
        busy_bots = set(self.trips.keys())
        free_bots = [bid for bid in range(self.num_bots) if bid not in busy_bots]

        # Items already claimed by active trips
        claimed_items = set()
        for trip in self.trips.values():
            claimed_items.add(trip.item_idx)

        # Active order needs
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Count items already being carried or in-trip for active
        active_assigned = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    active_assigned[t] = active_assigned.get(t, 0) + 1
        for trip in self.trips.values():
            if not trip.is_staging and trip.item_type in active_needs:
                active_assigned[trip.item_type] = active_assigned.get(trip.item_type, 0) + 1

        # Active shortfall
        active_short = {}
        for t, need in active_needs.items():
            s = need - active_assigned.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Preview needs
        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Preview already assigned
        preview_assigned = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in preview_needs:
                    preview_assigned[t] = preview_assigned.get(t, 0) + 1
        for trip in self.trips.values():
            if trip.is_staging and trip.item_type in preview_needs:
                preview_assigned[trip.item_type] = preview_assigned.get(trip.item_type, 0) + 1

        preview_short = {}
        for t, need in preview_needs.items():
            s = need - preview_assigned.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Dropoff loads for balancing
        dropoff_loads = {dz: 0 for dz in self.drop_zones}
        for trip in self.trips.values():
            if trip.dropoff in dropoff_loads:
                dropoff_loads[trip.dropoff] += 1

        # Sort free bots by proximity to needed items
        def bot_priority(bid):
            pos = bot_positions[bid]
            if active_short:
                d = min((self.tables.get_distance(pos, adj)
                         for tid in active_short
                         for _, adj_cells, _ in self.type_items.get(tid, [])
                         for adj in adj_cells), default=999)
                return d
            return 999

        free_bots.sort(key=bot_priority)

        # Assign active trips first
        for bid in list(free_bots):
            if not active_short:
                break
            pos = bot_positions[bid]
            inv = bot_inventories.get(bid, [])

            # If bot already has active items, send to dropoff (no new trip needed)
            if any(t in active_needs for t in inv):
                dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                # Create a delivery-only trip
                trip = Trip(bid, -1, -1, pos, dz)
                trip.phase = Trip.TRAVEL_DROPOFF
                self.trips[bid] = trip
                free_bots.remove(bid)
                continue

            # If bot has free slots, assign active pickup
            if len(inv) < INV_CAP:
                item_idx, adj_pos, item_type = self._find_item(
                    pos, active_short, claimed_items)
                if item_idx is not None:
                    dz = self._balanced_dropoff(adj_pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    trip = Trip(bid, item_idx, item_type, adj_pos, dz)
                    self.trips[bid] = trip
                    claimed_items.add(item_idx)
                    active_short[item_type] = active_short.get(item_type, 1) - 1
                    if active_short[item_type] <= 0:
                        del active_short[item_type]
                    free_bots.remove(bid)
                    continue

        # Assign preview/staging trips to remaining free bots
        staging_count = {dz: sum(1 for t in self.trips.values()
                                  if t.is_staging and t.dropoff == dz)
                         for dz in self.drop_zones}

        for bid in list(free_bots):
            if not preview_short:
                break
            pos = bot_positions[bid]
            inv = bot_inventories.get(bid, [])

            # If bot has preview items, stage at dropoff
            if any(t in preview_needs for t in inv):
                # Find best staging dropoff
                best_dz = None
                best_d = 9999
                for dz in self.drop_zones:
                    if staging_count.get(dz, 0) >= 1:
                        continue
                    d = self.tables.get_distance(pos, dz)
                    if d < best_d:
                        best_d = d
                        best_dz = dz
                if best_dz is not None:
                    trip = Trip(bid, -1, -1, pos, best_dz, is_staging=True)
                    trip.phase = Trip.TRAVEL_DROPOFF
                    self.trips[bid] = trip
                    staging_count[best_dz] = staging_count.get(best_dz, 0) + 1
                    free_bots.remove(bid)
                    continue

            # If bot is empty, assign preview pickup trip
            if len(inv) < INV_CAP:
                # Find a free dropoff for staging
                best_dz = None
                best_d = 9999
                for dz in self.drop_zones:
                    if staging_count.get(dz, 0) >= 1:
                        continue
                    best_dz = dz
                    break

                if best_dz is not None:
                    item_idx, adj_pos, item_type = self._find_item(
                        pos, preview_short, claimed_items)
                    if item_idx is not None:
                        trip = Trip(bid, item_idx, item_type, adj_pos, best_dz, is_staging=True)
                        self.trips[bid] = trip
                        claimed_items.add(item_idx)
                        preview_short[item_type] = preview_short.get(item_type, 1) - 1
                        if preview_short[item_type] <= 0:
                            del preview_short[item_type]
                        staging_count[best_dz] = staging_count.get(best_dz, 0) + 1
                        free_bots.remove(bid)
                        continue

    def _update_trips(self, bot_positions, bot_inventories, active_order):
        """Update trip phases based on bot positions."""
        active_needs = set()
        if active_order:
            for t in active_order.needs():
                active_needs.add(t)

        completed = []
        for bid, trip in list(self.trips.items()):
            pos = bot_positions[bid]
            inv = bot_inventories.get(bid, [])

            if trip.phase == Trip.TRAVEL_PICKUP:
                if pos == trip.pickup_pos:
                    trip.phase = Trip.PICKUP
            elif trip.phase == Trip.PICKUP:
                # Pickup happened last round (or will this round)
                if trip.item_type in [int(self.ms.item_types[trip.item_idx])] if trip.item_idx >= 0 else False:
                    has_item = trip.item_type in inv
                    if has_item:
                        trip.phase = Trip.TRAVEL_DROPOFF
            elif trip.phase == Trip.TRAVEL_DROPOFF:
                if pos == trip.dropoff:
                    if trip.is_staging:
                        trip.phase = Trip.WAIT_AT_DROPOFF
                    else:
                        trip.phase = Trip.DROPOFF
            elif trip.phase == Trip.DROPOFF:
                # Dropoff happened, trip done
                completed.append(bid)
            elif trip.phase == Trip.WAIT_AT_DROPOFF:
                # Staging: check if our items are still relevant
                if not any(t in active_needs for t in inv):
                    pass  # Still waiting for cascade
                else:
                    # Our items match active order - cascade will handle it
                    pass

            # Cancel trip if bot's items became dead
            if not trip.is_staging and trip.phase >= Trip.TRAVEL_DROPOFF:
                if inv and not any(t in active_needs for t in inv):
                    completed.append(bid)

        for bid in set(completed):
            if bid in self.trips:
                del self.trips[bid]

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

        # Track stalls
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Detect order change → clear staging trips
        order_id = active_order.id if active_order else -1
        if order_id != self._last_order_id:
            # Order changed - cancel all staging trips (items may now be active)
            for bid in list(self.trips.keys()):
                trip = self.trips[bid]
                if trip.is_staging:
                    # Check if the staging items now match the NEW active order
                    inv = bot_inventories.get(bid, [])
                    if active_order and any(active_order.needs_type(t) for t in inv):
                        # Convert staging trip to delivery trip
                        trip.is_staging = False
                        trip.phase = Trip.TRAVEL_DROPOFF
                    else:
                        del self.trips[bid]
            self._last_order_id = order_id

        # Update existing trips
        self._update_trips(bot_positions, bot_inventories, active_order)

        # Assign new trips to free bots
        self._assign_trips(bot_positions, bot_inventories,
                           active_order, preview_order, rnd)

        # Build goals from trips
        goals = {}
        goal_types = {}
        pickup_targets = {}

        for bid, trip in self.trips.items():
            if trip.phase == Trip.TRAVEL_PICKUP:
                goals[bid] = trip.pickup_pos
                goal_types[bid] = 'pickup'
            elif trip.phase == Trip.PICKUP:
                goals[bid] = trip.pickup_pos
                goal_types[bid] = 'pickup'
                if trip.item_idx >= 0:
                    pickup_targets[bid] = trip.item_idx
            elif trip.phase == Trip.TRAVEL_DROPOFF:
                goals[bid] = trip.dropoff
                goal_types[bid] = 'deliver' if not trip.is_staging else 'stage'
            elif trip.phase == Trip.DROPOFF:
                goals[bid] = trip.dropoff
                goal_types[bid] = 'deliver'
            elif trip.phase == Trip.WAIT_AT_DROPOFF:
                goals[bid] = trip.dropoff
                goal_types[bid] = 'stage'

        # Free bots without trips → park
        occupied = set(goals.values())
        for bid in range(num_bots):
            if bid not in goals:
                pos = bot_positions[bid]
                # Park in corridor
                best = self.spawn
                best_d = 9999
                for cell in self._corridor_cells:
                    if cell not in occupied:
                        d = self.tables.get_distance(pos, cell)
                        if 0 < d < best_d:
                            best_d = d
                            best = cell
                goals[bid] = best
                goal_types[bid] = 'park'
                occupied.add(best)

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'stage':
                return (1, dist)
            elif gt == 'pickup':
                return (2, dist)
            elif gt == 'flee':
                return (3, dist)
            else:
                return (4, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    if (nx, ny) in self.walkable:
                        actions[bid] = (a, -1)
                        break
                continue

            # At dropoff: deliver if it's a delivery trip
            if pos in self.drop_set:
                trip = self.trips.get(bid)
                if trip and not trip.is_staging and bot_inventories[bid]:
                    # Check if any inv matches active order
                    if active_order and any(active_order.needs_type(t)
                                            for t in bot_inventories[bid]):
                        actions[bid] = (ACT_DROPOFF, -1)
                        if trip:
                            trip.phase = Trip.DONE
                        continue
                # Staging trips: WAIT (cascade handles delivery)

            # At pickup target
            if bid in pickup_targets and pos == goal:
                actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                trip = self.trips.get(bid)
                if trip:
                    trip.phase = Trip.TRAVEL_DROPOFF
                continue

            # Opportunistic pickup: if adjacent to needed item and have space
            if len(bot_inventories[bid]) < INV_CAP and active_order:
                for item_idx in range(ms.num_items):
                    tid = int(ms.item_types[item_idx])
                    if not active_order.needs_type(tid):
                        continue
                    for adj in ms.item_adjacencies.get(item_idx, []):
                        if adj == pos:
                            actions[bid] = (ACT_PICKUP, item_idx)
                            break
                    if actions[bid][0] == ACT_PICKUP:
                        break
                if actions[bid][0] == ACT_PICKUP:
                    continue

            # Follow pathfinder
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareTripPlanner(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nTripPlanner done: {state.score} pts, {state.orders_completed} orders, "
                  f"{elapsed:.1f}s")

        return state.score, action_log


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7009
    score, actions = NightmareTripPlanner.run_sim(seed, verbose=True)
    print(f"\nFinal: {score}", file=sys.stderr)
