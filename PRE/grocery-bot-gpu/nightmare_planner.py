#!/usr/bin/env python3
"""Nightmare planner: trip-based solver with inventory management.

Key insight: dead inventory (41-47%) is the main bottleneck.
This solver plans complete trips (pickup sequence + delivery) and
avoids picking up items that will become dead.

Strategy:
1. Plan full trips: 1-3 items → dropoff
2. Only assign items when the bot CAN deliver before order likely completes
3. Preview items only in spare slots (will become active on next order)
4. PIBT pathfinding for collision avoidance
"""
from __future__ import annotations

import sys
import time
import random as _random

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

sys.stdout.reconfigure(encoding='utf-8')

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
NUM_ROUNDS = DIFF_ROUNDS['nightmare']
NUM_BOTS = CONFIGS['nightmare']['bots']


class TripPlanner:
    """Trip-based solver with careful inventory management."""

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None,
                 solver_seed: int = 0,
                 drop_d_weight: float = 0.6):
        self.rng = _random.Random(solver_seed) if solver_seed else _random.Random()
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.future_orders = future_orders or []
        self.drop_d_weight = drop_d_weight

        # Pathfinding
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)

        # Item lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # State
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Trip state: each bot has a planned trip
        self._trips: dict[int, dict] = {}
        # Trip format: {'items': [(item_idx, adj_pos, type_id), ...],
        #               'phase': 'fetch'|'deliver', 'step': 0,
        #               'dropoff': (x,y), 'age': 0}
        self._last_active_id = -1

    def _nearest_drop(self, pos):
        return min(self.drop_zones,
                   key=lambda dz: self.tables.get_distance(pos, dz))

    def _plan_trip(self, bid, pos, inv, needed_types, claimed_items,
                   preview_types=None, is_preview=False):
        """Plan a multi-item trip: pick up items → deliver at dropoff.

        Returns a trip dict or None if no useful trip can be planned.
        """
        items_to_fetch = []
        free_slots = INV_CAP - len(inv)
        if free_slots <= 0:
            return None

        # Sort needed types by closest item
        fetch_candidates = []
        for tid, count in needed_types.items():
            if count <= 0:
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in claimed_items:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    drop_d = min(self.tables.get_distance(adj, dz)
                                 for dz in self.drop_zones)
                    cost = d + drop_d * self.drop_d_weight
                    fetch_candidates.append((cost, item_idx, adj, tid))

        fetch_candidates.sort()

        # Greedily pick items along a route
        current_pos = pos
        used_types = set()
        for _, item_idx, adj, tid in fetch_candidates:
            if item_idx in claimed_items:
                continue
            if len(items_to_fetch) >= free_slots:
                break
            # Don't double-assign same type beyond needs
            type_count = sum(1 for _, _, t in items_to_fetch if t == tid)
            if type_count >= needed_types.get(tid, 0):
                continue

            items_to_fetch.append((item_idx, adj, tid))
            claimed_items.add(item_idx)
            current_pos = adj

        if not items_to_fetch:
            # Try preview items in spare slots
            if preview_types and not is_preview:
                return self._plan_trip(bid, pos, inv, preview_types,
                                       claimed_items, is_preview=True)
            return None

        # Pick dropoff
        last_item_pos = items_to_fetch[-1][1]
        dropoff = self._nearest_drop(last_item_pos)

        return {
            'items': items_to_fetch,
            'phase': 'fetch',
            'step': 0,
            'dropoff': dropoff,
            'age': 0,
            'is_preview': is_preview,
        }

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = NUM_ROUNDS

        # Extract state
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall tracking
        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Active analysis
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Count what's being carried
        carrying_active: dict[int, int] = {}
        for inv in bot_inventories.values():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Preview analysis
        preview_needs: dict[int, int] = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Reset trips on active order change
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            # Keep preview trips (they'll become active)
            for bid in list(self._trips.keys()):
                trip = self._trips[bid]
                if not trip.get('is_preview', False):
                    del self._trips[bid]
            self._last_active_id = active_id

        # Age and validate trips
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            trip['age'] += 1
            if trip['age'] > 30:
                del self._trips[bid]
                continue
            # Validate: are the trip's items still needed?
            if trip['phase'] == 'fetch' and not trip.get('is_preview'):
                step_idx = trip['step']
                if step_idx < len(trip['items']):
                    _, _, tid = trip['items'][step_idx]
                    if not (active_order and active_order.needs_type(tid)):
                        del self._trips[bid]
                        continue

        # Claimed items from existing trips
        claimed_items: set[int] = set()
        for bid, trip in self._trips.items():
            for item_idx, _, _ in trip['items']:
                claimed_items.add(item_idx)

        # Count en-route active types
        en_route_active: dict[int, int] = {}
        for bid, trip in self._trips.items():
            if trip.get('is_preview'):
                continue
            if trip['phase'] == 'fetch':
                for i in range(trip['step'], len(trip['items'])):
                    _, _, tid = trip['items'][i]
                    en_route_active[tid] = en_route_active.get(tid, 0) + 1

        # Reduced shortfall accounting for en-route
        reduced_short: dict[int, int] = {}
        for t, s in active_short.items():
            remaining = s - en_route_active.get(t, 0)
            if remaining > 0:
                reduced_short[t] = remaining

        # Assign new trips to bots without active trips
        bots_needing_trips = []
        for bid in range(num_bots):
            if bid in self._trips:
                continue
            inv = bot_inventories[bid]
            has_active = any(t in active_needs for t in inv)
            if has_active:
                # Bot has active items → deliver
                self._trips[bid] = {
                    'items': [],
                    'phase': 'deliver',
                    'step': 0,
                    'dropoff': self._nearest_drop(bot_positions[bid]),
                    'age': 0,
                    'is_preview': False,
                }
            elif len(inv) < INV_CAP:
                bots_needing_trips.append(bid)

        # Sort by proximity to needed items
        def _trip_priority(bid):
            pos = bot_positions[bid]
            if reduced_short:
                d = min((self.tables.get_distance(pos, adj)
                         for tid in reduced_short
                         for _, adjs in self.type_items.get(tid, [])
                         for adj in adjs), default=999)
            else:
                d = 999
            return d + self.rng.random() * 2

        bots_needing_trips.sort(key=_trip_priority)

        for bid in bots_needing_trips:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # Try active trip first
            if reduced_short:
                trip = self._plan_trip(bid, pos, inv, reduced_short,
                                       claimed_items, preview_needs)
                if trip:
                    self._trips[bid] = trip
                    # Update reduced_short
                    for _, _, tid in trip['items']:
                        if not trip.get('is_preview'):
                            if tid in reduced_short:
                                reduced_short[tid] -= 1
                                if reduced_short[tid] <= 0:
                                    del reduced_short[tid]
                    continue

            # Try preview trip
            if preview_needs:
                trip = self._plan_trip(bid, pos, inv, preview_needs,
                                       claimed_items, is_preview=True)
                if trip:
                    self._trips[bid] = trip
                    continue

        # Compute goals from trips
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}

        for bid in range(num_bots):
            if bid not in self._trips:
                # Park in corridor
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            trip = self._trips[bid]
            if trip['phase'] == 'fetch':
                step_idx = trip['step']
                if step_idx < len(trip['items']):
                    _, adj, _ = trip['items'][step_idx]
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                else:
                    # All items fetched → deliver
                    trip['phase'] = 'deliver'
                    goals[bid] = trip['dropoff']
                    goal_types[bid] = 'deliver'
            elif trip['phase'] == 'deliver':
                goals[bid] = trip['dropoff']
                goal_types[bid] = 'deliver'

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            noise = self.rng.random() * 0.5
            if gt == 'deliver':
                return (0, dist + noise)
            elif gt == 'pickup':
                return (2, dist + noise)
            else:
                return (5, dist + noise)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # PIBT pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            stall_count = self.stall_counts.get(bid, 0)
            if stall_count >= 3:
                if gt == 'deliver' and goal in self.drop_set:
                    drop_dist = self.tables.get_distance(pos, goal)
                    if drop_dist <= 4 and stall_count < 8:
                        pass
                    else:
                        actions[bid] = (self._escape_action(bid, pos), -1)
                        continue
                else:
                    actions[bid] = (self._escape_action(bid, pos), -1)
                    continue

            # AT DROPOFF: deliver
            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    if bid in self._trips:
                        del self._trips[bid]
                    continue

            # AT PICKUP TARGET
            if gt == 'pickup' and bid in self._trips:
                trip = self._trips[bid]
                if trip['phase'] == 'fetch':
                    step_idx = trip['step']
                    if step_idx < len(trip['items']):
                        item_idx, adj, _ = trip['items'][step_idx]
                        if pos == adj:
                            actions[bid] = (ACT_PICKUP, item_idx)
                            trip['step'] += 1
                            continue

            # Opportunistic adjacent pickup (active items only)
            if gt in ('pickup', 'deliver') and len(bot_inventories[bid]) < INV_CAP:
                opp = self._check_adjacent_active(pos, active_short)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_active(self, pos, active_short):
        if not active_short:
            return None
        ms = self.ms
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short or active_short[tid] <= 0:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid, pos):
        dirs = list(MOVES)
        self.rng.shuffle(dirs)
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT


def run_sim(seed, solver_seed=1, drop_d_weight=0.6, verbose=True):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = TripPlanner(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=drop_d_weight)
    action_log = []
    chains = 0

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        o_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > o_before + 1:
            chains += state.orders_completed - o_before - 1

    if verbose:
        print(f"  seed={seed} score={state.score} orders={state.orders_completed} "
              f"chains={chains}")
    return state.score, action_log, chains


def main():
    seeds = [7005, 11, 42, 45, 100, 200, 300, 500]
    scores = []
    total_chains = 0

    print("TripPlanner solver (8 seeds):")
    print("-" * 60)
    t0 = time.time()

    for seed in seeds:
        score, _, chains = run_sim(seed)
        scores.append(score)
        total_chains += chains

    mean = sum(scores) / len(scores)
    print("-" * 60)
    print(f"Mean: {mean:.1f}  Chains: {total_chains}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
