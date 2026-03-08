"""Nightmare solver V7: Cascade-optimized V6 variant.

Key difference from V6: dedicates 2-3 bots as "cascade stagers" who
pick up preview items and wait at dropoff tiles. When the active order
completes, cascade auto-delivers their items for the next order.

Target: 5+ cascade items per completion → halve inter-order gaps → 75+ orders.
"""
from __future__ import annotations

import time
import numpy as np

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
from nightmare_solver_v6 import V6Allocator, NightmareSolverV6

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class V7Allocator(V6Allocator):
    """Cascade-optimized allocator.

    Changes from V6:
    1. Dedicates 2-3 "cascade stagers" per order cycle
    2. Stagers pick up preview items and wait at free dropoffs
    3. Preview pickup starts immediately (not gated on remaining_active==0)
    4. Stagers get high urgency (same as deliverers)
    """

    def __init__(self, ms, tables, drop_zones, **kwargs):
        super().__init__(ms, tables, drop_zones, **kwargs)
        self._stager_assignments: dict[int, tuple[int, int]] = {}  # bid → target dropoff
        self._stager_bots: set[int] = set()
        self._last_active_needs_hash = None

    def allocate(self, bot_positions, bot_inventories,
                 active_order, preview_order, round_num, num_rounds=500,
                 future_orders=None):
        goals = {}
        goal_types = {}
        pickup_targets = {}

        # Active order analysis
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Preview order analysis
        preview_needs = {}
        preview_oid = -1
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1
            preview_oid = preview_order.id

        # Reset on order change
        active_hash = tuple(sorted(active_needs.items()))
        if active_hash != self._last_active_needs_hash:
            self._stager_bots.clear()
            self._stager_assignments.clear()
            self._last_active_needs_hash = active_hash

        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._committed_stages.clear()
            self._last_preview_id = preview_oid
            self._stager_bots.clear()
            self._stager_assignments.clear()
        for bid in list(self._preview_bot_types.keys()):
            inv = bot_inventories.get(bid, [])
            assigned_t = self._preview_bot_types[bid]
            if assigned_t not in inv and inv:
                del self._preview_bot_types[bid]

        # Classify bots
        carrying_active = {}
        carrying_preview = {}
        active_carriers = []
        preview_carriers = []
        dead_bots = []
        empty_bots = []

        for bid, inv in bot_inventories.items():
            if not inv:
                empty_bots.append(bid)
                continue
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            if has_active:
                active_carriers.append(bid)
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
            elif has_preview:
                preview_carriers.append(bid)
                for t in inv:
                    if t in preview_needs:
                        carrying_preview[t] = carrying_preview.get(t, 0) + 1
            elif len(inv) < INV_CAP:
                empty_bots.append(bid)
            else:
                dead_bots.append(bid)

        # Active shortfall
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())

        # Preview shortfall
        preview_assigned_types = dict(carrying_preview)
        for bid, t in self._preview_bot_types.items():
            if t in preview_needs:
                inv = bot_inventories.get(bid, [])
                if t not in inv:
                    preview_assigned_types[t] = preview_assigned_types.get(t, 0) + 1
        preview_short = {}
        for t, need in preview_needs.items():
            s = need - preview_assigned_types.get(t, 0)
            if s > 0:
                preview_short[t] = s

        type_assigned = {}
        preview_type_assigned = dict(preview_assigned_types)
        claimed_items = set()
        dropoff_loads = {dz: 0 for dz in self.drop_zones}

        # === DELIVER: active carriers ===
        fill_up_bots = []
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)
            if free_slots == 0 or total_short == 0:
                dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
            else:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                best_detour = 9999
                for tid in active_short:
                    for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                        for adj in adj_cells:
                            d_to = self.tables.get_distance(pos, adj)
                            d_back = self._drop_dist(adj)
                            detour = d_to + d_back - drop_dist
                            if detour < best_detour:
                                best_detour = detour
                if best_detour < 8:
                    fill_up_bots.append(bid)
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'

        # === FILL-UP (with cascade-aware preview fill) ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bot_types = set(inv)
            assigned_fill = False
            filtered_short = {t: s for t, s in active_short.items()
                              if t not in bot_types or s > 1}
            if filtered_short:
                item_idx, adj_pos = self._assign_item(
                    pos, filtered_short, type_assigned, claimed_items)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    assigned_fill = True

            # Try preview on the way (CASCADE fill-up: higher detour threshold)
            if not assigned_fill and preview_short:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                preview_filtered = {t: s for t, s in preview_short.items()
                                    if t not in bot_types}
                if preview_filtered:
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_filtered, preview_type_assigned,
                        claimed_items, strict=True)
                    if item_idx is not None:
                        d_to_item = self.tables.get_distance(pos, adj_pos)
                        d_item_to_drop = self._drop_dist(adj_pos)
                        detour = d_to_item + d_item_to_drop - drop_dist
                        if detour <= 8:  # V7: higher threshold for cascade fill
                            goals[bid] = adj_pos
                            goal_types[bid] = 'pickup'
                            pickup_targets[bid] = item_idx
                            tid = int(self.ms.item_types[item_idx])
                            preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                            claimed_items.add(item_idx)
                            assigned_fill = True

            if not assigned_fill:
                dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'

        # === PREVIEW CARRIERS: stage at dropoff (V7: no deliver_zones exclusion) ===
        deliver_zones = set()
        for bid_d in goals:
            if goal_types.get(bid_d) == 'deliver' and goals[bid_d] in self.drop_set:
                deliver_zones.add(goals[bid_d])

        staging_counts = {dz: 0 for dz in self.drop_zones}
        occupied_goals = set(goals.values())

        for bid in preview_carriers:
            pos = bot_positions[bid]
            best_zone = None
            best_d = 9999
            for dz in self.drop_zones:
                # V7: DON'T exclude deliver_zones - stagers go to any free dropoff
                # But limit to 1 stager per dropoff to avoid blocking
                if staging_counts[dz] >= 1:
                    continue
                # Skip if a deliverer is physically at this dropoff already
                if any(bot_positions.get(b) == dz for b in goals
                       if goal_types.get(b) == 'deliver' and goals[b] == dz
                       and bot_positions.get(b) == dz):
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_zone = dz
            if best_zone is not None and best_d < 25:
                staging_counts[best_zone] += 1
                goals[bid] = best_zone
                goal_types[bid] = 'stage'
                self._stager_bots.add(bid)
            else:
                park = self._corridor_parking(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'flee'

        # === DEAD BOTS (V7: recycle dead bots with free slots for preview) ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            if free_slots > 0 and preview_short:
                bot_types = set(inv)
                preview_filtered = {t: s for t, s in preview_short.items()
                                    if t not in bot_types}
                if preview_filtered:
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_filtered, preview_type_assigned,
                        claimed_items, strict=True)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        continue

            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === EMPTY BOTS (V7: active first, then early preview) ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        preview_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)

            # Active pickup FIRST (same as V6 - never steal from active)
            if active_short:
                item_idx, adj_pos = self._assign_item(
                    pos, active_short, type_assigned, claimed_items)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    continue

            # V7: Preview pickup WITHOUT remaining_active gate
            # Allow preview work even when some active items haven't been assigned yet
            if preview_short:
                item_idx, adj_pos = self._assign_item(
                    pos, preview_short, preview_type_assigned,
                    claimed_items, strict=True)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                    self._preview_bot_types[bid] = tid
                    claimed_items.add(item_idx)
                    preview_assigned += 1
                    continue

            # Park
            park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets


class NightmareSolverV7(NightmareSolverV6):
    """V7: Cascade-optimized nightmare solver."""

    def __init__(self, ms, tables, future_orders=None):
        super().__init__(ms, tables, future_orders=future_orders)
        # Replace allocator with V7
        self.allocator = V7Allocator(ms, tables, self.drop_zones,
                                     max_preview_pickers=99, drop_d_weight=0.4)

    def action(self, state, all_orders, rnd):
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))

        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos
            hist = self._pos_history.get(bid, [])
            hist.append(pos)
            if len(hist) > self._OSCILLATION_WINDOW:
                hist = hist[-self._OSCILLATION_WINDOW:]
            self._pos_history[bid] = hist

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        active_needs = {}
        carrying_active = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid2, inv in bot_inventories.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        future_orders = []
        if self.future_orders:
            idx = state.next_order_idx
            for i in range(8):
                if idx + i < len(self.future_orders):
                    future_orders.append(self.future_orders[idx + i])

        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future_orders)

        # V7: Override stager goals - after pickup, route to assigned dropoff
        for bid in list(self.allocator._stager_bots):
            if bid in self.allocator._stager_assignments:
                inv = bot_inventories.get(bid, [])
                has_preview = preview_order and any(
                    preview_order.needs_type(t) for t in inv) if inv else False
                if has_preview and goal_types.get(bid) not in ('pickup',):
                    target_dz = self.allocator._stager_assignments[bid]
                    goals[bid] = target_dz
                    goal_types[bid] = 'stage'

        # Urgency order (V7: stagers get priority 1)
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'stage' and bid in self.allocator._stager_bots:
                return (1, dist)
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 4, dist)
            elif gt == 'pickup':
                return (2, dist)
            elif gt in ('stage', 'preview'):
                return (3, dist)
            elif gt in ('future_pickup', 'future_stage'):
                return (4, dist)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                # V7: stagers at dropoff WAIT (cascade handles delivery)

            if gt in ('pickup', 'preview', 'future_pickup') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, active_order, preview_order, gt,
                    bot_inventories[bid], active_short)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        return NightmareSolverV7._run_internal(state, all_orders, verbose)

    @staticmethod
    def _run_internal(state, all_orders, verbose=False):
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV7(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nV7 done: {state.score} pts, {state.orders_completed} orders, "
                  f"{chains} chains (max={max_chain}), {elapsed:.1f}s")

        return state.score, action_log


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7009
    score, actions = NightmareSolverV7.run_sim(seed, verbose=True)
    print(f"\nFinal: {score}", file=sys.stderr)
