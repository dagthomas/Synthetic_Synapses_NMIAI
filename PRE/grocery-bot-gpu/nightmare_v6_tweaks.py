#!/usr/bin/env python3
"""Test V6 allocator tweaks for nightmare.

Hypothesis: the bottleneck is the LAST item of each order. Over-assigning
bots to the last 1-2 items should speed up order completion.
"""
from __future__ import annotations
import sys, time, copy
import numpy as np
from game_engine import init_game, step, INV_CAP
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
from nightmare_pathfinder import build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

sys.stdout.reconfigure(encoding='utf-8')


class TweakedAllocator(V6Allocator):
    """V6 allocator with tunable parameters."""

    def __init__(self, *args,
                 over_assign_bonus=0,     # extra bots per type when total_short <= threshold
                 over_assign_threshold=2, # total_short threshold for flooding
                 sticky_goals=False,      # keep previous assignments when possible
                 fill_detour=8,           # fill-up detour threshold
                 preview_fill_detour=6,   # preview fill-up detour threshold
                 no_park_bots=False,      # force idle bots to pick up ANYTHING (prevent parking)
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.over_assign_bonus = over_assign_bonus
        self.over_assign_threshold = over_assign_threshold
        self.sticky_goals = sticky_goals
        self.fill_detour = fill_detour
        self.preview_fill_detour = preview_fill_detour
        self.no_park_bots = no_park_bots
        self._sticky_assignments: dict[int, tuple[int, tuple[int, int]]] = {}  # bid → (item_idx, adj_cell)
        self._sticky_order_id = -1

    def _assign_item_overassign(self, bot_pos, needed, assigned_counts, claimed,
                                 total_short, zone_filter=-1, type_bonus=None):
        """Like _assign_item but with dynamic over-assignment limit."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            # Dynamic over-assignment: allow more bots when few items remain
            if total_short <= self.over_assign_threshold:
                max_assign = need_count + self.over_assign_bonus
            else:
                max_assign = need_count + 1
            if assigned_counts.get(tid, 0) >= max_assign:
                continue
            bonus = type_bonus.get(tid, 0) if type_bonus else 0
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                if zone_filter >= 0 and item_zone != zone_filter:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    if zone_filter >= 0:
                        drop_d = self.tables.get_distance(
                            adj, self.zone_dropoff.get(zone_filter, self.drop_zones[0]))
                    else:
                        drop_d = self._drop_dist(adj)
                    cost = d + drop_d * self.drop_d_weight - bonus
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def allocate(self, bot_positions, bot_inventories,
                 active_order, preview_order, round_num, num_rounds=500,
                 future_orders=None):
        """Modified allocate with over-assignment for last items."""
        goals = {}
        goal_types = {}
        pickup_targets = {}

        # Active order analysis
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        preview_oid = -1
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1
            preview_oid = preview_order.id

        # Future order analysis
        future_type_freq = {}
        future_all_needs = {}
        if future_orders:
            for fo in future_orders:
                seen_types = set()
                for t in fo.needs():
                    future_all_needs[t] = future_all_needs.get(t, 0) + 1
                    if t not in seen_types:
                        future_type_freq[t] = future_type_freq.get(t, 0) + 1
                        seen_types.add(t)

        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._committed_stages.clear()
            self._last_preview_id = preview_oid
        for bid in list(self._preview_bot_types.keys()):
            inv = bot_inventories.get(bid, [])
            assigned_t = self._preview_bot_types[bid]
            if assigned_t not in inv and inv:
                del self._preview_bot_types[bid]

        # Sticky goal tracking
        active_oid = active_order.id if active_order else -1
        if active_oid != self._sticky_order_id:
            self._sticky_assignments.clear()
            self._sticky_order_id = active_oid

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

        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())

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
                if best_detour < self.fill_detour:
                    fill_up_bots.append(bid)
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'

        # === FILL-UP ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bot_types = set(inv)
            assigned_fill = False
            filtered_short = {t: s for t, s in active_short.items()
                              if t not in bot_types or s > 1}
            if filtered_short:
                item_idx, adj_pos = self._assign_item_overassign(
                    pos, filtered_short, type_assigned, claimed_items, total_short)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    assigned_fill = True

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
                        if detour <= self.preview_fill_detour:
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

        # === PREVIEW CARRIERS: stage at dropoff ===
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
                if dz in deliver_zones:
                    continue
                if staging_counts[dz] >= 6:
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_zone = dz
            if best_zone is not None and best_d < 20:
                staging_counts[best_zone] += 1
                goals[bid] = best_zone
                goal_types[bid] = 'stage'
            else:
                park = self._corridor_parking(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'flee'

        # === DEAD BOTS ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === EMPTY BOTS ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        preview_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)

            # Sticky: if bot has a previous active assignment still valid, keep it
            if self.sticky_goals and bid in self._sticky_assignments:
                prev_item, prev_adj = self._sticky_assignments[bid]
                tid = int(self.ms.item_types[prev_item])
                if tid in active_short and active_short[tid] > 0:
                    # Still needed — keep assignment
                    goals[bid] = prev_adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = prev_item
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(prev_item)
                    continue
                else:
                    del self._sticky_assignments[bid]

            # Active pickup with over-assignment
            if active_short:
                item_idx, adj_pos = self._assign_item_overassign(
                    pos, active_short, type_assigned, claimed_items, total_short)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    if self.sticky_goals:
                        self._sticky_assignments[bid] = (item_idx, adj_pos)
                    continue

            # Preview pickup
            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active == 0 and preview_short:
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


class TweakedSolver(NightmareSolverV6):
    def __init__(self, ms, tables, future_orders=None, **alloc_kwargs):
        super().__init__(ms, tables, future_orders=future_orders)
        self.allocator = TweakedAllocator(
            ms, tables, self.drop_zones,
            max_preview_pickers=99,
            **alloc_kwargs)


def run_sim(seed, verbose=False, **alloc_kwargs):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = TweakedSolver(ms, tables, future_orders=all_orders, **alloc_kwargs)
    num_rounds = DIFF_ROUNDS['nightmare']
    action_log = []
    for rnd in range(num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(actions)
        step(state, actions, all_orders)
    return state.score, action_log


def main():
    seeds = list(range(1000, 1020))

    configs = [
        ("V6 baseline (ddw=0.4)", dict(drop_d_weight=0.4)),
        ("V6 ddw=0.8", dict(drop_d_weight=0.8)),
        ("Over-assign +3 (ts<=2)", dict(drop_d_weight=0.4, over_assign_bonus=3, over_assign_threshold=2)),
        ("Over-assign +5 (ts<=3)", dict(drop_d_weight=0.4, over_assign_bonus=5, over_assign_threshold=3)),
        ("Over-assign +3 ddw=0.8", dict(drop_d_weight=0.8, over_assign_bonus=3, over_assign_threshold=2)),
        ("Sticky goals", dict(drop_d_weight=0.4, sticky_goals=True)),
        ("Sticky + over-assign", dict(drop_d_weight=0.4, sticky_goals=True, over_assign_bonus=3)),
        ("Fill detour 12", dict(drop_d_weight=0.4, fill_detour=12)),
        ("Fill detour 4", dict(drop_d_weight=0.4, fill_detour=4)),
    ]

    for name, cfg in configs:
        scores = []
        t0 = time.time()
        for seed in seeds:
            score, _ = run_sim(seed, **cfg)
            scores.append(score)
        elapsed = time.time() - t0
        print(f"{name:30s}: mean={np.mean(scores):6.1f} min={min(scores):4d} max={max(scores):4d} ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()
