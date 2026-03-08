#!/usr/bin/env python3
"""Test zone-aware V6 vs baseline V6 on nightmare."""
from __future__ import annotations
import sys, time
import numpy as np
from game_engine import init_game, step, INV_CAP
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

sys.stdout.reconfigure(encoding='utf-8')


class ZoneV6Allocator(V6Allocator):
    """V6 allocator with zone-filtered item assignment."""

    def __init__(self, *args, use_zone_filter=True, zone_delivery=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_zone_filter = use_zone_filter
        self.zone_delivery = zone_delivery

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

        # Reset preview tracking on order change
        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._committed_stages.clear()
            self._last_preview_id = preview_oid
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

        # === DELIVER: active carriers → zone dropoff ===
        fill_up_bots = []
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bz = self.bot_zone.get(bid, 2)
            free_slots = INV_CAP - len(inv)
            if free_slots == 0 or total_short == 0:
                if self.zone_delivery:
                    dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
            else:
                if self.zone_delivery:
                    dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                else:
                    dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                best_detour = 9999
                zf = bz if self.use_zone_filter else -1
                for tid in active_short:
                    for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                        if zf >= 0 and item_zone != zf:
                            continue
                        for adj in adj_cells:
                            d_to = self.tables.get_distance(pos, adj)
                            d_back = self.tables.get_distance(adj, dz)
                            detour = d_to + d_back - drop_dist
                            if detour < best_detour:
                                best_detour = detour
                if best_detour < 8:
                    fill_up_bots.append(bid)
                else:
                    if self.zone_delivery:
                        dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                    else:
                        dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'

        # === FILL-UP with zone filter ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bz = self.bot_zone.get(bid, 2)
            bot_types = set(inv)
            assigned_fill = False
            filtered_short = {t: s for t, s in active_short.items()
                              if t not in bot_types or s > 1}
            if filtered_short:
                zf = bz if self.use_zone_filter else -1
                item_idx, adj_pos = self._assign_item(
                    pos, filtered_short, type_assigned, claimed_items,
                    zone_filter=zf)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    assigned_fill = True

            if not assigned_fill and preview_short:
                if self.zone_delivery:
                    dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                else:
                    dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                preview_filtered = {t: s for t, s in preview_short.items()
                                    if t not in bot_types}
                if preview_filtered:
                    zf = bz if self.use_zone_filter else -1
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_filtered, preview_type_assigned,
                        claimed_items, strict=True, zone_filter=zf)
                    if item_idx is not None:
                        d_to_item = self.tables.get_distance(pos, adj_pos)
                        d_item_to_drop = self.tables.get_distance(adj_pos, dz)
                        detour = d_to_item + d_item_to_drop - drop_dist
                        if detour <= 6:
                            goals[bid] = adj_pos
                            goal_types[bid] = 'pickup'
                            pickup_targets[bid] = item_idx
                            tid = int(self.ms.item_types[item_idx])
                            preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                            claimed_items.add(item_idx)
                            assigned_fill = True

            if not assigned_fill:
                if self.zone_delivery:
                    dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'

        # === PREVIEW CARRIERS: stage at zone dropoff ===
        deliver_zones = set()
        for bid_d in goals:
            if goal_types.get(bid_d) == 'deliver' and goals[bid_d] in self.drop_set:
                deliver_zones.add(goals[bid_d])

        staging_counts = {dz: 0 for dz in self.drop_zones}
        occupied_goals = set(goals.values())

        for bid in preview_carriers:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)
            if self.zone_delivery:
                dz = self.zone_dropoff.get(bz, self.drop_zones[0])
                if dz not in deliver_zones and staging_counts[dz] < 6:
                    d = self.tables.get_distance(pos, dz)
                    if d < 20:
                        staging_counts[dz] += 1
                        goals[bid] = dz
                        goal_types[bid] = 'stage'
                    else:
                        park = self._corridor_parking(pos, occupied_goals, zone=bz)
                        occupied_goals.add(park)
                        goals[bid] = park
                        goal_types[bid] = 'flee'
                else:
                    park = self._corridor_parking(pos, occupied_goals, zone=bz)
                    occupied_goals.add(park)
                    goals[bid] = park
                    goal_types[bid] = 'flee'
            else:
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
            bz = self.bot_zone.get(bid, 2)
            park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === EMPTY BOTS with zone filter ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        for bid in empty_by_proximity:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)

            # Active pickup - zone filtered
            if active_short:
                zf = bz if self.use_zone_filter else -1
                item_idx, adj_pos = self._assign_item(
                    pos, active_short, type_assigned, claimed_items,
                    zone_filter=zf)
                # Fallback: if zone has no matching items, try global
                if item_idx is None and self.use_zone_filter:
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

            # Preview pickup - zone filtered
            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active == 0 and preview_short:
                zf = bz if self.use_zone_filter else -1
                item_idx, adj_pos = self._assign_item(
                    pos, preview_short, preview_type_assigned,
                    claimed_items, strict=True, zone_filter=zf)
                if item_idx is None and self.use_zone_filter:
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
                    continue

            # Park in zone
            park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets


class ZoneNightmareSolver(NightmareSolverV6):
    """V6 with zone-aware allocation."""

    def __init__(self, ms, tables, future_orders=None,
                 use_zone_filter=True, zone_delivery=True,
                 drop_d_weight=0.8):
        super().__init__(ms, tables, future_orders=future_orders)
        self.allocator = ZoneV6Allocator(
            ms, tables, self.drop_zones,
            max_preview_pickers=99,
            drop_d_weight=drop_d_weight,
            use_zone_filter=use_zone_filter,
            zone_delivery=zone_delivery)


def run_sim(seed, use_zone_filter=True, zone_delivery=True, drop_d_weight=0.8, verbose=False):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = ZoneNightmareSolver(ms, tables, future_orders=all_orders,
                                  use_zone_filter=use_zone_filter,
                                  zone_delivery=zone_delivery,
                                  drop_d_weight=drop_d_weight)
    num_rounds = DIFF_ROUNDS['nightmare']
    action_log = []
    for rnd in range(num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(actions)
        step(state, actions, all_orders)
    if verbose:
        print(f"  Score={state.score} Ord={state.orders_completed} Items={state.items_delivered}")
    return state.score, action_log


def main():
    seeds = list(range(1000, 1010))

    configs = [
        ("V6 baseline (ddw=0.4)", dict(use_zone_filter=False, zone_delivery=False, drop_d_weight=0.4)),
        ("V6 ddw=0.8", dict(use_zone_filter=False, zone_delivery=False, drop_d_weight=0.8)),
        ("Zone filter only", dict(use_zone_filter=True, zone_delivery=False, drop_d_weight=0.8)),
        ("Zone delivery only", dict(use_zone_filter=False, zone_delivery=True, drop_d_weight=0.8)),
        ("Zone filter + delivery", dict(use_zone_filter=True, zone_delivery=True, drop_d_weight=0.8)),
    ]

    for name, cfg in configs:
        scores = []
        t0 = time.time()
        for seed in seeds:
            score, _ = run_sim(seed, **cfg)
            scores.append(score)
        elapsed = time.time() - t0
        print(f"{name:30s}: mean={np.mean(scores):6.1f} min={min(scores):4d} max={max(scores):4d} ({elapsed:.1f}s)")
        for i, seed in enumerate(seeds):
            print(f"  {seed}: {scores[i]}")


if __name__ == '__main__':
    main()
