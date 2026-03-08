#!/usr/bin/env python3
"""Test V6 with zone-filtered active item assignment."""
import time
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
from game_engine import init_game, step, INV_CAP
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

seeds = [1000, 1003, 1006, 1009, 7009]

# Save originals
orig_allocate = V6Allocator.allocate


def make_zone_allocate(use_zone_active, use_zone_preview, fallback_global):
    """Create allocator that filters items by bot zone."""
    orig = orig_allocate

    def patched(self, bot_positions, bot_inventories,
                active_order, preview_order, round_num, num_rounds=500,
                future_orders=None):
        goals = {}
        goal_types = {}
        pickup_targets = {}

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
        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._committed_stages.clear()
            self._last_preview_id = preview_oid
        for bid in list(self._preview_bot_types.keys()):
            inv = bot_inventories.get(bid, [])
            assigned_t = self._preview_bot_types[bid]
            if assigned_t not in inv and inv:
                del self._preview_bot_types[bid]

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

        # DELIVER
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
                min_item_dist = self._min_dist_to_types(pos, active_short.keys())
                if min_item_dist < drop_dist and min_item_dist < 10:
                    fill_up_bots.append(bid)
                else:
                    dz = self._balanced_dropoff(pos, dropoff_loads)
                    dropoff_loads[dz] += 1
                    goals[bid] = dz
                    goal_types[bid] = 'deliver'

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
                        if detour <= 6:
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

        # STAGING
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
                if dz in deliver_zones: continue
                if staging_counts[dz] >= 6: continue
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

        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # EMPTY BOTS — with optional zone filtering
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        for bid in empty_by_proximity:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)

            # Active pickup with zone filter
            if active_short:
                if use_zone_active:
                    item_idx, adj_pos = self._assign_item(
                        pos, active_short, type_assigned, claimed_items,
                        zone_filter=bz)
                    if item_idx is None and fallback_global:
                        item_idx, adj_pos = self._assign_item(
                            pos, active_short, type_assigned, claimed_items)
                else:
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

            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active == 0 and preview_short:
                if use_zone_preview:
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_short, preview_type_assigned,
                        claimed_items, strict=True, zone_filter=bz)
                    if item_idx is None and fallback_global:
                        item_idx, adj_pos = self._assign_item(
                            pos, preview_short, preview_type_assigned,
                            claimed_items, strict=True)
                else:
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

            park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets
    return patched


configs = [
    ("baseline", False, False, False),
    ("zone_active", True, False, True),
    ("zone_preview", False, True, True),
    ("zone_both", True, True, True),
    ("zone_both_strict", True, True, False),  # no global fallback
]

for name, za, zp, fb in configs:
    V6Allocator.allocate = make_zone_allocate(za, zp, fb)
    scores = []
    for seed in seeds:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
        for rnd in range(500):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            step(state, actions, all_orders)
        scores.append(state.score)
    print(f'{name:25s}: mean={sum(scores)/len(scores):.1f} scores={scores}')

V6Allocator.allocate = orig_allocate
