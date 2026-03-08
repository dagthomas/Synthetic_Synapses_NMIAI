"""V4 task allocator for nightmare mode - immune to external modifications.

Key improvements over baseline:
- drop_d weight = 0.0 (pure distance-to-item routing)
- Fill-up: 1.5x drop_dist threshold, 15 cell cap
- 1x active coverage (not 2x)
"""
from __future__ import annotations

from game_engine import MapState, Order, INV_CAP
from precompute import PrecomputedTables
from nightmare_chain_planner import ChainPlanner, ChainPlan


class NightmareAllocV4:

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        self.spawn = ms.spawn

        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            ix = int(ms.item_positions[idx, 0])
            adj = ms.item_adjacencies.get(idx, [])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        self.corridor_ys = [1, ms.height // 2, ms.height - 3]

        self.bot_zone: dict[int, int] = {}
        self.zone_dropoff: dict[int, tuple[int, int]] = {}
        sorted_drops = sorted(drop_zones, key=lambda d: d[0])
        for i, dz in enumerate(sorted_drops):
            self.zone_dropoff[i] = dz
        for bid in range(20):
            self.bot_zone[bid] = 0 if bid < 7 else (1 if bid < 14 else 2)

        self._preview_bot_types: dict[int, int] = {}
        self._last_preview_id: int = -1

        self._near_drop_cells: list[tuple[int, int]] = []
        for dz in drop_zones:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    cell = (dz[0] + dx, dz[1] + dy)
                    if cell in self.drop_set:
                        continue
                    if cell not in tables.pos_to_idx:
                        continue
                    d = tables.get_distance(cell, dz)
                    if 1 <= d <= 3:
                        self._near_drop_cells.append(cell)
        self._near_drop_cells = list(set(self._near_drop_cells))

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def allocate(self, bot_positions, bot_inventories,
                 active_order, preview_order,
                 round_num, num_rounds=500,
                 future_orders=None, chain_plan=None,
                 allow_preview_pickup=True,
                 pipeline_orders=None):
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

        # === DELIVER ===
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

        # === FILL-UP ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bot_types = set(inv)
            assigned_fill = False
            filtered_short = {t: s for t, s in active_short.items()
                              if t not in bot_types or s > 1}
            if filtered_short:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, filtered_short, type_assigned, claimed_items)
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
                        bid, pos, preview_filtered, preview_type_assigned,
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

        # === PREVIEW CARRIERS ===
        deliver_zones = set()
        for bid_d in goals:
            if goal_types.get(bid_d) == 'deliver' and goals[bid_d] in self.drop_set:
                deliver_zones.add(goals[bid_d])

        staging_counts = {dz: 0 for dz in self.drop_zones}
        occupied_goals = set(goals.values())

        # When active order almost done, rush preview carriers to ANY dropoff
        rush_mode = total_short <= 2 and active_order is not None

        for bid in preview_carriers:
            pos = bot_positions[bid]
            best_zone = None
            best_d = 9999
            for dz in self.drop_zones:
                if not rush_mode and dz in deliver_zones:
                    continue
                if staging_counts[dz] >= 2 if rush_mode else staging_counts[dz] >= 1:
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
                bz = self.bot_zone.get(bid, 2)
                park = self._corridor_parking(pos, occupied_goals, zone=bz)
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

        # === EMPTY BOTS ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        max_preview_pickers = min(4, len(empty_by_proximity)) if allow_preview_pickup else 0
        preview_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]
            bz = self.bot_zone.get(bid, 2)

            # 1x active coverage
            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active > 0:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, active_short, type_assigned, claimed_items,
                    zone_filter=bz)
                if item_idx is None:
                    item_idx, adj_pos = self._assign_item(
                        bid, pos, active_short, type_assigned, claimed_items)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    continue

            # Preview pickup
            if preview_short and preview_assigned < max_preview_pickers:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, preview_short, preview_type_assigned,
                    claimed_items, strict=True, zone_filter=bz)
                if item_idx is None:
                    item_idx, adj_pos = self._assign_item(
                        bid, pos, preview_short, preview_type_assigned,
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
            if not allow_preview_pickup:
                target_cy = self.corridor_ys[bid % len(self.corridor_ys)]
                park = self._corridor_parking(pos, occupied_goals, zone=bz,
                                              target_row=target_cy)
            else:
                park = self._corridor_parking(pos, occupied_goals, zone=bz)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets

    def _assign_item(self, bot_id, bot_pos, needed, assigned_counts,
                     claimed, strict=False, zone_filter=-1):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            max_assign = need_count if strict else need_count + 1
            if assigned_counts.get(tid, 0) >= max_assign:
                continue
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                if zone_filter >= 0 and item_zone != zone_filter:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    if d < best_cost:
                        best_cost = d
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

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

    def _corridor_parking(self, pos, occupied, zone=-1, target_row=-1):
        best = self.spawn
        best_d = 9999
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28)}
        x_lo, x_hi = zone_x_ranges.get(zone, (0, self.ms.width - 1))
        rows = [target_row] if target_row >= 0 else self.corridor_ys
        for cy in rows:
            for cx in range(x_lo, x_hi + 1):
                cell = (cx, cy)
                if cell in self.tables.pos_to_idx and cell not in occupied:
                    if any(self.tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                        continue
                    d = self.tables.get_distance(pos, cell)
                    if 0 < d < best_d:
                        best_d = d
                        best = cell
        if best == self.spawn:
            for cy in self.corridor_ys:
                for cx in range(x_lo, x_hi + 1):
                    cell = (cx, cy)
                    if cell in self.tables.pos_to_idx and cell not in occupied:
                        d = self.tables.get_distance(pos, cell)
                        if 0 < d < best_d:
                            best_d = d
                            best = cell
        return best

    def _min_dist_to_types(self, pos, types):
        best = 9999
        for tid in types:
            for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best
