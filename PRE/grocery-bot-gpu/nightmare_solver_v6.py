"""Nightmare solver V6: Clean reimplementation with improvements.

Based on proven V3+NightmareTaskAlloc approach (233 baseline).
Own file to avoid external modification conflicts.

Key improvements over baseline:
1. Rush mode: preview carriers rush to dropoff when active almost done
2. Larger staging cap to enable chains
3. Better dead bot handling
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

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class V6Allocator:
    """Task allocator: faithful to proven NightmareTaskAlloc + improvements."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]],
                 max_preview_pickers: int = 4,
                 drop_d_weight: float = 0.4):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        self.spawn = ms.spawn
        self.max_preview_pickers = max_preview_pickers
        self.drop_d_weight = drop_d_weight

        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        # Reverse lookup: position → [(item_idx, type_id)]
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            ix = int(ms.item_positions[idx, 0])
            adj = ms.item_adjacencies.get(idx, [])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))
            for a in adj:
                if a not in self.pos_to_items:
                    self.pos_to_items[a] = []
                self.pos_to_items[a].append((idx, tid))

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
        self._committed_stages: dict[int, tuple[int, int]] = {}  # bid → dropoff

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

        # Precompute corridor parking candidates per zone
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28), -1: (0, ms.width - 1)}
        self._corridor_park_cells: dict[int, list[tuple[int, int]]] = {}
        near_drop = set()
        for cell in tables.pos_to_idx:
            if any(tables.get_distance(cell, dz) <= 1 for dz in drop_zones):
                near_drop.add(cell)
        for zone_id in [-1, 0, 1, 2]:
            x_lo, x_hi = zone_x_ranges[zone_id]
            cells = []
            for cy in self.corridor_ys:
                for cx in range(x_lo, x_hi + 1):
                    cell = (cx, cy)
                    if cell in tables.pos_to_idx and cell not in near_drop:
                        cells.append(cell)
            self._corridor_park_cells[zone_id] = cells
        # Fallback: all corridor cells (no dropoff exclusion)
        self._corridor_park_fallback: dict[int, list[tuple[int, int]]] = {}
        for zone_id in [-1, 0, 1, 2]:
            x_lo, x_hi = zone_x_ranges[zone_id]
            cells = []
            for cy in self.corridor_ys:
                for cx in range(x_lo, x_hi + 1):
                    cell = (cx, cy)
                    if cell in tables.pos_to_idx:
                        cells.append(cell)
            self._corridor_park_fallback[zone_id] = cells

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

    def _balanced_dropoff(self, pos, loads, exclude_zone=None):
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            if exclude_zone and dz == exclude_zone:
                continue
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 5
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _assign_item(self, bot_pos, needed, assigned_counts, claimed,
                     strict=False, zone_filter=-1, type_bonus=None):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            max_assign = need_count if strict else need_count + 1
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

    def _corridor_parking(self, pos, occupied, zone=-1, target_row=-1):
        best = self.spawn
        best_d = 9999
        candidates = self._corridor_park_cells.get(zone, self._corridor_park_cells[-1])
        for cell in candidates:
            if target_row >= 0 and cell[1] != target_row:
                continue
            if cell not in occupied:
                d = self.tables.get_distance(pos, cell)
                if 0 < d < best_d:
                    best_d = d
                    best = cell
        if best == self.spawn:
            fallback = self._corridor_park_fallback.get(zone, self._corridor_park_fallback[-1])
            for cell in fallback:
                if cell not in occupied:
                    d = self.tables.get_distance(pos, cell)
                    if 0 < d < best_d:
                        best_d = d
                        best = cell
        return best

    def _shelf_parking(self, pos, occupied, zone=-1):
        """Park adjacent to a shelf item in the bot's zone for fast future pickup."""
        best = None
        best_d = 9999
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28)}
        x_lo, x_hi = zone_x_ranges.get(zone, (0, self.ms.width - 1))

        for item_idx in range(self.ms.num_items):
            ix = int(self.ms.item_positions[item_idx, 0])
            if ix < x_lo or ix > x_hi:
                continue
            for adj in self.ms.item_adjacencies.get(item_idx, []):
                if adj in occupied or adj in self.drop_set:
                    continue
                if any(self.tables.get_distance(adj, dz) <= 1 for dz in self.drop_zones):
                    continue
                d = self.tables.get_distance(pos, adj)
                if 0 < d < best_d:
                    best_d = d
                    best = adj

        if best is not None:
            return best
        return self._corridor_parking(pos, occupied, zone=zone)

    def _near_drop_parking(self, pos, occupied):
        """Park 2-3 cells from nearest dropoff, out of traffic."""
        best = self.spawn
        best_score = 9999
        for cell in self._near_drop_cells:
            if cell in occupied:
                continue
            d_from_bot = self.tables.get_distance(pos, cell)
            d_to_drop = self._drop_dist(cell)
            score = d_from_bot + d_to_drop * 2
            if score < best_score:
                best_score = score
                best = cell
        return best

    def _shelf_parking_for_types(self, pos, occupied, target_types, zone=-1):
        """Park adjacent to a shelf item of a target type for fast pickup."""
        best = None
        best_d = 9999
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28)}
        x_lo, x_hi = zone_x_ranges.get(zone, (0, self.ms.width - 1))

        for tid in target_types:
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                ix = int(self.ms.item_positions[item_idx, 0])
                if ix < x_lo or ix > x_hi:
                    continue
                for adj in adj_cells:
                    if adj in occupied or adj in self.drop_set:
                        continue
                    if any(self.tables.get_distance(adj, dz) <= 1 for dz in self.drop_zones):
                        continue
                    d = self.tables.get_distance(pos, adj)
                    if 0 < d < best_d:
                        best_d = d
                        best = adj

        if best is not None:
            return best
        return self._corridor_parking(pos, occupied, zone=zone)

    def _min_dist_to_types(self, pos, types):
        best = 9999
        for tid in types:
            for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

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

        # Future order analysis: types needed across next N orders
        future_type_freq = {}  # type → count of future orders needing it
        future_all_needs = {}  # type → total count needed
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
                # Dead inventory but free slots → can still pick up useful items
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
                # Detour-based fill-up: worth it if detour is small
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
                    pos, filtered_short, type_assigned, claimed_items)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    assigned_fill = True

            # Try preview on the way
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

            # Active pickup
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

            # Preview pickup (all surplus bots work on preview)
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


class NightmareSolverV6:
    """V6: Clean reimplementation of proven approach."""

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
        self.allocator = V6Allocator(ms, tables, self.drop_zones,
                                     max_preview_pickers=99, drop_d_weight=0.4)

        self.future_orders = future_orders or []
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        # Oscillation detection: track last N positions per bot
        self._pos_history: dict[int, list[tuple[int, int]]] = {}
        self._OSCILLATION_WINDOW = 16  # Check last 16 rounds
        self._OSCILLATION_UNIQUE_THRESHOLD = 4  # Oscillation: <=4 unique cells in 16 rounds

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
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
            # Track position history for oscillation detection
            hist = self._pos_history.get(bid, [])
            hist.append(pos)
            if len(hist) > self._OSCILLATION_WINDOW:
                hist = hist[-self._OSCILLATION_WINDOW:]
            self._pos_history[bid] = hist

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Active shortfall
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

        # Future orders beyond preview (look ahead 8 orders)
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

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
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

    def _check_adjacent_pickup(self, bid, pos, active_order, preview_order,
                                goal_type, bot_inv, active_short):
        adjacent_items = self.allocator.pos_to_items.get(pos, [])
        if not adjacent_items:
            return None
        bot_types = set(bot_inv)
        total_short = sum(active_short.values())
        for item_idx, tid in adjacent_items:
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
            elif total_short == 0 and preview_order and preview_order.needs_type(tid):
                if tid in bot_types:
                    continue
            elif goal_type == 'preview' and preview_order and preview_order.needs_type(tid):
                pass
            else:
                continue
            if tid in active_short and active_short[tid] > 0:
                active_short[tid] -= 1
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

    def ws_action(self, live_bots: list[dict], data: dict, map_state: MapState) -> list[dict]:
        """Per-round entry for live_gpu_stream.py WebSocket format."""
        ms = map_state or self.ms

        # Build order objects from WS data
        orders_data = data.get('orders', [])
        active_order = None
        preview_order = None
        for od in orders_data:
            items_req = od.get('items_required', [])
            items_del = od.get('items_delivered', [])
            req_ids = [ms.type_name_to_id.get(n, 0) for n in items_req]
            order = Order(0, req_ids, od.get('status', 'active'))
            for dn in items_del:
                tid = ms.type_name_to_id.get(dn, -1)
                if tid >= 0:
                    order.deliver_type(tid)
            if od.get('status') == 'active':
                active_order = order
            elif od.get('status') == 'preview':
                preview_order = order

        # Build state dicts
        bot_pos_dict = {}
        bot_inv_dict = {}
        for bot in live_bots:
            bid = bot['id']
            bot_pos_dict[bid] = tuple(bot['position'])
            inv = []
            for item_name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(item_name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv_dict[bid] = inv

        rnd = data.get('round', 0)
        num_rounds = data.get('max_rounds', 500)

        # Update congestion and stall
        self.congestion.update(list(bot_pos_dict.values()))
        for bid, pos in bot_pos_dict.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Compute active shortfall
        active_needs = {}
        carrying_active = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid2, inv in bot_inv_dict.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Task allocation
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_pos_dict, bot_inv_dict,
            active_order, preview_order, rnd, num_rounds)

        # Urgency order
        priority_map = {'deliver': 0, 'pickup': 1, 'stage': 2, 'preview': 3, 'flee': 4, 'park': 5}
        all_bids = [bot['id'] for bot in live_bots]
        urgency_order = sorted(all_bids, key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos_dict.get(bid, self.spawn),
                                     goals.get(bid, self.spawn))
        ))

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_pos_dict, goals, urgency_order, goal_types=goal_types)

        # Build WS actions
        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right',
                        'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            inv_names = bot.get('inventory', [])
            inv_ids = bot_inv_dict.get(bid, [])

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            # At dropoff: deliver if goal is 'deliver'
            if pos in self.drop_set and gt == 'deliver' and inv_names:
                ws_actions.append({'bot': bid, 'action': 'drop_off'})
                continue

            # At pickup target
            if gt in ('pickup', 'preview', 'future_pickup') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    })
                    continue

            # Opportunistic adjacent pickup (active items)
            if len(inv_names) < INV_CAP and active_short:
                opp = self._ws_adjacent_active(bid, pos, ms, active_short, set(inv_ids))
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # Opportunistic adjacent preview pickup when active fully covered
            if (len(inv_names) < INV_CAP and not active_short
                    and preview_order):
                opp = self._ws_adjacent_preview(bid, pos, ms, preview_order, set(inv_ids))
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # Use pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_adjacent_active(self, bid, pos, ms, active_short, bot_types):
        """Pick up adjacent item if type still needed by active order."""
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short:
                continue
            if tid in bot_types and active_short[tid] <= 1:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    def _ws_adjacent_preview(self, bid, pos, ms, preview_order, bot_types):
        """Pick up adjacent preview item when active is fully covered."""
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if not preview_order.needs_type(tid):
                continue
            if tid in bot_types:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    @staticmethod
    def run_from_capture(capture_data: dict, verbose: bool = False) -> tuple[int, list]:
        """Run V6 heuristic from capture data (for production pipeline)."""
        from game_engine import init_game_from_capture
        num_orders = len(capture_data.get('orders', []))
        state, all_orders = init_game_from_capture(capture_data, num_orders=max(num_orders, 40))
        return NightmareSolverV6._run_internal(state, all_orders, verbose)

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        return NightmareSolverV6._run_internal(state, all_orders, verbose)

    @staticmethod
    def _run_internal(state: GameState, all_orders: list[Order],
                      verbose: bool = False) -> tuple[int, list]:
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
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
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s")
        return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Nightmare solver V6')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = NightmareSolverV6.run_sim(seed, verbose=args.verbose)
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
