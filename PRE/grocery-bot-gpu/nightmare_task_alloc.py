"""Task allocation (MRTA) for nightmare mode.

Pipeline strategy: active carriers deliver, preview carriers park NEAR dropoff
for fast delivery when order changes, empty bots pre-fetch preview items.
"""
from __future__ import annotations

from game_engine import (
    MapState, Order, INV_CAP,
)
from precompute import PrecomputedTables
from nightmare_chain_planner import ChainPlanner, ChainPlan


class NightmareTaskAlloc:
    """Allocator: active-first with near-dropoff preview staging."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        self.spawn = ms.spawn

        self.chain_planner = ChainPlanner(ms, tables, drop_zones)

        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            ix = int(ms.item_positions[idx, 0])
            adj = ms.item_adjacencies.get(idx, [])
            zone = self._shelf_zone(ix)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        self.corridor_ys = [1, ms.height // 2, ms.height - 3]

        # Zone assignments: bots 0-6=LEFT, 7-13=MID, 14-19=RIGHT
        self.bot_zone: dict[int, int] = {}
        self.zone_dropoff: dict[int, tuple[int, int]] = {}
        # Sort dropoffs by x coordinate for LEFT/MID/RIGHT
        sorted_drops = sorted(drop_zones, key=lambda d: d[0])
        for i, dz in enumerate(sorted_drops):
            self.zone_dropoff[i] = dz
        for bid in range(20):
            if bid < 7:
                self.bot_zone[bid] = 0  # LEFT
            elif bid < 14:
                self.bot_zone[bid] = 1  # MID
            else:
                self.bot_zone[bid] = 2  # RIGHT

        self._preview_bot_types: dict[int, int] = {}
        self._last_preview_id: int = -1

        # Goal persistence: keep pickup assignments across rounds
        # {bid: (item_idx, adj_pos, goal_type, type_id)}
        self._persistent_goals: dict[int, tuple[int, tuple[int, int], str, int]] = {}
        self._last_active_oid: int = -1

        # Pre-compute near-dropoff parking spots (2-3 cells from each dropoff)
        self._near_drop_cells: list[tuple[int, int]] = []
        for dz in drop_zones:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    cell = (dz[0] + dx, dz[1] + dy)
                    if cell in self.drop_set:
                        continue  # Don't park ON dropoff
                    if cell not in tables.pos_to_idx:
                        continue  # Not walkable
                    d = tables.get_distance(cell, dz)
                    if 1 <= d <= 3:
                        self._near_drop_cells.append(cell)
        # Deduplicate
        self._near_drop_cells = list(set(self._near_drop_cells))

    def _shelf_zone(self, shelf_x: int) -> int:
        if shelf_x <= 9:
            return 0
        elif shelf_x <= 17:
            return 1
        else:
            return 2

    def _nearest_drop(self, pos: tuple[int, int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _drop_dist(self, pos: tuple[int, int]) -> int:
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _near_drop_parking(self, pos: tuple[int, int],
                           occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find a parking spot 2-3 cells from nearest dropoff."""
        best = self.spawn
        best_score = 9999
        for cell in self._near_drop_cells:
            if cell in occupied:
                continue
            d_from_bot = self.tables.get_distance(pos, cell)
            d_to_drop = self._drop_dist(cell)
            # Prefer cells close to bot AND close to dropoff
            score = d_from_bot + d_to_drop * 2
            if score < best_score:
                best_score = score
                best = cell
        return best

    def allocate(self, bot_positions: dict[int, tuple[int, int]],
                 bot_inventories: dict[int, list[int]],
                 active_order: Order | None,
                 preview_order: Order | None,
                 round_num: int,
                 num_rounds: int = 500,
                 future_orders: list[Order] | None = None,
                 chain_plan: ChainPlan | None = None,
                 allow_preview_pickup: bool = True,
                 pipeline_orders: list[Order] | None = None,
                 max_preview_pickers_override: int = -1):
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        # Active order analysis
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # (persistent goal tracking disabled — greedy replanning is better)

        # Preview order analysis
        preview_needs: dict[int, int] = {}
        preview_oid = -1
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1
            preview_oid = preview_order.id

        # Reset persistent preview tracking if preview order changed
        if preview_oid != self._last_preview_id:
            self._preview_bot_types.clear()
            self._last_preview_id = preview_oid
        for bid in list(self._preview_bot_types.keys()):
            inv = bot_inventories.get(bid, [])
            assigned_t = self._preview_bot_types[bid]
            if assigned_t not in inv and inv:
                del self._preview_bot_types[bid]

        # Classify bots
        carrying_active: dict[int, int] = {}
        carrying_preview: dict[int, int] = {}
        active_carriers: list[int] = []
        preview_carriers: list[int] = []
        dead_bots: list[int] = []
        empty_bots: list[int] = []

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

        # Shortfall
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())

        preview_assigned_types: dict[int, int] = dict(carrying_preview)
        for bid, t in self._preview_bot_types.items():
            if t in preview_needs:
                inv = bot_inventories.get(bid, [])
                if t not in inv:
                    preview_assigned_types[t] = preview_assigned_types.get(t, 0) + 1
        preview_short: dict[int, int] = {}
        for t, need in preview_needs.items():
            s = need - preview_assigned_types.get(t, 0)
            if s > 0:
                preview_short[t] = s

        type_assigned: dict[int, int] = {}
        preview_type_assigned: dict[int, int] = dict(preview_assigned_types)
        claimed_items: set[int] = set()
        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        # === DELIVER: active carriers ===
        fill_up_bots: list[int] = []
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

        # === FILL-UP: active carriers picking more items ===
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

            # No active fill → try preview items only if ON THE WAY to dropoff
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

        # === PREVIEW CARRIERS: stage at non-deliver dropoff zones ===
        deliver_zones: set[tuple[int, int]] = set()
        for bid_d in goals:
            if goal_types.get(bid_d) == 'deliver' and goals[bid_d] in self.drop_set:
                deliver_zones.add(goals[bid_d])

        staging_counts: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}
        max_staging_per_zone = 6

        occupied_goals = set(goals.values())
        for bid in preview_carriers:
            pos = bot_positions[bid]

            # Stage at non-deliver zone to avoid blocking deliveries
            best_zone = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_zones:
                    continue
                if staging_counts[dz] >= max_staging_per_zone:
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

        # === DEAD BOTS: flee to corridors ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === EMPTY BOTS: active pickup → preview fetch → park ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        if max_preview_pickers_override >= 0:
            max_preview_pickers = max_preview_pickers_override
        else:
            # Scale preview pickers with map openness (more walkable = less congestion)
            walkable_count = len(self.tables.pos_to_idx)
            if walkable_count >= 210:
                max_preview_pickers = min(10, len(empty_by_proximity)) if allow_preview_pickup else 0
            else:
                max_preview_pickers = min(4, len(empty_by_proximity)) if allow_preview_pickup else 0
        preview_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]

            # Active pickup first (global — urgency over zone preference)
            if active_short:
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

            # Preview pickup (relaxed: allow even when active not fully covered,
            # as long as enough bots are assigned to active)
            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if preview_short and preview_assigned < max_preview_pickers:
                # Only if active has enough assigned OR no active items needed
                active_pickers = sum(1 for b, gt in goal_types.items() if gt == 'pickup')
                if remaining_active == 0 or active_pickers >= remaining_active:
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

            # Park in corridor — out of the way
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets

    def _assign_item(self, bot_id: int, bot_pos: tuple[int, int],
                     needed: dict[int, int],
                     assigned_counts: dict[int, int],
                     claimed: set[int],
                     strict: bool = False,
                     zone_filter: int = -1) -> tuple[int | None, tuple[int, int] | None]:
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
                    if zone_filter >= 0:
                        # Use bot's zone dropoff for cost
                        drop_d = self.tables.get_distance(
                            adj, self.zone_dropoff[zone_filter])
                    else:
                        drop_d = self._drop_dist(adj)
                    cost = d + drop_d * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj

        return best_idx, best_adj

    def _staging_dropoff(self, pos: tuple[int, int],
                         staging_counts: dict[tuple[int, int], int],
                         delivery_loads: dict[tuple[int, int], int]) -> tuple[int, int] | None:
        """Find best dropoff for staging (waiting for auto-delivery).

        Prefers zones with fewer staging bots and fewer active deliverers.
        Returns None if all zones are too busy.
        """
        best = None
        best_score = 9999
        for dz in self.drop_zones:
            stage_count = staging_counts.get(dz, 0)
            if stage_count >= 2:
                continue  # Too many stagers at this zone
            d = self.tables.get_distance(pos, dz)
            deliver_load = delivery_loads.get(dz, 0)
            score = d + stage_count * 8 + deliver_load * 4
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _balanced_dropoff(self, pos: tuple[int, int],
                          loads: dict[tuple[int, int], int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 5
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _near_dropoff_cell(self, pos: tuple[int, int],
                           dz: tuple[int, int],
                           occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find a walkable cell 1-2 steps from dropoff zone."""
        best = dz
        best_d = 9999
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = (dz[0] + dx, dz[1] + dy)
                if cell in self.drop_set or cell in occupied:
                    continue
                if cell not in self.tables.pos_to_idx:
                    continue
                d_to_drop = self.tables.get_distance(cell, dz)
                if d_to_drop < 1 or d_to_drop > 2:
                    continue
                d_from_bot = self.tables.get_distance(pos, cell)
                score = d_from_bot + d_to_drop * 2
                if score < best_d:
                    best_d = score
                    best = cell
        return best

    def _corridor_parking(self, pos: tuple[int, int],
                          occupied: set[tuple[int, int]]) -> tuple[int, int]:
        best = self.spawn
        best_d = 9999
        for cy in self.corridor_ys:
            for dx in range(15):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width:
                        cell = (cx, cy)
                        if cell in self.tables.pos_to_idx and cell not in occupied:
                            if any(self.tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                                continue
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
        return best

    def _min_dist_to_types(self, pos: tuple[int, int], types) -> int:
        best = 9999
        for tid in types:
            for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best
