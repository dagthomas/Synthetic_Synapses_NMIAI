"""Pipeline task allocator for nightmare mode.

Key insight: 20 bots can pre-fetch items for 8-10 future orders simultaneously.
When the active order completes, chain reactions auto-deliver staged items,
potentially completing 3-5 orders in a single round.

Bot roles:
  TRIGGER (3-5 bots): Complete the active order ASAP
  STAGER (10-15 bots): Pre-fetch items for orders N+1 through N+K, stage at dropoff
  DEAD (0-3 bots): Items matching no upcoming order, parked out of the way
"""
from __future__ import annotations

from game_engine import MapState, Order, INV_CAP
from precompute import PrecomputedTables


class NightmarePipelineAlloc:
    """Pipeline allocator: trigger active + stage future orders across all 20 bots."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]]):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        self.spawn = ms.spawn

        # Item index: type_id -> [(item_idx, adjacencies, zone_idx)]
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

        # Zone assignments: sort dropoffs LEFT/MID/RIGHT by x
        sorted_drops = sorted(drop_zones, key=lambda d: d[0])
        self.zone_dropoff: dict[int, tuple[int, int]] = {}
        for i, dz in enumerate(sorted_drops):
            self.zone_dropoff[i] = dz
        self.bot_zone: dict[int, int] = {}
        for bid in range(20):
            if bid < 7:
                self.bot_zone[bid] = 0
            elif bid < 14:
                self.bot_zone[bid] = 1
            else:
                self.bot_zone[bid] = 2

    def _shelf_zone(self, x: int) -> int:
        best, best_d = 0, 9999
        for zi, dz in self.zone_dropoff.items() if hasattr(self, 'zone_dropoff') else enumerate(self.drop_zones):
            d = abs(x - dz[0]) if isinstance(dz, tuple) else abs(x - self.drop_zones[zi][0])
            if d < best_d:
                best_d = d
                best = zi
        return best

    def allocate(self, bot_positions: dict[int, tuple[int, int]],
                 bot_inventories: dict[int, list[int]],
                 active_order: Order | None,
                 preview_order: Order | None,
                 round_num: int,
                 num_rounds: int = 500,
                 future_orders: list[Order] | None = None,
                 chain_plan=None,
                 allow_preview_pickup: bool = True,
                 pipeline_orders: list[Order] | None = None,
                 max_preview_pickers_override: int = -1,
                 ) -> tuple[dict, dict, dict]:
        """Pipeline allocation: trigger team + stager teams.

        Returns (goals, goal_types, pickup_targets) same as NightmareTaskAlloc.
        """
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        # Build pipeline of upcoming orders
        pipeline: list[Order] = []
        if active_order:
            pipeline.append(active_order)
        if preview_order:
            pipeline.append(preview_order)
        if future_orders:
            pipeline.extend(future_orders[:8])  # up to 10 total

        if not pipeline:
            # No orders at all — everyone parks
            occupied = set()
            for bid in bot_positions:
                park = self._corridor_parking(bot_positions[bid], occupied)
                occupied.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'
            return goals, goal_types, pickup_targets

        # ── DEMAND: what types does each pipeline order still need? ──
        # demand[order_idx] = {type_id: count_needed}
        demand: list[dict[int, int]] = []
        for order in pipeline:
            needs: dict[int, int] = {}
            for t in order.needs():
                needs[t] = needs.get(t, 0) + 1
            demand.append(needs)

        # Aggregate: total demand across all pipeline orders
        total_demand: dict[int, int] = {}
        for d in demand:
            for t, n in d.items():
                total_demand[t] = total_demand.get(t, 0) + n

        # ── SUPPLY: what's already in bot inventories? ──
        active_needs = dict(demand[0]) if demand else {}

        # Classify bots by what they're carrying
        active_carriers: list[int] = []
        pipeline_carriers: list[int] = []  # carry items for ANY pipeline order
        dead_bots: list[int] = []
        empty_bots: list[int] = []

        # Track supply already committed
        supply_by_type: dict[int, int] = {}  # type -> count in all inventories
        active_supply: dict[int, int] = {}   # type -> count matching active

        for bid, inv in bot_inventories.items():
            if not inv:
                empty_bots.append(bid)
                continue

            has_active = any(t in active_needs for t in inv)
            has_pipeline = any(t in total_demand for t in inv)

            if has_active:
                active_carriers.append(bid)
                for t in inv:
                    supply_by_type[t] = supply_by_type.get(t, 0) + 1
                    if t in active_needs:
                        active_supply[t] = active_supply.get(t, 0) + 1
            elif has_pipeline:
                pipeline_carriers.append(bid)
                for t in inv:
                    supply_by_type[t] = supply_by_type.get(t, 0) + 1
            else:
                dead_bots.append(bid)
                for t in inv:
                    supply_by_type[t] = supply_by_type.get(t, 0) + 1

        # Active shortfall
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - active_supply.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Pipeline demand minus existing supply
        pipeline_unfulfilled: dict[int, int] = {}
        for t, total_need in total_demand.items():
            remaining = total_need - supply_by_type.get(t, 0)
            if remaining > 0:
                pipeline_unfulfilled[t] = remaining

        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}
        type_assigned: dict[int, int] = {}  # types assigned to empty bots this round
        claimed_items: set[int] = set()
        occupied_goals: set[tuple[int, int]] = set()

        # ════════════════════════════════════════════════════════════════
        # PASS 1: ACTIVE CARRIERS → deliver to dropoff
        # ════════════════════════════════════════════════════════════════
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # Try to fill spare slots with active-needed items on the way
            if len(inv) < INV_CAP and active_short:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, active_short, type_assigned, claimed_items)
                if item_idx is not None:
                    d_to_item = self.tables.get_distance(pos, adj_pos)
                    d_item_to_drop = self._drop_dist(adj_pos)
                    d_direct = self._drop_dist(pos)
                    detour = d_to_item + d_item_to_drop - d_direct
                    if detour <= 4:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        continue

            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] = dropoff_loads.get(dz, 0) + 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # ════════════════════════════════════════════════════════════════
        # PASS 2: PIPELINE CARRIERS → stage at dropoff (wait for chain)
        # ════════════════════════════════════════════════════════════════
        staging_counts: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        for bid in pipeline_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # If at a dropoff, stay (waiting for chain reaction)
            if pos in self.drop_set:
                goals[bid] = pos
                goal_types[bid] = 'stage'
                staging_counts[pos] = staging_counts.get(pos, 0) + 1
                continue

            # Fill spare slots with pipeline-needed items on the way to dropoff
            if len(inv) < INV_CAP and pipeline_unfulfilled:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, pipeline_unfulfilled, type_assigned,
                    claimed_items, strict=True)
                if item_idx is not None:
                    d_to_item = self.tables.get_distance(pos, adj_pos)
                    d_item_to_drop = self._drop_dist(adj_pos)
                    d_direct = self._drop_dist(pos)
                    detour = d_to_item + d_item_to_drop - d_direct
                    if detour <= 5:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        continue

            # Go to best dropoff zone (least busy, closest)
            best_zone = self._staging_dropoff(pos, staging_counts, dropoff_loads)
            if best_zone:
                staging_counts[best_zone] = staging_counts.get(best_zone, 0) + 1
                goals[bid] = best_zone
                goal_types[bid] = 'stage'
            else:
                park = self._corridor_parking(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'flee'

        # ════════════════════════════════════════════════════════════════
        # PASS 3: DEAD BOTS → park in corridors
        # ════════════════════════════════════════════════════════════════
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # ════════════════════════════════════════════════════════════════
        # PASS 4: EMPTY BOTS → active pickup, then pipeline prefetch
        # ════════════════════════════════════════════════════════════════
        # Sort by proximity to nearest needed item
        empty_by_proximity = sorted(empty_bots, key=lambda bid:
            self._min_dist_to_types(bot_positions[bid],
                active_short.keys() if active_short else pipeline_unfulfilled.keys()))

        # Cap pipeline prefetchers to avoid congestion (too many bots = gridlock)
        max_pipeline_pickers = 8
        pipeline_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]

            # 1) Active pickup: highest priority
            if active_short:
                remaining = sum(max(0, s - type_assigned.get(t, 0))
                                for t, s in active_short.items())
                if remaining > 0:
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

            # 2) Preview pickup (order N+1): second priority
            preview_needs_remaining: dict[int, int] = {}
            if len(demand) > 1:
                for t, n in demand[1].items():
                    assigned = type_assigned.get(t, 0)
                    if n - assigned > 0:
                        preview_needs_remaining[t] = n - assigned
            if preview_needs_remaining and pipeline_assigned < max_pipeline_pickers:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, preview_needs_remaining, type_assigned,
                    claimed_items, strict=True)
                if item_idx is not None:
                    goals[bid] = adj_pos
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = item_idx
                    tid = int(self.ms.item_types[item_idx])
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    claimed_items.add(item_idx)
                    pipeline_assigned += 1
                    continue

            # 3) Pipeline prefetch (orders N+2..N+K): pick high-frequency types
            if pipeline_unfulfilled and pipeline_assigned < max_pipeline_pickers:
                # Focus on types needed in 2+ pipeline orders (safer — less dead inv risk)
                type_frequency: dict[int, int] = {}
                for d in demand[2:]:  # skip active and preview
                    for t in d:
                        if t in pipeline_unfulfilled:
                            type_frequency[t] = type_frequency.get(t, 0) + 1

                # Only prefetch types that appear in multiple future orders
                safe_prefetch: dict[int, int] = {}
                for t, freq in type_frequency.items():
                    if freq >= 2:  # needed in 2+ future orders
                        remaining = pipeline_unfulfilled[t] - type_assigned.get(t, 0)
                        if remaining > 0:
                            safe_prefetch[t] = remaining

                if safe_prefetch:
                    type_urgency = {t: type_frequency.get(t, 1.0) for t in safe_prefetch}
                    item_idx, adj_pos = self._assign_item_weighted(
                        bid, pos, safe_prefetch, type_urgency,
                        type_assigned, claimed_items)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        pipeline_assigned += 1
                        continue

            # 4) No work available → park
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets

    # ── Item assignment helpers ──────────────────────────────────────

    def _assign_item(self, bot_id: int, bot_pos: tuple[int, int],
                     needed: dict[int, int],
                     assigned_counts: dict[int, int],
                     claimed: set[int],
                     strict: bool = False,
                     ) -> tuple[int | None, tuple[int, int] | None]:
        """Find nearest item matching needed types."""
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
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    drop_d = self._drop_dist(adj)
                    cost = d + drop_d * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj

        return best_idx, best_adj

    def _assign_item_weighted(self, bot_id: int, bot_pos: tuple[int, int],
                              needed: dict[int, int],
                              type_urgency: dict[int, float],
                              assigned_counts: dict[int, int],
                              claimed: set[int],
                              ) -> tuple[int | None, tuple[int, int] | None]:
        """Find item maximizing urgency-weighted value / distance."""
        best_idx = None
        best_adj = None
        best_score = -1.0

        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            if assigned_counts.get(tid, 0) >= need_count:
                continue
            urgency = type_urgency.get(tid, 1.0)
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    drop_d = self._drop_dist(adj)
                    total_d = max(1, d + drop_d * 0.4)
                    score = urgency / total_d
                    if score > best_score:
                        best_score = score
                        best_idx = item_idx
                        best_adj = adj

        return best_idx, best_adj

    # ── Distance helpers ──────────────────────────────────────────

    def _drop_dist(self, pos: tuple[int, int]) -> int:
        """Distance to nearest dropoff zone."""
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

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

    def _staging_dropoff(self, pos: tuple[int, int],
                         staging_counts: dict[tuple[int, int], int],
                         delivery_loads: dict[tuple[int, int], int]
                         ) -> tuple[int, int] | None:
        best = None
        best_score = 9999
        for dz in self.drop_zones:
            stage_count = staging_counts.get(dz, 0)
            if stage_count >= 5:
                continue
            d = self.tables.get_distance(pos, dz)
            deliver_load = delivery_loads.get(dz, 0)
            score = d + stage_count * 6 + deliver_load * 4
            if score < best_score:
                best_score = score
                best = dz
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
                            if any(self.tables.get_distance(cell, dz) <= 1
                                   for dz in self.drop_zones):
                                continue
                            d = self.tables.get_distance(pos, cell)
                            if 0 < d < best_d:
                                best_d = d
                                best = cell
        return best

    def _min_dist_to_types(self, pos: tuple[int, int], types) -> int:
        best = 9999
        for tid in types:
            for item_idx, adj_cells, zone in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best
