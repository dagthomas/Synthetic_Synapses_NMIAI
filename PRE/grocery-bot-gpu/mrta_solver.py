"""Generalized MRTA solver for all difficulties (medium, hard, expert, nightmare).

Adapts the NightmareSolverV2 architecture (MRTA + PIBT pathfinding + congestion)
to work with any bot count and dropoff configuration.

Usage:
    python mrta_solver.py hard --seeds 1000-1009 -v
    python mrta_solver.py expert --seeds 7005 -v
    python mrta_solver.py medium --seeds 42 -v
"""
from __future__ import annotations

import time

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF, actions_to_ws_format,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable


class MRTATaskAlloc:
    """Goal assignment for any bot count and dropoff configuration."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 drop_zones: list[tuple[int, int]], num_bots: int):
        self.ms = ms
        self.tables = tables
        self.drop_zones = drop_zones
        self.drop_set = set(drop_zones)
        self.spawn = ms.spawn
        self.num_bots = num_bots
        self.num_drops = len(drop_zones)

        # Build type_id -> [(item_idx, [adj_cells], zone)]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            ix = int(ms.item_positions[idx, 0])
            adj = ms.item_adjacencies.get(idx, [])
            zone = self._shelf_zone(ix) if self.num_drops > 1 else 0
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))

        # Zone assignment for multi-dropoff
        self.bot_zone: dict[int, int] = {}
        if self.num_drops > 1:
            bots_per_zone = num_bots // self.num_drops
            remainder = num_bots % self.num_drops
            idx = 0
            for z in range(self.num_drops):
                count = bots_per_zone + (1 if z < remainder else 0)
                for _ in range(count):
                    self.bot_zone[idx] = z
                    idx += 1
        else:
            for bid in range(num_bots):
                self.bot_zone[bid] = 0

        # Zone -> dropoff
        self.zone_drop = {}
        for i, dz in enumerate(drop_zones):
            self.zone_drop[i] = dz

        # Corridor parking rows
        self.corridor_ys = [1, ms.height // 2, ms.height - 3]

        # Persistent preview tracking
        self._preview_bot_types: dict[int, int] = {}
        self._last_preview_id: int = -1

    def _shelf_zone(self, shelf_x: int) -> int:
        if self.num_drops <= 1:
            return 0
        # Divide map into equal zones by x coordinate
        zone_width = self.ms.width / self.num_drops
        return min(int(shelf_x / zone_width), self.num_drops - 1)

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

    def allocate(self, bot_positions: dict[int, tuple[int, int]],
                 bot_inventories: dict[int, list[int]],
                 active_order: Order | None,
                 preview_order: Order | None,
                 round_num: int,
                 num_rounds: int = 300):
        """Assign goals to all bots.

        Returns:
            goals: {bid: (x, y)}
            goal_types: {bid: str}
            pickup_targets: {bid: int}
        """
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        # Active order analysis
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

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

        # Preview shortfall
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

        # Tracking
        type_assigned: dict[int, int] = {}
        preview_type_assigned: dict[int, int] = dict(preview_assigned_types)
        claimed_items: set[int] = set()
        dropoff_loads: dict[tuple[int, int], int] = {dz: 0 for dz in self.drop_zones}

        # === DELIVER vs FILL-UP: bots with active items ===
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

        # === STAGE: preview carriers ===
        # When active order almost done, rush to dropoff for chain reaction.
        # Otherwise park in corridors (avoid blocking active carriers).
        preview_carriers.sort(key=lambda b: min(
            self.tables.get_distance(bot_positions[b], dz) for dz in self.drop_zones))
        for bid in preview_carriers:
            pos = bot_positions[bid]
            if 0 < total_short <= 3:
                # Rush to dropoff for auto-delivery chain
                dz = self._balanced_dropoff(pos, dropoff_loads)
                dropoff_loads[dz] += 1
                goals[bid] = dz
                goal_types[bid] = 'deliver'
            else:
                park = self._corridor_parking(pos, set(goals.values()))
                goals[bid] = park
                goal_types[bid] = 'flee'

        # === FLEE: dead inventory bots ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._corridor_parking(pos, set(goals.values()))
            goals[bid] = park
            goal_types[bid] = 'flee'

        # === FILL-UP: active carriers that should pick more ===
        for bid in fill_up_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bot_types = set(inv)
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
                    continue
            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # === PICKUP: empty bots ===
        empty_by_proximity = sorted(empty_bots, key=lambda bid: self._min_dist_to_types(
            bot_positions[bid], active_short.keys() if active_short else preview_needs.keys()))

        # Scale preview pickers by bot count
        max_preview_pickers = max(1, self.num_bots // 4)
        preview_assigned = 0

        for bid in empty_by_proximity:
            pos = bot_positions[bid]

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

            remaining_active = sum(max(0, s - type_assigned.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active == 0 and preview_short and preview_assigned < max_preview_pickers:
                item_idx, adj_pos = self._assign_item(
                    bid, pos, preview_short, preview_type_assigned, claimed_items, strict=True)
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

            park = self._corridor_parking(pos, set(goals.values()))
            goals[bid] = park
            goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets

    def _assign_item(self, bot_id: int, bot_pos: tuple[int, int],
                     needed: dict[int, int],
                     assigned_counts: dict[int, int],
                     claimed: set[int],
                     strict: bool = False) -> tuple[int | None, tuple[int, int] | None]:
        """Find best item for bot. Returns (item_idx, adj_pos)."""
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

    def _balanced_dropoff(self, pos: tuple[int, int],
                          loads: dict[tuple[int, int], int]) -> tuple[int, int]:
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 3
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _orbit_cell(self, dropoff: tuple[int, int],
                    occupied: set[tuple[int, int]]) -> tuple[int, int]:
        dx, dy = dropoff
        candidates = [
            (dx, dy - 1), (dx - 1, dy), (dx + 1, dy),
            (dx, dy - 2), (dx - 2, dy), (dx + 2, dy),
        ]
        for cell in candidates:
            if cell in self.tables.pos_to_idx and cell not in occupied and cell not in self.drop_set:
                return cell
        return dropoff

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


class MRTASolver:
    """Per-round MRTA solver for any difficulty with 2+ bots."""

    def __init__(self, difficulty: str, map_state: MapState,
                 precomputed_tables: PrecomputedTables | None = None):
        self.difficulty = difficulty
        self.ms = map_state
        self.tables = precomputed_tables or PrecomputedTables.get(map_state)
        self.walkable = build_walkable(map_state)
        self.drop_zones = [tuple(dz) for dz in map_state.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = map_state.spawn
        self.num_bots = CONFIGS[difficulty]['bots']
        self.num_rounds = DIFF_ROUNDS[difficulty]

        # Subsystems
        self.traffic = TrafficRules(map_state)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(map_state, self.tables,
                                               self.traffic, self.congestion)
        self.allocator = MRTATaskAlloc(map_state, self.tables, self.drop_zones, self.num_bots)

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        """Per-round entry point. Returns [(action_type, item_idx), ...] per bot."""
        num_bots = len(state.bot_positions)

        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Active shortfall for opportunistic pickup
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid2, inv in bot_inventories.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Task allocation
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, self.num_rounds)

        # Urgency ordering
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
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        path_actions = self.pathfinder.plan_all(bot_positions, goals, urgency_order)

        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape (3+ rounds stuck at same position)
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # At dropoff: deliver or stage
            if pos in self.drop_set:
                if gt == 'deliver':
                    if bot_inventories[bid]:
                        actions[bid] = (ACT_DROPOFF, -1)
                        continue
                elif gt == 'stage':
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # At pickup target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if gt in ('pickup', 'preview', 'deliver') and len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, state, active_order, preview_order, gt,
                    bot_inventories[bid], active_short)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            # Use pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_pickup(self, bid: int, pos: tuple[int, int],
                                state: GameState,
                                active_order: Order | None,
                                preview_order: Order | None,
                                goal_type: str,
                                bot_inv: list[int],
                                active_short: dict[int, int]) -> tuple[int, int] | None:
        ms = self.ms
        bot_types = set(bot_inv)

        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])

            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
            elif goal_type == 'preview' and preview_order and preview_order.needs_type(tid):
                pass
            else:
                continue

            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)

        return None

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int) -> int:
        """Anti-stall: pick a deterministic but varied direction."""
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
        num_bots = len(live_bots)

        bot_pos_dict = {}
        bot_inv_dict = {}
        for i, bot in enumerate(live_bots):
            bid = bot['id']
            bot_pos_dict[bid] = tuple(bot['position'])
            inv = []
            for item_name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(item_name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv_dict[bid] = inv

        rnd = data.get('round', 0)

        # Build order objects
        orders_data = data.get('orders', [])
        active_order = None
        preview_order = None
        for od in orders_data:
            items_req = od.get('items_required', [])
            req_ids = [ms.type_name_to_id.get(n, 0) for n in items_req]
            order = Order(0, req_ids, od.get('status', 'active'))
            for dn in od.get('items_delivered', []):
                tid = ms.type_name_to_id.get(dn, -1)
                if tid >= 0:
                    order.deliver_type(tid)
            if od.get('status') == 'active':
                active_order = order
            elif od.get('status') == 'preview':
                preview_order = order

        # Update congestion and stall
        self.congestion.update(list(bot_pos_dict.values()))
        for bid, pos in bot_pos_dict.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Active shortfall
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid2, inv in bot_inv_dict.items():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Task allocation
        num_rounds = data.get('max_rounds', self.num_rounds)
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

        path_actions = self.pathfinder.plan_all(bot_pos_dict, goals, urgency_order)

        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            inv_names = bot.get('inventory', [])

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            # At dropoff: deliver or wait
            if pos in self.drop_set:
                if gt == 'deliver' and inv_names:
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    continue
                elif gt == 'stage':
                    ws_actions.append({'bot': bid, 'action': 'wait'})
                    continue

            # At pickup target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id']
                    })
                    continue

            # Opportunistic adjacent pickup
            if len(inv_names) < INV_CAP and gt in ('pickup', 'preview', 'deliver', 'stage'):
                opp_pickup = self._ws_check_adjacent(bid, pos, ms, orders_data, active_short)
                if opp_pickup is not None:
                    ws_actions.append(opp_pickup)
                    continue

            # Pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_check_adjacent(self, bid: int, pos: tuple[int, int],
                            ms: MapState, orders_data: list,
                            active_short: dict[int, int]) -> dict | None:
        """Opportunistic adjacent pickup: only active-needed items."""
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short or active_short[tid] <= 0:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos and item_idx < len(ms.items):
                    return {
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    @staticmethod
    def run_sim(seed: int, difficulty: str, verbose: bool = False) -> tuple[int, list]:
        """Run full simulation. Returns (score, action_log)."""
        state, all_orders = init_game(seed, difficulty, num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = MRTASolver(difficulty, ms, tables)
        num_rounds = DIFF_ROUNDS[difficulty]
        chains = 0
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

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" +{c}!" if c > 1 else ""
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


DB_URL = "postgres://grocery:grocery123@localhost:5433/grocery_bot"


def record_to_pg(seed, difficulty, score, orders_completed, items_delivered, action_log, elapsed):
    """Record run to PostgreSQL."""
    import json
    import os
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("  psycopg2 not installed, skipping DB recording", file=__import__('sys').stderr)
        return None

    db_url = os.environ.get("GROCERY_DB_URL", DB_URL)
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        from game_engine import build_map, CELL_WALL, CELL_SHELF, state_to_ws_format, actions_to_ws_format
        ms = build_map(difficulty)
        cfg = CONFIGS[difficulty]

        walls = []
        shelves = []
        for y in range(ms.height):
            for x in range(ms.width):
                c = int(ms.grid[y, x])
                if c == CELL_WALL:
                    walls.append([x, y])
                elif c == CELL_SHELF:
                    shelves.append([x, y])

        items = [{"id": it["id"], "type": it["type"], "position": list(it["position"])}
                 for it in ms.items]

        cur.execute("""
            INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
                              item_types, order_size_min, order_size_max,
                              walls, shelves, items, drop_off, spawn,
                              final_score, items_delivered, orders_completed, run_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            seed, difficulty, ms.width, ms.height, cfg['bots'],
            ms.num_types, cfg['order_size'][0], cfg['order_size'][1],
            json.dumps(walls), json.dumps(shelves),
            json.dumps(items), json.dumps(list(ms.drop_off)),
            json.dumps(list(ms.spawn)),
            score, items_delivered, orders_completed,
            'synthetic',
        ))
        run_id = cur.fetchone()[0]

        if action_log:
            from game_engine import init_game, step as game_step
            gs, all_orders = init_game(seed, difficulty, num_orders=100)
            num_rounds = DIFF_ROUNDS[difficulty]
            round_tuples = []
            for rnd in range(min(len(action_log), num_rounds)):
                gs.round = rnd
                ws_data = state_to_ws_format(gs, all_orders)
                ws_acts = actions_to_ws_format(action_log[rnd], gs.map_state)
                bots = [{"id": b["id"], "position": b["position"],
                         "inventory": b.get("inventory", [])} for b in ws_data["bots"]]
                orders = [{"id": o["id"], "items_required": o["items_required"],
                           "items_delivered": o.get("items_delivered", []),
                           "status": o.get("status", "active")}
                          for o in ws_data.get("orders", [])]
                round_tuples.append((
                    run_id, rnd, json.dumps(bots), json.dumps(orders),
                    json.dumps(ws_acts), ws_data["score"], json.dumps([])
                ))
                game_step(gs, action_log[rnd], all_orders)

            execute_values(cur, """
                INSERT INTO rounds (run_id, round_number, bots, orders, actions, score, events)
                VALUES %s
            """, round_tuples, page_size=100)

        conn.commit()
        conn.close()
        print(f"  Recorded to DB: run_id={run_id}", file=__import__('sys').stderr)
        return run_id
    except Exception as e:
        print(f"  DB error: {e}", file=__import__('sys').stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MRTA solver for all difficulties')
    parser.add_argument('difficulty', choices=['medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-record', action='store_true', help='Skip PostgreSQL recording')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        score, action_log = MRTASolver.run_sim(seed, args.difficulty, verbose=args.verbose)
        elapsed = time.time() - st
        scores.append(score)
        print(f"Seed {seed}: {score}")

        if not args.no_record:
            state2, all_orders2 = init_game(seed, args.difficulty, num_orders=100)
            num_rounds = DIFF_ROUNDS[args.difficulty]
            for rnd, acts in enumerate(action_log):
                if rnd >= num_rounds:
                    break
                state2.round = rnd
                step(state2, acts, all_orders2)
            record_to_pg(seed, args.difficulty, score, state2.orders_completed,
                         state2.items_delivered, action_log, elapsed)

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()
