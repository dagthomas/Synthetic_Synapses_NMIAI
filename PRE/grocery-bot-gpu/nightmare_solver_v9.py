"""Nightmare solver V9: Committed trip planner with chain optimization.

Key improvements over V6:
1. Plan persistence — goals don't change every round (eliminates oscillation)
2. State machine per bot — PICK → DELIVER → PICK or STAGE → WAIT
3. Order-change triggered replanning — not per-round
4. Future-order aware preview pickup — reduces dead inventory
5. Chain-optimized staging — preview items staged at dropoffs before chain fires
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


class BotPlan:
    __slots__ = ['goal', 'goal_type', 'item_idx', 'dropoff_target',
                 'order_version', 'created_round']

    def __init__(self, goal, goal_type, item_idx=-1, dropoff_target=None,
                 order_version=0, created_round=0):
        self.goal = goal
        self.goal_type = goal_type  # deliver, active_pick, preview_pick, stage, park, flee
        self.item_idx = item_idx
        self.dropoff_target = dropoff_target
        self.order_version = order_version
        self.created_round = created_round


class NightmareSolverV9:
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

        # Item lookup by type
        self.type_items: dict[int, list[tuple[int, list, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            ix = int(ms.item_positions[idx, 0])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            self.type_items.setdefault(tid, []).append((idx, adj, zone))

        # Persistent plans
        self.plans: dict[int, BotPlan] = {}
        self.order_version = 0
        self.last_active_id = -1
        self.last_preview_id = -1

        # Future orders
        self.future_orders = future_orders or []
        self.future_type_set: set[int] = set()

        # Stall detection
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Bot zones
        self.bot_zone = {}
        for bid in range(self.num_bots):
            self.bot_zone[bid] = 0 if bid < 7 else (1 if bid < 14 else 2)

        self.corridor_ys = [1, ms.height // 2, ms.height - 3]

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        ms = self.ms
        num_bots = min(self.num_bots, len(state.bot_positions))

        # Build current state
        positions = {}
        inventories = {}
        for bid in range(num_bots):
            positions[bid] = (int(state.bot_positions[bid, 0]),
                              int(state.bot_positions[bid, 1]))
            inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(positions.values()))

        for bid in range(num_bots):
            pos = positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active = state.get_active_order()
        preview = state.get_preview_order()

        # Compute needs
        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Detect order change
        active_id = active.id if active else -1
        preview_id = preview.id if preview else -1

        if active_id != self.last_active_id:
            self.order_version += 1
            self.plans.clear()
            self.last_active_id = active_id
            self.last_preview_id = preview_id
            self._update_future_types(state, all_orders)
        elif preview_id != self.last_preview_id:
            self.last_preview_id = preview_id
            for bid in list(self.plans.keys()):
                if self.plans[bid].goal_type in ('preview_pick', 'stage'):
                    del self.plans[bid]
            self._update_future_types(state, all_orders)

        # Active shortfall (carrying only, not planned)
        carrying_active = {}
        for bid in range(num_bots):
            for t in inventories[bid]:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # Validate plans, collect bots needing replan
        needs_replan = self._validate_plans(
            num_bots, positions, inventories, active_needs, preview_needs,
            preview, active_short, rnd)

        if needs_replan:
            self._assign_plans(needs_replan, positions, inventories,
                               active, preview, active_needs, preview_needs, rnd)

        # Build goals from plans
        goals = {}
        goal_types = {}
        pickup_targets = {}
        for bid in range(num_bots):
            plan = self.plans.get(bid)
            if plan:
                goals[bid] = plan.goal
                gt = plan.goal_type
                if gt == 'active_pick':
                    gt = 'pickup'
                elif gt == 'preview_pick':
                    gt = 'preview'
                goal_types[bid] = gt
                if plan.item_idx >= 0:
                    pickup_targets[bid] = plan.item_idx
            else:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'

        # Urgency order
        def _urg(bid):
            gt = goal_types.get(bid, 'park')
            d = self.tables.get_distance(positions[bid], goals[bid])
            pri = {'deliver': 0, 'flee': 1, 'pickup': 2, 'preview': 3,
                   'stage': 3, 'park': 5}.get(gt, 5)
            return (pri, d)
        urgency_order = sorted(range(num_bots), key=_urg)

        path_actions = self.pathfinder.plan_all(
            positions, goals, urgency_order, goal_types=goal_types)

        # Execute
        actions = [(ACT_WAIT, -1)] * num_bots
        for bid in range(num_bots):
            pos = positions[bid]
            gt = goal_types.get(bid, 'park')
            inv = inventories.get(bid, [])

            if self.stall_counts.get(bid, 0) >= 5:
                actions[bid] = (self._escape_action(bid, pos, rnd), -1)
                continue

            if pos in self.drop_set and gt == 'deliver' and inv:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if pos == goals[bid]:
                    actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                    continue

            if len(inv) < INV_CAP:
                opp = self._adjacent_pickup(bid, pos, active, preview, inv,
                                            active_short, gt)
                if opp is not None:
                    actions[bid] = opp
                    continue

            actions[bid] = (path_actions.get(bid, ACT_WAIT), -1)

        return actions

    def _update_future_types(self, state, all_orders):
        self.future_type_set = set()
        idx = state.next_order_idx
        for i in range(5):
            if idx + i < len(all_orders):
                for t in all_orders[idx + i].required:
                    self.future_type_set.add(int(t))

    def _validate_plans(self, num_bots, positions, inventories,
                        active_needs, preview_needs, preview,
                        active_short, rnd):
        needs_replan = []

        # Count what's planned for active picking
        planned_active = {}
        for bid, plan in self.plans.items():
            if plan.goal_type == 'active_pick' and plan.item_idx >= 0:
                tid = int(self.ms.item_types[plan.item_idx])
                planned_active[tid] = planned_active.get(tid, 0) + 1

        for bid in range(num_bots):
            plan = self.plans.get(bid)
            pos = positions[bid]
            inv = inventories.get(bid, [])

            if plan is None:
                needs_replan.append(bid)
                continue

            # Bot acquired active items via adjacent pickup → switch to deliver
            if plan.goal_type not in ('deliver', 'active_pick'):
                if any(t in active_needs for t in inv):
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue

            if plan.goal_type == 'deliver':
                if pos in self.drop_set and not any(t in active_needs for t in inv):
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue
                if not inv:
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue
                if not any(t in active_needs for t in inv):
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue

            elif plan.goal_type == 'active_pick':
                if plan.item_idx >= 0:
                    tid = int(self.ms.item_types[plan.item_idx])
                    if pos == plan.goal and tid in inv:
                        needs_replan.append(bid)
                        del self.plans[bid]
                        continue
                    # Over-assigned? More pickers than shortfall
                    total = planned_active.get(tid, 0)
                    short = active_short.get(tid, 0)
                    if total > short + 1:
                        # This bot is redundant — replan
                        planned_active[tid] -= 1
                        needs_replan.append(bid)
                        del self.plans[bid]
                        continue

            elif plan.goal_type == 'preview_pick':
                if plan.item_idx >= 0:
                    tid = int(self.ms.item_types[plan.item_idx])
                    if pos == plan.goal and tid in inv:
                        needs_replan.append(bid)
                        del self.plans[bid]
                        continue
                    if preview and not preview.needs_type(tid):
                        needs_replan.append(bid)
                        del self.plans[bid]
                        continue

            elif plan.goal_type == 'stage':
                if pos in self.drop_set:
                    continue  # Waiting — don't replan

            elif plan.goal_type in ('park', 'flee'):
                if not inv and (active_short or preview_needs):
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue
                if len(inv) < INV_CAP and active_short:
                    needs_replan.append(bid)
                    del self.plans[bid]
                    continue

            # Preview pickers should switch to active when there's shortfall
            if plan.goal_type == 'preview_pick' and active_short:
                # Only switch if bot hasn't picked up preview item yet
                if plan.item_idx >= 0:
                    tid = int(self.ms.item_types[plan.item_idx])
                    if tid not in inv:
                        needs_replan.append(bid)
                        del self.plans[bid]
                        continue

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 8:
                if bid in self.plans:
                    del self.plans[bid]
                needs_replan.append(bid)

        return needs_replan

    def _assign_plans(self, unplanned, positions, inventories,
                      active, preview, active_needs, preview_needs, rnd):
        # Compute coverage from existing plans
        carrying_active = {}
        picking_active = {}
        carrying_preview = {}
        picking_preview = {}

        for bid, plan in self.plans.items():
            inv = inventories.get(bid, [])
            if plan.goal_type == 'deliver':
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
            elif plan.goal_type == 'active_pick' and plan.item_idx >= 0:
                tid = int(self.ms.item_types[plan.item_idx])
                picking_active[tid] = picking_active.get(tid, 0) + 1
            elif plan.goal_type == 'stage':
                for t in inv:
                    if t in preview_needs:
                        carrying_preview[t] = carrying_preview.get(t, 0) + 1
            elif plan.goal_type == 'preview_pick' and plan.item_idx >= 0:
                tid = int(self.ms.item_types[plan.item_idx])
                picking_preview[tid] = picking_preview.get(tid, 0) + 1

        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0) - picking_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        preview_short = {}
        for t, need in preview_needs.items():
            s = need - carrying_preview.get(t, 0) - picking_preview.get(t, 0)
            if s > 0:
                preview_short[t] = s

        dropoff_loads = {dz: 0 for dz in self.drop_zones}
        stage_counts = {dz: 0 for dz in self.drop_zones}
        for plan in self.plans.values():
            if plan.goal_type == 'deliver' and plan.dropoff_target:
                dropoff_loads[plan.dropoff_target] += 1
            if plan.goal_type == 'stage' and plan.dropoff_target:
                stage_counts[plan.dropoff_target] += 1

        claimed_adj = set()
        for plan in self.plans.values():
            if plan.goal_type in ('active_pick', 'preview_pick') and plan.goal:
                claimed_adj.add(plan.goal)

        # Classify unplanned bots
        to_deliver = []
        to_stage = []
        empties = []
        deads = []

        for bid in unplanned:
            inv = inventories.get(bid, [])
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv) if preview else False
            if has_active:
                to_deliver.append(bid)
            elif has_preview:
                to_stage.append(bid)
            elif len(inv) < INV_CAP:
                empties.append(bid)
            else:
                deads.append(bid)

        # DELIVER: active carriers
        for bid in to_deliver:
            pos = positions[bid]
            inv = inventories[bid]
            free_slots = INV_CAP - len(inv)

            if free_slots > 0 and sum(active_short.values()) > 0:
                item_idx, adj = self._find_best_item(pos, active_short, claimed_adj)
                if item_idx is not None:
                    dz = self._nearest_drop(pos)
                    pick_d = self.tables.get_distance(pos, adj)
                    drop_d = self.tables.get_distance(pos, dz)
                    if pick_d < drop_d and pick_d < 10:
                        self.plans[bid] = BotPlan(
                            adj, 'active_pick', item_idx,
                            order_version=self.order_version, created_round=rnd)
                        claimed_adj.add(adj)
                        tid = int(self.ms.item_types[item_idx])
                        active_short[tid] = max(0, active_short.get(tid, 0) - 1)
                        continue

            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            self.plans[bid] = BotPlan(
                dz, 'deliver', dropoff_target=dz,
                order_version=self.order_version, created_round=rnd)

        # STAGE: preview carriers
        deliver_drops = set()
        for plan in self.plans.values():
            if plan.goal_type == 'deliver' and plan.dropoff_target:
                deliver_drops.add(plan.dropoff_target)

        for bid in to_stage:
            pos = positions[bid]
            best_dz = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_drops:
                    continue  # Never stage where deliverers go
                if stage_counts.get(dz, 0) >= 1:
                    continue  # Max 1 stager per dropoff
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_dz = dz

            if best_dz and best_d < 30:
                stage_counts[best_dz] += 1
                self.plans[bid] = BotPlan(
                    best_dz, 'stage', dropoff_target=best_dz,
                    order_version=self.order_version, created_round=rnd)
            else:
                park = self._find_parking(pos, claimed_adj)
                claimed_adj.add(park)
                self.plans[bid] = BotPlan(
                    park, 'flee',
                    order_version=self.order_version, created_round=rnd)

        # EMPTY BOTS: active pickup, then preview pickup
        empties_sorted = sorted(empties, key=lambda bid:
            self._min_dist_to_types(positions[bid],
                active_short if active_short else preview_short))

        for bid in empties_sorted:
            pos = positions[bid]

            if active_short and sum(active_short.values()) > 0:
                item_idx, adj = self._find_best_item(pos, active_short, claimed_adj)
                if item_idx is not None:
                    self.plans[bid] = BotPlan(
                        adj, 'active_pick', item_idx,
                        order_version=self.order_version, created_round=rnd)
                    claimed_adj.add(adj)
                    tid = int(self.ms.item_types[item_idx])
                    active_short[tid] = max(0, active_short.get(tid, 0) - 1)
                    continue

            if preview_short and sum(preview_short.values()) > 0:
                item_idx, adj = self._find_best_item(
                    pos, preview_short, claimed_adj, strict=True)
                if item_idx is not None:
                    self.plans[bid] = BotPlan(
                        adj, 'preview_pick', item_idx,
                        order_version=self.order_version, created_round=rnd)
                    claimed_adj.add(adj)
                    tid = int(self.ms.item_types[item_idx])
                    preview_short[tid] = max(0, preview_short.get(tid, 0) - 1)
                    continue

            park = self._find_parking(pos, claimed_adj,
                                       zone=self.bot_zone.get(bid, 2))
            claimed_adj.add(park)
            self.plans[bid] = BotPlan(
                park, 'park',
                order_version=self.order_version, created_round=rnd)

        # DEAD BOTS
        for bid in deads:
            pos = positions[bid]
            park = self._find_parking(pos, claimed_adj,
                                       zone=self.bot_zone.get(bid, 2))
            claimed_adj.add(park)
            self.plans[bid] = BotPlan(
                park, 'flee',
                order_version=self.order_version, created_round=rnd)

    def _find_best_item(self, pos, needed, claimed_adj, strict=False):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            for item_idx, adj_cells, zone in self.type_items.get(tid, []):
                for adj in adj_cells:
                    if adj in claimed_adj:
                        continue
                    d = self.tables.get_distance(pos, adj)
                    drop_d = min(self.tables.get_distance(adj, dz)
                                 for dz in self.drop_zones)
                    future_bonus = -2 if tid in self.future_type_set else 0
                    cost = d + drop_d * 0.4 + future_bonus
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def _nearest_drop(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

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

    def _find_parking(self, pos, occupied=None, zone=-1):
        occupied = occupied or set()
        best = self.spawn
        best_d = 9999
        zone_x_ranges = {0: (1, 9), 1: (10, 19), 2: (20, 28)}
        x_lo, x_hi = zone_x_ranges.get(zone, (0, self.ms.width - 1))

        for cy in self.corridor_ys:
            for cx in range(x_lo, x_hi + 1):
                cell = (cx, cy)
                if cell in self.tables.pos_to_idx and cell not in occupied:
                    if any(self.tables.get_distance(cell, dz) <= 1
                           for dz in self.drop_zones):
                        continue
                    d = self.tables.get_distance(pos, cell)
                    if 0 < d < best_d:
                        best_d = d
                        best = cell

        if best == self.spawn:
            for cy in self.corridor_ys:
                for cx in range(1, self.ms.width - 1):
                    cell = (cx, cy)
                    if cell in self.tables.pos_to_idx and cell not in occupied:
                        d = self.tables.get_distance(pos, cell)
                        if 0 < d < best_d:
                            best_d = d
                            best = cell
        return best

    def _min_dist_to_types(self, pos, types):
        if not types:
            return 9999
        best = 9999
        for tid in types:
            for _, adj_cells, _ in self.type_items.get(tid, []):
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    if d < best:
                        best = d
        return best

    def _escape_action(self, bid, pos, rnd):
        dirs = list(MOVES)
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def _adjacent_pickup(self, bid, pos, active, preview, inv,
                         active_short, gt):
        ms = self.ms
        bot_types = set(inv)
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
            elif not active_short and preview and preview.needs_type(tid):
                if tid in bot_types:
                    continue
            elif gt == 'preview' and preview and preview.needs_type(tid):
                if tid in bot_types:
                    continue
            else:
                continue

            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    if tid in active_short and active_short[tid] > 0:
                        active_short[tid] -= 1
                    return (ACT_PICKUP, item_idx)
        return None

    # === WebSocket interface ===

    def ws_action(self, live_bots: list[dict], data: dict,
                  map_state: MapState) -> list[dict]:
        ms = map_state or self.ms

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

        bot_pos = {}
        bot_inv = {}
        for bot in live_bots:
            bid = bot['id']
            bot_pos[bid] = tuple(bot['position'])
            inv = []
            for name in bot.get('inventory', []):
                tid = ms.type_name_to_id.get(name, -1)
                if tid >= 0:
                    inv.append(tid)
            bot_inv[bid] = inv

        rnd = data.get('round', 0)
        num_bots = len(live_bots)

        self.congestion.update(list(bot_pos.values()))
        for bid, pos in bot_pos.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Order change detection
        active_id = id(active_order) if active_order else -1
        preview_id = id(preview_order) if preview_order else -1

        # Use needs hash for WS (no persistent order IDs)
        needs_hash = tuple(sorted(active_needs.items()))
        preview_hash = tuple(sorted(preview_needs.items()))
        ah = hash(needs_hash)
        ph = hash(preview_hash)

        if ah != self.last_active_id:
            self.order_version += 1
            self.plans.clear()
            self.last_active_id = ah
            self.last_preview_id = ph
        elif ph != self.last_preview_id:
            self.last_preview_id = ph
            for bid in list(self.plans.keys()):
                if self.plans[bid].goal_type in ('preview_pick', 'stage'):
                    del self.plans[bid]

        carrying_active = {}
        for bid, inv in bot_inv.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        needs_replan = self._validate_plans(
            num_bots, bot_pos, bot_inv, active_needs, preview_needs,
            preview_order, active_short, rnd)

        if needs_replan:
            self._assign_plans(needs_replan, bot_pos, bot_inv,
                               active_order, preview_order,
                               active_needs, preview_needs, rnd)

        goals = {}
        goal_types = {}
        pickup_targets = {}
        all_bids = [bot['id'] for bot in live_bots]
        for bid in all_bids:
            plan = self.plans.get(bid)
            if plan:
                goals[bid] = plan.goal
                gt = plan.goal_type
                if gt == 'active_pick':
                    gt = 'pickup'
                elif gt == 'preview_pick':
                    gt = 'preview'
                goal_types[bid] = gt
                if plan.item_idx >= 0:
                    pickup_targets[bid] = plan.item_idx
            else:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'

        pri_map = {'deliver': 0, 'pickup': 1, 'stage': 2, 'preview': 3,
                   'flee': 4, 'park': 5}
        urgency_order = sorted(all_bids, key=lambda bid: (
            pri_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos.get(bid, self.spawn),
                                     goals.get(bid, self.spawn))))

        path_actions = self.pathfinder.plan_all(
            bot_pos, goals, urgency_order, goal_types=goal_types)

        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left',
                        'move_right', 'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            inv_names = bot.get('inventory', [])
            inv_ids = bot_inv.get(bid, [])

            if self.stall_counts.get(bid, 0) >= 5:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            if pos in self.drop_set and gt == 'deliver' and inv_names:
                ws_actions.append({'bot': bid, 'action': 'drop_off'})
                continue

            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goals[bid] and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    })
                    continue

            if len(inv_names) < INV_CAP and active_short:
                opp = self._ws_adjacent_active(bid, pos, ms, active_short,
                                               set(inv_ids))
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            if len(inv_names) < INV_CAP and not active_short and preview_order:
                opp = self._ws_adjacent_preview(bid, pos, ms, preview_order,
                                                set(inv_ids))
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_adjacent_active(self, bid, pos, ms, active_short, bot_types):
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short or active_short[tid] <= 0:
                continue
            if tid in bot_types and active_short[tid] <= 1:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {
                        'bot': bid, 'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    def _ws_adjacent_preview(self, bid, pos, ms, preview_order, bot_types):
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if not preview_order.needs_type(tid):
                continue
            if tid in bot_types:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {
                        'bot': bid, 'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    # === Simulation ===

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV9(ms, tables, future_orders=all_orders)
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
    parser = argparse.ArgumentParser(description='Nightmare solver V9')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = NightmareSolverV9.run_sim(seed, verbose=args.verbose)
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
