#!/usr/bin/env python3
"""Nightmare solver V7: Task-queue solver with persistent assignments.

Key differences from V6:
1. Tasks persist across rounds (bots keep their assignments until complete)
2. Split between active and preview work from the start
3. Preview bots stage at dropoffs for cascade potential
4. Dead bots stage near dropoffs (not corridors) if items match future orders
"""
from __future__ import annotations
import sys, time, copy
import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class BotTask:
    """Persistent task for a bot."""
    __slots__ = ['task_type', 'target', 'item_idx', 'item_type', 'dropoff',
                 'order_id', 'phase']

    def __init__(self, task_type, target=None, item_idx=-1, item_type=-1,
                 dropoff=None, order_id=-1, phase='go'):
        self.task_type = task_type  # 'pickup_active', 'pickup_preview', 'deliver', 'stage', 'park', 'flee'
        self.target = target       # (x, y) goal position
        self.item_idx = item_idx   # shelf item index for pickup
        self.item_type = item_type # type being picked up
        self.dropoff = dropoff     # which dropoff to deliver/stage at
        self.order_id = order_id   # which order this task is for
        self.phase = phase         # 'go' (to item) or 'return' (to dropoff)


class TaskQueueSolver:
    """Task-based solver with persistent cross-round assignments."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables
        self.drop_zones = sorted([tuple(dz) for dz in ms.drop_off_zones],
                                 key=lambda d: d[0])
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']
        self.future_orders = future_orders or []

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # Item lookup
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]], int]]] = {}
        self.pos_to_items: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            ix = int(ms.item_positions[idx, 0])
            zone = 0 if ix <= 9 else (1 if ix <= 17 else 2)
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj, zone))
            for a in adj:
                if a not in self.pos_to_items:
                    self.pos_to_items[a] = []
                self.pos_to_items[a].append((idx, tid))

        # Corridor cells for parking
        corridor_ys = [1, ms.height // 2, ms.height - 3]
        near_drop = set()
        for cell in tables.pos_to_idx:
            if any(tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                near_drop.add(cell)
        self._corridor_cells = []
        for cy in corridor_ys:
            for cx in range(ms.width):
                cell = (cx, cy)
                if cell in tables.pos_to_idx and cell not in near_drop:
                    self._corridor_cells.append(cell)

        # Near-drop parking cells
        self._near_drop_cells = []
        for dz in self.drop_zones:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    cell = (dz[0] + dx, dz[1] + dy)
                    if cell in self.drop_set:
                        continue
                    if cell not in tables.pos_to_idx:
                        continue
                    d = tables.get_distance(cell, dz)
                    if 1 <= d <= 3:
                        self._near_drop_cells.append(cell)
        self._near_drop_cells = list(set(self._near_drop_cells))

        # Persistent state
        self.bot_tasks: dict[int, BotTask] = {}
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._last_active_id = -1
        self._last_preview_id = -1

    def _nearest_drop(self, pos):
        return min(self.drop_zones, key=lambda dz: self.tables.get_distance(pos, dz))

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _balanced_dropoff(self, pos, loads):
        best = self.drop_zones[0]
        best_score = 9999
        for dz in self.drop_zones:
            d = self.tables.get_distance(pos, dz)
            score = d + loads.get(dz, 0) * 4
            if score < best_score:
                best_score = score
                best = dz
        return best

    def _corridor_parking(self, pos, occupied):
        best = self.spawn
        best_d = 9999
        for cell in self._corridor_cells:
            if cell not in occupied:
                d = self.tables.get_distance(pos, cell)
                if 0 < d < best_d:
                    best_d = d
                    best = cell
        return best

    def _near_drop_parking(self, pos, occupied):
        best = self.spawn
        best_score = 9999
        for cell in self._near_drop_cells:
            if cell in occupied:
                continue
            d_from = self.tables.get_distance(pos, cell)
            d_to_drop = self._drop_dist(cell)
            score = d_from + d_to_drop * 2
            if score < best_score:
                best_score = score
                best = cell
        return best

    def _find_item(self, bot_pos, needs, assigned_counts, claimed, drop_d_weight=0.4):
        """Find best item for given needs."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        best_tid = -1
        for tid, need_count in needs.items():
            if need_count <= 0:
                continue
            if assigned_counts.get(tid, 0) >= need_count + 1:
                continue
            for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    drop_d = self._drop_dist(adj)
                    cost = d + drop_d * drop_d_weight
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
                        best_tid = tid
        return best_idx, best_adj, best_tid

    def _update_tasks(self, state, active_order, preview_order):
        """Update persistent tasks based on current state."""
        active_id = active_order.id if active_order else -1
        preview_id = preview_order.id if preview_order else -1

        # Detect order change
        order_changed = (active_id != self._last_active_id)
        self._last_active_id = active_id
        self._last_preview_id = preview_id

        active_needs = set()
        if active_order:
            active_needs = set(active_order.needs())
        preview_needs = set()
        if preview_order:
            preview_needs = set(preview_order.needs())

        for bid in list(self.bot_tasks.keys()):
            task = self.bot_tasks[bid]
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)

            if task.task_type == 'pickup_active':
                if order_changed:
                    # Active order changed; check if our item still matches
                    if task.item_type not in active_needs:
                        # Item no longer needed for active
                        if task.item_type in inv:
                            # Already picked up, check if matches preview
                            if task.item_type in preview_needs:
                                self.bot_tasks[bid] = BotTask(
                                    'stage', self._nearest_drop(pos),
                                    order_id=preview_id)
                            else:
                                del self.bot_tasks[bid]
                        else:
                            del self.bot_tasks[bid]
                        continue

                if task.phase == 'go':
                    # Check if picked up
                    if task.item_type in inv:
                        # Transition to deliver
                        dz = self._nearest_drop(pos)
                        task.task_type = 'deliver'
                        task.target = dz
                        task.dropoff = dz
                        task.phase = 'return'
                elif task.phase == 'return':
                    # Check if delivered (at dropoff and did ACT_DROPOFF)
                    if pos in self.drop_set and task.item_type not in inv:
                        del self.bot_tasks[bid]

            elif task.task_type == 'pickup_preview':
                if order_changed:
                    # Preview order is now active! Change task to deliver
                    if task.item_type in inv:
                        dz = self._nearest_drop(pos)
                        self.bot_tasks[bid] = BotTask(
                            'deliver', dz, item_type=task.item_type,
                            dropoff=dz, phase='return')
                    elif task.item_type in active_needs:
                        # Still needed (now as active), keep going
                        task.task_type = 'pickup_active'
                    else:
                        del self.bot_tasks[bid]
                    continue

                if task.phase == 'go':
                    if task.item_type in inv:
                        # Picked up! Go to dropoff to stage
                        dz = self._nearest_drop(pos)
                        task.task_type = 'stage'
                        task.target = dz
                        task.dropoff = dz
                        task.phase = 'return'

            elif task.task_type == 'deliver':
                if pos in self.drop_set and task.item_type not in inv:
                    del self.bot_tasks[bid]
                elif task.item_type not in inv and pos not in self.drop_set:
                    # Item was auto-delivered via cascade while en route
                    del self.bot_tasks[bid]

            elif task.task_type == 'stage':
                if order_changed:
                    # Order changed; check if our items match the new active
                    has_active_items = any(t in active_needs for t in inv)
                    if has_active_items:
                        # Our items are now active! Deliver
                        self.bot_tasks[bid] = BotTask(
                            'deliver', self._nearest_drop(pos),
                            dropoff=self._nearest_drop(pos), phase='return')
                    else:
                        has_preview_items = any(t in preview_needs for t in inv)
                        if has_preview_items:
                            # Still useful for preview
                            pass
                        else:
                            del self.bot_tasks[bid]

            elif task.task_type in ('park', 'flee'):
                # Re-evaluate periodically
                if order_changed:
                    del self.bot_tasks[bid]

    def _assign_new_tasks(self, state, active_order, preview_order, rnd, num_rounds):
        """Assign tasks to unassigned bots."""
        bot_positions = {}
        bot_inventories = {}
        for bid in range(self.num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Compute needs
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Count already-assigned bots per type
        active_assigned = {}
        preview_assigned = {}
        for bid, task in self.bot_tasks.items():
            if task.task_type in ('pickup_active', 'deliver'):
                if task.item_type >= 0:
                    active_assigned[task.item_type] = active_assigned.get(task.item_type, 0) + 1
            elif task.task_type in ('pickup_preview', 'stage'):
                if task.item_type >= 0:
                    preview_assigned[task.item_type] = preview_assigned.get(task.item_type, 0) + 1

        # Active shortfall (accounting for assigned bots AND carrying bots)
        carrying_active = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            covered = carrying_active.get(t, 0) + active_assigned.get(t, 0)
            s = need - covered
            if s > 0:
                active_short[t] = s

        # Preview shortfall
        carrying_preview = {}
        for bid, inv in bot_inventories.items():
            if bid in self.bot_tasks:
                continue
            for t in inv:
                if t in preview_needs:
                    carrying_preview[t] = carrying_preview.get(t, 0) + 1
        preview_short = {}
        for t, need in preview_needs.items():
            covered = carrying_preview.get(t, 0) + preview_assigned.get(t, 0)
            s = need - covered
            if s > 0:
                preview_short[t] = s

        # Find unassigned bots
        unassigned = []
        for bid in range(self.num_bots):
            if bid not in self.bot_tasks:
                unassigned.append(bid)

        if not unassigned:
            return

        # Classify unassigned bots
        empty_bots = []
        active_carrier_bots = []
        preview_carrier_bots = []
        dead_bots = []

        for bid in unassigned:
            inv = bot_inventories[bid]
            pos = bot_positions[bid]
            if not inv:
                empty_bots.append(bid)
            elif any(t in active_needs for t in inv):
                active_carrier_bots.append(bid)
            elif any(t in preview_needs for t in inv):
                preview_carrier_bots.append(bid)
            elif len(inv) < INV_CAP:
                empty_bots.append(bid)
            else:
                dead_bots.append(bid)

        occupied_goals = set(t.target for t in self.bot_tasks.values() if t.target)
        claimed_items = set(t.item_idx for t in self.bot_tasks.values() if t.item_idx >= 0)
        dropoff_loads = {dz: 0 for dz in self.drop_zones}
        for t in self.bot_tasks.values():
            if t.task_type == 'deliver' and t.dropoff in self.drop_set:
                dropoff_loads[t.dropoff] = dropoff_loads.get(t.dropoff, 0) + 1

        # === Active carriers → deliver ===
        for bid in active_carrier_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            # Check for fill-up
            if free_slots > 0 and active_short:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                best_fill_item = None
                best_fill_adj = None
                best_fill_tid = -1
                best_detour = 9999
                for tid in active_short:
                    for item_idx, adj_cells, _ in self.type_items.get(tid, []):
                        if item_idx in claimed_items:
                            continue
                        for adj in adj_cells:
                            d_to = self.tables.get_distance(pos, adj)
                            d_back = self._drop_dist(adj)
                            detour = d_to + d_back - drop_dist
                            if detour < best_detour and detour < 6:
                                best_detour = detour
                                best_fill_item = item_idx
                                best_fill_adj = adj
                                best_fill_tid = tid

                if best_fill_item is not None:
                    self.bot_tasks[bid] = BotTask(
                        'pickup_active', best_fill_adj,
                        item_idx=best_fill_item, item_type=best_fill_tid,
                        dropoff=dz, order_id=self._last_active_id, phase='go')
                    active_short[best_fill_tid] = active_short.get(best_fill_tid, 0) - 1
                    claimed_items.add(best_fill_item)
                    continue

            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            # Find the active type in inventory
            active_type = -1
            for t in inv:
                if t in active_needs:
                    active_type = t
                    break
            self.bot_tasks[bid] = BotTask(
                'deliver', dz, item_type=active_type,
                dropoff=dz, order_id=self._last_active_id, phase='return')

        # === Preview carriers → stage ===
        deliver_zones = set()
        for t in self.bot_tasks.values():
            if t.task_type == 'deliver' and t.target in self.drop_set:
                deliver_zones.add(t.target)

        staging_counts = {dz: 0 for dz in self.drop_zones}
        for t in self.bot_tasks.values():
            if t.task_type == 'stage' and t.target in self.drop_set:
                staging_counts[t.target] = staging_counts.get(t.target, 0) + 1

        for bid in preview_carrier_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            preview_type = -1
            for t in inv:
                if t in preview_needs:
                    preview_type = t
                    break

            best_dz = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_zones:
                    continue
                if staging_counts[dz] >= 3:
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_dz = dz
            if best_dz is None:
                for dz in self.drop_zones:
                    if staging_counts[dz] >= 3:
                        continue
                    d = self.tables.get_distance(pos, dz)
                    if d < best_d:
                        best_d = d
                        best_dz = dz

            if best_dz is not None:
                staging_counts[best_dz] += 1
                self.bot_tasks[bid] = BotTask(
                    'stage', best_dz, item_type=preview_type,
                    dropoff=best_dz, order_id=self._last_preview_id, phase='return')
            else:
                park = self._near_drop_parking(pos, occupied_goals)
                occupied_goals.add(park)
                self.bot_tasks[bid] = BotTask('park', park)

        # === Dead bots ===
        for bid in dead_bots:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            # Check if items match future orders
            has_future = False
            if self.future_orders:
                idx = state.next_order_idx
                for i in range(8):
                    if idx + i < len(self.future_orders):
                        fo = self.future_orders[idx + i]
                        if any(t in set(fo.needs()) for t in inv):
                            has_future = True
                            break
            if has_future:
                park = self._near_drop_parking(pos, occupied_goals)
                occupied_goals.add(park)
                self.bot_tasks[bid] = BotTask('park', park)
            else:
                park = self._corridor_parking(pos, occupied_goals)
                occupied_goals.add(park)
                self.bot_tasks[bid] = BotTask('flee', park)

        # === Empty bots: split between active and preview ===
        # Budget: cover active shortfall + 2, rest go to preview
        active_pickup_budget = sum(active_short.values()) + 2
        active_picked = 0

        # Sort by proximity to needed items
        def _prox_key(bid):
            pos = bot_positions[bid]
            targets = active_short if active_short else preview_short
            if not targets:
                return 9999
            best_d = 9999
            for tid in targets:
                for _, adj_cells, _ in self.type_items.get(tid, []):
                    for adj in adj_cells:
                        d = self.tables.get_distance(pos, adj)
                        if d < best_d:
                            best_d = d
            return best_d

        empty_sorted = sorted(empty_bots, key=_prox_key)

        for bid in empty_sorted:
            pos = bot_positions[bid]

            # Active pickup
            if active_short and active_picked < active_pickup_budget:
                item_idx, adj_pos, tid = self._find_item(
                    pos, active_short, active_assigned, claimed_items)
                if item_idx is not None:
                    dz = self._nearest_drop(adj_pos)
                    self.bot_tasks[bid] = BotTask(
                        'pickup_active', adj_pos,
                        item_idx=item_idx, item_type=tid,
                        dropoff=dz, order_id=self._last_active_id, phase='go')
                    active_assigned[tid] = active_assigned.get(tid, 0) + 1
                    active_short[tid] = max(0, active_short.get(tid, 0) - 1)
                    claimed_items.add(item_idx)
                    active_picked += 1
                    continue

            # Preview pickup (even if active_short > 0)
            if preview_short:
                item_idx, adj_pos, tid = self._find_item(
                    pos, preview_short, preview_assigned, claimed_items,
                    drop_d_weight=0.6)  # Weight toward nearby dropoff
                if item_idx is not None:
                    dz = self._nearest_drop(adj_pos)
                    self.bot_tasks[bid] = BotTask(
                        'pickup_preview', adj_pos,
                        item_idx=item_idx, item_type=tid,
                        dropoff=dz, order_id=self._last_preview_id, phase='go')
                    preview_assigned[tid] = preview_assigned.get(tid, 0) + 1
                    preview_short[tid] = max(0, preview_short.get(tid, 0) - 1)
                    claimed_items.add(item_idx)
                    continue

            # Park
            park = self._corridor_parking(pos, occupied_goals)
            occupied_goals.add(park)
            self.bot_tasks[bid] = BotTask('park', park)

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        num_bots = self.num_bots
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # Update positions/stall tracking
        bot_positions = {}
        for bid in range(num_bots):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            bot_positions[bid] = pos
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        self.congestion.update(list(bot_positions.values()))

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Phase 1: Update existing tasks
        self._update_tasks(state, active_order, preview_order)

        # Phase 2: Assign new tasks to unassigned bots
        self._assign_new_tasks(state, active_order, preview_order, rnd, num_rounds)

        # Phase 3: Build goals from tasks
        goals = {}
        goal_types = {}
        pickup_targets = {}

        for bid in range(num_bots):
            task = self.bot_tasks.get(bid)
            if task is None:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            goals[bid] = task.target or self.spawn
            goal_types[bid] = task.task_type

            if task.task_type in ('pickup_active', 'pickup_preview') and task.item_idx >= 0:
                pickup_targets[bid] = task.item_idx

        # Phase 4: Urgency ordering
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'flee':
                return (1, dist)
            elif gt in ('pickup_active',):
                return (2, dist)
            elif gt in ('stage', 'pickup_preview'):
                return (3, dist)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # Phase 5: Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Phase 6: Execute actions
        actions = [(ACT_WAIT, -1)] * num_bots
        bot_inventories = {bid: state.bot_inv_list(bid) for bid in range(num_bots)}

        # Compute active needs for opportunistic pickup
        active_short = {}
        if active_order:
            active_needs = {}
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            carrying = {}
            for bid, inv in bot_inventories.items():
                for t in inv:
                    if t in active_needs:
                        carrying[t] = carrying.get(t, 0) + 1
            for t, need in active_needs.items():
                s = need - carrying.get(t, 0)
                if s > 0:
                    active_short[t] = s

        for bid in range(num_bots):
            pos = bot_positions[bid]
            task = self.bot_tasks.get(bid)
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # At dropoff: deliver or wait (stage)
            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                if gt == 'stage' and pos == goal:
                    # Check if we should deliver (items match active order)
                    inv = bot_inventories[bid]
                    if active_order and any(active_order.needs_type(t) for t in inv):
                        actions[bid] = (ACT_DROPOFF, -1)
                        continue
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # At pickup target: pick up
            if gt in ('pickup_active', 'pickup_preview') and bid in pickup_targets:
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, pickup_targets[bid])
                    continue

            # Opportunistic adjacent pickup
            if len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, active_order, preview_order, gt,
                    bot_inventories[bid], active_short)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            # Follow pathfinder
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_pickup(self, bid, pos, active_order, preview_order,
                                goal_type, bot_inv, active_short):
        adjacent_items = self.pos_to_items.get(pos, [])
        if not adjacent_items:
            return None
        bot_types = set(bot_inv)
        total_short = sum(active_short.values())
        for item_idx, tid in adjacent_items:
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
                active_short[tid] -= 1
                return (ACT_PICKUP, item_idx)
            elif total_short == 0 and preview_order and preview_order.needs_type(tid):
                if tid not in bot_types:
                    return (ACT_PICKUP, item_idx)
            elif goal_type == 'pickup_preview' and preview_order and preview_order.needs_type(tid):
                if tid not in bot_types:
                    return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid, pos, rnd):
        dirs = MOVES[:]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed, verbose=True):
        t0 = time.time()
        num_rounds = DIFF_ROUNDS['nightmare']
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = TaskQueueSolver(ms, tables, future_orders=all_orders)

        action_log = []
        prev_completed = 0
        chains = 0

        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))
            step(state, actions, all_orders)

            if state.orders_completed > prev_completed:
                new = state.orders_completed - prev_completed
                if new > 1:
                    chains += 1
                if verbose:
                    elapsed = time.time() - t0
                    print(f'  R{rnd:3d}: orders={state.orders_completed} '
                          f'(+{new}) score={state.score}'
                          f'{" CASCADE!" if new > 1 else ""}',
                          file=sys.stderr)
                prev_completed = state.orders_completed

        elapsed = time.time() - t0
        if verbose:
            print(f'V7 result: score={state.score}, orders={state.orders_completed}, '
                  f'chains={chains}, time={elapsed:.1f}s', file=sys.stderr)

        return state.score, action_log


# Alias for compatibility
NightmareSolverV7 = TaskQueueSolver


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    args = parser.parse_args()

    score, actions = TaskQueueSolver.run_sim(args.seed)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)
