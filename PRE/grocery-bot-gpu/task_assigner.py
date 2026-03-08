"""Tiered Hungarian assignment with multi-drop-zone coordination.

Assigns bots to items using a 4-tier role system:
  Tier 1 - Active pickers: assigned to current active order items
  Tier 2 - Preview pre-pickers: pre-picking items for preview order
  Tier 3 - Delivery shuttles: carrying items toward drop zones
  Tier 4 - Scouts/idle: pre-positioned for fast response

Multi-zone drop-off load balancing:
  score(zone) = dist(bot, zone) + 4 * queue_len(zone)

Order transition handling:
  When active order has <=2 items remaining, promote preview pickers to delivery.

Usage:
    from task_assigner import TaskAssigner
    assigner = TaskAssigner(tables, ms)
    goals, roles = assigner.assign(state, all_orders, bot_states)
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from game_engine import (
    GameState, MapState, Order, INV_CAP,
    ACT_PICKUP, ACT_DROPOFF,
)
from action_gen import find_items_of_type
from precompute import PrecomputedTables
from pibt import PRIO_DELIVERING, PRIO_CARRYING, PRIO_PICKING, PRIO_PREPICKING, PRIO_IDLE


class Goal:
    """A high-level goal for a bot."""
    __slots__ = ['type', 'target', 'item_idx', 'zone_idx', 'items_to_pick']

    PICK = 'pick'
    DELIVER = 'deliver'
    MOVE = 'move'
    WAIT = 'wait'

    def __init__(self, goal_type: str, target: tuple[int, int] | None = None,
                 item_idx: int = -1, zone_idx: int = -1):
        self.type = goal_type
        self.target = target
        self.item_idx = item_idx
        self.zone_idx = zone_idx
        self.items_to_pick = []  # [(item_idx, adj_cell), ...] for multi-item trips

    def __repr__(self):
        return f'Goal({self.type}, {self.target}, item={self.item_idx})'


class BotState:
    """Tracked state for one bot across rounds."""
    __slots__ = ['bot_id', 'goal', 'role', 'stuck_count', 'last_pos',
                 'prev_pos', 'no_progress_count', 'blocked_targets',
                 'assignment_round']

    def __init__(self, bot_id: int):
        self.bot_id = bot_id
        self.goal: Goal | None = None
        self.role: int = PRIO_IDLE
        self.stuck_count: int = 0
        self.last_pos: tuple[int, int] | None = None
        self.prev_pos: tuple[int, int] | None = None  # 2-back for oscillation
        self.no_progress_count: int = 0  # rounds without getting closer to goal
        self.blocked_targets: set = set()  # targets that caused stuck, avoid for 30 rounds
        self.assignment_round: int = -1

    def set_goal(self, goal: Goal, role: int, round_num: int = -1):
        self.goal = goal
        self.role = role
        self.stuck_count = 0
        self.no_progress_count = 0
        self.assignment_round = round_num

    def clear(self):
        # Record blocked target before clearing
        if self.goal and self.goal.target:
            self.blocked_targets.add(self.goal.target)
        self.goal = None
        self.role = PRIO_IDLE
        self.stuck_count = 0
        self.no_progress_count = 0
        self.assignment_round = -1


class TaskAssigner:
    """Centralized task assignment for nightmare mode."""

    def __init__(self, tables: PrecomputedTables, ms: MapState):
        self.tables = tables
        self.ms = ms
        self.drop_off_zones = getattr(ms, 'drop_off_zones', [ms.drop_off])
        self._last_active_id = -1

    def assign(
        self,
        state: GameState,
        all_orders: list[Order],
        bot_states: list[BotState],
    ) -> tuple[list[Goal | None], list[int]]:
        """Run full assignment pipeline.

        Returns:
            goals: list of Goal per bot (or None for wait)
            roles: list of PRIO_* role per bot
        """
        ms = self.ms
        tables = self.tables
        n_bots = len(state.bot_positions)
        active = state.get_active_order()
        preview = state.get_preview_order()
        self._current_round = state.round

        # Detect order transitions
        active_id = active.id if active else -1
        order_changed = active_id != self._last_active_id
        self._last_active_id = active_id

        if order_changed:
            # Reset pickers whose assignments are stale + clear blocked targets
            for bs in bot_states:
                if bs.role in (PRIO_PICKING, PRIO_PREPICKING):
                    bs.clear()
                bs.blocked_targets.clear()  # new order = fresh start

        # Periodic decay of blocked_targets (every 30 rounds)
        if state.round > 0 and state.round % 30 == 0:
            for bs in bot_states:
                bs.blocked_targets.clear()

        # Detect stuck bots (stationary or oscillating)
        for bs in bot_states:
            bid = bs.bot_id
            pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
            # Detect stationary stuck
            if bs.last_pos == pos and bs.goal is not None:
                bs.stuck_count += 1
            # Detect oscillation: pos == prev_pos means alternating between 2 cells
            elif bs.prev_pos == pos and bs.goal is not None:
                bs.stuck_count += 1
            else:
                bs.stuck_count = 0

            # Track no-progress: not getting closer to goal target
            if bs.goal and bs.goal.target and bs.goal.target != pos:
                goal_dist = tables.get_distance(pos, bs.goal.target)
                prev_dist = tables.get_distance(bs.last_pos, bs.goal.target) if bs.last_pos else 9999
                if goal_dist >= prev_dist:
                    bs.no_progress_count += 1
                else:
                    bs.no_progress_count = 0
            else:
                bs.no_progress_count = 0

            bs.prev_pos = bs.last_pos
            bs.last_pos = pos

            # Role-specific stuck thresholds: deliverers should always make progress
            np_limit = 7 if bs.role in (PRIO_DELIVERING, PRIO_CARRYING) else 12
            if bs.stuck_count > 8 or bs.no_progress_count > np_limit:
                bs.clear()

        goals: list[Goal | None] = [None] * n_bots
        roles: list[int] = [PRIO_IDLE] * n_bots

        # Phase 1: Handle bots already carrying items
        self._assign_deliveries(state, bot_states, goals, roles, active, preview)

        # Phase 2: Assign active order items to idle bots (Hungarian)
        if active:
            self._assign_active_picks(state, bot_states, goals, roles, active, all_orders)

        # Phase 3: Pre-pick preview items with bots that have 2+ free slots.
        if preview and active:
            self._assign_preview_picks(state, bot_states, goals, roles, active, preview)

        # Phase 4: Keep existing goals for bots still working
        for bs in bot_states:
            bid = bs.bot_id
            if goals[bid] is None and bs.goal is not None:
                goals[bid] = bs.goal
                roles[bid] = bs.role

        # Update bot_states with new assignments
        for bid in range(n_bots):
            if goals[bid] is not None:
                bot_states[bid].goal = goals[bid]
                bot_states[bid].role = roles[bid]

        return goals, roles

    def _assign_deliveries(
        self,
        state: GameState,
        bot_states: list[BotState],
        goals: list[Goal | None],
        roles: list[int],
        active: Order | None,
        preview: Order | None,
    ):
        """Route bots with active-matching inventory items to best drop zone.

        Only delivers items matching the ACTIVE order. Preview items stay
        in inventory until the preview becomes active (order transition).
        """
        n_bots = len(state.bot_positions)

        # Count queue length per zone (bots already heading there)
        zone_queues = [0] * len(self.drop_off_zones)
        for bs in bot_states:
            if bs.goal and bs.goal.type == Goal.DELIVER:
                zi = bs.goal.zone_idx
                if 0 <= zi < len(self.drop_off_zones):
                    zone_queues[zi] += 1

        for bid in range(n_bots):
            inv = state.bot_inv_list(bid)
            if not inv:
                # Bot has no items - clear any stale delivery goal
                if bot_states[bid].role in (PRIO_DELIVERING, PRIO_CARRYING):
                    bot_states[bid].clear()
                continue

            pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))

            # Only deliver items matching the ACTIVE order
            has_active = active and any(active.needs_type(t) for t in inv)

            # Already delivering - keep goal only if still has active items
            if bot_states[bid].role == PRIO_DELIVERING:
                if has_active:
                    goals[bid] = bot_states[bid].goal
                    roles[bid] = PRIO_DELIVERING
                    continue
                else:
                    bot_states[bid].clear()

            # If carrying active items, go deliver
            if has_active:
                zone_idx, zone_pos = self._select_drop_zone(
                    pos, zone_queues, bot_states[bid].blocked_targets)
                goal = Goal(Goal.DELIVER, zone_pos, zone_idx=zone_idx)
                goals[bid] = goal
                roles[bid] = PRIO_DELIVERING
                zone_queues[zone_idx] += 1
                bot_states[bid].set_goal(goal, PRIO_DELIVERING, self._current_round)
                continue

            # Bot has items but none match active order - treat as idle
            # (items stay in inventory, delivered when order transitions)

    def _assign_active_picks(
        self,
        state: GameState,
        bot_states: list[BotState],
        goals: list[Goal | None],
        roles: list[int],
        active: Order,
        all_orders: list[Order],
    ):
        """Assign active order items to idle bots via Hungarian matching."""
        remaining = self._compute_remaining_needs(state, bot_states, active,
                                                   pick_role=PRIO_PICKING)
        needed_items = []
        for tid, count in remaining.items():
            for _ in range(max(0, count)):
                needed_items.append(tid)

        if not needed_items:
            return

        idle_bots = [bid for bid in range(len(state.bot_positions))
                     if goals[bid] is None
                     and bot_states[bid].role in (PRIO_IDLE, PRIO_PREPICKING, PRIO_PICKING)
                     and state.bot_inv_count(bid) < INV_CAP]

        if not idle_bots:
            return

        self._hungarian_assign(state, idle_bots, needed_items, goals, roles,
                               bot_states, PRIO_PICKING)

    def _assign_preview_picks(
        self,
        state: GameState,
        bot_states: list[BotState],
        goals: list[Goal | None],
        roles: list[int],
        active: Order,
        preview: Order,
    ):
        """Assign preview order items to idle bots with 2+ free inventory slots.

        Safety: requires 2+ free slots so bot can still pick 1 active item
        if needed, preventing inventory deadlocks.
        """
        # Only pre-pick when active is fully covered (only count active pickers)
        active_needs = self._compute_remaining_needs(state, bot_states, active,
                                                     pick_role=PRIO_PICKING)
        active_remaining = sum(max(0, v) for v in active_needs.values())
        if active_remaining > 0:
            return

        remaining = self._compute_remaining_needs(state, bot_states, preview,
                                                   pick_role=PRIO_PREPICKING)
        needed_items = []
        for tid, count in remaining.items():
            for _ in range(max(0, count)):
                needed_items.append(tid)

        if not needed_items:
            return

        # Require 2+ free inventory slots (keep 1 for active items)
        idle_bots = [bid for bid in range(len(state.bot_positions))
                     if goals[bid] is None
                     and bot_states[bid].role == PRIO_IDLE
                     and state.bot_inv_count(bid) <= INV_CAP - 2]

        if not idle_bots:
            return

        self._hungarian_assign(state, idle_bots, needed_items, goals, roles,
                               bot_states, PRIO_PREPICKING)

    def _hungarian_assign(
        self,
        state: GameState,
        available_bots: list[int],
        needed_types: list[int],
        goals: list[Goal | None],
        roles: list[int],
        bot_states: list[BotState],
        role: int,
    ):
        """Core Hungarian assignment: bots -> item types."""
        ms = self.ms
        tables = self.tables
        n_bots = len(available_bots)
        n_needs = len(needed_types)

        if n_bots == 0 or n_needs == 0:
            return

        # Compute aisle congestion: count bots already picking at each x-column
        AISLE_PENALTY = 1
        available_set = set(available_bots)
        aisle_congestion: dict[int, int] = {}
        if AISLE_PENALTY > 0:
            for bs in bot_states:
                if bs.bot_id in available_set:
                    continue
                if bs.goal and bs.goal.type == Goal.PICK and bs.goal.target:
                    ax = bs.goal.target[0]
                    aisle_congestion[ax] = aisle_congestion.get(ax, 0) + 1

        BIG = 100_000
        cost_matrix = np.full((n_bots, n_needs), BIG, dtype=np.float64)
        best_items: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

        for row, bid in enumerate(available_bots):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            bot_pos = (bx, by)
            blocked = bot_states[bid].blocked_targets

            for col, tid in enumerate(needed_types):
                best_d = 9999
                best_item = None
                best_adj = None
                for item_idx in find_items_of_type(ms, tid):
                    result = tables.get_nearest_item_cell(bot_pos, item_idx, ms)
                    if result:
                        adj = (result[0], result[1])
                        d = result[2]
                        # Penalize previously blocked targets
                        if adj in blocked:
                            d += 50
                        # Penalize congested aisles to spread bots
                        d += AISLE_PENALTY * aisle_congestion.get(adj[0], 0)
                        if d < best_d:
                            best_d = d
                            best_item = item_idx
                            best_adj = adj
                if best_item is not None:
                    cost_matrix[row, col] = best_d
                    best_items[(row, col)] = (best_item, best_adj)

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
            return
        assigned_needs = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= BIG:
                continue
            if c in assigned_needs:
                continue
            bid = available_bots[r]
            if (r, c) in best_items:
                item_idx, adj_cell = best_items[(r, c)]
                goal = Goal(Goal.PICK, adj_cell, item_idx=item_idx)
                goals[bid] = goal
                roles[bid] = role
                bot_states[bid].set_goal(goal, role, self._current_round)
                assigned_needs.add(c)

    def _select_drop_zone(
        self,
        bot_pos: tuple[int, int],
        zone_queues: list[int],
        blocked_targets: set | None = None,
    ) -> tuple[int, tuple[int, int]]:
        """Select best drop zone: distance + congestion + blocked penalty."""
        best_score = 999999
        best_idx = 0
        for zi, dz in enumerate(self.drop_off_zones):
            dist = self.tables.get_distance(bot_pos, dz)
            score = dist + 4 * zone_queues[zi]
            # Penalize zones that previously caused stuck
            if blocked_targets and dz in blocked_targets:
                score += 50
            if score < best_score:
                best_score = score
                best_idx = zi
        return best_idx, self.drop_off_zones[best_idx]

    def _compute_remaining_needs(
        self,
        state: GameState,
        bot_states: list[BotState],
        order: Order,
        pick_role: int | None = None,
    ) -> dict[int, int]:
        """Compute what an order still needs, accounting for inventories and assignments.

        Args:
            pick_role: If set, only subtract picks from bots with this role.
                       Prevents preview picks from being subtracted from active needs.
        """
        needs: dict[int, int] = {}
        for tid in order.needs():
            needs[tid] = needs.get(tid, 0) + 1

        n_bots = len(state.bot_positions)
        ms = self.ms

        # Subtract ALL bots' inventories
        for bid in range(n_bots):
            for t in state.bot_inv_list(bid):
                if t in needs and needs[t] > 0:
                    needs[t] -= 1

        # Subtract items assigned to picking bots (filtered by role if specified)
        for bs in bot_states:
            if bs.goal and bs.goal.type == Goal.PICK:
                if pick_role is not None and bs.role != pick_role:
                    continue
                item_idx = bs.goal.item_idx
                if 0 <= item_idx < ms.num_items:
                    tid = int(ms.item_types[item_idx])
                    if tid in needs and needs[tid] > 0:
                        needs[tid] -= 1

        return needs
