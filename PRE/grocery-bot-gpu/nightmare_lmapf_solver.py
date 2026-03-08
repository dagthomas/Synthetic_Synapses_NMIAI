"""LMAPF V4 solver for Nightmare mode.

Builds on V3's proven allocator + PIBT pathfinding (255 score baseline).
Adds chain reaction orchestration to multiply scoring throughput.

Strategy:
1. V3's NightmareTaskAlloc for task assignment (proven)
2. V3's PIBT pathfinding (proven)
3. Chain orchestration: batch delivery with staged preview items
4. Delivery pipeline: all 20 bots productive via pre-fetching
5. Future order lookahead for deep pre-fetch

Usage:
    python nightmare_lmapf_solver.py --seeds 7005 -v
    python nightmare_lmapf_solver.py --seeds 7005 -v --compare
"""
from __future__ import annotations

import time

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
from nightmare_task_alloc import NightmareTaskAlloc
from nightmare_chain_planner import ChainPlanner

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right',
                'pick_up', 'drop_off']


class LMAPFSolver:
    """V4: V3 allocation + PIBT + chain orchestration.

    Identical to V3 for basic delivery, but adds:
    - Aggressive preview/future pre-fetching (all idle bots fetch)
    - Chain trigger timing: hold delivery when chain is almost ready
    - Delivery pipelining: stage 3+ bots at dropoffs before trigger
    """

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = CONFIGS['nightmare']['bots']
        self.future_orders = future_orders or []

        # V3's proven subsystems
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)
        self.allocator = NightmareTaskAlloc(
            ms, self.tables, self.drop_zones)
        self.chain_planner = ChainPlanner(
            ms, self.tables, self.drop_zones)

        # Item type -> [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # State
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self.chain_events: list[tuple[int, int]] = []
        self._seq_pos = -1

        # Chain state
        self._hold_rounds = 0
        self._last_staged_count = 0

        # Persistent trip tracking
        self._trips: dict[int, dict] = {}  # bid -> {goal, type, item_idx, age}
        self._last_active_id = -1

    # ------------------------------------------------------------------
    # Future order lookahead
    # ------------------------------------------------------------------

    def _get_future_orders(self, state, all_orders, depth=8):
        future = []
        preview = state.get_preview_order()
        if preview:
            future.append(preview)
        if all_orders:
            for i in range(state.next_order_idx,
                           min(state.next_order_idx + depth, len(all_orders))):
                future.append(all_orders[i])
        elif self.future_orders:
            start = state.orders_completed + 2
            for i in range(start, min(start + depth, len(self.future_orders))):
                future.append(self.future_orders[i])
        return future[:depth]

    # ------------------------------------------------------------------
    # Chain orchestration
    # ------------------------------------------------------------------

    def _count_staged_preview(self, preview, bot_positions, bot_inventories):
        """Count how many preview items are at/near dropoffs."""
        if not preview:
            return 0, 0
        needs = list(preview.needs())
        total = len(needs)
        if total == 0:
            return 0, 0

        need_count: dict[int, int] = {}
        for t in needs:
            need_count[t] = need_count.get(t, 0) + 1

        staged = 0
        for bid, inv in bot_inventories.items():
            pos = bot_positions.get(bid)
            if pos is None:
                continue
            d_drop = min(self.tables.get_distance(pos, dz)
                         for dz in self.drop_zones)
            if d_drop > 3:
                continue
            for t in inv:
                if need_count.get(t, 0) > 0:
                    need_count[t] -= 1
                    staged += 1
        return staged, total

    def _should_hold(self, active, preview, bot_positions,
                     bot_inventories, rnd, num_rounds):
        """Hold delivery to build chain? Returns True if we should wait."""
        if not active or not preview:
            return False
        if num_rounds - rnd < 40:
            return False
        if self._hold_rounds > 15:
            return False  # max hold time

        remaining = len(active.needs())
        if remaining > 2:
            return False

        # Check if active is completable with current carriers
        active_needs_copy: dict[int, int] = {}
        for t in active.needs():
            active_needs_copy[t] = active_needs_copy.get(t, 0) + 1
        all_carried = True
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs_copy and active_needs_copy[t] > 0:
                    active_needs_copy[t] -= 1
        for v in active_needs_copy.values():
            if v > 0:
                all_carried = False
                break

        if not all_carried:
            return False  # can't hold what we don't have

        staged, total = self._count_staged_preview(
            preview, bot_positions, bot_inventories)
        if staged >= total:
            return False  # chain ready, FIRE!

        # Hold if staging is progressing (>30% done)
        if total > 0 and staged / total >= 0.3:
            return True

        return False

    # ------------------------------------------------------------------
    # Per-round action
    # ------------------------------------------------------------------

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # Extract state
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall + congestion tracking
        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()
        future = self._get_future_orders(state, all_orders)

        # Reset hold state when active order changes
        active_hash = (hash(tuple(sorted(active_order.required)))
                       if active_order else -1)
        if not hasattr(self, '_last_active_hash'):
            self._last_active_hash = -1
        if active_hash != self._last_active_hash:
            self._hold_rounds = 0
            self._last_staged_count = 0
            self._last_active_hash = active_hash

        # Chain planning
        chain_plan = self.chain_planner.plan_chain(
            active_order, future, bot_positions, bot_inventories)

        # Active shortfall
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for inv in bot_inventories.values():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        total_short = sum(active_short.values())

        # Clear persistent trips on active order change
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            self._trips.clear()
            self._last_active_id = active_id

        # Age out stale trips
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            trip['age'] += 1
            if trip['age'] > 20:  # 20 round max
                del self._trips[bid]

        # V3 allocation (proven baseline)
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future, chain_plan=chain_plan,
            allow_preview_pickup=True)

        # Apply persistent trips: override V3's assignment for bots
        # that are still en route to their assigned item
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            pos = bot_positions[bid]
            # Trip completed: at goal or picked up item
            if pos == trip['goal'] or len(bot_inventories[bid]) > trip.get('inv_count', 0):
                del self._trips[bid]
                continue
            # Check the item is still relevant
            tid = trip.get('type_id', -1)
            if trip['goal_type'] == 'pickup':
                if not (active_order and active_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            elif trip['goal_type'] == 'preview':
                if not (preview_order and preview_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            # Override V3 assignment
            goals[bid] = trip['goal']
            goal_types[bid] = trip['goal_type']
            if trip.get('item_idx') is not None:
                pickup_targets[bid] = trip['item_idx']

        # Record new trip assignments from V3
        for bid in range(num_bots):
            gt = goal_types.get(bid)
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if bid not in self._trips:
                    item_idx = pickup_targets[bid]
                    tid = int(self.ms.item_types[item_idx]) if item_idx >= 0 else -1
                    self._trips[bid] = {
                        'goal': goals[bid],
                        'goal_type': gt,
                        'item_idx': item_idx,
                        'type_id': tid,
                        'inv_count': len(bot_inventories[bid]),
                        'age': 0,
                    }

        # POST-PROCESS: Recycle dead/idle bots with free slots
        claimed_items = set(pickup_targets.values())
        for bid in range(num_bots):
            gt = goal_types.get(bid, 'park')
            if gt not in ('flee', 'park'):
                continue
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free = INV_CAP - len(inv)
            if free <= 0:
                continue

            # Try active pickup first
            if active_short:
                idx, adj = self._find_best_item(
                    pos, active_short, claimed_items)
                if idx is not None:
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = idx
                    claimed_items.add(idx)
                    self._trips[bid] = {
                        'goal': adj,
                        'goal_type': 'pickup',
                        'item_idx': idx,
                        'type_id': int(self.ms.item_types[idx]),
                        'inv_count': len(inv),
                        'age': 0,
                    }
                    continue

            # Then preview
            if preview_order:
                preview_n: dict[int, int] = {}
                for t in preview_order.needs():
                    preview_n[t] = preview_n.get(t, 0) + 1
                for t in inv:
                    if t in preview_n:
                        preview_n[t] -= 1
                        if preview_n[t] <= 0:
                            del preview_n[t]
                if preview_n:
                    idx, adj = self._find_best_item(
                        pos, preview_n, claimed_items)
                    if idx is not None:
                        goals[bid] = adj
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = idx
                        claimed_items.add(idx)
                        self._trips[bid] = {
                            'goal': adj,
                            'goal_type': 'preview',
                            'item_idx': idx,
                            'type_id': int(self.ms.item_types[idx]),
                            'inv_count': len(inv),
                            'age': 0,
                        }

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
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

        # PIBT pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order,
            goal_types=goal_types)

        # Build actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF: deliver
            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup (active items)
            if gt in ('pickup', 'preview', 'deliver') and len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, active_order, preview_order, gt,
                    bot_inventories[bid], active_short, chain_plan)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_pickup(self, bid, pos, active_order, preview_order,
                                goal_type, bot_inv, active_short,
                                chain_plan=None):
        """Adjacent item pickup (V3-identical logic)."""
        ms = self.ms
        bot_types = set(bot_inv)
        total_short = sum(active_short.values())

        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])

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

            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)

        return None

    def _should_hold_at_dropoff(self, bid, active, preview,
                                bot_positions, bot_inventories,
                                rnd, num_rounds):
        """Hold delivery at dropoff to wait for chain staging.

        Only holds if:
        1. This delivery would likely complete the active order
        2. Preview items are being staged but not ready yet
        3. We haven't held too long
        4. Enough game time remains
        """
        if not active or not preview:
            return False
        if num_rounds - rnd < 50:
            return False  # too late, just deliver
        if self._hold_rounds > 8:
            return False  # max hold time exceeded

        # Would this delivery complete the active order?
        remaining = list(active.needs())
        if len(remaining) > 3:
            return False  # too far from completion

        # Simulate: would this bot's items complete the order?
        inv = bot_inventories.get(bid, [])
        remaining_after = list(remaining)
        for t in inv:
            if t in remaining_after:
                remaining_after.remove(t)

        # Also account for other deliver bots at dropoff
        for bid2, inv2 in bot_inventories.items():
            if bid2 == bid:
                continue
            pos2 = bot_positions.get(bid2)
            if pos2 is None or pos2 not in self.drop_set:
                continue
            for t in inv2:
                if t in remaining_after:
                    remaining_after.remove(t)

        if len(remaining_after) > 0:
            return False  # delivery won't complete the order anyway

        # Delivery WOULD complete the order. Check preview staging.
        staged, total = self._count_staged_preview(
            preview, bot_positions, bot_inventories)

        if staged >= total:
            return False  # chain fully ready, FIRE!

        # Hold if staging is making progress
        if staged > self._last_staged_count or staged >= total * 0.4:
            self._hold_rounds += 1
            self._last_staged_count = staged
            return True

        return False

    def _find_best_item(self, pos: tuple[int, int],
                        needed: dict[int, int],
                        claimed: set[int]) -> tuple[int | None, tuple[int, int] | None]:
        """Find nearest item of a needed type, not already claimed."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, count in needed.items():
            if count <= 0:
                continue
            for item_idx, adj_cells in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(pos, adj)
                    drop_d = min(self.tables.get_distance(adj, dz)
                                 for dz in self.drop_zones)
                    cost = d + drop_d * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def _escape_action(self, bid, pos, rnd):
        dirs = list(MOVES)
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    # ------------------------------------------------------------------
    # WebSocket interface
    # ------------------------------------------------------------------

    def ws_action(self, live_bots: list[dict], data: dict,
                  map_state: MapState) -> list[dict]:
        ms = map_state or self.ms
        rnd = data.get('round', 0)
        num_bots = len(live_bots)

        active_order, preview_order = None, None
        for od in data.get('orders', []):
            req_ids = [ms.type_name_to_id.get(n, 0)
                       for n in od.get('items_required', [])]
            order = Order(0, req_ids, od.get('status', 'active'))
            for dn in od.get('items_delivered', []):
                tid = ms.type_name_to_id.get(dn, -1)
                if tid >= 0:
                    order.deliver_type(tid)
            if od.get('status') == 'active':
                active_order = order
            elif od.get('status') == 'preview':
                preview_order = order

        bp, bi = {}, {}
        all_bids = []
        for bot in live_bots:
            bid = bot['id']
            all_bids.append(bid)
            bp[bid] = tuple(bot['position'])
            inv = [ms.type_name_to_id.get(n, -1) for n in bot.get('inventory', [])]
            bi[bid] = [t for t in inv if t >= 0]

        self.congestion.update(list(bp.values()))
        for bid, pos in bp.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Active shortfall
        active_needs: dict[int, int] = {}
        carrying: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for inv in bi.values():
                for t in inv:
                    if t in active_needs:
                        carrying[t] = carrying.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, n in active_needs.items():
            s = n - carrying.get(t, 0)
            if s > 0:
                active_short[t] = s

        num_rounds = data.get('max_rounds', 500)

        total_short = sum(active_short.values())

        # Clear persistent trips on active order change
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            self._trips.clear()
            self._last_active_id = active_id
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            trip['age'] += 1
            if trip['age'] > 20:
                del self._trips[bid]

        # V3 allocation
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bp, bi, active_order, preview_order, rnd, num_rounds,
            future_orders=None, chain_plan=None,
            allow_preview_pickup=True)

        # Apply persistent trips
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            pos = bp.get(bid)
            if pos is None:
                del self._trips[bid]
                continue
            if pos == trip['goal'] or len(bi.get(bid, [])) > trip.get('inv_count', 0):
                del self._trips[bid]
                continue
            tid = trip.get('type_id', -1)
            if trip['goal_type'] == 'pickup':
                if not (active_order and active_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            elif trip['goal_type'] == 'preview':
                if not (preview_order and preview_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            goals[bid] = trip['goal']
            goal_types[bid] = trip['goal_type']
            if trip.get('item_idx') is not None:
                pickup_targets[bid] = trip['item_idx']

        # Record new trips
        for bid in all_bids:
            gt = goal_types.get(bid)
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if bid not in self._trips:
                    item_idx = pickup_targets[bid]
                    tid = int(ms.item_types[item_idx]) if item_idx >= 0 else -1
                    self._trips[bid] = {
                        'goal': goals[bid], 'goal_type': gt,
                        'item_idx': item_idx, 'type_id': tid,
                        'inv_count': len(bi.get(bid, [])), 'age': 0,
                    }

        # Recycle dead/idle bots with free slots
        claimed_items = set(pickup_targets.values())
        for bid in all_bids:
            gt = goal_types.get(bid, 'park')
            if gt not in ('flee', 'park'):
                continue
            inv = bi.get(bid, [])
            if INV_CAP - len(inv) <= 0:
                continue
            pos = bp.get(bid, self.spawn)
            if active_short:
                idx, adj = self._find_best_item(pos, active_short, claimed_items)
                if idx is not None:
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = idx
                    claimed_items.add(idx)
                    self._trips[bid] = {
                        'goal': adj, 'goal_type': 'pickup',
                        'item_idx': idx, 'type_id': int(ms.item_types[idx]),
                        'inv_count': len(inv), 'age': 0,
                    }
                    continue
            if preview_order:
                pn = {}
                for t in preview_order.needs():
                    pn[t] = pn.get(t, 0) + 1
                for t in inv:
                    if t in pn:
                        pn[t] -= 1
                        if pn[t] <= 0:
                            del pn[t]
                if pn:
                    idx, adj = self._find_best_item(pos, pn, claimed_items)
                    if idx is not None:
                        goals[bid] = adj
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = idx
                        claimed_items.add(idx)
                        self._trips[bid] = {
                            'goal': adj, 'goal_type': 'preview',
                            'item_idx': idx, 'type_id': int(ms.item_types[idx]),
                            'inv_count': len(inv), 'age': 0,
                        }

        # Urgency + PIBT
        prio_map = {'deliver': 0, 'pickup': 1, 'stage': 2,
                    'preview': 3, 'flee': 4, 'park': 5}
        urgency_order = sorted(all_bids, key=lambda bid: (
            prio_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bp.get(bid, self.spawn),
                                     goals.get(bid, self.spawn))))

        path_actions = self.pathfinder.plan_all(
            bp, goals, urgency_order, goal_types=goal_types)

        # Build WS actions
        ws_actions = []
        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            inv_names = bot.get('inventory', [])
            inv = bi.get(bid, [])

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            if pos in self.drop_set and gt == 'deliver' and inv_names:
                ws_actions.append({'bot': bid, 'action': 'drop_off'})
                continue

            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({'bot': bid, 'action': 'pick_up',
                                       'item_id': ms.items[item_idx]['id']})
                    continue

            # Opportunistic active pickup
            if len(inv_names) < INV_CAP and gt in ('pickup', 'deliver') and active_short:
                opp = self._ws_active_adjacent(bid, pos, ms, active_short)
                if opp:
                    ws_actions.append(opp)
                    continue

            # Fill with preview on delivery
            if (len(inv_names) < INV_CAP and gt == 'deliver'
                    and not active_short and preview_order):
                opp = self._ws_preview_adjacent(bid, pos, ms, preview_order,
                                                set(inv))
                if opp:
                    ws_actions.append(opp)
                    continue

            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_active_adjacent(self, bid, pos, ms, active_short):
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {'bot': bid, 'action': 'pick_up',
                            'item_id': ms.items[item_idx]['id']}
        return None

    def _ws_preview_adjacent(self, bid, pos, ms, preview, bot_types):
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if not preview.needs_type(tid) or tid in bot_types:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {'bot': bid, 'action': 'pick_up',
                            'item_id': ms.items[item_idx]['id']}
        return None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = LMAPFSolver(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains, max_chain = 0, 0
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
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                drop_info = ""
                if c >= 1:
                    at = []
                    for b in range(len(state.bot_positions)):
                        bpos = (int(state.bot_positions[b, 0]),
                                int(state.bot_positions[b, 1]))
                        if bpos in solver.drop_set:
                            inv = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv}")
                    drop_info = f" Drop=[{','.join(at)}]"
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}"
                         if active else " DONE")
                      + extra + drop_info)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"Time={elapsed:.1f}s "
                  f"({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_lmapf, scores_v3 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - LMAPF V4")
        print(f"{'='*50}")
        score, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
        scores_lmapf.append(score)

        if args.compare:
            from nightmare_solver_v2 import NightmareSolverV3
            print(f"\n--- V3 ---")
            s3, _ = NightmareSolverV3.run_sim(seed, verbose=args.verbose)
            scores_v3.append(s3)
            print(f"\nV4={score} vs V3={s3} (delta={score - s3:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nV4: mean={statistics.mean(scores_lmapf):.1f} "
              f"max={max(scores_lmapf)} min={min(scores_lmapf)}")
        if scores_v3:
            print(f"V3: mean={statistics.mean(scores_v3):.1f}")
