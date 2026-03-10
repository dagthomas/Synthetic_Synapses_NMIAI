"""Nightmare mode solvers: V2 (legacy) + V3 (chain reaction pipeline).

V3 exploits the game engine's chain reaction mechanic: when a TRIGGER bot
completes the active order, ALL bots at dropoff zones auto-deliver matching
items. If the new active order also completes, the chain continues.

Usage:
    python nightmare_solver_v2.py --seeds 7005 -v
    python nightmare_solver_v2.py --seeds 7005 -v --v2  # force old solver
    python nightmare_solver_v2.py --seeds 1000-1009 -v
"""
from __future__ import annotations

import time

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    build_map_from_capture, generate_all_orders,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF, actions_to_ws_format,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_task_alloc import NightmareTaskAlloc
from nightmare_chain_planner import ChainPlanner


class NightmareSolverV3:
    """V3: Chain reaction pipeline solver for nightmare.

    20 bots, 3 dropoffs, 500 rounds. Uses full future order lookahead
    to stage bots at dropoff zones with chain-valuable items.
    """

    def __init__(self, map_state: MapState,
                 precomputed_tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None):
        self.ms = map_state
        self.tables = precomputed_tables or PrecomputedTables.get(map_state)
        self.walkable = build_walkable(map_state)
        self.drop_zones = [tuple(dz) for dz in map_state.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = map_state.spawn
        self.num_bots = CONFIGS['nightmare']['bots']

        # Subsystems
        self.traffic = TrafficRules(map_state)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(map_state, self.tables,
                                               self.traffic, self.congestion)
        self.allocator = NightmareTaskAlloc(map_state, self.tables, self.drop_zones)
        self.chain_planner = ChainPlanner(map_state, self.tables, self.drop_zones)
        # Pipeline allocator for live mode (uses future orders for staging)
        self._pipeline_alloc = None
        if future_orders:
            try:
                from nightmare_pipeline_alloc import NightmarePipelineAlloc
                self._pipeline_alloc = NightmarePipelineAlloc(
                    map_state, self.tables, self.drop_zones)
            except Exception:
                pass

        # Pre-loaded future orders (from capture or all_orders in sim)
        self.future_orders = future_orders or []
        self._seq_pos = -1  # cached sequence position

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Chain event tracking (for verbose/stats)
        self.chain_events: list[tuple[int, int]] = []  # (round, chain_len)

    def _get_future_orders(self, state: GameState, all_orders: list[Order],
                           depth: int = 8) -> list[Order]:
        """Extract future orders from state or pre-loaded list.

        In sim mode: all_orders gives full lookahead.
        In live mode: use self.future_orders from capture DB.
        """
        future: list[Order] = []

        # Preview order is always the first future order
        preview = state.get_preview_order()
        if preview:
            future.append(preview)

        # From all_orders: look at orders not yet issued
        if all_orders:
            for i in range(state.next_order_idx, min(state.next_order_idx + depth, len(all_orders))):
                future.append(all_orders[i])
        elif self.future_orders:
            # Fall back to pre-loaded orders
            # Estimate which orders are still upcoming based on orders_completed
            start_idx = state.orders_completed + 2  # +2 because active + preview already exist
            for i in range(start_idx, min(start_idx + depth, len(self.future_orders))):
                future.append(self.future_orders[i])

        return future[:depth]

    def _find_sequence_pos(self, active_order: Order | None) -> int:
        """Match active order to captured sequence index."""
        if not active_order or not self.future_orders:
            return -1
        active_req = tuple(sorted(active_order.required))
        # Search near cached position first, then broaden
        start = max(0, self._seq_pos - 1)
        for i in range(start, len(self.future_orders)):
            if tuple(sorted(self.future_orders[i].required)) == active_req:
                if i != self._seq_pos:
                    import sys
                    print(f"  [seq] Matched active order at pos {i} "
                          f"(types={active_req})", file=sys.stderr)
                self._seq_pos = i
                return i
        # Wrap-around search (shouldn't happen normally)
        for i in range(0, start):
            if tuple(sorted(self.future_orders[i].required)) == active_req:
                self._seq_pos = i
                return i
        # Debug: log first mismatch
        if not hasattr(self, '_seq_miss_logged'):
            self._seq_miss_logged = True
            import sys
            print(f"  [seq] NO MATCH: active={active_req}, "
                  f"future[0]={tuple(sorted(self.future_orders[0].required)) if self.future_orders else 'empty'}",
                  file=sys.stderr)
        return -1

    def _build_lookahead(self, active_order: Order | None,
                         preview_order: Order | None) -> Order | None:
        """Combine preview + next 2 captured orders into a lookahead order.

        Bots assigned to this mega-preview will pre-fetch items for upcoming
        orders. When those orders become active, bots at dropoff auto-deliver
        via chain reaction.
        """
        if not self.future_orders:
            return preview_order

        seq_pos = self._find_sequence_pos(active_order)
        if seq_pos < 0:
            return preview_order

        # Collect types: preview + orders seq_pos+2 and seq_pos+3
        all_types: list[int] = []
        if preview_order:
            all_types.extend(preview_order.required)
        lookahead_count = 0
        for i in range(seq_pos + 2, min(seq_pos + 4, len(self.future_orders))):
            all_types.extend(self.future_orders[i].required)
            lookahead_count += 1

        if not all_types:
            return preview_order

        # Log only on sequence position change
        if seq_pos != getattr(self, '_last_log_seq', -1):
            self._last_log_seq = seq_pos
            import sys
            print(f"  [lookahead] seq={seq_pos} +{lookahead_count} future, "
                  f"{len(all_types)} types", file=sys.stderr)
        return Order(0, all_types, 'preview')

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        """Per-round entry point. Returns [(action_type, item_idx), ...] per bot."""
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # Extract bot positions and inventories
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Update congestion heatmap
        self.congestion.update(list(bot_positions.values()))

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Get orders
        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Future orders for chain planning
        future = self._get_future_orders(state, all_orders)

        # Chain planning
        chain_plan = self.chain_planner.plan_chain(
            active_order, future, bot_positions, bot_inventories)

        # Compute active shortfall for opportunistic pickup logic
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

        # Task allocation with future orders and chain plan
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future, chain_plan=chain_plan)
        self._last_goal_types = goal_types  # for diagnostics

        # Build urgency order with tiebreak rotation
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions[bid], goals.get(bid, self.spawn))
            rotation = (bid + rnd) % 100
            if gt == 'deliver':
                return (0, dist, rotation)
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 4, dist, rotation)
            elif gt == 'pickup':
                return (2, dist, rotation)
            elif gt in ('stage', 'preview'):
                return (3, dist, rotation)
            else:
                return (5, dist, rotation)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # Pathfinding with recursive PIBT
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types,
            round_number=rnd)

        # Build final actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        # Track preview types picked this round to prevent surplus
        preview_picked_round: dict[int, int] = {}  # type_id → count picked

        total_short = sum(active_short.values())

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape: override everything after 3+ rounds stuck
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF: deliver if carrying items AND goal is 'deliver'
            # Staging bots (gt='stage') WAIT at dropoff for chain reactions —
            # their items auto-deliver when another bot completes active order
            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET: pick up item
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # ADJACENT to needed item (opportunistic — all bot types)
            if len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, active_order, preview_order, gt,
                    bot_inventories[bid], active_short, chain_plan,
                    preview_picked_round)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            # Use pathfinder's action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_pickup(self, bid: int, pos: tuple[int, int],
                                active_order: Order | None,
                                preview_order: Order | None,
                                goal_type: str,
                                bot_inv: list[int],
                                active_short: dict[int, int],
                                chain_plan=None,
                                preview_picked_round=None) -> tuple[int, int] | None:
        """Check if any adjacent item is worth picking up."""
        ms = self.ms
        bot_types = set(bot_inv)
        total_short = sum(active_short.values())

        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])

            # Active shortfall: highest priority
            if tid in active_short and active_short[tid] > 0:
                if tid in bot_types and active_short[tid] <= 1:
                    continue
            elif total_short == 0 and preview_order and preview_order.needs_type(tid):
                # Guard: bots near full inventory shouldn't pick preview speculatively
                if len(bot_inv) >= 2 and goal_type not in ('pickup', 'preview', 'deliver'):
                    continue
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

        # Update congestion and stall
        self.congestion.update(list(bot_pos_dict.values()))
        for bid, pos in bot_pos_dict.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        # Future orders for task allocation (no chain planning — too aggressive for live)
        future = []
        chain_plan = None
        if self.future_orders and active_order:
            seq_pos = self._find_sequence_pos(active_order)
            if seq_pos >= 0:
                for i in range(seq_pos + 2, min(seq_pos + 6, len(self.future_orders))):
                    future.append(self.future_orders[i])

        # Compute active shortfall
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

        # Task allocation (standard allocator — pipeline too congestion-prone)
        num_rounds = data.get('max_rounds', 500)
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_pos_dict, bot_inv_dict,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future, chain_plan=None,
            allow_preview_pickup=True)

        # Urgency order with tiebreak rotation (idle bots get lowest priority)
        priority_map = {'deliver': 0, 'pickup': 1, 'stage': 2, 'preview': 3, 'flee': 4, 'park': 5}
        all_bids = [bot['id'] for bot in live_bots]
        urgency_order = sorted(all_bids, key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_pos_dict.get(bid, self.spawn),
                                     goals.get(bid, self.spawn)),
            (bid + rnd) % 100
        ))

        # Pathfinding with recursive PIBT
        path_actions = self.pathfinder.plan_all(
            bot_pos_dict, goals, urgency_order, goal_types=goal_types,
            round_number=rnd)

        # Build WS actions
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

            # At dropoff: deliver only if goal is 'deliver' (not 'stage')
            if pos in self.drop_set and gt == 'deliver' and inv_names:
                ws_actions.append({'bot': bid, 'action': 'drop_off'})
                continue

            # At pickup target (active, preview, or future items)
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id']
                    })
                    continue

            # Opportunistic: pick active-needed items (all bot types)
            if len(inv_names) < INV_CAP and active_short:
                opp = self._ws_active_adjacent(bid, pos, ms, active_short)
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # Fill spare slots with preview items (all bot types, guard: <2 items for idle)
            if (len(inv_names) < INV_CAP
                    and not active_short and preview_order):
                # Idle bots (park/flee/stage): only if <2 items to avoid dead inv
                if gt in ('park', 'flee', 'stage') and len(inv_names) >= 2:
                    pass
                else:
                    opp = self._ws_preview_adjacent(bid, pos, ms, preview_order,
                                                    set(bot_inv_dict.get(bid, [])))
                    if opp is not None:
                        ws_actions.append(opp)
                        continue

            # Use pathfinder action
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_active_adjacent(self, bid: int, pos: tuple[int, int],
                            ms: MapState, active_short: dict[int, int]) -> dict | None:
        """Pick up adjacent item only if type is still needed by active order."""
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {
                        'bot': bid,
                        'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    }
        return None

    def _ws_preview_adjacent(self, bid: int, pos: tuple[int, int],
                             ms: MapState, preview_order: Order,
                             bot_types: set) -> dict | None:
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

    def _ws_check_adjacent(self, bid: int, pos: tuple[int, int],
                            ms: MapState, orders_data: list,
                            chain_plan=None) -> dict | None:
        """Check for opportunistic adjacent pickups in WS format.

        V3: picks active-needed AND chain-valuable items.
        """
        active_needs = set()
        for od in orders_data:
            if od.get('status') != 'active':
                continue
            req = od.get('items_required', [])
            delivered = od.get('items_delivered', [])
            remaining = list(req)
            for d in delivered:
                if d in remaining:
                    remaining.remove(d)
            active_needs = {ms.type_name_to_id.get(n, -1) for n in remaining}
            active_needs.discard(-1)

        type_values = chain_plan.future_type_values if chain_plan else {}

        best_idx = None
        best_val = -1
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            # Active items: always valuable
            if tid in active_needs:
                val = 10.0
            elif type_values.get(tid, 0) > 0.5:
                val = type_values[tid]
            else:
                continue
            if val <= best_val:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    best_idx = item_idx
                    best_val = val
                    break

        if best_idx is not None and best_idx < len(ms.items):
            return {
                'bot': bid,
                'action': 'pick_up',
                'item_id': ms.items[best_idx]['id'],
            }
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False, live_map: MapState | None = None) -> tuple[int, list]:
        """Run full simulation with V3 chain pipeline. Returns (score, action_log).

        If live_map is provided, uses the live server map layout with seed-based
        orders. Otherwise falls back to procedural map (legacy).
        """
        if live_map is not None:
            # Use live map layout + seed-based orders
            all_orders = generate_all_orders(seed, live_map, 'nightmare', count=100)
            num_bots = CONFIGS['nightmare']['bots']
            state = GameState(live_map)
            state.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
            state.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
            for i in range(num_bots):
                state.bot_positions[i] = [live_map.spawn[0], live_map.spawn[1]]
            state.orders = [all_orders[0].copy(), all_orders[1].copy()]
            state.orders[0].status = 'active'
            state.orders[1].status = 'preview'
            state.next_order_idx = 2
            state.active_idx = 0
            ms = live_map
        else:
            state, all_orders = init_game(seed, 'nightmare', num_orders=100)
            ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV3(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains = 0
        max_chain = 0
        action_log = []

        # Utilization tracking
        goal_totals = {'deliver': 0, 'pickup': 0, 'preview': 0, 'stage': 0, 'flee': 0, 'park': 0}
        order_rounds = []
        stall_total = 0
        escape_total = 0

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd

            # PRE-STEP: snapshot bots at dropoff with items (for chain diagnosis)
            drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
            pre_at_drop = {}
            for b in range(len(state.bot_positions)):
                bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                if bp in drop_set:
                    inv = state.bot_inv_list(b)
                    if inv:
                        pre_at_drop[b] = (bp, inv)

            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            # Track utilization
            for gt in getattr(solver, '_last_goal_types', {}).values():
                if gt in goal_totals:
                    goal_totals[gt] += 1
            # Track stalls
            for b in range(len(state.bot_positions)):
                if solver.stall_counts.get(b, 0) >= 1:
                    stall_total += 1
                if solver.stall_counts.get(b, 0) >= 3:
                    escape_total += 1
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 0:
                order_rounds.append(rnd)
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN×{c}!" if c > 1 else ""
                # Post-step: bots at dropoff
                dropoff_info = ""
                if c >= 1:
                    at_drop = []
                    for b in range(len(state.bot_positions)):
                        bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                        if bp in drop_set:
                            inv = state.bot_inv_list(b)
                            at_drop.append(f"b{b}:{inv}")
                    if at_drop:
                        dropoff_info = f" AtDrop=[{', '.join(at_drop)}]"
                    # Pre-step staging diagnostic
                    if pre_at_drop:
                        pre_info = ", ".join(f"b{b}@{p}:{i}" for b, (p, i) in pre_at_drop.items())
                        dropoff_info += f" PRE=[{pre_info}]"
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra + dropoff_info)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
            if solver.chain_events:
                print(f"Chain events: {solver.chain_events}")
            # Utilization summary
            avg_per_rnd = {gt: cnt / num_rounds for gt, cnt in sorted(goal_totals.items())}
            working = avg_per_rnd.get('deliver', 0) + avg_per_rnd.get('pickup', 0) + avg_per_rnd.get('preview', 0) + avg_per_rnd.get('stage', 0)
            idle = avg_per_rnd.get('flee', 0) + avg_per_rnd.get('park', 0)
            print(f"Avg/rnd: {' '.join(f'{gt}={v:.1f}' for gt, v in avg_per_rnd.items())}")
            print(f"Working={working:.1f} Idle={idle:.1f} ({idle/(working+idle)*100:.0f}% idle)")
            if len(order_rounds) > 1:
                gaps = [order_rounds[i+1] - order_rounds[i] for i in range(len(order_rounds)-1)]
                print(f"Order gaps: avg={np.mean(gaps):.1f} min={min(gaps)} max={max(gaps)}")
            print(f"Stalls: {stall_total} ({stall_total/num_rounds:.1f}/rnd) "
                  f"Escapes: {escape_total} ({escape_total/num_rounds:.2f}/rnd)")
        return state.score, action_log


# Keep V2 as alias for backward compatibility
class NightmareSolverV2(NightmareSolverV3):
    """Backward-compatible alias. V3 is a strict superset of V2."""
    pass


DB_URL = "postgres://grocery:grocery123@localhost:5433/grocery_bot"


def record_to_pg(seed, score, orders_completed, items_delivered, action_log, elapsed):
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
        ms = build_map('nightmare')
        cfg = CONFIGS['nightmare']

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
            seed, 'nightmare', ms.width, ms.height, cfg['bots'],
            ms.num_types, cfg['order_size'][0], cfg['order_size'][1],
            json.dumps(walls), json.dumps(shelves),
            json.dumps(items), json.dumps(list(ms.drop_off)),
            json.dumps(list(ms.spawn)),
            score, items_delivered, orders_completed,
            'synthetic',
        ))
        run_id = cur.fetchone()[0]

        # Insert round data from action_log
        if action_log:
            from game_engine import init_game, step as game_step
            gs, all_orders = init_game(seed, 'nightmare', num_orders=100)
            round_tuples = []
            for rnd in range(min(len(action_log), 500)):
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
    import sys
    parser = argparse.ArgumentParser(description='Nightmare solver V3 (chain pipeline)')
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-record', action='store_true', help='Skip PostgreSQL recording')
    parser.add_argument('--v2', action='store_true', help='Use V2 solver (no chain planning)')
    parser.add_argument('--no-live-map', action='store_true',
                        help='Use procedural map instead of live server map')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    # Load live map from captured data (default for nightmare)
    live_map = None
    if not args.no_live_map:
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap and cap.get('grid'):
                live_map = build_map_from_capture(cap)
                print(f"Using live map: {live_map.width}x{live_map.height}, "
                      f"{live_map.num_items} items, "
                      f"{sum(1 for y in range(live_map.height) for x in range(live_map.width) if live_map.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF))} walkable",
                      file=sys.stderr)
            else:
                print("No capture data found, using procedural map", file=sys.stderr)
        except Exception as e:
            print(f"Could not load live map: {e}, using procedural map", file=sys.stderr)

    scores = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        score, action_log = NightmareSolverV3.run_sim(seed, verbose=args.verbose,
                                                       live_map=live_map)
        elapsed = time.time() - st
        scores.append(score)
        print(f"Seed {seed}: {score}")

        if not args.no_record:
            # Replay for DB recording using same map
            if live_map is not None:
                all_orders2 = generate_all_orders(seed, live_map, 'nightmare', count=100)
                num_bots = CONFIGS['nightmare']['bots']
                state2 = GameState(live_map)
                state2.bot_positions = np.zeros((num_bots, 2), dtype=np.int16)
                state2.bot_inventories = np.full((num_bots, INV_CAP), -1, dtype=np.int8)
                for i in range(num_bots):
                    state2.bot_positions[i] = [live_map.spawn[0], live_map.spawn[1]]
                state2.orders = [all_orders2[0].copy(), all_orders2[1].copy()]
                state2.orders[0].status = 'active'
                state2.orders[1].status = 'preview'
                state2.next_order_idx = 2
                state2.active_idx = 0
            else:
                state2, all_orders2 = init_game(seed, 'nightmare', num_orders=100)
            for rnd, acts in enumerate(action_log):
                state2.round = rnd
                step(state2, acts, all_orders2)
            record_to_pg(seed, score, state2.orders_completed,
                         state2.items_delivered, action_log, elapsed)

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()
