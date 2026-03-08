"""Token Passing Pickup-and-Delivery solver for Nightmare mode.

Based on "Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks"
(Ma et al., 2017). Hybrid approach:
  - Global cost-optimal task assignment (not zone-based)
  - PIBT-style reactive pathfinding (proven reliable)
  - Strict active/preview/future separation to avoid dead inventory
  - Conveyor belt delivery with chain reaction staging

20 bots, 3 dropoff zones, 500 rounds, 21 item types.
"""
from __future__ import annotations

import time
from itertools import permutations

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class NightmareTPSolver:
    """Hybrid Token Passing + PIBT solver.

    Per-round reactive allocation with global cost function.
    Active/preview/future item separation prevents dead inventory.
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

        self.future_orders = future_orders or []
        self._seq_pos = -1

        # PIBT pathfinder
        self.traffic = TrafficRules(map_state)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(map_state, self.tables,
                                               self.traffic, self.congestion)

        # Item type → [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            adj = map_state.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Persistent goal tracking: {bid: (goal_pos, goal_type, item_idx, type_id, assigned_round)}
        self._persistent_goals: dict[int, tuple[tuple[int, int], str, int, int, int]] = {}

        # Stall tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}

        # Chain tracking
        self.chain_events: list[tuple[int, int]] = []
        self._debug = False

    # ------------------------------------------------------------------
    # Per-round allocation
    # ------------------------------------------------------------------

    def _allocate(self, bot_positions: dict[int, tuple[int, int]],
                  bot_inventories: dict[int, list[int]],
                  active: Order | None, preview: Order | None,
                  rnd: int, num_rounds: int,
                  future: list[Order] | None = None
                  ) -> tuple[dict, dict, dict]:
        """Compute per-bot goals for this round.

        Returns: (goals, goal_types, pickup_targets)
        """
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        # --- Phase 1: Identify active carriers and active shortfall ---
        active_needs: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Who's carrying active items?
        active_carriers: set[int] = set()
        carrying_active: dict[int, int] = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs and active_needs.get(t, 0) > 0:
                    active_carriers.add(bid)
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        # Active shortfall: what's still needed after accounting for carriers
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        assigned_bots: set[int] = set()

        # --- Phase 2: Active carriers → deliver to nearest dropoff ---
        for bid in sorted(active_carriers):
            pos = bot_positions.get(bid)
            if pos is None:
                continue
            drop = min(self.drop_zones,
                       key=lambda dz: self.tables.get_distance(pos, dz))
            goals[bid] = drop
            goal_types[bid] = 'deliver'
            assigned_bots.add(bid)

        # --- Phase 3: Assign bots to fetch active shortfall items ---
        # Track which bots are already heading to fetch active items (persistent)
        fetching_active: dict[int, int] = {}  # type_id -> count
        for bid, (_, gt, _, tid, _arnd) in self._persistent_goals.items():
            if gt == 'pickup' and bid not in assigned_bots and tid in active_short:
                fetching_active[tid] = fetching_active.get(tid, 0) + 1

        for t, shortage in list(active_short.items()):
            shortage -= fetching_active.get(t, 0)
            if shortage <= 0:
                continue

            for _ in range(shortage):
                best_bid = self._find_best_fetcher(
                    t, bot_positions, bot_inventories, assigned_bots)
                if best_bid is None:
                    break
                pos = bot_positions[best_bid]
                shelf = self._best_shelf_from(t, pos)
                if shelf is None:
                    break
                item_idx, adj_pos = shelf
                goals[best_bid] = adj_pos
                goal_types[best_bid] = 'pickup'
                pickup_targets[best_bid] = item_idx
                self._persistent_goals[best_bid] = (adj_pos, 'pickup', item_idx, t, rnd)
                assigned_bots.add(best_bid)

        # --- Phase 4: Non-active carriers → stage near dropoff (max 1 per zone) ---
        # Stage at most 1 bot per dropoff zone (for chain reactions)
        staged_zones: set[tuple[int, int]] = set()
        # Don't stage at zones where active carriers are heading
        for bid in active_carriers:
            if bid in goals:
                staged_zones.add(goals[bid])

        for bid, inv in bot_inventories.items():
            if bid in assigned_bots or not inv:
                continue
            pos = bot_positions.get(bid)
            if pos is None:
                continue
            # Find a dropoff zone that isn't clogged
            free_zones = [dz for dz in self.drop_zones if dz not in staged_zones]
            if not free_zones:
                # Park near dropoff instead
                goals[bid] = self._near_dropoff(pos)
                goal_types[bid] = 'stage'
                assigned_bots.add(bid)
                continue
            drop = min(free_zones,
                       key=lambda dz: self.tables.get_distance(pos, dz))
            goals[bid] = drop
            goal_types[bid] = 'stage'
            staged_zones.add(drop)
            assigned_bots.add(bid)

        # --- Phase 5: Empty bots → fetch preview/future items ---
        if preview:
            preview_needs: dict[int, int] = {}
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

            # Subtract what's already being carried/fetched for preview
            for bid, inv in bot_inventories.items():
                for t in inv:
                    if t in preview_needs and preview_needs[t] > 0:
                        preview_needs[t] -= 1

            for t, need in list(preview_needs.items()):
                if need <= 0:
                    continue
                best_bid = self._find_best_fetcher(
                    t, bot_positions, bot_inventories, assigned_bots)
                if best_bid is None:
                    continue
                pos = bot_positions[best_bid]
                shelf = self._best_shelf_from(t, pos)
                if shelf is None:
                    continue
                item_idx, adj_pos = shelf
                goals[best_bid] = adj_pos
                goal_types[best_bid] = 'preview'
                pickup_targets[best_bid] = item_idx
                self._persistent_goals[best_bid] = (adj_pos, 'preview', item_idx, t, rnd)
                assigned_bots.add(best_bid)
                preview_needs[t] -= 1

        # --- Phase 6: Remaining bots → fetch future items or park ---
        if future:
            for order in future[:4]:
                for t in order.needs():
                    best_bid = self._find_best_fetcher(
                        t, bot_positions, bot_inventories, assigned_bots,
                        max_inv=INV_CAP - 1)  # leave 1 slot for active
                    if best_bid is None:
                        continue
                    pos = bot_positions[best_bid]
                    shelf = self._best_shelf_from(t, pos)
                    if shelf is None:
                        continue
                    item_idx, adj_pos = shelf
                    goals[best_bid] = adj_pos
                    goal_types[best_bid] = 'preview'
                    pickup_targets[best_bid] = item_idx
                    self._persistent_goals[best_bid] = (adj_pos, 'preview', item_idx, t, rnd)
                    assigned_bots.add(best_bid)

        # --- Phase 7: Unassigned bots → park or use persistent goal ---
        for bid in range(self.num_bots):
            if bid in assigned_bots or bid in goals:
                continue
            pos = bot_positions.get(bid)
            if pos is None:
                goals[bid] = self.spawn
                goal_types[bid] = 'park'
                continue

            # Keep persistent goal if still valid
            if bid in self._persistent_goals:
                pgoal, pgt, pidx, ptid, _arnd = self._persistent_goals[bid]
                if pgt == 'pickup' and ptid in active_short:
                    goals[bid] = pgoal
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = pidx
                    continue
                elif pgt == 'preview':
                    goals[bid] = pgoal
                    goal_types[bid] = 'preview'
                    pickup_targets[bid] = pidx
                    continue

            # Park: stay where we are if not blocking
            inv = bot_inventories.get(bid, [])
            if inv:
                # Has items but no task → deliver (they might match future active)
                drop = min(self.drop_zones,
                           key=lambda dz: self.tables.get_distance(pos, dz))
                goals[bid] = drop
                goal_types[bid] = 'stage'
            elif pos in self.drop_set:
                # Empty bot at dropoff → flee
                goals[bid] = self._flee_target(pos)
                goal_types[bid] = 'flee'
            else:
                goals[bid] = pos
                goal_types[bid] = 'park'

        return goals, goal_types, pickup_targets

    def _find_best_fetcher(self, type_id: int,
                            bot_positions: dict[int, tuple[int, int]],
                            bot_inventories: dict[int, list[int]],
                            assigned: set[int],
                            max_inv: int = INV_CAP) -> int | None:
        """Find the best unassigned bot to fetch given type.

        Uses zone bias: bots prefer shelves in their "home zone" to reduce congestion.
        """
        best_bid = None
        best_cost = 9999

        for bid in range(self.num_bots):
            if bid in assigned:
                continue
            inv = bot_inventories.get(bid, [])
            if len(inv) >= max_inv:
                continue
            pos = bot_positions.get(bid)
            if pos is None:
                continue

            shelf = self._best_shelf_from(type_id, pos)
            if shelf is None:
                continue
            _, adj_pos = shelf

            cost = self.tables.get_distance(pos, adj_pos)
            # Zone bias: bots 0-6=LEFT(x<10), 7-13=MID(10-20), 14-19=RIGHT(x>20)
            bot_zone = 0 if bid < 7 else (1 if bid < 14 else 2)
            shelf_x = adj_pos[0]
            shelf_zone = 0 if shelf_x < 10 else (1 if shelf_x < 20 else 2)
            if bot_zone != shelf_zone:
                cost += 5  # zone crossing penalty
            # Prefer bots with lower IDs (move first)
            cost += bid * 0.05
            # Prefer empty bots
            if not inv:
                cost -= 1

            if cost < best_cost:
                best_cost = cost
                best_bid = bid

        return best_bid

    def _best_shelf_from(self, type_id: int, from_pos: tuple[int, int]
                          ) -> tuple[int, tuple[int, int]] | None:
        """Find closest (item_idx, adj_pos) for type."""
        candidates = self.type_items.get(type_id, [])
        if not candidates:
            return None
        best = None
        best_dist = 9999
        for item_idx, adj_list in candidates:
            for adj in adj_list:
                d = self.tables.get_distance(from_pos, adj)
                if d < best_dist:
                    best_dist = d
                    best = (item_idx, adj)
        return best

    def _near_dropoff(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Find a cell 1-2 steps from nearest dropoff (not on it)."""
        drop = min(self.drop_zones,
                   key=lambda dz: self.tables.get_distance(pos, dz))
        best = pos
        best_dist = 9999
        for cell in self.walkable:
            if cell in self.drop_set:
                continue
            d_to_drop = self.tables.get_distance(cell, drop)
            if 1 <= d_to_drop <= 3:
                d_from_me = self.tables.get_distance(pos, cell)
                if d_from_me < best_dist:
                    best_dist = d_from_me
                    best = cell
        return best

    def _flee_target(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Find a cell away from dropoff zones."""
        best = pos
        best_score = -9999
        for cell in self.walkable:
            if cell in self.drop_set or cell == self.spawn:
                continue
            drop_dist = min(self.tables.get_distance(cell, dz) for dz in self.drop_zones)
            my_dist = self.tables.get_distance(pos, cell)
            if my_dist > 8:
                continue
            score = drop_dist * 2 - my_dist
            if score > best_score:
                best_score = score
                best = cell
        return best

    # ------------------------------------------------------------------
    # Per-round action
    # ------------------------------------------------------------------

    def action(self, state: GameState, all_orders: list[Order], rnd: int) -> list[tuple[int, int]]:
        """Per-round action for sim mode."""
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # Extract state
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall detection
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        self.congestion.update(list(bot_positions.values()))

        active = state.get_active_order()
        preview = state.get_preview_order()
        future = self._get_future_orders(state, all_orders)

        # Clear persistent goals for items we already picked up
        self._clean_persistent_goals(bot_inventories, rnd)

        # Allocate goals
        goals, goal_types, pickup_targets = self._allocate(
            bot_positions, bot_inventories, active, preview, rnd, num_rounds, future)

        # PIBT urgency order
        urgency_order = self._urgency_order(bot_positions, goals, goal_types)

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build final actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF: deliver only if carrying active-matching items
            if pos in self.drop_set and inv and gt == 'deliver':
                if active and any(active.needs_type(t) for t in inv):
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                else:
                    # Items don't match active → flee dropoff
                    act = self._escape_action(bid, pos, rnd)
                    actions[bid] = (act, -1)
                    continue

            # AT DROPOFF staging: deliver if preview/chain items match
            if pos in self.drop_set and inv and gt == 'stage':
                # Stage bots WAIT at dropoff. When another bot completes
                # the active order, chain reaction auto-delivers their items.
                actions[bid] = (ACT_WAIT, -1)
                continue

            # AT PICKUP TARGET: pick up
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    # Clear persistent goal (pickup done)
                    self._persistent_goals.pop(bid, None)
                    continue

            # Opportunistic active pickup (only for active-shortfall items)
            if len(inv) < INV_CAP and active and gt != 'stage':
                opp = self._adjacent_active_pickup(bid, pos, active, inv)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Deliver bots: pick up active items en route
            if gt == 'deliver' and len(inv) < INV_CAP and active:
                opp = self._adjacent_active_pickup(bid, pos, active, inv)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _clean_persistent_goals(self, bot_inventories: dict[int, list[int]], rnd: int = 0):
        """Remove persistent goals for items already picked up or stale goals."""
        to_remove = []
        for bid, (_, gt, _, tid, assigned_rnd) in self._persistent_goals.items():
            if gt in ('pickup', 'preview'):
                inv = bot_inventories.get(bid, [])
                if tid in inv:
                    to_remove.append(bid)
                elif len(inv) >= INV_CAP:
                    to_remove.append(bid)
                # Stale goal: been heading there for 25+ rounds
                elif rnd - assigned_rnd > 25:
                    to_remove.append(bid)
        for bid in to_remove:
            self._persistent_goals.pop(bid, None)

    def _adjacent_active_pickup(self, bid: int, pos: tuple[int, int],
                                 active: Order, inv: list[int]) -> tuple[int, int] | None:
        """Pick up adjacent item needed by active order (avoid duplicates)."""
        ms = self.ms
        # Compute what's still needed
        needs: dict[int, int] = {}
        for t in active.needs():
            needs[t] = needs.get(t, 0) + 1
        for t in inv:
            if t in needs:
                needs[t] -= 1
                if needs[t] <= 0:
                    del needs[t]
        if not needs:
            return None

        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in needs:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    def _urgency_order(self, bot_positions, goals, goal_types):
        priority_map = {'deliver': 0, 'flee': 1, 'pickup': 2, 'stage': 3,
                        'preview': 4, 'park': 5}

        def key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(bot_positions.get(bid, self.spawn),
                                            goals.get(bid, self.spawn))
            return (priority_map.get(gt, 5), dist)

        return sorted(range(self.num_bots), key=key)

    def _escape_action(self, bid: int, pos: tuple[int, int], rnd: int) -> int:
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def _get_future_orders(self, state: GameState, all_orders: list[Order],
                            depth: int = 8) -> list[Order]:
        future: list[Order] = []
        preview = state.get_preview_order()
        if preview:
            future.append(preview)
        if all_orders:
            for i in range(state.next_order_idx,
                           min(state.next_order_idx + depth, len(all_orders))):
                future.append(all_orders[i])
        elif self.future_orders:
            start_idx = state.orders_completed + 2
            for i in range(start_idx, min(start_idx + depth, len(self.future_orders))):
                future.append(self.future_orders[i])
        return future[:depth]

    # ------------------------------------------------------------------
    # WebSocket interface
    # ------------------------------------------------------------------

    def ws_action(self, live_bots: list[dict], data: dict, map_state: MapState) -> list[dict]:
        """Per-round entry for live WebSocket games."""
        ms = map_state or self.ms
        rnd = data.get('round', 0)

        # Parse orders
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

        # Parse bots
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

        # Stall/congestion
        for bid, pos in bot_pos_dict.items():
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos
        self.congestion.update(list(bot_pos_dict.values()))

        num_rounds = data.get('max_rounds', 500)
        self._clean_persistent_goals(bot_inv_dict, rnd)

        # Future orders
        future = []
        if self.future_orders:
            est_completed = data.get('score', 0) // 5
            start_idx = est_completed + 2
            for i in range(start_idx, min(start_idx + 8, len(self.future_orders))):
                future.append(self.future_orders[i])

        # Allocate
        goals, goal_types, pickup_targets = self._allocate(
            bot_pos_dict, bot_inv_dict, active_order, preview_order, rnd, num_rounds, future)

        # Pathfinding
        urgency_order = self._urgency_order(bot_pos_dict, goals, goal_types)
        path_actions = self.pathfinder.plan_all(
            bot_pos_dict, goals, urgency_order, goal_types=goal_types)

        # Build WS actions
        ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right',
                        'pick_up', 'drop_off']
        ws_actions = []

        for bot in live_bots:
            bid = bot['id']
            pos = tuple(bot['position'])
            inv_names = bot.get('inventory', [])
            inv = bot_inv_dict.get(bid, [])
            gt = goal_types.get(bid, 'park')

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                continue

            # Deliver at dropoff (only if active items match)
            if pos in self.drop_set and inv_names and gt == 'deliver':
                if active_order and any(active_order.needs_type(t) for t in inv):
                    ws_actions.append({'bot': bid, 'action': 'drop_off'})
                    continue
                else:
                    act = self._escape_action(bid, pos, rnd)
                    ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})
                    continue

            # Stage at dropoff
            if pos in self.drop_set and inv_names and gt == 'stage':
                ws_actions.append({'bot': bid, 'action': 'wait'})
                continue

            # Pickup at target
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                goal = goals.get(bid)
                if pos == goal and item_idx < len(ms.items):
                    ws_actions.append({
                        'bot': bid, 'action': 'pick_up',
                        'item_id': ms.items[item_idx]['id'],
                    })
                    self._persistent_goals.pop(bid, None)
                    continue

            # Opportunistic active pickup
            if len(inv_names) < INV_CAP and active_order and gt != 'stage':
                opp = self._ws_adjacent_active(bid, pos, ms, active_order, inv)
                if opp is not None:
                    ws_actions.append(opp)
                    continue

            # PIBT
            act = path_actions.get(bid, ACT_WAIT)
            ws_actions.append({'bot': bid, 'action': ACTION_NAMES[act]})

        return ws_actions

    def _ws_adjacent_active(self, bid, pos, ms, active, inv):
        needs: dict[int, int] = {}
        for t in active.needs():
            needs[t] = needs.get(t, 0) + 1
        for t in inv:
            if t in needs:
                needs[t] -= 1
                if needs[t] <= 0:
                    del needs[t]
        if not needs:
            return None
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in needs:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return {'bot': bid, 'action': 'pick_up',
                            'item_id': ms.items[item_idx]['id']}
        return None

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    @staticmethod
    def run_sim(seed: int, verbose: bool = False, debug: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareTPSolver(ms, tables, future_orders=all_orders)
        solver._debug = debug
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
                solver.chain_events.append((rnd, c))

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                # Bots at dropoff
                drop_info = ""
                if c >= 1:
                    at = []
                    for b in range(len(state.bot_positions)):
                        bp = (int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1]))
                        if bp in solver.drop_set:
                            inv = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv}")
                    drop_info = f" Drop=[{','.join(at)}]"
                print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}" if active else " DONE")
                      + extra + drop_info)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
                  f" Items={state.items_delivered} Chains={chains} MaxChain={max_chain}"
                  f" Time={elapsed:.1f}s ({elapsed/num_rounds*1000:.1f}ms/rnd)")
            if solver.chain_events:
                print(f"Chain events: {solver.chain_events}")
        return state.score, action_log


# CLI
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_tp = []
    scores_v3 = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - Token Passing Solver")
        print(f"{'='*50}")
        score, _ = NightmareTPSolver.run_sim(seed, verbose=args.verbose, debug=args.debug)
        scores_tp.append(score)

        if args.compare:
            from nightmare_solver_v2 import NightmareSolverV3
            print(f"\n--- V3 ---")
            s3, _ = NightmareSolverV3.run_sim(seed, verbose=args.verbose)
            scores_v3.append(s3)
            print(f"\nTP={score} vs V3={s3} (delta={score - s3:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nTP: mean={statistics.mean(scores_tp):.1f}")
        if scores_v3:
            print(f"V3: mean={statistics.mean(scores_v3):.1f}")
