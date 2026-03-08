"""Assembly Pipeline Solver for Nightmare mode.

Fundamental redesign: plan ONCE per order change, then execute.

Key differences from V4:
1. Per-order planning (not round-by-round re-planning)
2. Parallel assignment: active + preview fetchers assigned simultaneously
3. Role persistence: bots keep assignments until delivery
4. Stagers ACT_WAIT at dropoff (don't consume items for chain)
5. All deliver bots ACT_DROPOFF (trigger chain when active completes)

Architecture:
- When order changes: compute optimal bot-to-item assignments for both active AND preview
- Each bot gets a persistent trip: (goal_pos, item_idx, role, dropoff)
- Bots execute trips via PIBT pathfinding
- At dropoff: trigger bots ACT_DROPOFF, stager bots ACT_WAIT
- Chain fires when active completes, auto-delivering stager items
"""
from __future__ import annotations

import time

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class AssemblyPipelineSolver:
    """Plan-once-per-order solver with chain orchestration."""

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None):
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.future_orders = future_orders or []

        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)

        # Item index: type_id → [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Zone assignment for load balancing
        self.bot_zone: dict[int, int] = {}
        sorted_drops = sorted(self.drop_zones, key=lambda d: d[0])
        self.zone_dropoff = {i: dz for i, dz in enumerate(sorted_drops)}
        for bid in range(20):
            if bid < 7:
                self.bot_zone[bid] = 0
            elif bid < 14:
                self.bot_zone[bid] = 1
            else:
                self.bot_zone[bid] = 2

        # State tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self.chain_events: list[tuple[int, int]] = []

        # Per-order plan
        self._plan: dict[int, dict] = {}  # bid → {goal, item_idx, role, dropoff, phase}
        self._last_order_id = -1
        self._last_preview_id = -1

    def _compute_plan(self, bot_positions, bot_inventories,
                      active, preview, future_orders):
        """Compute bot assignments for current active + preview orders."""
        plan: dict[int, dict] = {}
        claimed_items: set[int] = set()
        claimed_bots: set[int] = set()

        # Determine what active still needs (not yet carried)
        active_needs: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        # Determine what preview needs
        preview_needs: dict[int, int] = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Future order needs (for pipe/prefetch)
        future_needs: dict[int, int] = {}
        if future_orders:
            for order in future_orders[:3]:
                for t in order.needs():
                    future_needs[t] = future_needs.get(t, 0) + 1

        # Step 1: Classify bots with existing inventory
        for bid, inv in bot_inventories.items():
            if not inv:
                continue
            has_active = any(t in active_needs for t in inv)
            has_preview = any(t in preview_needs for t in inv)
            has_future = any(t in future_needs for t in inv)

            if has_active:
                # Route to nearest dropoff for delivery
                dz = self._nearest_dropoff(bot_positions[bid])
                plan[bid] = {
                    'goal': dz,
                    'role': 'trigger',
                    'phase': 'deliver',
                }
                claimed_bots.add(bid)
                # Subtract from needs
                for t in inv:
                    if active_needs.get(t, 0) > 0:
                        active_needs[t] -= 1
            elif has_preview:
                # Route to staging dropoff
                dz = self._staging_dropoff(bot_positions[bid], plan)
                plan[bid] = {
                    'goal': dz,
                    'role': 'stager',
                    'phase': 'deliver',
                }
                claimed_bots.add(bid)
                for t in inv:
                    if preview_needs.get(t, 0) > 0:
                        preview_needs[t] -= 1
            elif has_future:
                # Hold for future - go to nearest dropoff area
                dz = self._nearest_dropoff(bot_positions[bid])
                plan[bid] = {
                    'goal': dz,
                    'role': 'pipe',
                    'phase': 'deliver',
                }
                claimed_bots.add(bid)
            else:
                # Dead inventory - park
                plan[bid] = {
                    'goal': self._park_spot(bot_positions[bid], plan),
                    'role': 'dead',
                    'phase': 'park',
                }
                claimed_bots.add(bid)

        # Step 2: Assign empty bots to fetch items
        # Active items first, then preview, then future
        empty_bots = [bid for bid in bot_inventories
                      if not bot_inventories[bid] and bid not in claimed_bots]

        # Sort by proximity to nearest needed item
        def _bot_priority(bid):
            pos = bot_positions[bid]
            min_d = 9999
            for needs in [active_needs, preview_needs]:
                for tid in needs:
                    if needs[tid] <= 0:
                        continue
                    for _, adj_cells in self.type_items.get(tid, []):
                        for adj in adj_cells:
                            d = self.tables.get_distance(pos, adj)
                            if d < min_d:
                                min_d = d
            return min_d

        empty_bots.sort(key=_bot_priority)

        # Assign to active needs
        active_type_assigned: dict[int, int] = {}
        for bid in list(empty_bots):
            if not active_needs or all(v <= 0 for v in active_needs.values()):
                break
            pos = bot_positions[bid]
            item_idx, adj_pos = self._find_best_item(
                pos, active_needs, claimed_items, active_type_assigned)
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._nearest_dropoff(adj_pos)
                plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'trigger',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                }
                claimed_bots.add(bid)
                claimed_items.add(item_idx)
                active_type_assigned[tid] = active_type_assigned.get(tid, 0) + 1
                active_needs[tid] = active_needs.get(tid, 0) - 1
                empty_bots.remove(bid)

        # Assign to preview needs
        preview_type_assigned: dict[int, int] = {}
        for bid in list(empty_bots):
            if not preview_needs or all(v <= 0 for v in preview_needs.values()):
                break
            pos = bot_positions[bid]
            item_idx, adj_pos = self._find_best_item(
                pos, preview_needs, claimed_items, preview_type_assigned)
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._staging_dropoff(adj_pos, plan)
                plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'stager',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                }
                claimed_bots.add(bid)
                claimed_items.add(item_idx)
                preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                preview_needs[tid] = preview_needs.get(tid, 0) - 1
                empty_bots.remove(bid)

        # Assign remaining to future needs
        future_type_assigned: dict[int, int] = {}
        for bid in list(empty_bots):
            if not future_needs or all(v <= 0 for v in future_needs.values()):
                break
            pos = bot_positions[bid]
            item_idx, adj_pos = self._find_best_item(
                pos, future_needs, claimed_items, future_type_assigned)
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._nearest_dropoff(adj_pos)
                plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'pipe',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                }
                claimed_bots.add(bid)
                claimed_items.add(item_idx)
                future_type_assigned[tid] = future_type_assigned.get(tid, 0) + 1
                future_needs[tid] = future_needs.get(tid, 0) - 1
                empty_bots.remove(bid)

        # Park remaining bots
        for bid in empty_bots:
            plan[bid] = {
                'goal': self._park_spot(bot_positions[bid], plan),
                'role': 'idle',
                'phase': 'park',
            }

        return plan

    def _find_best_item(self, pos, needs, claimed, type_assigned):
        """Find nearest item of a needed type."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, count in needs.items():
            if count <= 0:
                continue
            if type_assigned.get(tid, 0) >= count:
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

    def _nearest_dropoff(self, pos):
        best = self.drop_zones[0]
        best_d = self.tables.get_distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _staging_dropoff(self, pos, plan):
        """Find dropoff not heavily used by trigger bots."""
        deliver_zones: set[tuple[int, int]] = set()
        for p in plan.values():
            if p.get('role') == 'trigger' and p.get('phase') == 'deliver':
                g = p.get('goal')
                if g in self.drop_set:
                    deliver_zones.add(g)

        best = None
        best_d = 9999
        for dz in self.drop_zones:
            if dz in deliver_zones:
                continue
            d = self.tables.get_distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz

        if best is None:
            best = self._nearest_dropoff(pos)
        return best

    def _park_spot(self, pos, plan):
        """Find parking spot away from dropoffs."""
        occupied = {p.get('goal') for p in plan.values() if p.get('goal')}
        corridor_ys = [1, self.ms.height // 2]
        best = self.spawn
        best_d = 9999
        for cy in corridor_ys:
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

        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active = state.get_active_order()
        preview = state.get_preview_order()

        # Get future orders
        future = []
        if all_orders:
            start = getattr(state, 'next_order_idx', 0)
            for i in range(start, min(start + 5, len(all_orders))):
                future.append(all_orders[i])

        # Detect order change → replan
        active_id = active.id if active else -1
        if active_id != self._last_order_id:
            self._plan = self._compute_plan(
                bot_positions, bot_inventories,
                active, preview, future)
            self._last_order_id = active_id

        # Update plan: handle completed trips + stale cleanup
        for bid in range(num_bots):
            p = self._plan.get(bid)
            if not p:
                continue

            pos = bot_positions[bid]
            inv = bot_inventories[bid]

            # Track trip age for stale timeout
            p['age'] = p.get('age', 0) + 1

            if p['phase'] == 'fetch':
                # Bot picked up item → switch to deliver phase
                if inv:
                    tid = p.get('type_id')
                    if tid is not None and tid in inv:
                        dz = p.get('dropoff', self._nearest_dropoff(pos))
                        p['phase'] = 'deliver'
                        p['goal'] = dz
                        p.pop('item_idx', None)
                        p['age'] = 0
                    elif len(inv) >= INV_CAP:
                        # Full inventory but not with assigned type → deliver what we have
                        dz = self._nearest_dropoff(pos)
                        p['phase'] = 'deliver'
                        p['goal'] = dz
                        p.pop('item_idx', None)
                        p['age'] = 0
                        # Reclassify role based on what we're carrying
                        has_active = active and any(active.needs_type(t) for t in inv)
                        p['role'] = 'trigger' if has_active else 'stager'

                # Stale fetch: bot stuck for 25+ rounds
                if p.get('age', 0) > 25:
                    self._reassign_bot(bid, pos, active, preview, future)

            elif p['phase'] == 'deliver':
                if pos in self.drop_set:
                    if not inv:
                        # Empty after delivery → reassign
                        self._reassign_bot(bid, pos, active, preview, future)
                    elif p['role'] == 'trigger':
                        # Trigger delivered but has leftovers
                        has_active = active and any(active.needs_type(t) for t in inv)
                        if not has_active:
                            # Leftovers don't match active → check if useful
                            has_preview = preview and any(preview.needs_type(t) for t in inv)
                            if has_preview:
                                p['role'] = 'stager'  # Hold for chain
                            else:
                                # Dead leftovers → reassign (bot still has dead items)
                                p['role'] = 'dead'
                                p['goal'] = self._park_spot(pos, self._plan)
                                p['phase'] = 'park'

                # Stager items consumed by chain
                if p['role'] == 'stager' and not inv:
                    self._reassign_bot(bid, pos, active, preview, future)

                # Stale delivery: bot can't reach dropoff for 25+ rounds
                if p.get('age', 0) > 25 and pos not in self.drop_set:
                    self._reassign_bot(bid, pos, active, preview, future)

        # Handle bots with no plan
        for bid in range(num_bots):
            if bid not in self._plan:
                self._plan[bid] = {
                    'goal': self.spawn,
                    'role': 'idle',
                    'phase': 'park',
                }

        # Build goals for PIBT
        goals: dict[int, tuple[int, int]] = {}
        goal_types: dict[int, str] = {}
        pickup_targets: dict[int, int] = {}

        for bid in range(num_bots):
            p = self._plan[bid]
            goals[bid] = p['goal']

            if p['phase'] == 'fetch':
                if p['role'] == 'trigger':
                    goal_types[bid] = 'pickup'
                else:
                    goal_types[bid] = 'preview'
                if 'item_idx' in p:
                    pickup_targets[bid] = p['item_idx']
            elif p['phase'] == 'deliver':
                if p['role'] == 'trigger':
                    goal_types[bid] = 'deliver'
                elif p['role'] == 'stager':
                    goal_types[bid] = 'stage'
                elif p['role'] == 'pipe':
                    goal_types[bid] = 'deliver'  # pipe items also deliver
                else:
                    goal_types[bid] = 'flee'
            else:
                goal_types[bid] = 'park'

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'flee':
                return (4, dist)
            elif gt == 'pickup':
                return (1, dist)
            elif gt in ('stage', 'preview'):
                return (2, dist)
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
            inv = bot_inventories[bid]
            p = self._plan[bid]

            # Stall escape
            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF
            if pos in self.drop_set and inv:
                if p['role'] in ('trigger', 'pipe'):
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                elif p['role'] == 'stager':
                    # Stagers WAIT - their items are for chain auto-delivery
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # AT PICKUP TARGET
            if p['phase'] == 'fetch' and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup (active items only)
            if gt in ('pickup', 'deliver') and len(inv) < INV_CAP and active:
                opp = self._check_adjacent_active(bid, pos, active, inv)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # PIBT move
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _reassign_bot(self, bid, pos, active, preview, future):
        """Reassign an empty bot after completing its trip."""
        claimed = {p.get('item_idx') for p in self._plan.values()
                   if p.get('item_idx') is not None and p is not self._plan.get(bid)}

        # Try ACTIVE first (highest priority)
        active_needs: dict[int, int] = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        if active_needs:
            type_assigned: dict[int, int] = {}
            for pbid, p in self._plan.items():
                if pbid == bid:
                    continue
                if p.get('role') == 'trigger' and p.get('type_id') is not None:
                    tid = p['type_id']
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1

            item_idx, adj_pos = self._find_best_item(
                pos, active_needs, claimed, type_assigned)
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._nearest_dropoff(adj_pos)
                self._plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'trigger',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                    'age': 0,
                }
                return

        # Then preview (for chain staging)
        preview_needs: dict[int, int] = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        if preview_needs:
            type_assigned = {}
            for pbid, p in self._plan.items():
                if pbid == bid:
                    continue
                if p.get('role') == 'stager' and p.get('type_id') is not None:
                    tid = p['type_id']
                    type_assigned[tid] = type_assigned.get(tid, 0) + 1

            item_idx, adj_pos = self._find_best_item(
                pos, preview_needs, claimed, type_assigned)
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._staging_dropoff(pos, self._plan)
                self._plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'stager',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                    'age': 0,
                }
                return

        # Try future
        future_needs: dict[int, int] = {}
        if future:
            for order in future[:3]:
                for t in order.needs():
                    future_needs[t] = future_needs.get(t, 0) + 1

        if future_needs:
            claimed = {p.get('item_idx') for p in self._plan.values()
                       if p.get('item_idx') is not None}
            item_idx, adj_pos = self._find_best_item(
                pos, future_needs, claimed, {})
            if item_idx is not None:
                tid = int(self.ms.item_types[item_idx])
                dz = self._nearest_dropoff(adj_pos)
                self._plan[bid] = {
                    'goal': adj_pos,
                    'item_idx': item_idx,
                    'role': 'pipe',
                    'phase': 'fetch',
                    'dropoff': dz,
                    'type_id': tid,
                }
                return

        # Park
        self._plan[bid] = {
            'goal': self._park_spot(pos, self._plan),
            'role': 'idle',
            'phase': 'park',
        }

    def _check_adjacent_active(self, bid, pos, active, inv):
        """Check for adjacent active item pickup."""
        ms = self.ms
        active_needs: dict[int, int] = {}
        for t in active.needs():
            active_needs[t] = active_needs.get(t, 0) + 1
        bot_types = set(inv)

        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_needs or active_needs[tid] <= 0:
                continue
            if tid in bot_types and active_needs[tid] <= 1:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid, pos, rnd):
        dirs = list(MOVES)
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=200)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = AssemblyPipelineSolver(ms, tables, future_orders=all_orders)
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
                            inv_b = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv_b}")
                    drop_info = f" Drop=[{','.join(at)}]"
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}"
                         if active else " DONE")
                      + extra + drop_info)

        elapsed = time.time() - t0
        if verbose:
            dead = sum(1 for b in range(len(state.bot_positions))
                       if state.bot_inv_list(b))
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"DeadBots={dead} "
                  f"Time={elapsed:.1f}s "
                  f"({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Assembly Pipeline Solver')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores, scores_v4 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - Assembly Pipeline")
        print(f"{'='*50}")
        score, _ = AssemblyPipelineSolver.run_sim(seed, verbose=args.verbose)
        scores.append(score)

        if args.compare:
            from nightmare_lmapf_solver import LMAPFSolver
            print(f"\n--- V4 ---")
            s4, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
            scores_v4.append(s4)
            print(f"\nAssembly={score} vs V4={s4} (delta={score - s4:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nAssembly: mean={statistics.mean(scores):.1f} "
              f"max={max(scores)} min={min(scores)}")
        if scores_v4:
            print(f"V4: mean={statistics.mean(scores_v4):.1f}")
