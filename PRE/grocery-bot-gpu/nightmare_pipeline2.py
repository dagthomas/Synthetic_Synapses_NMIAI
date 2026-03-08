"""Nightmare pipeline solver v2: dedicated active/preview teams with cascading.

Design:
- Active team: fetch and deliver active order items ASAP
- Preview team: fetch ALL preview order items, stage at dropoffs
- When active completes → preview auto-delivers → cascade
- Uses V6's pathfinder for collision avoidance
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
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap


class PipelineSolver2:
    def __init__(self, ms, tables, future_orders=None):
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
        self.future_orders = future_orders or []
        self.stall_counts = {}
        self.prev_positions = {}
        self._pos_history = {}

        # Item lookup
        self.type_items = {}
        self.pos_to_items = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))
            for a in adj:
                if a not in self.pos_to_items:
                    self.pos_to_items[a] = []
                self.pos_to_items[a].append((idx, tid))

        # Corridor parking cells
        self._park_cells = []
        near_drop = set()
        for cell in tables.pos_to_idx:
            if any(tables.get_distance(cell, dz) <= 1 for dz in self.drop_zones):
                near_drop.add(cell)
        for cy in [1, ms.height // 2, ms.height - 3]:
            for cx in range(1, ms.width - 1):
                cell = (cx, cy)
                if cell in tables.pos_to_idx and cell not in near_drop and cell not in self.drop_set:
                    self._park_cells.append(cell)

    def _nearest_drop(self, pos):
        return min(self.drop_zones, key=lambda dz: self.tables.get_distance(pos, dz))

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

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

    def _find_item(self, tid, bot_pos, claimed):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for item_idx, adj_cells in self.type_items.get(tid, []):
            if item_idx in claimed:
                continue
            for adj in adj_cells:
                d = self.tables.get_distance(bot_pos, adj)
                dd = self._drop_dist(adj)
                cost = d + dd * 0.4
                if cost < best_cost:
                    best_cost = cost
                    best_idx = item_idx
                    best_adj = adj
        return best_idx, best_adj

    def _escape_action(self, bid, pos, rnd):
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        h = (bid * 7 + rnd * 13) % 4
        dirs = dirs[h:] + dirs[:h]
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def _park(self, pos, occupied):
        best = self.spawn
        best_d = 9999
        for cell in self._park_cells:
            if cell not in occupied:
                d = self.tables.get_distance(pos, cell)
                if 0 < d < best_d:
                    best_d = d
                    best = cell
        return best

    def action(self, state, all_orders, rnd):
        ms = self.ms
        num_bots = len(state.bot_positions)

        bot_positions = {}
        bot_inventories = {}
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

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        # Classify bots
        carrying_active = {}
        carrying_preview = {}
        active_carriers = []
        preview_carriers = []
        dead_bots = []
        empty_bots = []

        for bid in range(num_bots):
            inv = bot_inventories[bid]
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
            elif len(inv) < INV_CAP:
                empty_bots.append(bid)
            else:
                dead_bots.append(bid)

        # Shortfalls
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_active_short = sum(active_short.values())

        preview_short = {}
        for t, need in preview_needs.items():
            s = need - carrying_preview.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Allocate
        goals = {}
        goal_types = {}
        pickup_targets = {}
        claimed_items = set()
        dropoff_loads = {dz: 0 for dz in self.drop_zones}
        type_assigned_active = {}
        type_assigned_preview = {}
        occupied_goals = set()

        # 1. Active carriers → deliver
        for bid in active_carriers:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            # Check if we can detour for more active items
            if free_slots > 0 and total_active_short > 0:
                dz = self._nearest_drop(pos)
                drop_dist = self.tables.get_distance(pos, dz)
                best_detour = 9999
                best_item = None
                best_adj = None
                for tid in active_short:
                    if tid in set(inv) and active_short[tid] <= 1:
                        continue
                    for item_idx, adj_cells in self.type_items.get(tid, []):
                        if item_idx in claimed_items:
                            continue
                        for adj in adj_cells:
                            d_to = self.tables.get_distance(pos, adj)
                            d_back = self._drop_dist(adj)
                            detour = d_to + d_back - drop_dist
                            if detour < best_detour:
                                best_detour = detour
                                best_item = item_idx
                                best_adj = adj
                if best_detour < 6 and best_item is not None:
                    goals[bid] = best_adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = best_item
                    tid = int(ms.item_types[best_item])
                    type_assigned_active[tid] = type_assigned_active.get(tid, 0) + 1
                    claimed_items.add(best_item)
                    continue

            dz = self._balanced_dropoff(pos, dropoff_loads)
            dropoff_loads[dz] += 1
            goals[bid] = dz
            goal_types[bid] = 'deliver'

        # 2. Preview carriers → stage at dropoff (avoid delivery dropoffs)
        deliver_drops = set()
        for bid_d in goals:
            if goal_types.get(bid_d) == 'deliver' and goals[bid_d] in self.drop_set:
                deliver_drops.add(goals[bid_d])

        staging_counts = {dz: 0 for dz in self.drop_zones}
        for bid in preview_carriers:
            pos = bot_positions[bid]
            best_dz = None
            best_d = 9999
            for dz in self.drop_zones:
                if dz in deliver_drops:
                    continue
                if staging_counts[dz] >= 3:
                    continue
                d = self.tables.get_distance(pos, dz)
                if d < best_d:
                    best_d = d
                    best_dz = dz
            if best_dz is None:
                # All dropoffs have delivery or staging, try any
                for dz in self.drop_zones:
                    if staging_counts[dz] >= 3:
                        continue
                    d = self.tables.get_distance(pos, dz)
                    if d < best_d:
                        best_d = d
                        best_dz = dz
            if best_dz is not None and best_d < 25:
                staging_counts[best_dz] += 1
                goals[bid] = best_dz
                goal_types[bid] = 'stage'
            else:
                park = self._park(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'flee'

        # 3. Empty bots → active pickup, then preview pickup
        empty_sorted = sorted(empty_bots, key=lambda bid:
            min((self.tables.get_distance(bot_positions[bid], adj)
                 for tid in (active_short.keys() or preview_short.keys() or [0])
                 for _, adj_cells in self.type_items.get(tid, [])
                 for adj in adj_cells), default=9999))

        for bid in empty_sorted:
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            bot_types = set(inv)
            assigned = False

            # Active pickup
            remaining_active = sum(max(0, s - type_assigned_active.get(t, 0))
                                   for t, s in active_short.items())
            if remaining_active > 0:
                for tid in sorted(active_short.keys(),
                                   key=lambda t: active_short[t] - type_assigned_active.get(t, 0),
                                   reverse=True):
                    if type_assigned_active.get(tid, 0) >= active_short[tid] + 1:
                        continue
                    if tid in bot_types and active_short[tid] <= 1:
                        continue
                    item_idx, adj_pos = self._find_item(tid, pos, claimed_items)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'pickup'
                        pickup_targets[bid] = item_idx
                        type_assigned_active[tid] = type_assigned_active.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        assigned = True
                        break

            # Preview pickup (when active is covered)
            if not assigned and remaining_active <= 0 and preview_short:
                for tid in sorted(preview_short.keys(),
                                   key=lambda t: preview_short[t] - type_assigned_preview.get(t, 0),
                                   reverse=True):
                    if type_assigned_preview.get(tid, 0) >= preview_short[tid]:
                        continue
                    if tid in bot_types:
                        continue
                    item_idx, adj_pos = self._find_item(tid, pos, claimed_items)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        type_assigned_preview[tid] = type_assigned_preview.get(tid, 0) + 1
                        claimed_items.add(item_idx)
                        assigned = True
                        break

            if not assigned:
                park = self._park(pos, occupied_goals)
                occupied_goals.add(park)
                goals[bid] = park
                goal_types[bid] = 'park'

        # 4. Dead bots → park
        for bid in dead_bots:
            pos = bot_positions[bid]
            park = self._park(pos, occupied_goals)
            occupied_goals.add(park)
            goals[bid] = park
            goal_types[bid] = 'flee'

        # Urgency order
        priority_map = {'deliver': 0, 'pickup': 1, 'stage': 2,
                        'preview': 3, 'flee': 4, 'park': 5}
        urgency_order = sorted(range(num_bots), key=lambda bid: (
            priority_map.get(goal_types.get(bid, 'park'), 5),
            self.tables.get_distance(bot_positions[bid],
                                     goals.get(bid, self.spawn))))

        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions = [(ACT_WAIT, -1)] * num_bots
        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if len(bot_inventories[bid]) < INV_CAP:
                adj_items = self.pos_to_items.get(pos, [])
                for item_idx, tid in adj_items:
                    if tid in active_short and active_short[tid] > 0:
                        actions[bid] = (ACT_PICKUP, item_idx)
                        active_short[tid] -= 1
                        break
                    elif preview_order and preview_order.needs_type(tid):
                        if tid not in set(bot_inventories[bid]):
                            total_short_now = sum(active_short.values())
                            if total_short_now == 0:
                                actions[bid] = (ACT_PICKUP, item_idx)
                                break
                else:
                    act = path_actions.get(bid, ACT_WAIT)
                    actions[bid] = (act, -1)
                    continue
                continue

            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    @staticmethod
    def run_sim(seed, verbose=True):
        num_rounds = DIFF_ROUNDS['nightmare']
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = PipelineSolver2(ms, tables, future_orders=all_orders)

        t_start = time.time()
        chains = 0
        max_chain = 0
        action_log = []

        for rnd in range(num_rounds):
            state.round = rnd
            old_orders = state.orders_completed
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))
            step(state, actions, all_orders)

            d = state.orders_completed - old_orders
            if d > 1:
                chains += 1
                max_chain = max(max_chain, d)

            if verbose and (rnd < 5 or rnd % 50 == 0 or rnd == num_rounds - 1):
                active = state.get_active_order()
                need = len(active.needs()) if active else 0
                print(f'R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d} Need={need}')

        elapsed = time.time() - t_start
        if verbose:
            print(f'\nFinal: Score={state.score} Ord={state.orders_completed} '
                  f'Items={state.items_delivered} '
                  f'Chains={chains} MaxChain={max_chain} Time={elapsed:.1f}s')
        return state.score, action_log


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7009
    score, actions = PipelineSolver2.run_sim(seed)
    print(f'Score: {score}')
