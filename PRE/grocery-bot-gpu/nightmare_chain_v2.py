#!/usr/bin/env python3
"""Chain-V2: Pipeline solver that coordinates synchronized delivery + chain.

Key insight from testing: chains work perfectly when 3 bots at 3 dropoffs
have the right items. The solver must coordinate:
1. Active workers: fetch and deliver active items (but HOLD last delivery)
2. Preview workers: fetch preview items and stage near dropoffs
3. Synchronized delivery: when preview workers are staged, active worker
   delivers the last item → chain fires → preview items auto-deliver

The trick: delivery bots near dropoff WAIT until enough preview items are
staged, then all rush to dropoffs and deliver simultaneously.
"""
from __future__ import annotations

import sys
import time
import random as _random

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import NightmarePathfinder, build_walkable
from nightmare_traffic import TrafficRules, CongestionMap
from nightmare_task_alloc import NightmareTaskAlloc

sys.stdout.reconfigure(encoding='utf-8')

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
NUM_ROUNDS = DIFF_ROUNDS['nightmare']
NUM_BOTS = CONFIGS['nightmare']['bots']


class ChainV2Solver:
    """Pipeline solver with synchronized chain delivery."""

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None,
                 solver_seed: int = 0):
        self.rng = _random.Random(solver_seed) if solver_seed else _random.Random()
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.future_orders = future_orders or []

        # Pathfinding
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)
        self.allocator = NightmareTaskAlloc(
            ms, self.tables, self.drop_zones, rng=self.rng)

        # Item lookup
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
        self._trips: dict[int, dict] = {}
        self._last_active_id = -1

        # Chain state
        self._preview_staged: dict[int, set[int]] = {}  # drop_zone_idx -> set of bot_ids
        self._chain_ready = False
        self._hold_count = 0

    def _find_best_item(self, pos, needed, claimed, prefer_drop=None):
        """Find nearest item of needed type."""
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
                    if prefer_drop:
                        drop_d = self.tables.get_distance(adj, prefer_drop)
                    else:
                        drop_d = min(self.tables.get_distance(adj, dz)
                                     for dz in self.drop_zones)
                    cost = d + drop_d * 0.4
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)

        # Extract state
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Stall tracking
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

        # Future orders
        future = []
        if preview_order:
            future.append(preview_order)
        if all_orders:
            for i in range(state.next_order_idx,
                           min(state.next_order_idx + 8, len(all_orders))):
                future.append(all_orders[i])

        # Reset on active change
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            self._trips.clear()
            self._last_active_id = active_id
            self._hold_count = 0
            self._chain_ready = False

        # Age trips
        for bid in list(self._trips.keys()):
            self._trips[bid]['age'] += 1
            if self._trips[bid]['age'] > 25:
                del self._trips[bid]

        # Active analysis
        active_needs: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        active_types = set(active_needs.keys())
        carrying_active: dict[int, int] = {}
        for inv in bot_inventories.values():
            for t in inv:
                if t in active_needs:
                    carrying_active[t] = carrying_active.get(t, 0) + 1

        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_active_remaining = sum(active_needs.values())

        # Preview analysis
        preview_needs: dict[int, int] = {}
        preview_only_needs: dict[int, int] = {}  # Types in preview but NOT in active
        if preview_order:
            for t in preview_order.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1
            for t, n in preview_needs.items():
                if t not in active_types:
                    preview_only_needs[t] = n

        # === Use V3 allocator as base ===
        from nightmare_chain_planner import ChainPlan
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, NUM_ROUNDS,
            future_orders=future, chain_plan=None,
            allow_preview_pickup=True)

        # Apply persistent trips
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            pos = bot_positions[bid]
            if pos == trip['goal'] or len(bot_inventories[bid]) > trip.get('inv_count', 0):
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
        for bid in range(num_bots):
            gt = goal_types.get(bid)
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if bid not in self._trips:
                    item_idx = pickup_targets[bid]
                    tid = int(self.ms.item_types[item_idx]) if item_idx >= 0 else -1
                    self._trips[bid] = {
                        'goal': goals[bid], 'goal_type': gt,
                        'item_idx': item_idx, 'type_id': tid,
                        'inv_count': len(bot_inventories[bid]), 'age': 0,
                    }

        # === POST-PROCESS: Recycle dead/idle ===
        claimed_items = set(pickup_targets.values())
        for bid in range(num_bots):
            gt = goal_types.get(bid, 'park')
            if gt not in ('flee', 'park'):
                continue
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            if INV_CAP - len(inv) <= 0:
                continue

            if active_short:
                idx, adj = self._find_best_item(pos, active_short, claimed_items)
                if idx is not None:
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = idx
                    claimed_items.add(idx)
                    self._trips[bid] = {
                        'goal': adj, 'goal_type': 'pickup',
                        'item_idx': idx, 'type_id': int(self.ms.item_types[idx]),
                        'inv_count': len(inv), 'age': 0,
                    }
                    continue

            if preview_order:
                pn = dict(preview_needs)
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
                            'item_idx': idx, 'type_id': int(self.ms.item_types[idx]),
                            'inv_count': len(inv), 'age': 0,
                        }

        # === CHAIN COORDINATION ===
        # Count how many preview items are at/near dropoffs
        preview_at_drops = 0
        preview_total = sum(preview_needs.values())
        if preview_order and total_active_remaining <= 3:
            # Active order almost done — count preview readiness
            pn_remaining = dict(preview_needs)
            for bid in range(num_bots):
                pos = bot_positions[bid]
                inv = bot_inventories[bid]
                d_drop = min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)
                if d_drop <= 2:  # at or very near dropoff
                    for t in inv:
                        if t in pn_remaining and pn_remaining[t] > 0:
                            pn_remaining[t] -= 1
                            preview_at_drops += 1

        # Urgency order
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            noise = self.rng.random() * 0.5
            if gt == 'deliver':
                return (0, dist + noise)
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 4, dist + noise)
            elif gt == 'pickup':
                return (2, dist + noise)
            elif gt in ('stage', 'preview'):
                return (3, dist + noise)
            else:
                return (5, dist + noise)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # PIBT pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order, goal_types=goal_types)

        # Build actions
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)
            inv = bot_inventories[bid]

            # Stall escape
            stall_count = self.stall_counts.get(bid, 0)
            if stall_count >= 3:
                if gt == 'deliver' and goal in self.drop_set:
                    drop_dist = self.tables.get_distance(pos, goal)
                    if drop_dist <= 4 and stall_count < 8:
                        pass
                    else:
                        actions[bid] = (self._escape_action(bid, pos), -1)
                        continue
                else:
                    actions[bid] = (self._escape_action(bid, pos), -1)
                    continue

            # AT DROPOFF: deliver
            if pos in self.drop_set:
                if gt == 'deliver' and inv:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if gt in ('pickup', 'preview', 'deliver') and len(inv) < INV_CAP:
                opp = self._check_adjacent(bid, pos, active_order, active_short)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent(self, bid, pos, active_order, active_short):
        ms = self.ms
        if not active_short:
            return None
        for item_idx in range(ms.num_items):
            tid = int(ms.item_types[item_idx])
            if tid not in active_short or active_short[tid] <= 0:
                continue
            for adj in ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    def _escape_action(self, bid, pos):
        dirs = list(MOVES)
        self.rng.shuffle(dirs)
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    @staticmethod
    def run_sim(seed, solver_seed=0, verbose=True):
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = ChainV2Solver(ms, tables, future_orders=all_orders,
                                solver_seed=solver_seed)
        action_log = []
        chains = 0

        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))

            completed_before = state.orders_completed
            step(state, actions, all_orders)
            if state.orders_completed > completed_before + 1:
                chains += state.orders_completed - completed_before - 1

        if verbose:
            print(f"  seed={seed} score={state.score} orders={state.orders_completed} "
                  f"chains={chains}")
        return state.score, action_log, chains


def main():
    seeds = [7005, 11, 42, 45, 100, 200, 300, 500]
    scores = []
    total_chains = 0

    print("Chain V2 solver (8 seeds):")
    print("-" * 60)
    t0 = time.time()

    for seed in seeds:
        score, _, chains = ChainV2Solver.run_sim(seed)
        scores.append(score)
        total_chains += chains

    mean = sum(scores) / len(scores)
    print("-" * 60)
    print(f"Mean: {mean:.1f}  Chains: {total_chains}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
