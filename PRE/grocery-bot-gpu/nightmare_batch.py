#!/usr/bin/env python3
"""Nightmare Batch Solver: Pre-stage items for N orders, trigger chain cascade.

Key insight: 20 bots × 3 inv = 60 capacity. Orders need ~5.5 items.
We can carry items for 10 orders at once. One delivery triggers chain
cascade through all 10 orders.

Phase 1 (LOAD): Bots pick up items for orders 1-K
Phase 2 (STAGE): All bots at dropoffs with items
Phase 3 (TRIGGER): Deliver last order-1 item → chain cascade

Usage: python nightmare_batch.py --seed 7009
"""
from __future__ import annotations
import sys, time, heapq, random
from collections import defaultdict

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


class BatchSolver:
    """Batch solver: pick up items for N orders, chain-deliver."""

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 all_orders: list[Order]):
        self.ms = ms
        self.tables = tables
        self.all_orders = all_orders
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = tuple(ms.spawn)
        self.walkable = build_walkable(ms)
        self.num_bots = CONFIGS['nightmare']['bots']

        # Item info indexed by type
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # Pathfinder
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(ms, tables, self.traffic, self.congestion)

        # State
        self.phase = 'INIT'  # INIT, LOAD, STAGE, TRIGGER, DELIVER
        self.bot_goals: dict[int, tuple[int, int]] = {}
        self.bot_goal_types: dict[int, str] = {}
        self.bot_pickup_targets: dict[int, int] = {}
        self.bot_item_types: dict[int, list[int]] = {i: [] for i in range(self.num_bots)}
        self.batch_orders: list[int] = []  # Indices into all_orders
        self.batch_needs: dict[int, int] = {}  # type_id → count needed
        self.batch_assigned: dict[int, int] = {}  # type_id → count assigned
        self.claimed_items: set[int] = set()
        self.stall: dict[int, int] = {i: 0 for i in range(self.num_bots)}
        self.prev_pos: dict[int, tuple[int, int]] = {}

    def _nearest_drop(self, pos):
        return min(self.drop_zones, key=lambda dz: self.tables.get_distance(pos, dz))

    def _drop_dist(self, pos):
        return min(self.tables.get_distance(pos, dz) for dz in self.drop_zones)

    def _plan_batch(self, state: GameState):
        """Plan a batch: determine which orders to pre-stage and assign items to bots."""
        order_idx = state.next_order_idx - 2  # Current active order index
        if order_idx < 0:
            order_idx = 0

        # Active order is already partially done — handle it separately
        active = state.get_active_order()
        preview = state.get_preview_order()

        # Compute how many items we can carry
        total_free = 0
        for bid in range(self.num_bots):
            inv_count = state.bot_inv_count(bid)
            # Count items that DON'T match active/preview as wasted
            total_free += INV_CAP - inv_count

        # How many orders can we batch? Start with preview, then future
        self.batch_needs.clear()
        self.batch_assigned.clear()
        self.batch_orders.clear()
        self.claimed_items.clear()

        # Active order items still needed
        active_needs = {}
        if active:
            for t in active.needs():
                t = int(t)
                active_needs[t] = active_needs.get(t, 0) + 1
            # Subtract what's already being carried
            for bid in range(self.num_bots):
                for t in state.bot_inv_list(bid):
                    if t in active_needs and active_needs[t] > 0:
                        active_needs[t] -= 1
                        if active_needs[t] == 0:
                            del active_needs[t]

        # Start batch with active needs
        items_in_batch = sum(active_needs.values())
        for t, c in active_needs.items():
            self.batch_needs[t] = self.batch_needs.get(t, 0) + c

        # Add preview + future orders to batch
        future_start = state.next_order_idx
        max_batch = min(8, total_free // 4)  # Don't try to batch too many

        for i in range(max_batch):
            idx = future_start + i
            if idx >= len(self.all_orders):
                break
            order = self.all_orders[idx]
            order_items = len(order.required)
            if items_in_batch + order_items > total_free:
                break
            self.batch_orders.append(idx)
            for t in order.required:
                t = int(t)
                self.batch_needs[t] = self.batch_needs.get(t, 0) + 1
            items_in_batch += order_items

        return items_in_batch

    def _assign_pickups(self, state: GameState):
        """Assign item pickups to bots for the current batch."""
        bot_positions = {}
        bot_inventories = {}
        for bid in range(self.num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.bot_goals.clear()
        self.bot_goal_types.clear()
        self.bot_pickup_targets.clear()

        # Which types do we still need to pick up?
        remaining_needs = dict(self.batch_needs)

        # Subtract items already in bot inventories
        for bid in range(self.num_bots):
            for t in bot_inventories[bid]:
                if t in remaining_needs and remaining_needs[t] > 0:
                    remaining_needs[t] -= 1
                    if remaining_needs[t] == 0:
                        del remaining_needs[t]

        # Active order: bots with active items → deliver FIRST
        active = state.get_active_order()
        active_types = set()
        if active:
            for t in active.needs():
                active_types.add(int(t))

        # Bots with active items → deliver
        for bid in range(self.num_bots):
            inv = bot_inventories[bid]
            if any(t in active_types for t in inv):
                dz = self._nearest_drop(bot_positions[bid])
                self.bot_goals[bid] = dz
                self.bot_goal_types[bid] = 'deliver'

        # Remaining bots → pick up batch items
        pickup_bots = []
        for bid in range(self.num_bots):
            if bid in self.bot_goals:
                continue
            if state.bot_inv_count(bid) >= INV_CAP:
                # Full inventory with non-active items → stage at dropoff
                inv = bot_inventories[bid]
                any_useful = any(t in self.batch_needs for t in inv)
                if any_useful:
                    dz = self._nearest_drop(bot_positions[bid])
                    self.bot_goals[bid] = dz
                    self.bot_goal_types[bid] = 'stage'
                continue
            pickup_bots.append(bid)

        # Sort pickup bots by proximity to items
        pickup_bots.sort(key=lambda bid: min(
            (self.tables.get_distance(bot_positions[bid], adj)
             for tid in remaining_needs
             for _, adjs in self.type_items.get(tid, [])
             for adj in adjs),
            default=9999))

        # Assign items to bots greedily
        for bid in pickup_bots:
            pos = bot_positions[bid]
            free_slots = INV_CAP - state.bot_inv_count(bid)

            if not remaining_needs or free_slots <= 0:
                # Stage at dropoff
                dz = self._nearest_drop(pos)
                self.bot_goals[bid] = dz
                self.bot_goal_types[bid] = 'stage'
                continue

            # Find best item to pick up
            best_item = None
            best_adj = None
            best_cost = 9999
            best_tid = -1

            for tid, count in remaining_needs.items():
                if count <= 0:
                    continue
                for item_idx, adj_cells in self.type_items.get(tid, []):
                    if item_idx in self.claimed_items:
                        continue
                    for adj in adj_cells:
                        d = self.tables.get_distance(pos, adj)
                        drop_d = self._drop_dist(adj)
                        cost = d + drop_d * 0.3
                        if cost < best_cost:
                            best_cost = cost
                            best_item = item_idx
                            best_adj = adj
                            best_tid = tid

            if best_item is not None:
                self.bot_goals[bid] = best_adj
                self.bot_goal_types[bid] = 'pickup'
                self.bot_pickup_targets[bid] = best_item
                self.claimed_items.add(best_item)
                remaining_needs[best_tid] -= 1
                if remaining_needs[best_tid] <= 0:
                    del remaining_needs[best_tid]
                self.batch_assigned[best_tid] = self.batch_assigned.get(best_tid, 0) + 1
            else:
                dz = self._nearest_drop(pos)
                self.bot_goals[bid] = dz
                self.bot_goal_types[bid] = 'stage'

    def action(self, state: GameState, all_orders: list[Order], rnd: int):
        """Generate actions for round rnd."""
        num_bots = self.num_bots
        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Update stall tracking
        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_pos.get(bid) == pos:
                self.stall[bid] = self.stall.get(bid, 0) + 1
            else:
                self.stall[bid] = 0
            self.prev_pos[bid] = pos

        # Re-plan every few rounds or when needed
        if rnd % 5 == 0 or not self.bot_goals:
            self._plan_batch(state)
            self._assign_pickups(state)

        # Unassigned bots get goals
        for bid in range(num_bots):
            if bid not in self.bot_goals:
                dz = self._nearest_drop(bot_positions[bid])
                self.bot_goals[bid] = dz
                self.bot_goal_types[bid] = 'stage'

        # Urgency ordering
        def _urgency(bid):
            gt = self.bot_goal_types.get(bid, 'stage')
            d = self.tables.get_distance(bot_positions[bid],
                                          self.bot_goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, d)
            elif gt == 'pickup':
                return (1, d)
            elif gt == 'stage':
                return (2, d)
            return (3, d)

        urgency_order = sorted(range(num_bots), key=_urgency)

        # Pathfinding
        path_actions = self.pathfinder.plan_all(
            bot_positions, self.bot_goals, urgency_order,
            goal_types=self.bot_goal_types)

        actions = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = self.bot_goal_types.get(bid, 'stage')
            goal = self.bot_goals.get(bid, self.spawn)
            inv = bot_inventories[bid]

            # Stall escape
            if self.stall.get(bid, 0) >= 3:
                dirs = list(MOVES)
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = pos[0] + DX[a], pos[1] + DY[a]
                    if (nx, ny) in self.walkable:
                        actions[bid] = (a, -1)
                        break
                continue

            # At dropoff
            if pos in self.drop_set:
                if gt == 'deliver' and inv:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue
                elif gt == 'stage':
                    # Wait at dropoff for chain
                    actions[bid] = (ACT_WAIT, -1)
                    continue

            # At pickup target
            if gt == 'pickup' and bid in self.bot_pickup_targets:
                item_idx = self.bot_pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    # After pickup, re-assign next round
                    del self.bot_goals[bid]
                    del self.bot_goal_types[bid]
                    del self.bot_pickup_targets[bid]
                    continue

            # Opportunistic pickup of needed items
            if len(inv) < INV_CAP and self.batch_needs:
                opp = self._opp_pickup(state, bid, pos)
                if opp is not None:
                    actions[bid] = opp
                    continue

            # Follow pathfinder
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _opp_pickup(self, state, bid, pos):
        """Opportunistic pickup of adjacent batch-needed items."""
        active = state.get_active_order()
        active_types = set()
        if active:
            for t in active.needs():
                active_types.add(int(t))

        for item_idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[item_idx])
            if tid not in active_types and tid not in self.batch_needs:
                continue
            for adj in self.ms.item_adjacencies.get(item_idx, []):
                if adj == pos:
                    return (ACT_PICKUP, item_idx)
        return None

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = BatchSolver(ms, tables, all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']

        action_log = []
        chains = 0
        max_chain = 0
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if '-' in args.seeds:
        lo, hi = args.seeds.split('-')
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]

    scores = []
    for seed in seeds:
        score, _ = BatchSolver.run_sim(seed, verbose=args.verbose)
        scores.append(score)
        print(f'Seed {seed}: {score}')

    if len(scores) > 1:
        print(f'Mean: {sum(scores)/len(scores):.1f}  '
              f'Min: {min(scores)}  Max: {max(scores)}')


if __name__ == '__main__':
    main()
