#!/usr/bin/env python3
"""Chain-optimized LMAPF solver for nightmare mode.

Key insight: chains fire when ACT_DROPOFF completes the active order AND
other bots at dropoff tiles have items matching the new active (old preview).
Auto-delivery then delivers those items. If the new active also completes,
another chain fires.

Strategy:
- 3 "chain stager" bots (one per dropoff zone) fetch preview-only items
  and park AT dropoff tiles
- 17 "worker" bots fetch and deliver active items normally
- When a worker's delivery completes the active order, chain stagers'
  preview items auto-deliver to the new active order
- Chain depth 1 doubles order completion rate

Bot ordering matters: stagers should have lower IDs than workers so they
move to dropoffs BEFORE workers deliver in the same round.
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


class ChainLMAPFSolver:
    """Chain-optimized LMAPF with dedicated stager bots."""

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None,
                 solver_seed: int = 0,
                 num_stagers: int = 3):
        self.rng = _random.Random(solver_seed) if solver_seed else _random.Random()
        self.ms = ms
        self.tables = tables or PrecomputedTables.get(ms)
        self.walkable = build_walkable(ms)
        self.drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = ms.spawn
        self.num_bots = NUM_BOTS
        self.future_orders = future_orders or []
        self.num_stagers = min(num_stagers, len(self.drop_zones))

        # Stager assignments: bot_id -> dropoff zone
        # Use lowest bot IDs as stagers (processed first in step())
        self.stager_bots: dict[int, tuple[int, int]] = {}
        sorted_drops = sorted(self.drop_zones, key=lambda d: d[0])
        for i in range(self.num_stagers):
            self.stager_bots[i] = sorted_drops[i]
        self.worker_bots = [b for b in range(self.num_bots) if b not in self.stager_bots]

        # Pathfinding
        self.traffic = TrafficRules(ms)
        self.congestion = CongestionMap()
        self.pathfinder = NightmarePathfinder(
            ms, self.tables, self.traffic, self.congestion)

        # Worker allocator (uses 17 worker bots)
        self.allocator = NightmareTaskAlloc(
            ms, self.tables, self.drop_zones, rng=self.rng)

        # Item type -> [(item_idx, [adj_positions])]
        self.type_items: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
        for idx in range(ms.num_items):
            tid = int(ms.item_types[idx])
            adj = ms.item_adjacencies.get(idx, [])
            if tid not in self.type_items:
                self.type_items[tid] = []
            self.type_items[tid].append((idx, adj))

        # State tracking
        self.stall_counts: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self._trips: dict[int, dict] = {}
        self._last_active_id = -1
        self._stager_state: dict[int, str] = {}  # bid -> 'fetch' | 'stage' | 'wait'
        self._stager_targets: dict[int, int] = {}  # bid -> item_idx being fetched
        self._stager_target_types: dict[int, int] = {}  # bid -> type_id being fetched

    def _get_preview_only_needs(self, active_order, preview_order):
        """Get types needed by preview but NOT by active (safe for chain)."""
        if not preview_order:
            return {}
        active_types = set()
        if active_order:
            for t in active_order.required:
                active_types.add(int(t))

        needs: dict[int, int] = {}
        for t in preview_order.needs():
            if t not in active_types:
                needs[t] = needs.get(t, 0) + 1
        return needs

    def _get_all_preview_needs(self, preview_order):
        """Get all types needed by preview order."""
        if not preview_order:
            return {}
        needs: dict[int, int] = {}
        for t in preview_order.needs():
            needs[t] = needs.get(t, 0) + 1
        return needs

    def _find_best_item(self, pos, needed, claimed, zone_drop=None):
        """Find nearest item of a needed type."""
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
                    # Prefer items near the stager's dropoff
                    if zone_drop:
                        drop_d = self.tables.get_distance(adj, zone_drop)
                    else:
                        drop_d = min(self.tables.get_distance(adj, dz)
                                     for dz in self.drop_zones)
                    cost = d + drop_d * 0.5
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = NUM_ROUNDS

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

        # Future order lookahead
        future = []
        if preview_order:
            future.append(preview_order)
        if all_orders:
            for i in range(state.next_order_idx,
                           min(state.next_order_idx + 8, len(all_orders))):
                future.append(all_orders[i])

        # Reset trips on active order change
        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            # Keep stager trips, clear worker trips
            worker_trips = {b: t for b, t in self._trips.items()
                           if b not in self.stager_bots}
            for bid in worker_trips:
                del self._trips[bid]
            self._last_active_id = active_id

        # Age out stale trips
        for bid in list(self._trips.keys()):
            self._trips[bid]['age'] += 1
            if self._trips[bid]['age'] > 20:
                del self._trips[bid]

        # Active shortfall
        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for bid in self.worker_bots:
                for t in bot_inventories.get(bid, []):
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s

        # --- STAGER GOALS ---
        stager_goals: dict[int, tuple[int, int]] = {}
        stager_goal_types: dict[int, str] = {}
        stager_pickup_targets: dict[int, int] = {}
        claimed_items: set[int] = set()

        preview_needs = self._get_all_preview_needs(preview_order)
        # Track what stagers are already carrying for preview
        stager_carrying: dict[int, int] = {}
        for bid in self.stager_bots:
            for t in bot_inventories.get(bid, []):
                if t in preview_needs:
                    stager_carrying[t] = stager_carrying.get(t, 0) + 1

        # Remaining preview needs after accounting for what stagers carry
        preview_remaining: dict[int, int] = dict(preview_needs)
        for t, count in stager_carrying.items():
            if t in preview_remaining:
                preview_remaining[t] = max(0, preview_remaining[t] - count)
                if preview_remaining[t] == 0:
                    del preview_remaining[t]

        for bid, drop_zone in self.stager_bots.items():
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free_slots = INV_CAP - len(inv)

            # Stager at its dropoff with items → WAIT (ready for chain)
            if pos == drop_zone and inv:
                has_preview = any(t in preview_needs for t in inv)
                if has_preview:
                    stager_goals[bid] = drop_zone
                    stager_goal_types[bid] = 'stage'
                    continue

            # Stager has full inventory → go to dropoff
            if free_slots == 0:
                stager_goals[bid] = drop_zone
                stager_goal_types[bid] = 'stage'
                continue

            # Stager has preview items but not full → check if should go to dropoff
            has_useful = any(t in preview_needs for t in inv)
            if has_useful and free_slots <= 1:
                # Almost full with useful items, go to dropoff
                stager_goals[bid] = drop_zone
                stager_goal_types[bid] = 'stage'
                continue

            # Fetch preview items (prefer preview-only types to avoid active consumption)
            if preview_remaining and free_slots > 0:
                item_idx, adj_pos = self._find_best_item(
                    pos, preview_remaining, claimed_items, zone_drop=drop_zone)
                if item_idx is not None:
                    stager_goals[bid] = adj_pos
                    stager_goal_types[bid] = 'preview'
                    stager_pickup_targets[bid] = item_idx
                    claimed_items.add(item_idx)
                    tid = int(self.ms.item_types[item_idx])
                    if tid in preview_remaining:
                        preview_remaining[tid] = max(0, preview_remaining[tid] - 1)
                        if preview_remaining[tid] == 0:
                            del preview_remaining[tid]
                    continue

            # Nothing useful to fetch → park at dropoff anyway
            if inv:
                stager_goals[bid] = drop_zone
                stager_goal_types[bid] = 'stage'
            else:
                # Empty stager, no preview items available → park near dropoff
                stager_goals[bid] = drop_zone
                stager_goal_types[bid] = 'stage'

        # --- WORKER GOALS (using V3 allocator) ---
        # Create worker-only position/inventory dicts
        worker_positions = {bid: bot_positions[bid] for bid in self.worker_bots
                           if bid in bot_positions}
        worker_inventories = {bid: bot_inventories[bid] for bid in self.worker_bots
                             if bid in bot_inventories}

        from nightmare_chain_planner import ChainPlan
        goals, goal_types, pickup_targets = self.allocator.allocate(
            worker_positions, worker_inventories,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future, chain_plan=None,
            allow_preview_pickup=True)

        # Merge stager and worker goals
        for bid in self.stager_bots:
            goals[bid] = stager_goals.get(bid, self.stager_bots[bid])
            goal_types[bid] = stager_goal_types.get(bid, 'stage')
            if bid in stager_pickup_targets:
                pickup_targets[bid] = stager_pickup_targets[bid]

        # Apply persistent trips for workers
        for bid in list(self._trips.keys()):
            if bid in self.stager_bots:
                continue
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
                        'goal': goals[bid],
                        'goal_type': gt,
                        'item_idx': item_idx,
                        'type_id': tid,
                        'inv_count': len(bot_inventories[bid]),
                        'age': 0,
                    }

        # POST-PROCESS: Recycle dead/idle workers with free slots
        claimed_items = set(pickup_targets.values())
        for bid in self.worker_bots:
            gt = goal_types.get(bid, 'park')
            if gt not in ('flee', 'park'):
                continue
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free = INV_CAP - len(inv)
            if free <= 0:
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
                            'item_idx': idx, 'type_id': int(self.ms.item_types[idx]),
                            'inv_count': len(inv), 'age': 0,
                        }

        # Urgency order: stagers HIGH priority (need to reach dropoff first)
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            noise = self.rng.random() * 0.5

            if bid in self.stager_bots:
                if gt == 'stage':
                    return (-1, dist + noise)  # Highest priority
                else:
                    return (1.5, dist + noise)  # Fetching: medium priority

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
            bot_positions, goals, urgency_order,
            goal_types=goal_types)

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
                if bid in self.stager_bots and pos == self.stager_bots[bid]:
                    # Stager at dropoff: just wait, don't escape
                    actions[bid] = (ACT_WAIT, -1)
                    continue
                elif gt == 'deliver' and goal in self.drop_set:
                    drop_dist = self.tables.get_distance(pos, goal)
                    if drop_dist <= 4 and stall_count < 8:
                        pass  # Let PIBT handle
                    else:
                        actions[bid] = (self._escape_action(bid, pos), -1)
                        continue
                else:
                    actions[bid] = (self._escape_action(bid, pos), -1)
                    continue

            # AT DROPOFF
            if pos in self.drop_set:
                if bid in self.stager_bots:
                    if pos == self.stager_bots[bid]:
                        # Stager at its dropoff: WAIT (items stay for chain)
                        actions[bid] = (ACT_WAIT, -1)
                        continue
                    # Stager at wrong dropoff: move toward correct one
                elif gt == 'deliver' and inv:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup (active items only for workers)
            if bid not in self.stager_bots:
                if gt in ('pickup', 'preview', 'deliver') and len(inv) < INV_CAP:
                    pickup_act = self._check_adjacent_active(
                        bid, pos, active_order, active_short)
                    if pickup_act is not None:
                        actions[bid] = pickup_act
                        continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    def _check_adjacent_active(self, bid, pos, active_order, active_short):
        """Adjacent pickup for active items only."""
        if not active_short:
            return None
        ms = self.ms
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
    def run_sim(seed, solver_seed=0, verbose=True, num_stagers=3):
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = ChainLMAPFSolver(ms, tables, future_orders=all_orders,
                                   solver_seed=solver_seed,
                                   num_stagers=num_stagers)
        action_log = []
        chains = 0

        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))

            completed_before = state.orders_completed
            step(state, actions, all_orders)
            completed_after = state.orders_completed
            if completed_after > completed_before + 1:
                chains += completed_after - completed_before - 1

        if verbose:
            print(f"  seed={seed} score={state.score} orders={state.orders_completed} "
                  f"chains={chains} stagers={num_stagers}")
        return state.score, action_log, chains


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='7005,11,42,45,100,200,300,500')
    parser.add_argument('--stagers', type=int, default=3)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    scores = []
    total_chains = 0

    print(f"Chain LMAPF (stagers={args.stagers}):")
    print("-" * 60)
    t0 = time.time()

    for seed in seeds:
        score, _, chains = ChainLMAPFSolver.run_sim(
            seed, num_stagers=args.stagers)
        scores.append(score)
        total_chains += chains

    mean = sum(scores) / len(scores)
    print("-" * 60)
    print(f"Mean: {mean:.1f}  Total chains: {total_chains}  ({time.time()-t0:.1f}s)")
    print(f"Scores: {scores}")


if __name__ == '__main__':
    main()
