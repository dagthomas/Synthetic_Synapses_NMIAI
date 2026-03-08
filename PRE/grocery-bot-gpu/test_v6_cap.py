#!/usr/bin/env python3
"""Test V6 with capped active empty-bot assignments."""
import time
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
from game_engine import init_game, step, INV_CAP
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

seeds = [1000, 1003, 1006, 1009, 7009]

# Save original allocate
original_allocate = V6Allocator.allocate

def make_patched(max_extras):
    orig = original_allocate
    def patched(self, bot_positions, bot_inventories,
                active_order, preview_order, round_num, num_rounds=500,
                future_orders=None):
        # Run original to get baseline allocation
        goals, goal_types, pickup_targets = orig(
            self, bot_positions, bot_inventories,
            active_order, preview_order, round_num, num_rounds,
            future_orders=future_orders)

        # Count active pickups assigned to empty bots
        active_needs = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
        carrying = {}
        for bid, inv in bot_inventories.items():
            for t in inv:
                if t in active_needs:
                    carrying[t] = carrying.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - carrying.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())
        max_active = total_short + max_extras

        # Find empty bots assigned to active pickup beyond cap
        active_pickup_bots = [bid for bid, gt in goal_types.items()
                              if gt == 'pickup' and not bot_inventories.get(bid, [])]
        if len(active_pickup_bots) > max_active:
            # Convert excess active pickups to preview
            preview_needs = {}
            if preview_order:
                for t in preview_order.needs():
                    preview_needs[t] = preview_needs.get(t, 0) + 1
            preview_type_assigned = {}
            claimed = set(pickup_targets.values())

            for bid in active_pickup_bots[max_active:]:
                pos = bot_positions[bid]
                bz = self.bot_zone.get(bid, 2)
                if preview_needs:
                    item_idx, adj_pos = self._assign_item(
                        pos, preview_needs, preview_type_assigned,
                        claimed, strict=True)
                    if item_idx is not None:
                        goals[bid] = adj_pos
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = item_idx
                        tid = int(self.ms.item_types[item_idx])
                        preview_type_assigned[tid] = preview_type_assigned.get(tid, 0) + 1
                        claimed.add(item_idx)
                        continue
                # Fallback: park
                occupied = set(goals.values())
                park = self._corridor_parking(pos, occupied, zone=bz)
                goals[bid] = park
                goal_types[bid] = 'park'
                if bid in pickup_targets:
                    del pickup_targets[bid]

        return goals, goal_types, pickup_targets
    return patched


for extras in [0, 1, 2, 3, 5, 99]:
    V6Allocator.allocate = make_patched(extras)
    scores = []
    for seed in seeds:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
        for rnd in range(500):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            step(state, actions, all_orders)
        scores.append(state.score)
    print(f'extras={extras}: mean={sum(scores)/len(scores):.1f} scores={scores}')

V6Allocator.allocate = original_allocate
