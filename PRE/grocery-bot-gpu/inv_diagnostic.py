#!/usr/bin/env python3
"""Diagnose inventory utilization: active vs preview vs dead items."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import (
    init_game, step, ACT_WAIT, ACT_DROPOFF, ACT_PICKUP,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT, INV_CAP,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver

NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def run_inv_diagnostic(seed, solver_seed=1):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)

    solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=0.6)

    # Per-round aggregate stats
    total_slots = 0
    active_slots = 0
    preview_slots = 0
    dead_slots = 0
    empty_slots = 0

    # Track order transitions
    order_changes = 0
    last_active_id = -1
    items_dead_at_transition = 0
    items_preview_at_transition = 0

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        active_types = set()
        preview_types = set()
        if active:
            for t in active.needs():
                active_types.add(int(t))
        if preview:
            for t in preview.needs():
                preview_types.add(int(t))

        # Check order transition
        active_id = active.id if active else -1
        if active_id != last_active_id and last_active_id >= 0:
            order_changes += 1
            # Count inventory status at transition
            for bid in range(20):
                inv = state.bot_inv_list(bid)
                for t in inv:
                    if t in active_types:
                        pass  # active
                    elif t in preview_types:
                        items_preview_at_transition += 1
                    else:
                        items_dead_at_transition += 1
        last_active_id = active_id

        actions = solver.action(state, all_orders, rnd)

        # Count inventory utilization
        for bid in range(20):
            inv = state.bot_inv_list(bid)
            total_slots += INV_CAP
            empty_slots += INV_CAP - len(inv)
            for t in inv:
                if t in active_types:
                    active_slots += 1
                elif t in preview_types:
                    preview_slots += 1
                else:
                    dead_slots += 1

        step(state, actions, all_orders)

    print(f"\nSeed {seed}: Score={state.score}, Orders={state.orders_completed}")
    print(f"Inventory utilization (avg per round):")
    print(f"  Total slots: {total_slots/NUM_ROUNDS:.1f}")
    print(f"  Active items: {active_slots/NUM_ROUNDS:.1f} ({active_slots/total_slots*100:.1f}%)")
    print(f"  Preview items: {preview_slots/NUM_ROUNDS:.1f} ({preview_slots/total_slots*100:.1f}%)")
    print(f"  Dead items: {dead_slots/NUM_ROUNDS:.1f} ({dead_slots/total_slots*100:.1f}%)")
    print(f"  Empty slots: {empty_slots/NUM_ROUNDS:.1f} ({empty_slots/total_slots*100:.1f}%)")
    print(f"Order transitions: {order_changes}")
    if order_changes:
        print(f"  Avg dead items at transition: {items_dead_at_transition/order_changes:.1f}")
        print(f"  Avg preview items at transition: {items_preview_at_transition/order_changes:.1f}")

    return state.score


if __name__ == '__main__':
    for seed in [7005, 11, 42, 45]:
        run_inv_diagnostic(seed)
