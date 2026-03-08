"""Diagnostics: measure V6 bottlenecks in nightmare mode."""
import numpy as np
from game_engine import (
    init_game, step, ACT_WAIT, ACT_PICKUP, ACT_DROPOFF,
    INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6


def run_diagnostics(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables)

    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = len(state.bot_positions)

    # Tracking
    order_start_round = 0
    order_durations = []
    chain_count = 0
    auto_delivers = 0
    last_active_id = -1
    bot_role_counts = {'active_carrier': 0, 'preview_carrier': 0, 'dead': 0,
                       'empty_active': 0, 'empty_preview': 0, 'empty_park': 0,
                       'stage': 0, 'deliver': 0, 'flee': 0, 'park': 0}
    stall_rounds = 0
    total_bot_rounds = 0
    dead_inv_items = 0
    total_inv_items = 0
    rounds_with_preview_at_drop = 0
    preview_items_at_drop = 0

    for rnd in range(num_rounds):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        # Track order transitions
        active_id = active.id if active else -1
        if active_id != last_active_id and last_active_id != -1:
            dur = rnd - order_start_round
            order_durations.append(dur)
            if dur == 0:
                chain_count += 1
            order_start_round = rnd
        last_active_id = active_id

        # Analyze bot state
        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
        preview_needs = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        drop_set = set(tuple(dz) for dz in ms.drop_off_zones)
        preview_at_drop_this_round = 0

        for bid in range(num_bots):
            pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)
            total_bot_rounds += 1

            # Count dead inventory
            for t in inv:
                total_inv_items += 1
                if t not in active_needs and t not in preview_needs:
                    dead_inv_items += 1

            # Count preview items at dropoff
            if pos in drop_set and inv:
                for t in inv:
                    if t in preview_needs:
                        preview_at_drop_this_round += 1

        if preview_at_drop_this_round > 0:
            rounds_with_preview_at_drop += 1
        preview_items_at_drop += preview_at_drop_this_round

        # Get actions from solver
        actions = solver.action(state, all_orders, rnd)
        step(state, actions, all_orders)

    # Final order
    if order_start_round < num_rounds:
        order_durations.append(num_rounds - order_start_round)

    return {
        'score': state.score,
        'orders': state.orders_completed,
        'order_durations': order_durations,
        'chain_count': chain_count,
        'avg_order_dur': np.mean(order_durations) if order_durations else 0,
        'dead_inv_pct': dead_inv_items / max(total_inv_items, 1) * 100,
        'preview_at_drop_rounds': rounds_with_preview_at_drop,
        'avg_preview_at_drop': preview_items_at_drop / num_rounds,
    }


if __name__ == '__main__':
    all_results = []
    for seed in range(1000, 1003):
        r = run_diagnostics(seed)
        print(f"\nSeed {seed}: {r['score']} pts, {r['orders']} orders")
        print(f"  Avg order duration: {r['avg_order_dur']:.1f} rounds")
        print(f"  Chain reactions: {r['chain_count']}")
        print(f"  Dead inventory: {r['dead_inv_pct']:.1f}%")
        print(f"  Rounds with preview at dropoff: {r['preview_at_drop_rounds']}/{500}")
        print(f"  Avg preview items at dropoff/round: {r['avg_preview_at_drop']:.2f}")
        dur = r['order_durations']
        if dur:
            print(f"  Order durations: min={min(dur)}, max={max(dur)}, "
                  f"med={np.median(dur):.0f}")
            fast = sum(1 for d in dur if d <= 5)
            slow = sum(1 for d in dur if d > 15)
            print(f"  Fast (<=5 rnds): {fast}, Slow (>15 rnds): {slow}")
        all_results.append(r)

    print(f"\n{'='*50}")
    print(f"AVERAGES across 5 seeds:")
    print(f"  Score: {np.mean([r['score'] for r in all_results]):.1f}")
    print(f"  Orders: {np.mean([r['orders'] for r in all_results]):.1f}")
    print(f"  Avg order duration: {np.mean([r['avg_order_dur'] for r in all_results]):.1f}")
    print(f"  Chain reactions: {np.mean([r['chain_count'] for r in all_results]):.1f}")
    print(f"  Dead inventory: {np.mean([r['dead_inv_pct'] for r in all_results]):.1f}%")
    print(f"  Avg preview at dropoff: {np.mean([r['avg_preview_at_drop'] for r in all_results]):.2f}")
