"""Deep per-round diagnostics: track exactly what each bot does per order."""
import numpy as np
from game_engine import (
    init_game, step, ACT_WAIT, ACT_PICKUP, ACT_DROPOFF,
    INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6


def run_deep_diag(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables)
    drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = len(state.bot_positions)

    order_start = 0
    last_active_id = -1
    order_stats = []

    # Per-order tracking
    active_bots_per_round = []
    preview_bots_per_round = []
    dead_bots_per_round = []
    empty_bots_per_round = []

    # Stall tracking
    prev_positions = {}
    total_stalls = 0
    total_moves = 0

    for rnd in range(num_rounds):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        active_id = active.id if active else -1
        if active_id != last_active_id and last_active_id != -1:
            dur = rnd - order_start
            order_stats.append(dur)
            order_start = rnd
        last_active_id = active_id

        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
        preview_needs = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        n_active = n_preview = n_dead = n_empty = 0
        for bid in range(num_bots):
            pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)

            # Stall detection
            if prev_positions.get(bid) == pos:
                total_stalls += 1
            else:
                total_moves += 1
            prev_positions[bid] = pos

            if not inv:
                n_empty += 1
            elif any(t in active_needs for t in inv):
                n_active += 1
            elif any(t in preview_needs for t in inv):
                n_preview += 1
            elif len(inv) < INV_CAP:
                n_empty += 1
            else:
                n_dead += 1

        active_bots_per_round.append(n_active)
        preview_bots_per_round.append(n_preview)
        dead_bots_per_round.append(n_dead)
        empty_bots_per_round.append(n_empty)

        actions = solver.action(state, all_orders, rnd)
        step(state, actions, all_orders)

    if order_start < num_rounds:
        order_stats.append(num_rounds - order_start)

    return {
        'score': state.score,
        'orders': state.orders_completed,
        'order_durs': order_stats,
        'stall_rate': total_stalls / max(total_stalls + total_moves, 1) * 100,
        'avg_active': np.mean(active_bots_per_round),
        'avg_preview': np.mean(preview_bots_per_round),
        'avg_dead': np.mean(dead_bots_per_round),
        'avg_empty': np.mean(empty_bots_per_round),
        # Dead bot progression
        'dead_r50': np.mean(dead_bots_per_round[:50]),
        'dead_r250': np.mean(dead_bots_per_round[200:300]) if len(dead_bots_per_round) >= 300 else 0,
        'dead_r450': np.mean(dead_bots_per_round[400:]) if len(dead_bots_per_round) >= 450 else 0,
    }


if __name__ == '__main__':
    for seed in [1000, 1001, 1002]:
        r = run_deep_diag(seed)
        print(f"\nSeed {seed}: {r['score']} pts, {r['orders']} orders")
        print(f"  Bot types (avg): active={r['avg_active']:.1f} preview={r['avg_preview']:.1f} "
              f"dead={r['avg_dead']:.1f} empty={r['avg_empty']:.1f}")
        print(f"  Dead progression: R50={r['dead_r50']:.1f} R250={r['dead_r250']:.1f} R450={r['dead_r450']:.1f}")
        print(f"  Stall rate: {r['stall_rate']:.1f}%")
        durs = r['order_durs']
        if durs:
            print(f"  Order durs: mean={np.mean(durs):.1f} med={np.median(durs):.0f} "
                  f"min={min(durs)} max={max(durs)}")
            # Histogram
            bins = [0, 6, 10, 15, 20, 30, 100]
            for i in range(len(bins)-1):
                count = sum(1 for d in durs if bins[i] <= d < bins[i+1])
                print(f"    [{bins[i]:2d}-{bins[i+1]:2d}): {count} orders")
