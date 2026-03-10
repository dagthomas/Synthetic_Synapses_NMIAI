#!/usr/bin/env python3
"""Diagnose delivery synchronization — how many bots do ACT_DROPOFF per round."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game, step, ACT_DROPOFF, INV_CAP
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def diagnose_seed(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=0)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

    dropoff_counts = []  # per round: how many bots do ACT_DROPOFF
    completion_rounds = []  # rounds where orders complete

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)

        # Count ACT_DROPOFF actions
        n_dropoff = sum(1 for bid in range(20) if actions[bid][0] == ACT_DROPOFF)
        dropoff_counts.append(n_dropoff)

        completed_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > completed_before:
            # Count ALL bots at dropoffs in this round (position after their action)
            bots_at_drop = 0
            for bid in range(20):
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                if any(bx == dz[0] and by == dz[1] for dz in drop_zones):
                    bots_at_drop += 1
            completion_rounds.append({
                'round': rnd,
                'dropoffs_this_round': n_dropoff,
                'bots_at_drop_after': bots_at_drop,
            })

    return state.score, completion_rounds, dropoff_counts


for seed in [7005, 42, 300]:
    score, completions, dcounts = diagnose_seed(seed)

    # Stats on dropoff actions per round
    nonzero = [d for d in dcounts if d > 0]
    print(f"\nSeed {seed}: score={score}")
    print(f"  Rounds with ACT_DROPOFF: {len(nonzero)}/{NUM_ROUNDS}")
    if nonzero:
        from collections import Counter
        c = Counter(nonzero)
        print(f"  Distribution: {dict(sorted(c.items()))}")

    print(f"  Order completions: {len(completions)}")
    for cr in completions[:10]:
        print(f"    R{cr['round']:3d}: {cr['dropoffs_this_round']} dropoffs, "
              f"{cr['bots_at_drop_after']} at drop after")

    # How many simultaneous dropoffs at completion time
    if completions:
        avg_sync = sum(cr['dropoffs_this_round'] for cr in completions) / len(completions)
        avg_at_drop = sum(cr['bots_at_drop_after'] for cr in completions) / len(completions)
        print(f"  Avg dropoffs at completion: {avg_sync:.1f}, avg bots at drop: {avg_at_drop:.1f}")
