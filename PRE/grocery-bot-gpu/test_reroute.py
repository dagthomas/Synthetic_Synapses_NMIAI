#!/usr/bin/env python3
"""Test delivery path rerouting with chain counting."""
import sys, copy, time
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game, step, ACT_DROPOFF
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']

def run_with_chains(seed, solver_seed=0):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=solver_seed)

    chains = 0
    total_deliveries = 0

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)

        # Count chain events: track active order before/after step
        active_before = state.active_idx
        step(state, actions, all_orders)
        active_after = state.active_idx

        # If active order advanced by more than 1, chains happened
        if active_after > active_before + 1:
            chains += (active_after - active_before - 1)

        total_deliveries = state.score

    return state.score, chains

seeds = [7005, 11, 42, 45, 100, 200, 300, 500]
scores = []
all_chains = 0

print("Testing delivery path rerouting (8 seeds):")
print("-" * 50)
t0 = time.time()

for seed in seeds:
    t1 = time.time()
    score, chains = run_with_chains(seed)
    dt = time.time() - t1
    scores.append(score)
    all_chains += chains
    print(f"  seed {seed:4d}: score={score:3d}  chains={chains}  ({dt:.1f}s)")

mean = sum(scores) / len(scores)
print("-" * 50)
print(f"Mean: {mean:.1f}  Total chains: {all_chains}  ({time.time()-t0:.1f}s)")
print(f"Scores: {scores}")
