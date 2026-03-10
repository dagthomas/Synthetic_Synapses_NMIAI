#!/usr/bin/env python3
"""Test max_preview_pickers=8 with 8 seeds."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game, step
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']

# Patch allocator
from nightmare_task_alloc import NightmareTaskAlloc
_orig_allocate = NightmareTaskAlloc.allocate
def _patched(self, *args, **kwargs):
    kwargs['max_preview_pickers_override'] = 8
    return _orig_allocate(self, *args, **kwargs)
NightmareTaskAlloc.allocate = _patched

seeds = [7005, 11, 42, 45, 100, 200, 300, 500]
scores = []
print("max_preview=8, 8 seeds:")
for seed in seeds:
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=0)
    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        step(state, actions, all_orders)
    scores.append(state.score)
    print(f"  seed {seed:4d}: {state.score}")

print(f"\nMean: {sum(scores)/len(scores):.1f}")
print(f"Scores: {scores}")
