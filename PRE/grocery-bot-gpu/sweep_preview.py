#!/usr/bin/env python3
"""Sweep max_preview_pickers to find optimal value."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game, step
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']
seeds = [7005, 42, 300, 500]


def test_preview(max_preview, seeds):
    scores = []
    for seed in seeds:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=0)

        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            # Override max_preview_pickers
            solver.allocator._max_preview_override = max_preview
            actions = solver.action(state, all_orders, rnd)
            step(state, actions, all_orders)
        scores.append(state.score)
    return sum(scores) / len(scores), scores


# Patch allocator to support override
from nightmare_task_alloc import NightmareTaskAlloc
_orig_allocate = NightmareTaskAlloc.allocate

def _patched_allocate(self, *args, **kwargs):
    override = getattr(self, '_max_preview_override', -1)
    if override >= 0:
        kwargs['max_preview_pickers_override'] = override
    return _orig_allocate(self, *args, **kwargs)

NightmareTaskAlloc.allocate = _patched_allocate

print(f"Sweeping max_preview_pickers (seeds: {seeds}):")
print("-" * 60)
for n in [0, 1, 2, 3, 4, 6, 8, 10, 15]:
    mean, scores = test_preview(n, seeds)
    print(f"  max_preview={n:2d}: mean={mean:.1f}  {scores}")
