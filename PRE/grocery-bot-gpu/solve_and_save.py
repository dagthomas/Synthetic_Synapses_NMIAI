"""Solve a game and save the optimized action sequence for WebSocket replay.

Usage:
    python solve_and_save.py <difficulty> [seed] [--time <seconds>]

Runs multi-strategy solver + optimizer, saves best action sequence.
"""
import sys
import time
import json
from multi_solve import multi_solve
from ws_client import save_actions


def solve_and_save(difficulty, seed=7001, time_limit=120.0, verbose=True):
    """Solve a game and save the action sequence."""
    t0 = time.time()

    if verbose:
        print(f"Solving {difficulty} seed={seed} (time limit: {time_limit}s)")

    score, actions = multi_solve(seed, difficulty, time_limit=time_limit, verbose=verbose)

    filename = f"solution_{difficulty}_{seed}.json"
    save_actions(actions, filename)

    if verbose:
        print(f"\nFinal score: {score}")
        print(f"Saved to: {filename}")
        print(f"Total time: {time.time()-t0:.1f}s")
        print(f"\nTo replay: python ws_client.py <ws_url> {filename} --difficulty {difficulty}")

    return score, filename


if __name__ == '__main__':
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    time_lim = 120.0

    for i, arg in enumerate(sys.argv):
        if arg == '--time' and i + 1 < len(sys.argv):
            time_lim = float(sys.argv[i + 1])

    solve_and_save(difficulty, seed, time_limit=time_lim)
