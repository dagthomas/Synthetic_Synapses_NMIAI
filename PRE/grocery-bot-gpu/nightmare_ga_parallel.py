#!/usr/bin/env python3
"""Nightmare GA with multiprocessing: massively parallel multi-restart + perturbation.

Supports both seed-based (sim) and capture-based (live) modes.
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
import multiprocessing as mp
from functools import partial

sys.stdout.reconfigure(encoding='utf-8')

# Module-level capture data path for workers
_CAPTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'nightmare_capture.json')


def _init_worker():
    """Initialize worker process."""
    sys.stdout.reconfigure(encoding='utf-8')


def _init_game(seed, capture_path):
    """Init game from capture file or seed."""
    from game_engine import init_game, init_game_from_capture, step
    if capture_path and os.path.exists(capture_path):
        with open(capture_path) as f:
            cap = json.load(f)
        n = len(cap.get('orders', []))
        return init_game_from_capture(cap, num_orders=max(n, 50))
    else:
        return init_game(seed, 'nightmare', num_orders=100)


def _run_lmapf_worker(args):
    """Worker: run LMAPF solver with given params."""
    seed, solver_seed, drop_d_weight, capture_path = args
    from game_engine import step
    from nightmare_lmapf_solver import LMAPFSolver
    from precompute import PrecomputedTables
    from configs import DIFF_ROUNDS

    NUM_ROUNDS = DIFF_ROUNDS['nightmare']
    state, all_orders = _init_game(seed, capture_path)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=drop_d_weight)

    action_log = []
    chains = 0
    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        o_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > o_before + 1:
            chains += state.orders_completed - o_before - 1

    return state.score, action_log, state.orders_completed, chains, solver_seed, drop_d_weight


def _replay_worker(args):
    """Worker: replay base actions + fresh solver from checkpoint."""
    seed, base_actions, perturb_round, solver_seed, drop_d_weight, capture_path = args
    from game_engine import step
    from nightmare_lmapf_solver import LMAPFSolver
    from precompute import PrecomputedTables
    from configs import DIFF_ROUNDS

    NUM_ROUNDS = DIFF_ROUNDS['nightmare']
    state, all_orders = _init_game(seed, capture_path)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)

    action_log = []
    chains = 0

    for rnd in range(min(perturb_round, NUM_ROUNDS)):
        state.round = rnd
        actions = base_actions[rnd] if rnd < len(base_actions) else [(0, -1)] * 20
        action_log.append(list(actions))
        o_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > o_before + 1:
            chains += state.orders_completed - o_before - 1

    solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=drop_d_weight)
    for rnd in range(perturb_round, NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        o_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > o_before + 1:
            chains += state.orders_completed - o_before - 1

    return state.score, action_log, state.orders_completed, chains, perturb_round


def optimize_parallel(game_seed: int, budget: float = 60.0,
                      n_workers: int = 6, verbose: bool = True,
                      capture_path: str = None):
    """Parallel GA optimizer. Uses capture_path if provided, else seed."""
    from configs import DIFF_ROUNDS
    NUM_ROUNDS = DIFF_ROUNDS['nightmare']

    t0 = time.time()
    rng = random.Random(game_seed * 7 + 1)

    best_score = 0
    best_actions = None
    best_orders = 0
    best_chains = 0
    n_evals = 0

    ddw_choices = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    checkpoints = list(range(0, NUM_ROUNDS, 10))  # every 10 rounds
    cp = capture_path  # shorthand

    with mp.Pool(n_workers, initializer=_init_worker) as pool:

        # Phase 1: Parallel multi-restart (40% of budget)
        phase1_end = t0 + budget * 0.4
        while time.time() < phase1_end:
            # Submit a batch of evals
            batch_size = n_workers * 2
            batch = []
            for _ in range(batch_size):
                ss = rng.randint(1, 100000)
                ddw = rng.choice(ddw_choices)
                batch.append((game_seed, ss, ddw, cp))

            results = pool.map(_run_lmapf_worker, batch)
            n_evals += len(results)

            for score, actions, orders, chains, ss, ddw in results:
                if score > best_score:
                    best_score = score
                    best_actions = actions
                    best_orders = orders
                    best_chains = chains
                    if verbose:
                        print(f"    [{n_evals}] score={score} ord={orders} "
                              f"ch={chains} (ss={ss}, ddw={ddw:.1f})")

        if verbose:
            print(f"  Phase 1: {n_evals} evals, best={best_score}")

        # Phase 2: Parallel checkpoint perturbation (50% of budget)
        phase2_end = t0 + budget * 0.9
        n_perturbs = 0

        while time.time() < phase2_end and best_actions is not None:
            batch_size = n_workers * 2
            batch = []
            for _ in range(batch_size):
                ckpt = rng.choice(checkpoints)
                ss = rng.randint(1, 100000)
                ddw = rng.choice(ddw_choices)
                batch.append((game_seed, best_actions, ckpt, ss, ddw, cp))

            results = pool.map(_replay_worker, batch)
            n_evals += len(results)
            n_perturbs += len(results)

            for score, actions, orders, chains, ckpt in results:
                if score > best_score:
                    best_score = score
                    best_actions = actions
                    best_orders = orders
                    best_chains = chains
                    if verbose:
                        print(f"    [{n_evals}] PERTURB cp={ckpt}: score={score} "
                              f"ord={orders} ch={chains}")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Final: score={best_score} orders={best_orders} "
              f"chains={best_chains} ({n_evals} evals, {elapsed:.1f}s)")

    return best_score, best_actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('--budget', type=float, default=60.0)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--capture', action='store_true',
                        help='Use captured live data instead of sim seed')
    parser.add_argument('--save', action='store_true',
                        help='Save best solution to data/nightmare_best_live.json')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    capture_path = None
    if args.capture:
        # Try DB first, fall back to file
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap and cap.get('orders'):
                # Save to temp file for workers
                capture_path = _CAPTURE_PATH
                os.makedirs(os.path.dirname(capture_path), exist_ok=True)
                with open(capture_path, 'w') as f:
                    json.dump(cap, f)
                print(f"Using captured data: {len(cap['orders'])} orders")
        except Exception as e:
            print(f"DB load failed ({e}), trying file...")
        if not capture_path and os.path.exists(_CAPTURE_PATH):
            capture_path = _CAPTURE_PATH
            with open(capture_path) as f:
                cap = json.load(f)
            print(f"Using captured file: {len(cap.get('orders', []))} orders")
        if not capture_path:
            print("No capture data found, falling back to sim seeds")

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    best_overall_score = 0
    best_overall_actions = None
    t0 = time.time()

    for seed in seeds:
        if not args.quiet:
            print(f"\nSeed {seed}:")
        score, actions = optimize_parallel(seed, args.budget, args.workers,
                                            not args.quiet, capture_path)
        scores.append(score)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_actions = actions

    if len(seeds) > 1:
        print(f"\nSummary: mean={sum(scores)/len(scores):.1f} "
              f"max={max(scores)} min={min(scores)}")
        for seed, score in zip(seeds, scores):
            print(f"  seed {seed}: {score}")
        print(f"Total time: {time.time()-t0:.1f}s")

    if args.save and best_overall_actions:
        os.makedirs('data', exist_ok=True)
        with open('data/nightmare_best_live.json', 'w') as f:
            json.dump({'score': best_overall_score,
                       'action_log': best_overall_actions}, f)
        print(f"Saved best solution: {best_overall_score}")


if __name__ == '__main__':
    main()
