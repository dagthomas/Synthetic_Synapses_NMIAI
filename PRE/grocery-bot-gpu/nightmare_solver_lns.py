"""Integrated nightmare solver combining LNS2, PIBT-v3, scheduler, and adaptive traffic.

Pipeline:
  Phase A (2s):   Generate initial solution via LMAPFSolver + PathfinderV3
  Phase B (60s):  Run LNS2 iterations on the full 500-round solution
  Phase C (live): Replay optimized plan via WebSocket
  Phase D:        If desync, replan from current state using V4 reactive

Usage:
    python nightmare_solver_lns.py --seed 7005 -v --time 60
    python nightmare_solver_lns.py --seed 7005 --windowed -v

Integration:
    Called from nightmare_offline.py as LNS2-enhanced training pipeline.
"""
from __future__ import annotations

import random
import time

from game_engine import (
    GameState, MapState, Order, step, init_game, init_game_from_capture,
    ACT_WAIT, ACT_PICKUP, ACT_DROPOFF, INV_CAP,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver
from nightmare_lns2 import NightmareLNS2, WindowedLNS2, lns2_optimize
from nightmare_scheduler import NightmareScheduler
from nightmare_pathfinder import build_walkable

NUM_ROUNDS = 500
NUM_BOTS = 20


class LNSSolver:
    """Integrated solver: baseline generation + LNS2 optimization.

    Combines:
    1. V4 (LMAPF) baseline generation with optional V3 multi-restart
    2. LNS2 large neighborhood search optimization
    3. Optional windowed LNS2 for faster convergence
    4. Optional scheduler-informed initial solution
    """

    def __init__(self, ms: MapState, tables: PrecomputedTables,
                 all_orders: list[Order], state0: GameState,
                 verbose: bool = False):
        self.ms = ms
        self.tables = tables
        self.all_orders = all_orders
        self.state0 = state0
        self.verbose = verbose
        self.walkable = build_walkable(ms)

    def solve(self, max_time: float = 120, lns_fraction: float = 0.7,
              num_restarts: int = 6, use_windowed: bool = False,
              ) -> tuple[int, list[list[tuple[int, int]]]]:
        """Full solve pipeline.

        Args:
            max_time: Total time budget in seconds
            lns_fraction: Fraction of time for LNS2 (rest for baseline generation)
            num_restarts: Number of baseline restarts to try
            use_windowed: Use windowed LNS2 variant

        Returns:
            (best_score, best_action_log)
        """
        t0 = time.time()
        baseline_budget = max_time * (1.0 - lns_fraction)
        lns_budget = max_time * lns_fraction

        # Phase A: Generate baseline solution(s)
        best_score, best_log = self._generate_baselines(
            baseline_budget, num_restarts)

        elapsed_baseline = time.time() - t0
        remaining = max_time - elapsed_baseline
        if remaining < 3:
            return best_score, best_log

        if self.verbose:
            print(f"  Baseline: score={best_score} ({elapsed_baseline:.1f}s)")

        # Phase B: LNS2 optimization
        lns_time = min(remaining - 1, lns_budget)

        if use_windowed:
            lns = WindowedLNS2(
                self.ms, self.tables, self.all_orders, self.state0,
                best_log, verbose=self.verbose)
            lns_score, lns_log = lns.optimize_windowed(max_time=lns_time)
        else:
            lns = NightmareLNS2(
                self.ms, self.tables, self.all_orders, self.state0,
                best_log, verbose=self.verbose)
            lns_score, lns_log = lns.optimize(max_time=lns_time)

        if lns_score > best_score:
            best_score = lns_score
            best_log = lns_log

        total = time.time() - t0
        if self.verbose:
            print(f"  LNS solve: score={best_score} ({total:.1f}s)")

        return best_score, best_log

    def _generate_baselines(self, budget: float, num_restarts: int,
                            ) -> tuple[int, list[list[tuple[int, int]]]]:
        """Generate baseline solutions via V4 + perturbed restarts."""
        t0 = time.time()

        # V4 baseline
        best_score, _, best_log = self._run_v4(seed=0)
        if self.verbose:
            print(f"    V4 baseline: {best_score}")

        # V4 perturbed restarts
        for i in range(1, num_restarts):
            if time.time() - t0 > budget:
                break
            rate = random.uniform(0.005, 0.04)
            score, _, log = self._run_v4(seed=i * 17 + 42, perturbation_rate=rate)
            if score > best_score:
                best_score = score
                best_log = log
                if self.verbose:
                    print(f"    Restart {i}: {score} (new best, rate={rate:.3f})")

        return best_score, best_log

    def _run_v4(self, seed: int = 0, perturbation_rate: float = 0.0,
                ) -> tuple[int, int, list[list[tuple[int, int]]]]:
        """Run V4 solver. Returns (score, orders_completed, action_log)."""
        solver = LMAPFSolver(
            self.ms, self.tables, future_orders=self.all_orders,
            solver_seed=seed)

        if perturbation_rate > 0:
            rng = random.Random(seed)

        state = self.state0.copy()
        action_log = []

        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            if perturbation_rate > 0:
                for bid in range(NUM_BOTS):
                    if rng.random() < perturbation_rate:
                        solver.stall_counts[bid] = rng.randint(0, 5)
            actions = solver.action(state, self.all_orders, rnd)
            action_log.append(actions)
            step(state, actions, self.all_orders)

        return state.score, state.orders_completed, action_log


def solve_nightmare(state0: GameState, all_orders: list[Order],
                    max_time: float = 120, verbose: bool = False,
                    use_windowed: bool = False,
                    ) -> tuple[int, list[list[tuple[int, int]]]]:
    """Convenience function: full LNS-enhanced nightmare solve.

    Args:
        state0: Initial game state
        all_orders: Complete order list
        max_time: Time budget in seconds
        verbose: Print progress
        use_windowed: Use windowed LNS2

    Returns:
        (best_score, best_action_log)
    """
    ms = state0.map_state
    tables = PrecomputedTables.get(ms)
    solver = LNSSolver(ms, tables, all_orders, state0, verbose=verbose)
    return solver.solve(max_time=max_time, use_windowed=use_windowed)


# ── CLI ──

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='LNS-integrated nightmare solver')
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--time', type=float, default=120, help='Total time budget (s)')
    parser.add_argument('--windowed', action='store_true', help='Use windowed LNS2')
    parser.add_argument('--lns-fraction', type=float, default=0.7,
                        help='Fraction of time for LNS2 (default: 0.7)')
    parser.add_argument('--restarts', type=int, default=6,
                        help='Number of baseline restarts (default: 6)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true',
                        help='Compare vs V4-only baseline')
    args = parser.parse_args()

    state0, all_orders = init_game(args.seed, 'nightmare', num_orders=100)

    if args.compare:
        # Run V4-only baseline
        ms = state0.map_state
        tables = PrecomputedTables.get(ms)
        print("Running V4-only baseline...")
        solver = LMAPFSolver(ms, tables, future_orders=all_orders)
        state = state0.copy()
        action_log = []
        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            step(state, actions, all_orders)
        v4_score = state.score
        print(f"V4-only: score={v4_score}")
        print()

    # Run LNS solver
    print(f"Running LNS solver (seed={args.seed}, time={args.time}s)...")
    score, log = solve_nightmare(
        state0, all_orders,
        max_time=args.time,
        verbose=args.verbose,
        use_windowed=args.windowed,
    )

    print(f"\nFinal: score={score}")
    if args.compare:
        print(f"Improvement: {v4_score} -> {score} ({score - v4_score:+d})")
