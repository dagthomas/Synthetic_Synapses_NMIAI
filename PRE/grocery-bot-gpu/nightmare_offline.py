"""Offline training + iterate pipeline for nightmare mode.

Generates optimized action sequences via V3 solver with stochastic restarts
and checkpoint-based local search. Supports iterate pipeline for live games.

TRAINING (sim mode):
    V3 baseline -> multi-restart with perturbations -> checkpoint search -> save best

LIVE PIPELINE:
    V3 live game -> capture orders -> train offline -> replay -> discover orders -> iterate

Usage:
    python nightmare_offline.py --seed 7005 -v                     # sim training
    python nightmare_offline.py --seeds 1000-1009 --train-time 60  # sweep
    python nightmare_offline.py --ws-url "wss://..." --time-budget 275  # live pipeline
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

from game_engine import (
    init_game, init_game_from_capture, step, GameState, Order, MapState,
    ACT_WAIT, ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v2 import NightmareSolverV3, record_to_pg
from nightmare_pathfinder import build_walkable

NUM_ROUNDS = 500
NUM_BOTS = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class PerturbedV3(NightmareSolverV3):
    """V3 with controlled stochastic perturbations for multi-restart search.

    Two perturbation mechanisms:
    1. Randomized stall escape direction (instead of deterministic hash)
    2. Random stall count jitter (triggers/suppresses escapes, changing congestion)
    """

    def __init__(self, *args, rng_seed=None, perturbation_rate=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(rng_seed)
        self.perturbation_rate = perturbation_rate

    def _escape_action(self, bid, pos, rnd):
        """Randomized stall escape direction."""
        dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
        self.rng.shuffle(dirs)
        for a in dirs:
            nx, ny = pos[0] + DX[a], pos[1] + DY[a]
            if (nx, ny) in self.walkable:
                return a
        return ACT_WAIT

    def action(self, state, all_orders, rnd):
        """V3 action with occasional stall count perturbation."""
        if self.perturbation_rate > 0:
            for bid in range(len(state.bot_positions)):
                if self.rng.random() < self.perturbation_rate:
                    self.stall_counts[bid] = self.rng.randint(0, 5)
        return super().action(state, all_orders, rnd)


class NightmareTrainer:
    """Offline trainer: V3 multi-restart + checkpoint perturbation search."""

    def __init__(self, seed=None, capture_data=None, verbose=False):
        """Init from seed (sim) or capture_data (live orders)."""
        if seed is not None:
            self.state0, self.all_orders = init_game(seed, 'nightmare', num_orders=100)
        elif capture_data is not None:
            n = len(capture_data.get('orders', []))
            self.state0, self.all_orders = init_game_from_capture(
                capture_data, num_orders=max(n, 50))
        else:
            raise ValueError("Need seed or capture_data")

        self.ms = self.state0.map_state
        self.tables = PrecomputedTables.get(self.ms)
        self.walkable = build_walkable(self.ms)
        self.verbose = verbose
        self.seed = seed

    def _run_solver(self, solver):
        """Run a solver instance on a fresh game copy. Returns (score, action_log)."""
        state = self.state0.copy()
        action_log = []
        for rnd in range(NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, self.all_orders, rnd)
            action_log.append(actions)
            step(state, actions, self.all_orders)
        return state.score, state.orders_completed, action_log

    def run_v3_baseline(self):
        """Run vanilla V3. Returns (score, orders_completed, action_log)."""
        solver = NightmareSolverV3(self.ms, self.tables, future_orders=self.all_orders)
        return self._run_solver(solver)

    def run_v3_perturbed(self, rng_seed, perturbation_rate=0.02):
        """Run V3 with stochastic perturbations."""
        solver = PerturbedV3(
            self.ms, self.tables, future_orders=self.all_orders,
            rng_seed=rng_seed, perturbation_rate=perturbation_rate)
        return self._run_solver(solver)

    def train(self, max_time=120, num_restarts=20):
        """Multi-restart training + checkpoint search. Returns (best_score, best_log)."""
        t0 = time.time()

        # Phase 1: V3 baseline
        best_score, best_ord, best_log = self.run_v3_baseline()
        if self.verbose:
            print(f"  Baseline: score={best_score} orders={best_ord}")

        # Phase 2: Quick stochastic restarts (15% of budget)
        # Perturbations occasionally find better solutions via different
        # stall escape patterns and congestion avoidance
        restart_budget = max_time * 0.15
        for i in range(num_restarts - 1):
            if time.time() - t0 > restart_budget:
                break
            rate = random.uniform(0.005, 0.06)
            score, ords, log = self.run_v3_perturbed(
                rng_seed=i * 17 + 42, perturbation_rate=rate)
            if score > best_score:
                best_score = score
                best_ord = ords
                best_log = log
                if self.verbose:
                    print(f"  Restart {i+1}: score={score} orders={ords} "
                          f"(rate={rate:.3f}) NEW BEST")
            elif self.verbose and i < 3:
                print(f"  Restart {i+1}: score={score} orders={ords}")

        # Phase 3: Checkpoint perturbation search with remaining time
        remaining = max_time - (time.time() - t0)
        if remaining > 5:
            cp_score, cp_log = self._checkpoint_search(
                best_log, best_score, max_time=remaining)
            if cp_score > best_score:
                best_score = cp_score
                best_log = cp_log

        elapsed = time.time() - t0
        if self.verbose:
            print(f"  Training done: score={best_score} ({elapsed:.1f}s)")

        return best_score, best_log

    def _checkpoint_search(self, baseline_log, baseline_score, max_time=60):
        """Checkpoint-based local search.

        At each checkpoint: force one random bot to take a different action,
        then re-run V3 (with optional perturbation) to the end.
        The forced action changes congestion patterns, leading to different
        allocation decisions downstream.
        """
        t0 = time.time()
        CP_INTERVAL = 25

        # Build checkpoints by simulating baseline
        checkpoints = {}
        state = self.state0.copy()
        for rnd in range(NUM_ROUNDS):
            if rnd % CP_INTERVAL == 0:
                checkpoints[rnd] = state.copy()
            state.round = rnd
            step(state, baseline_log[rnd], self.all_orders)

        best_score = baseline_score
        best_log = list(baseline_log)
        trials = 0
        improvements = 0
        cp_rounds = sorted(checkpoints.keys())

        while time.time() - t0 < max_time:
            # Pick random checkpoint (bias toward earlier rounds for more impact)
            weights = [1.0 / (1 + cp_rnd / 100) for cp_rnd in cp_rounds]
            cp_rnd = random.choices(cp_rounds, weights=weights, k=1)[0]
            state = checkpoints[cp_rnd].copy()

            # Fresh solver with optional light perturbation
            rng_seed = trials * 31 + 7
            solver = PerturbedV3(
                self.ms, self.tables, future_orders=self.all_orders,
                rng_seed=rng_seed,
                perturbation_rate=random.uniform(0.0, 0.02))

            # Get V3's normal action, then force one bot to move differently
            state.round = cp_rnd
            normal_actions = solver.action(state, self.all_orders, cp_rnd)

            bid = random.randint(0, NUM_BOTS - 1)
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))

            moves = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
            random.shuffle(moves)
            # Also consider wait as forced action (creates a "yield")
            if random.random() < 0.3:
                moves.append(ACT_WAIT)

            forced = list(normal_actions)
            forced_ok = False
            for m in moves:
                if m == ACT_WAIT:
                    forced[bid] = (ACT_WAIT, -1)
                    forced_ok = True
                    break
                nx, ny = pos[0] + DX[m], pos[1] + DY[m]
                if (nx, ny) in self.walkable:
                    forced[bid] = (m, -1)
                    forced_ok = True
                    break

            if not forced_ok:
                trials += 1
                continue

            # Build new action log: baseline[:cp_rnd] + forced + V3 re-run
            new_log = list(best_log[:cp_rnd])
            new_log.append(forced)
            step(state, forced, self.all_orders)

            # Re-run V3 from cp_rnd+1 to end
            for rnd in range(cp_rnd + 1, NUM_ROUNDS):
                state.round = rnd
                actions = solver.action(state, self.all_orders, rnd)
                new_log.append(actions)
                step(state, actions, self.all_orders)

            trials += 1
            if state.score > best_score:
                improvements += 1
                best_score = state.score
                best_log = new_log
                if self.verbose:
                    print(f"    CP: score={best_score} (trial={trials}, "
                          f"rnd={cp_rnd}, bid={bid})")

                # Recompute checkpoints from improved solution
                gs = self.state0.copy()
                for rnd in range(NUM_ROUNDS):
                    if rnd % CP_INTERVAL == 0:
                        checkpoints[rnd] = gs.copy()
                    gs.round = rnd
                    step(gs, best_log[rnd], self.all_orders)

        if self.verbose:
            print(f"    CP: {trials} trials, {improvements} improvements "
                  f"({time.time()-t0:.1f}s)")

        return best_score, best_log


# ── Pipeline functions ──

def train_sim(seed, max_time=120, verbose=False, no_record=False):
    """Train on a known seed and save best solution."""
    trainer = NightmareTrainer(seed=seed, verbose=verbose)
    score, action_log = trainer.train(max_time=max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, action_log, seed=seed)
    if verbose:
        status = "SAVED" if saved else "not saved (existing better)"
        print(f"  Seed {seed}: {score} [{status}]")

    if not no_record:
        try:
            state, all_orders = init_game(seed, 'nightmare', num_orders=100)
            for rnd, acts in enumerate(action_log):
                state.round = rnd
                step(state, acts, all_orders)
            record_to_pg(seed, score, state.orders_completed,
                         state.items_delivered, action_log, max_time)
        except Exception as e:
            print(f"  DB record error: {e}", file=sys.stderr)

    return score, action_log


def train_from_capture(capture_data, max_time=120, verbose=False):
    """Train from captured live game data."""
    trainer = NightmareTrainer(capture_data=capture_data, verbose=verbose)
    score, action_log = trainer.train(max_time=max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, action_log)
    if verbose:
        status = "SAVED" if saved else "not saved (existing better)"
        print(f"  Capture training: {score} [{status}]")

    return score, action_log


def _run_subprocess(cmd, timeout=180, keywords=None):
    """Run subprocess, stream filtered stderr, return (score, stderr_text)."""
    keywords = keywords or ['Score', 'GAME_OVER', 'score', 'ERROR', 'Final', 'desync']

    stderr_lines = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1, text=True)

    def read_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            stderr_lines.append(line)
            if any(k in line for k in keywords):
                print(f"      {line}", file=sys.stderr)

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()
    proc.wait(timeout=timeout)
    t.join(timeout=5)

    from subprocess_helpers import parse_game_score
    stderr_text = '\n'.join(stderr_lines)
    return parse_game_score(stderr_text), stderr_text


def live_pipeline(ws_url, time_budget=275, verbose=False):
    """Full iterate pipeline: V3 live -> train -> replay -> iterate.

    Phase 1: V3 live game via live_gpu_stream.py (captures orders + map)
    Phase 2+: Train from captured orders -> replay solution -> discover more orders
    """
    t_start = time.time()
    best_score = 0
    phases = []

    print(f"{'='*60}", file=sys.stderr)
    print(f"NIGHTMARE ITERATE PIPELINE", file=sys.stderr)
    print(f"  Time budget: {time_budget}s", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Phase 1: First live game with V3
    print(f"\n--- Phase 1: V3 live game ---", file=sys.stderr)
    live_script = os.path.join(SCRIPT_DIR, 'live_gpu_stream.py')
    cmd = [sys.executable, live_script, ws_url,
           '--max-states', '5000', '--no-refine', '--save',
           '--preload-capture', '--record']

    try:
        score, _ = _run_subprocess(cmd, timeout=180)
    except Exception as e:
        print(f"    Live game error: {e}", file=sys.stderr)
        score = 0

    elapsed = time.time() - t_start
    print(f"    V3 live: score={score}, time={elapsed:.0f}s", file=sys.stderr)
    best_score = score
    phases.append(('v3_live', score, elapsed))

    # Phase 2+: Iterate: train -> replay -> discover -> train
    for iteration in range(10):
        elapsed_total = time.time() - t_start
        remaining = time_budget - elapsed_total

        if remaining < 30:
            print(f"\n  Budget exhausted ({remaining:.0f}s left)", file=sys.stderr)
            break

        print(f"\n--- Iteration {iteration+1} "
              f"(best={best_score}, {remaining:.0f}s left) ---", file=sys.stderr)

        # Load captured orders
        from solution_store import load_capture
        capture = load_capture('nightmare')
        if not capture:
            print(f"  No capture data, cannot train", file=sys.stderr)
            break

        num_orders = len(capture.get('orders', []))

        # Train offline
        train_budget = min(remaining * 0.4, 60)
        print(f"  Training ({num_orders} orders, {train_budget:.0f}s)...",
              file=sys.stderr)

        iter_t0 = time.time()
        try:
            cap_score, _ = train_from_capture(
                capture, max_time=train_budget, verbose=verbose)
        except Exception as e:
            print(f"  Training error: {e}", file=sys.stderr)
            continue

        train_elapsed = time.time() - iter_t0
        if cap_score > best_score:
            best_score = cap_score
            print(f"  Sim score: {cap_score} (new best)", file=sys.stderr)
        else:
            print(f"  Sim score: {cap_score}", file=sys.stderr)

        # Replay trained solution on live
        remaining = time_budget - (time.time() - t_start)
        if remaining < 15:
            phases.append(('train', cap_score, train_elapsed))
            break

        print(f"  Replaying solution...", file=sys.stderr)
        replay_script = os.path.join(SCRIPT_DIR, 'replay_solution.py')
        replay_cmd = [sys.executable, replay_script, ws_url,
                      '--difficulty', 'nightmare']

        try:
            replay_score, _ = _run_subprocess(replay_cmd, timeout=180)
        except Exception as e:
            print(f"    Replay error: {e}", file=sys.stderr)
            replay_score = 0

        replay_elapsed = time.time() - iter_t0
        phases.append(('iter_' + str(iteration + 1), replay_score, replay_elapsed))

        if replay_score > best_score:
            best_score = replay_score
        print(f"  Replay: {replay_score}", file=sys.stderr)

    # Summary
    total = time.time() - t_start
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PIPELINE COMPLETE: nightmare", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for phase, sc, t in phases:
        print(f"  {phase:>12}: score={sc:>4} ({t:.0f}s)", file=sys.stderr)
    print(f"  Best: {best_score}  Total: {total:.0f}s", file=sys.stderr)

    import json
    print(json.dumps({
        'type': 'pipeline_complete',
        'difficulty': 'nightmare',
        'best_score': best_score,
        'iterations': len(phases),
        'total_time': round(total, 1),
    }))

    return best_score


def main():
    parser = argparse.ArgumentParser(
        description='Offline training + iterate pipeline for nightmare')
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--seeds', type=str,
                        help='Seed range (e.g. 1000-1009)')
    parser.add_argument('--train-time', type=int, default=120,
                        help='Training time per seed in seconds (default: 120)')
    parser.add_argument('--num-restarts', type=int, default=20,
                        help='Number of V3 restarts to try (default: 20)')
    parser.add_argument('--ws-url', type=str,
                        help='WebSocket URL for live iterate pipeline')
    parser.add_argument('--time-budget', type=int, default=275,
                        help='Total time budget for live pipeline (default: 275)')
    parser.add_argument('--no-record', action='store_true',
                        help='Skip PostgreSQL recording')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.ws_url:
        live_pipeline(args.ws_url, args.time_budget, args.verbose)
        return

    if args.seeds:
        from configs import parse_seeds
        seeds = parse_seeds(args.seeds)
    else:
        seeds = [args.seed]

    scores = []
    t0 = time.time()
    for seed in seeds:
        st = time.time()
        score, _ = train_sim(seed, args.train_time, args.verbose, args.no_record)
        elapsed = time.time() - st
        scores.append(score)
        print(f"Seed {seed}: {score} ({elapsed:.1f}s)")

    if len(scores) > 1:
        print(f"\n{'='*40}")
        print(f"Mean: {np.mean(scores):.1f}  Max: {max(scores)}  Min: {min(scores)}")
        print(f"Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
