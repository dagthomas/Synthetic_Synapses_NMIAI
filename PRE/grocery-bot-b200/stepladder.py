"""Multi-hour stepladder pipeline — day-long automated optimization.

Exploits 24-hour game determinism: same day = same map + orders.
Each token window (288s) discovers more orders via replay, then
deep offline training between windows uses the expanded order set.

State machine: INIT -> FETCH_TOKEN -> CAPTURE/REPLAY -> ITERATE -> DEEP_TRAIN -> loop

Usage:
    python stepladder.py hard --hours 6 --max-states 500000
    python stepladder.py expert --hours 12 --max-states 200000
    python stepladder.py hard --resume
    python stepladder.py hard --final-replay
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import _shared  # noqa: F401
from b200_config import get_params, detect_gpu, print_gpu_info
from deep_optimize import deep_optimize
from solution_store import (
    save_solution, load_capture, load_solution, load_meta, merge_capture,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'grocery-bot-gpu'))
STATE_DIR = os.path.join(SCRIPT_DIR, 'stepladder_state')


def _state_path(difficulty: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f'{difficulty}.json')


def _save_state(difficulty: str, state: dict):
    with open(_state_path(difficulty), 'w') as f:
        json.dump(state, f, indent=2)


def _load_state(difficulty: str) -> dict | None:
    path = _state_path(difficulty)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _fetch_token(difficulty: str) -> str | None:
    """Fetch a game token with cooldown awareness."""
    try:
        from fetch_token import fetch_token
        return fetch_token(difficulty, timeout_s=90)
    except Exception as e:
        print(f"  Token fetch failed: {e}", file=sys.stderr)
        return None


def _run_zig_capture(ws_url: str, difficulty: str) -> tuple[int, int]:
    """Run Zig bot for initial capture. Returns (score, num_orders_discovered)."""
    try:
        sys.path.insert(0, GPU_DIR)
        from production_run import run_zig_bot, capture_from_log
        score, log_path, elapsed = run_zig_bot(ws_url, difficulty)
        n_orders = 0
        if log_path:
            n_orders = capture_from_log(log_path, difficulty)
        return score, n_orders
    except Exception as e:
        print(f"  Zig capture failed: {e}", file=sys.stderr)
        return 0, 0


def _replay_and_discover(ws_url: str, difficulty: str) -> tuple[int, int]:
    """Replay current best solution, discover new orders. Returns (score, new_orders)."""
    try:
        sys.path.insert(0, GPU_DIR)
        from production_run import replay_solution_ws, capture_from_log
        score, log_path, elapsed = replay_solution_ws(ws_url, difficulty)
        n_new = 0
        if log_path:
            n_new = capture_from_log(log_path, difficulty)
        return score, n_new
    except Exception as e:
        print(f"  Replay failed: {e}", file=sys.stderr)
        return 0, 0


def _gpu_optimize_quick(difficulty: str, max_states: int,
                        max_time_s: float) -> tuple[int, float]:
    """Quick GPU optimization within token window."""
    try:
        sys.path.insert(0, GPU_DIR)
        from production_run import gpu_optimize
        return gpu_optimize(
            difficulty, max_states=max_states, max_time_s=max_time_s,
            speed_bonus=100.0)
    except Exception as e:
        print(f"  GPU optimize failed: {e}", file=sys.stderr)
        return 0, 0.0


def run_stepladder(difficulty: str, hours: float = 6.0,
                   max_states: int | None = None,
                   gpu: str = 'auto',
                   resume: bool = False,
                   final_replay_only: bool = False):
    """Run stepladder pipeline for given hours."""
    gpu_type = gpu if gpu != 'auto' else detect_gpu()
    params = get_params(difficulty, gpu_type)
    if max_states:
        params.max_states = max_states

    t0 = time.time()
    deadline = t0 + hours * 3600

    # Load or init state
    state = _load_state(difficulty) if resume else None
    if state is None:
        state = {
            'phase': 'init',
            'iteration': 0,
            'scores': [],
            'orders_discovered': [],
            'best_score': 0,
            'last_token_fetch': 0,
        }

    if final_replay_only:
        print(f"Final replay for {difficulty}...", file=sys.stderr)
        ws_url = _fetch_token(difficulty)
        if ws_url:
            score, _ = _replay_and_discover(ws_url, difficulty)
            print(f"Final replay score: {score}", file=sys.stderr)
        return

    print(f"\nStepladder: {difficulty}", file=sys.stderr)
    print(f"  Hours: {hours}, GPU: {gpu_type}", file=sys.stderr)
    print(f"  max_states={params.max_states:,}, joint_states={params.joint_states:,}",
          file=sys.stderr)
    print(f"  Starting iteration: {state['iteration']}", file=sys.stderr)

    while time.time() < deadline:
        remaining_hours = (deadline - time.time()) / 3600
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Iteration {state['iteration']}, "
              f"{remaining_hours:.1f}h remaining", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        # --- 1. Fetch token (with 60s cooldown) ---
        now = time.time()
        cooldown_remaining = max(0, 65 - (now - state['last_token_fetch']))
        if cooldown_remaining > 0:
            print(f"  Cooldown: {cooldown_remaining:.0f}s...", file=sys.stderr)
            time.sleep(cooldown_remaining)

        ws_url = _fetch_token(difficulty)
        state['last_token_fetch'] = time.time()
        _save_state(difficulty, state)

        if not ws_url:
            print(f"  Token fetch failed, waiting 120s...", file=sys.stderr)
            time.sleep(120)
            continue

        token_deadline = time.time() + 270  # 270s safe window (288s token - margin)

        # --- 2. Capture (first iter) or Replay (subsequent) ---
        has_capture = load_capture(difficulty) is not None
        has_solution = load_solution(difficulty) is not None

        if state['iteration'] == 0 and not has_capture:
            print(f"  Initial capture via Zig bot...", file=sys.stderr)
            score, n_orders = _run_zig_capture(ws_url, difficulty)
            state['scores'].append(score)
            state['orders_discovered'].append(n_orders)
        elif has_solution:
            print(f"  Replaying best solution...", file=sys.stderr)
            score, n_new = _replay_and_discover(ws_url, difficulty)
            state['scores'].append(score)
            state['orders_discovered'].append(n_new)
            if score > state['best_score']:
                state['best_score'] = score
        else:
            # Have capture but no solution — just do a quick Zig run
            print(f"  Zig bot (no solution yet)...", file=sys.stderr)
            score, n_orders = _run_zig_capture(ws_url, difficulty)
            state['scores'].append(score)
            state['orders_discovered'].append(n_orders)

        _save_state(difficulty, state)

        # --- 3. Iterate within token window (fast GPU passes) ---
        token_remaining = token_deadline - time.time()
        if token_remaining > 30:
            print(f"  GPU optimize ({token_remaining:.0f}s budget)...", file=sys.stderr)
            gpu_score, gpu_time = _gpu_optimize_quick(
                difficulty, params.max_states, max_time_s=token_remaining - 10)

            if gpu_score > state['best_score']:
                state['best_score'] = gpu_score
                print(f"  GPU optimize: {gpu_score}", file=sys.stderr)

            # Quick replay if time permits
            token_remaining = token_deadline - time.time()
            if token_remaining > 60 and load_solution(difficulty):
                print(f"  Quick replay ({token_remaining:.0f}s left)...", file=sys.stderr)
                # Would need a new token here — skip if no time
                pass

        _save_state(difficulty, state)

        # --- 4. Deep train offline (hours-long, no token needed) ---
        remaining_to_deadline = deadline - time.time()
        # Reserve 5 min for final token window
        deep_budget = remaining_to_deadline - 400

        if deep_budget > 120:
            # Scale deep budget: more time in later iterations (order set is larger)
            # First iteration: 30min max. Later: up to 2h.
            iter_factor = min(2.0, 0.5 + state['iteration'] * 0.3)
            deep_budget = min(deep_budget, iter_factor * 3600)

            print(f"\n  Deep training ({deep_budget:.0f}s = {deep_budget/3600:.1f}h)...",
                  file=sys.stderr)
            deep_optimize(
                difficulty,
                budget_s=deep_budget,
                max_states=params.max_states,
                joint_states=params.joint_states,
                gpu=gpu_type,
            )

            meta = load_meta(difficulty)
            if meta and meta.get('score', 0) > state['best_score']:
                state['best_score'] = meta['score']

        state['iteration'] += 1
        _save_state(difficulty, state)

    # --- Final replay ---
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Final replay for {difficulty}...", file=sys.stderr)

    ws_url = _fetch_token(difficulty)
    if ws_url and load_solution(difficulty):
        score, _ = _replay_and_discover(ws_url, difficulty)
        print(f"Final score: {score}", file=sys.stderr)
        if score > state['best_score']:
            state['best_score'] = score
    else:
        print(f"  No solution to replay or token failed", file=sys.stderr)

    _save_state(difficulty, state)

    print(f"\nStepladder complete: {difficulty}", file=sys.stderr)
    print(f"  Iterations: {state['iteration']}", file=sys.stderr)
    print(f"  Best score: {state['best_score']}", file=sys.stderr)
    print(f"  Score history: {state['scores']}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stepladder pipeline')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--hours', type=float, default=6.0)
    parser.add_argument('--max-states', type=int, default=None)
    parser.add_argument('--gpu', default='auto', choices=['auto', 'b200', '5090', 'generic'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--final-replay', action='store_true')
    args = parser.parse_args()

    print_gpu_info()
    run_stepladder(
        args.difficulty,
        hours=args.hours,
        max_states=args.max_states,
        gpu=args.gpu,
        resume=args.resume,
        final_replay_only=args.final_replay,
    )
