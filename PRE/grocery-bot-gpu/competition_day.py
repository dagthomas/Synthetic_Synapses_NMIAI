#!/usr/bin/env python3
"""Competition day script: exploit 24-hour determinism for unlimited offline training.

The "Time Millionaire" strategy: orders are deterministic per day, so we can:
  1. Play initial game to discover orders
  2. Train deeply offline (no time limit — minutes to hours)
  3. Fetch new token, replay to discover more orders
  4. Train again with expanded order set
  5. Repeat until score converges

Between token windows we have UNLIMITED time for offline training.
Each replay discovers 2-5 new orders → re-train → higher score → more orders.

Usage:
    python competition_day.py hard                    # Interactive (prompts for WS URLs)
    python competition_day.py hard --auto-token       # Auto-fetch tokens
    python competition_day.py hard --max-cycles 20    # Limit discovery cycles
    python competition_day.py expert --deep-states 200000 --deep-refine 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_state(difficulty: str) -> dict:
    """Load current solution state: orders known, best score, capture exists."""
    from solution_store import load_capture, load_meta
    capture = load_capture(difficulty)
    meta = load_meta(difficulty)
    n_orders = len(capture.get('orders', [])) if capture else 0
    score = meta.get('score', 0) if meta else 0
    return {
        'orders': n_orders,
        'score': score,
        'has_capture': capture is not None,
        'capture': capture,
    }


def get_ws_url(difficulty: str, auto_token: bool = False) -> Optional[str]:
    """Get a WebSocket URL, either auto-fetched or from user input."""
    if auto_token:
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, 'fetch_token.py'),
                 difficulty, '--json'],
                capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    try:
                        data = json.loads(line)
                        ws_url = data.get('ws_url')
                        if ws_url:
                            exp = data.get('expires_in', '?')
                            print(f"  Auto-fetched token (expires in {exp}s)",
                                  file=sys.stderr)
                            return ws_url
                    except json.JSONDecodeError:
                        pass
            print(f"  Auto-fetch failed (exit {result.returncode})", file=sys.stderr)
            if result.stderr:
                print(f"  {result.stderr[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"  Auto-fetch error: {e}", file=sys.stderr)

    # Manual fallback
    print(f"\nPaste WebSocket URL for {difficulty} (or 'q' to quit):")
    url = input().strip()
    if url.lower() in ('q', 'quit', 'exit', ''):
        return None
    return url


def run_initial_pipeline(ws_url: str, difficulty: str, max_states: int = 50000) -> int:
    """Run quick pipeline within one token to bootstrap order discovery.

    Returns best score achieved.
    """
    from production_run import (
        run_zig_bot, gpu_optimize, replay_solution_ws,
        capture_from_log, import_log_to_db,
    )

    print(f"\n--- Initial Pipeline (Zig → GPU → Replay → Repeat) ---",
          file=sys.stderr)

    t0 = time.time()
    best_score = 0

    # Phase 1: Zig bot
    score, log_path, elapsed = run_zig_bot(ws_url, difficulty)
    best_score = max(best_score, score)
    if log_path:
        import_log_to_db(log_path, run_type='zig')
        capture_from_log(log_path, difficulty)

    # Quick iterate: GPU optimize + replay, 3 times
    for i in range(3):
        remaining = 275 - (time.time() - t0)
        if remaining < 30:
            break

        gpu_time = min(25, remaining - 20)
        print(f"\n  Quick iter {i+1}/3 (budget={gpu_time:.0f}s)...", file=sys.stderr)

        opt_score, _ = gpu_optimize(
            difficulty, max_states=max_states, max_time_s=gpu_time,
            warm_only=(i > 0), orderings=2 if i == 0 else 1,
            refine_iters=2, speed_bonus=100.0)
        best_score = max(best_score, opt_score)

        remaining = 275 - (time.time() - t0)
        if remaining < 15:
            break

        replay_score, replay_log, _ = replay_solution_ws(ws_url, difficulty)
        best_score = max(best_score, replay_score)
        if replay_log:
            import_log_to_db(replay_log, run_type='replay')
            capture_from_log(replay_log, difficulty)

    return best_score


def train_deep(difficulty: str, max_states: int = 100000,
               refine_iters: int = 30, orderings: int = 3,
               speed_bonus: float = 150.0, max_dp_bots: Optional[int] = None,
               max_time: Optional[float] = None) -> int:
    """Deep offline training with no time pressure.

    Returns best score achieved.
    """
    from gpu_sequential_solver import solve_sequential, refine_from_solution
    from solution_store import load_capture, save_solution, load_meta, load_solution

    capture = load_capture(difficulty)
    if not capture:
        print(f"  No capture data for {difficulty}", file=sys.stderr)
        return 0

    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    n_orders = len(capture.get('orders', []))

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"DEEP TRAINING: {difficulty}", file=sys.stderr)
    print(f"  Orders: {n_orders}", file=sys.stderr)
    print(f"  Current best: {prev_score}", file=sys.stderr)
    print(f"  States: {max_states:,}", file=sys.stderr)
    print(f"  Refine iters: {refine_iters}", file=sys.stderr)
    print(f"  Orderings: {orderings}", file=sys.stderr)
    if max_time:
        print(f"  Time limit: {max_time:.0f}s", file=sys.stderr)
    else:
        print(f"  Time limit: unlimited", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    t0 = time.time()
    best_score = prev_score

    # Phase 1: Cold-start solve
    print(f"\n--- Cold start ({orderings} orderings, {max_states//1000}K states) ---",
          file=sys.stderr)
    kwargs = {
        'capture_data': capture,
        'difficulty': difficulty,
        'device': 'cuda',
        'verbose': True,
        'no_filler': True,
        'max_states': max_states,
        'num_pass1_orderings': orderings,
        'max_refine_iters': refine_iters,
        'speed_bonus': speed_bonus,
    }
    if max_dp_bots is not None:
        kwargs['max_dp_bots'] = max_dp_bots
    if max_time is not None:
        kwargs['max_time_s'] = max_time * 0.6  # 60% for cold start

    try:
        score, actions = solve_sequential(**kwargs)
        if score > 0:
            saved = save_solution(difficulty, score, actions)
            best_score = max(best_score, score)
            print(f"\n  Cold start: score={score} (saved={saved})", file=sys.stderr)
    except Exception as e:
        print(f"\n  Cold start failed: {e}", file=sys.stderr)

    # Phase 2: Warm-start refinement from best solution
    existing = load_solution(difficulty)
    if existing:
        remaining_time = None
        if max_time is not None:
            remaining_time = max_time - (time.time() - t0)
            if remaining_time < 10:
                print(f"\n  No time for warm refinement", file=sys.stderr)
                return best_score

        print(f"\n--- Warm refinement (from existing best={best_score}) ---",
              file=sys.stderr)
        try:
            ref_kwargs = {
                'capture_data': capture,
                'difficulty': difficulty,
                'device': 'cuda',
                'no_filler': True,
                'max_states': max_states,
                'max_refine_iters': refine_iters,
                'speed_bonus': speed_bonus,
            }
            if max_dp_bots is not None:
                ref_kwargs['max_dp_bots'] = max_dp_bots
            if remaining_time is not None:
                ref_kwargs['max_time_s'] = remaining_time

            ref_score, ref_actions = refine_from_solution(existing, **ref_kwargs)
            if ref_score > 0:
                saved = save_solution(difficulty, ref_score, ref_actions)
                best_score = max(best_score, ref_score)
                print(f"\n  Warm refine: score={ref_score} (saved={saved})",
                      file=sys.stderr)
        except Exception as e:
            print(f"\n  Warm refine failed: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n  Deep training complete: {prev_score} → {best_score} "
          f"(+{best_score - prev_score}) in {elapsed:.0f}s", file=sys.stderr)
    return best_score


def replay_and_discover(ws_url: str, difficulty: str) -> tuple[int, int]:
    """Replay best solution and capture new orders.

    Returns (replay_score, new_order_count).
    """
    from production_run import replay_solution_ws, capture_from_log, import_log_to_db

    state_before = load_state(difficulty)
    replay_score, log_path, elapsed = replay_solution_ws(ws_url, difficulty)

    new_orders = 0
    if log_path:
        import_log_to_db(log_path, run_type='replay')
        new_orders = capture_from_log(log_path, difficulty)

    state_after = load_state(difficulty)
    orders_gained = state_after['orders'] - state_before['orders']

    print(f"\n  Replay: score={replay_score}, "
          f"orders {state_before['orders']}→{state_after['orders']} "
          f"(+{orders_gained})", file=sys.stderr)

    return replay_score, state_after['orders']


def main():
    parser = argparse.ArgumentParser(
        description='Competition day: unlimited offline training with stepladder order discovery')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--max-cycles', type=int, default=50,
                        help='Max discovery cycles (default: 50)')
    parser.add_argument('--deep-states', type=int, default=100000,
                        help='Max states for deep training (default: 100K)')
    parser.add_argument('--deep-refine', type=int, default=30,
                        help='Refine iterations for deep training (default: 30)')
    parser.add_argument('--deep-orderings', type=int, default=3,
                        help='Pass1 orderings for deep training (default: 3)')
    parser.add_argument('--speed-bonus', type=float, default=150.0,
                        help='Speed bonus for deep training (default: 150)')
    parser.add_argument('--max-dp-bots', type=int, default=None,
                        help='Max bots to DP plan (rest get CPU greedy)')
    parser.add_argument('--auto-token', action='store_true',
                        help='Auto-fetch tokens via fetch_token.py')
    parser.add_argument('--deep-time', type=float, default=None,
                        help='Max seconds per deep training pass (default: unlimited)')
    parser.add_argument('--pipeline-states', type=int, default=50000,
                        help='Max states for quick pipeline iterations (default: 50K)')
    args = parser.parse_args()

    diff = args.difficulty
    t_start = time.time()
    history = []  # (cycle, orders, score, elapsed)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"COMPETITION DAY — {diff.upper()}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Strategy: Stepladder order discovery + deep offline training",
          file=sys.stderr)
    print(f"  Deep training: {args.deep_states//1000}K states, "
          f"{args.deep_refine} refine iters, {args.deep_orderings} orderings",
          file=sys.stderr)
    print(f"  Token mode: {'auto' if args.auto_token else 'manual'}",
          file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    state = load_state(diff)
    print(f"  Current state: {state['orders']} orders, score={state['score']}",
          file=sys.stderr)

    # Phase 1: Bootstrap — need initial capture if none exists
    if not state['has_capture']:
        print(f"\n--- Phase 1: Initial Capture ---", file=sys.stderr)
        print(f"  No capture data. Need to play initial game.", file=sys.stderr)
        ws_url = get_ws_url(diff, args.auto_token)
        if not ws_url:
            print("Aborted.", file=sys.stderr)
            return

        score = run_initial_pipeline(ws_url, diff, max_states=args.pipeline_states)
        state = load_state(diff)
        history.append((0, state['orders'], state['score'],
                        time.time() - t_start))
        print(f"\n  After bootstrap: {state['orders']} orders, "
              f"score={state['score']}", file=sys.stderr)

    # Phase 2: Stepladder discovery cycles
    prev_orders = state['orders']
    stall_count = 0

    for cycle in range(1, args.max_cycles + 1):
        state = load_state(diff)
        elapsed = time.time() - t_start

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"CYCLE {cycle}/{args.max_cycles} — "
              f"{state['orders']} orders, score={state['score']}, "
              f"elapsed={elapsed/60:.0f}m", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Step A: Deep offline training
        score = train_deep(
            diff,
            max_states=args.deep_states,
            refine_iters=args.deep_refine,
            orderings=args.deep_orderings,
            speed_bonus=args.speed_bonus,
            max_dp_bots=args.max_dp_bots,
            max_time=args.deep_time,
        )

        state = load_state(diff)
        print(f"\n  After training: score={state['score']}", file=sys.stderr)

        # Step B: Replay to discover more orders
        print(f"\n--- Replay (cycle {cycle}) ---", file=sys.stderr)
        ws_url = get_ws_url(diff, args.auto_token)
        if not ws_url:
            print(f"\n  No URL provided. Stopping.", file=sys.stderr)
            break

        replay_score, new_order_count = replay_and_discover(ws_url, diff)

        state = load_state(diff)
        orders_gained = state['orders'] - prev_orders
        history.append((cycle, state['orders'], state['score'],
                        time.time() - t_start))

        # Step C: Check progress
        if orders_gained > 0:
            print(f"\n  +{orders_gained} new orders! "
                  f"({prev_orders}→{state['orders']})", file=sys.stderr)
            stall_count = 0
            prev_orders = state['orders']
        else:
            stall_count += 1
            print(f"\n  No new orders (stall {stall_count}/3)", file=sys.stderr)

            if stall_count >= 3:
                print(f"\n  Order discovery stalled after 3 consecutive cycles.",
                      file=sys.stderr)
                print(f"  Options: (c)ontinue training, (r)eplay again, (q)uit",
                      file=sys.stderr)
                choice = input().strip().lower()
                if choice == 'q':
                    break
                elif choice == 'r':
                    stall_count = 0  # Reset and try replay again
                    continue
                else:
                    stall_count = 0  # Reset and continue training

    # Summary
    total_time = time.time() - t_start
    state = load_state(diff)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"COMPETITION DAY COMPLETE — {diff.upper()}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"\n{'Cycle':>6} {'Orders':>7} {'Score':>6} {'Elapsed':>10}",
          file=sys.stderr)
    print(f"{'-'*33}", file=sys.stderr)
    for cycle, orders, score, elapsed in history:
        mins = elapsed / 60
        print(f"{cycle:>6} {orders:>7} {score:>6} {mins:>9.1f}m",
              file=sys.stderr)

    print(f"\nFinal: {state['orders']} orders, score={state['score']}",
          file=sys.stderr)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)",
          file=sys.stderr)

    # Machine-readable output
    print(json.dumps({
        'type': 'competition_day_complete',
        'difficulty': diff,
        'final_score': state['score'],
        'final_orders': state['orders'],
        'cycles': len(history),
        'total_time': round(total_time, 1),
        'history': history,
    }))


if __name__ == '__main__':
    main()
