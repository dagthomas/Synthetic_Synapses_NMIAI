#!/usr/bin/env python3
"""Offline GPU optimize: load capture -> solve -> save solution.

Outputs JSON events to stdout for SSE streaming.
Usage: python optimize_and_save.py <difficulty> [--max-time 60] [--max-states 50000]

Error contract (script entry point):
  - Calls sys.exit(1) on fatal errors (missing capture, solver exception).
  - Emits {"type": "optimize_error", ...} or {"type": "optimize_done", "score": 0, ...}
    before exiting so SSE consumers get structured error info.
  - On success, emits {"type": "optimize_done", "score": N, ...} and exits 0.
"""
from __future__ import annotations

import sys
import json
import time
import argparse


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--max-time', type=int, default=20, help='Time budget in seconds')
    parser.add_argument('--max-states', type=int, default=None)
    parser.add_argument('--refine-iters', type=int, default=20)
    parser.add_argument('--orderings', type=int, default=None,
                        help='Pass1 orderings (default: 3 for hard/expert, 1 for easy/medium)')
    parser.add_argument('--warm-only', action='store_true',
                        help='Skip cold-start solve, only refine from existing solution')
    parser.add_argument('--speed-bonus', type=float, default=0.0,
                        help='Speed bonus coefficient (0=off, 100=aggressive)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (default: compile enabled for 3.5x speedup)')
    args = parser.parse_args()

    # Default orderings: multi-bot difficulties benefit from trying multiple bot orders
    if args.orderings is None:
        args.orderings = 3 if args.difficulty in ('hard', 'expert') else 1

    from solution_store import load_capture, save_solution, load_meta

    capture = load_capture(args.difficulty)
    if not capture:
        emit({"type": "optimize_error", "message": f"No capture data for {args.difficulty}"})
        sys.exit(1)

    meta = load_meta(args.difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    n_orders = len(capture.get('orders', []))

    emit({
        "type": "optimize_start",
        "difficulty": args.difficulty,
        "max_time": args.max_time,
        "prev_score": prev_score,
        "orders": n_orders,
    })

    from gpu_sequential_solver import solve_sequential, refine_from_solution
    from solution_store import load_solution

    t0 = time.time()

    def on_bot_progress(bot_id, num_bots, score, bot_time):
        emit({"type": "gpu_bot_done", "bot": bot_id, "num_bots": num_bots,
              "score": score, "elapsed": round(bot_time, 1)})

    def on_phase(phase_name, iteration, cpu_score):
        emit({"type": "gpu_phase", "phase": phase_name, "iteration": iteration,
              "score": cpu_score, "elapsed": round(time.time() - t0, 1)})

    kwargs = {
        'capture_data': capture,
        'difficulty': args.difficulty,
        'device': 'cuda',
        'verbose': True,
        'no_filler': True,
        'no_compile': args.no_compile,
        'max_time_s': args.max_time,
        'max_refine_iters': args.refine_iters,
        'num_pass1_orderings': args.orderings,
        'on_bot_progress': on_bot_progress,
        'on_phase': on_phase,
        'speed_bonus': args.speed_bonus,
    }
    if args.max_states:
        kwargs['max_states'] = args.max_states

    existing_actions = load_solution(args.difficulty)

    if args.warm_only and existing_actions:
        # Warm-only mode: skip cold-start, refine existing solution with new orders
        emit({"type": "gpu_phase", "phase": "warm_only_refine",
              "iteration": 0, "score": prev_score,
              "elapsed": round(time.time() - t0, 1)})
        try:
            score, actions = refine_from_solution(
                existing_actions, capture_data=capture,
                difficulty=args.difficulty, device='cuda',
                no_filler=True, no_compile=args.no_compile,
                max_time_s=args.max_time, max_refine_iters=args.refine_iters,
                on_bot_progress=on_bot_progress, on_phase=on_phase,
                speed_bonus=args.speed_bonus)
        except Exception as e:
            emit({"type": "optimize_done", "score": 0, "error": str(e),
                  "elapsed": round(time.time() - t0, 1)})
            sys.exit(1)
    elif args.warm_only:
        # Warm-only requested but no existing solution — fall back to cold-start
        emit({"type": "gpu_phase", "phase": "warm_only_fallback",
              "iteration": 0, "score": 0,
              "elapsed": round(time.time() - t0, 1)})
        try:
            score, actions = solve_sequential(**kwargs)
        except Exception as e:
            emit({"type": "optimize_done", "score": 0, "error": str(e),
                  "elapsed": round(time.time() - t0, 1)})
            sys.exit(1)
    else:
        # Normal mode: cold-start + optional warm-start refinement
        try:
            score, actions = solve_sequential(**kwargs)
        except Exception as e:
            emit({"type": "optimize_done", "score": 0, "error": str(e),
                  "elapsed": round(time.time() - t0, 1)})
            sys.exit(1)

        # Warm-start: if existing solution is better, try refining it
        if existing_actions and prev_score > score:
            remaining = args.max_time - (time.time() - t0)
            if remaining > 10:
                emit({"type": "gpu_phase", "phase": "warm_refine",
                      "iteration": 0, "score": prev_score,
                      "elapsed": round(time.time() - t0, 1)})
                try:
                    ref_score, ref_actions = refine_from_solution(
                        existing_actions, capture_data=capture,
                        difficulty=args.difficulty, device='cuda',
                        no_filler=True, no_compile=args.no_compile,
                        max_time_s=remaining, max_refine_iters=args.refine_iters,
                        on_bot_progress=on_bot_progress, on_phase=on_phase,
                        speed_bonus=args.speed_bonus)
                    if ref_score > score:
                        emit({"type": "gpu_phase", "phase": "warm_refine_improved",
                              "iteration": 0, "score": ref_score,
                              "elapsed": round(time.time() - t0, 1)})
                        score, actions = ref_score, ref_actions
                except Exception:
                    emit({"type": "gpu_phase", "phase": "warm_refine_failed",
                          "iteration": 0, "score": score,
                          "elapsed": round(time.time() - t0, 1)})

    elapsed = time.time() - t0

    if score > 0:
        saved = save_solution(args.difficulty, score, actions)
        emit({
            "type": "optimize_done",
            "score": score,
            "prev_score": prev_score,
            "saved": saved,
            "elapsed": round(elapsed, 1),
            "orders": n_orders,
        })
    else:
        emit({
            "type": "optimize_done",
            "score": 0,
            "prev_score": prev_score,
            "saved": False,
            "elapsed": round(elapsed, 1),
            "error": "solver returned score 0",
        })


if __name__ == '__main__':
    main()
