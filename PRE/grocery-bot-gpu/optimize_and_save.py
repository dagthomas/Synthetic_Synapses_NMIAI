#!/usr/bin/env python3
"""Offline GPU optimize: load capture -> solve -> save solution.

Outputs JSON events to stdout for SSE streaming.
Usage: python optimize_and_save.py <difficulty> [--max-time 60] [--max-states 50000]
"""
import sys
import json
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
parser.add_argument('--max-time', type=int, default=60, help='Time budget in seconds')
parser.add_argument('--max-states', type=int, default=None)
parser.add_argument('--refine-iters', type=int, default=20)
args = parser.parse_args()

def emit(event):
    print(json.dumps(event), flush=True)

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
    'no_compile': True,
    'max_time_s': args.max_time,
    'max_refine_iters': args.refine_iters,
    'on_bot_progress': on_bot_progress,
    'on_phase': on_phase,
}
if args.max_states:
    kwargs['max_states'] = args.max_states

t0 = time.time()

try:
    score, actions = solve_sequential(**kwargs)
except Exception as e:
    emit({"type": "optimize_done", "score": 0, "error": str(e),
          "elapsed": round(time.time() - t0, 1)})
    sys.exit(1)

# Warm-start: if existing solution (e.g., from game actions) is better,
# try refining from it to beat cold-start result
existing_actions = load_solution(args.difficulty)
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
                no_filler=True, no_compile=True,
                max_time_s=remaining, max_refine_iters=args.refine_iters,
                on_bot_progress=on_bot_progress, on_phase=on_phase)
            if ref_score > score:
                emit({"type": "gpu_phase", "phase": "warm_refine_improved",
                      "iteration": 0, "score": ref_score,
                      "elapsed": round(time.time() - t0, 1)})
                score, actions = ref_score, ref_actions
        except Exception as e:
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
