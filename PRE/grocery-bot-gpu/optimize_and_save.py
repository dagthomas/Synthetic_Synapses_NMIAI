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
parser.add_argument('--refine-iters', type=int, default=2)
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

from gpu_sequential_solver import solve_sequential

kwargs = {
    'capture_data': capture,
    'difficulty': args.difficulty,
    'device': 'cuda',
    'verbose': True,
    'no_filler': True,
    'no_compile': True,
    'max_time_s': args.max_time,
    'max_refine_iters': args.refine_iters,
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
