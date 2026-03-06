"""March 19 competition day automation — orchestrates all 4 difficulties.

B200 schedule (sequential, max GPU utilization):
  07:00-07:05   Easy: single-bot DP (trivially optimal)
  07:05-08:30   Medium: stepladder 1h25m
  08:30-13:00   Hard: stepladder 4.5h
  13:00-20:00   Expert: stepladder 7h
  20:00-20:30   Final replays all difficulties

Usage:
    python competition_day.py --gpu b200
    python competition_day.py --gpu b200 --config schedule.json
    python competition_day.py --difficulty hard
    python competition_day.py --final-replays
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import _shared  # noqa: F401
from b200_config import get_params, detect_gpu, print_gpu_info
from stepladder import run_stepladder, _fetch_token, _replay_and_discover
from deep_optimize import deep_optimize
from solution_store import load_meta, load_solution

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default schedule (hours allocated per difficulty)
DEFAULT_SCHEDULE = {
    'b200': {
        'easy': 0.1,      # 6 min — trivially optimal
        'medium': 1.4,     # 1h24m
        'hard': 4.5,       # 4h30m
        'expert': 7.0,     # 7h
    },
    '5090': {
        'easy': 0.1,
        'medium': 1.0,
        'hard': 3.0,
        'expert': 4.0,
    },
}


def run_competition_day(gpu: str = 'auto',
                        schedule: dict | None = None,
                        difficulty: str | None = None,
                        final_replays: bool = False):
    """Run the full competition day pipeline."""
    gpu_type = gpu if gpu != 'auto' else detect_gpu()
    if schedule is None:
        schedule = DEFAULT_SCHEDULE.get(gpu_type, DEFAULT_SCHEDULE['5090'])

    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"COMPETITION DAY — {now_str}", file=sys.stderr)
    print(f"GPU: {gpu_type}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if final_replays:
        _do_final_replays()
        return

    difficulties = ['easy', 'medium', 'hard', 'expert']
    if difficulty:
        difficulties = [difficulty]

    for diff in difficulties:
        hours = schedule.get(diff, 1.0)
        if hours <= 0:
            continue

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Starting {diff.upper()} — {hours:.1f}h budget", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        t0 = time.time()

        if diff == 'easy' and hours < 0.5:
            # Easy: just run deep_optimize for a few minutes
            deep_optimize(diff, budget_s=hours * 3600, gpu=gpu_type)
        else:
            run_stepladder(diff, hours=hours, gpu=gpu_type)

        elapsed = time.time() - t0
        meta = load_meta(diff)
        score = meta.get('score', 0) if meta else 0
        print(f"\n{diff.upper()} complete: score={score}, time={elapsed/3600:.1f}h",
              file=sys.stderr)

    # Final replays
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"FINAL REPLAYS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    _do_final_replays()

    # Summary
    _print_summary()


def _do_final_replays():
    """Replay best solutions for all difficulties."""
    for diff in ['easy', 'medium', 'hard', 'expert']:
        if not load_solution(diff):
            print(f"  {diff}: no solution", file=sys.stderr)
            continue

        meta = load_meta(diff)
        expected = meta.get('score', 0) if meta else 0

        ws_url = _fetch_token(diff)
        if ws_url:
            score, _ = _replay_and_discover(ws_url, diff)
            print(f"  {diff}: expected={expected}, actual={score}", file=sys.stderr)
            # Wait for cooldown
            time.sleep(65)
        else:
            print(f"  {diff}: token failed", file=sys.stderr)
            time.sleep(65)


def _print_summary():
    """Print final score summary."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"FINAL SCORES", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    total = 0
    for diff in ['easy', 'medium', 'hard', 'expert']:
        meta = load_meta(diff)
        score = meta.get('score', 0) if meta else 0
        total += score
        print(f"  {diff:8s}: {score:4d}", file=sys.stderr)
    print(f"  {'TOTAL':8s}: {total:4d}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Competition day automation')
    parser.add_argument('--gpu', default='auto', choices=['auto', 'b200', '5090', 'generic'])
    parser.add_argument('--config', type=str, default=None,
                        help='JSON schedule config file')
    parser.add_argument('--difficulty', type=str, default=None,
                        choices=['easy', 'medium', 'hard', 'expert'],
                        help='Run single difficulty')
    parser.add_argument('--final-replays', action='store_true',
                        help='Just do final replays')
    args = parser.parse_args()

    schedule = None
    if args.config:
        with open(args.config) as f:
            schedule = json.load(f)

    print_gpu_info()
    run_competition_day(
        gpu=args.gpu,
        schedule=schedule,
        difficulty=args.difficulty,
        final_replays=args.final_replays,
    )
