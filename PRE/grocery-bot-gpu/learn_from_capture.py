"""Learn from a captured game: run parallel optimizer on saved capture data.

Loads capture from solutions/<difficulty>/capture.json, runs heavy optimization,
and saves improved solution if score beats existing best.

Usage:
    python learn_from_capture.py <difficulty> [--time 120] [--workers 12]
"""
import argparse
import os
import sys
import time

from solution_store import load_capture, load_meta, save_solution, increment_optimizations


def learn(difficulty, time_limit=120.0, num_workers=None):
    """Run optimizer on saved capture, update best solution if improved."""
    capture = load_capture(difficulty)
    if capture is None:
        print(f"ERROR: No capture found for {difficulty}", file=sys.stderr)
        print(f"Run a game first: python live_solver.py <url> --save-capture", file=sys.stderr)
        return None

    meta = load_meta(difficulty)
    existing_score = meta.get('score', 0) if meta else 0
    print(f"Learning: {difficulty} (existing score: {existing_score})", file=sys.stderr)

    if num_workers is None:
        num_workers = min(12, os.cpu_count() or 4)

    from game_engine import init_game_from_capture
    from configs import CONFIGS
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    game_factory = lambda: init_game_from_capture(capture)

    t0 = time.time()

    if num_bots == 1:
        # Single-bot: use multi_solve (beam search + planner + optimizer)
        from multi_solve import multi_solve
        score, actions = multi_solve(
            difficulty=difficulty,
            time_limit=time_limit,
            verbose=True,
            game_factory=game_factory,
            parallel=False,
        )
    else:
        # Multi-bot: use parallel optimizer
        from parallel_optimizer import parallel_optimize
        score, actions = parallel_optimize(
            capture_data=capture,
            difficulty=difficulty,
            time_limit=time_limit,
            num_workers=num_workers,
            verbose=True,
        )
    elapsed = time.time() - t0

    increment_optimizations(difficulty)

    if score > existing_score:
        save_solution(difficulty, score, actions)
        print(f"IMPROVED: {existing_score} -> {score} (+{score - existing_score}) in {elapsed:.1f}s", file=sys.stderr)
    else:
        print(f"NO IMPROVEMENT: {score} <= {existing_score} ({elapsed:.1f}s)", file=sys.stderr)

    print(f"LEARN_DONE score={score} prev={existing_score}", file=sys.stderr)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn from captured game')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--time', type=float, default=120.0, help='Time limit in seconds')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    learn(args.difficulty, time_limit=args.time, num_workers=args.workers)
