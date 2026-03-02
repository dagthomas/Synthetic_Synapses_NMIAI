"""Run 40-seed sweeps for all difficulties using parallel optimizer.

Usage:
    python sweep_all.py                     # All difficulties, 60s/seed, 12 workers
    python sweep_all.py expert              # Single difficulty
    python sweep_all.py expert --time 120   # More time per seed
    python sweep_all.py --seeds 10          # Fewer seeds for quick test
"""
import sys
import time
import statistics

from multi_solve import multi_solve


def sweep(difficulty, seeds=range(7001, 7041), time_per_seed=60.0,
          num_workers=None, verbose=True):
    """Sweep multiple seeds for one difficulty."""
    scores = []
    t0 = time.time()

    for seed in seeds:
        score, _ = multi_solve(seed, difficulty, time_limit=time_per_seed,
                               verbose=verbose, parallel=True,
                               num_workers=num_workers)
        scores.append(score)
        elapsed = time.time() - t0
        print(f"  seed={seed}: score={score}  (running mean={statistics.mean(scores):.1f}, "
              f"max={max(scores)}, {elapsed:.0f}s elapsed)")
        print()

    mx = max(scores)
    mn = min(scores)
    mean = statistics.mean(scores)
    total = time.time() - t0

    print(f"\n=== {difficulty.upper()} RESULTS ===")
    print(f"Max={mx}  Mean={mean:.1f}  Min={mn}  ({total:.0f}s total)")
    print(f"Scores: {scores}")
    return scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('difficulty', nargs='?', default='all',
                        choices=['easy', 'medium', 'hard', 'expert', 'all'])
    parser.add_argument('--time', type=float, default=60.0)
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    diffs = ['easy', 'medium', 'hard', 'expert'] if args.difficulty == 'all' else [args.difficulty]
    seeds = range(7001, 7001 + args.seeds)

    all_results = {}
    for diff in diffs:
        print(f"\n{'='*60}")
        print(f"  SWEEP: {diff.upper()} ({args.seeds} seeds, {args.time}s/seed)")
        print(f"{'='*60}")
        scores = sweep(diff, seeds, time_per_seed=args.time,
                       num_workers=args.workers, verbose=True)
        all_results[diff] = scores

    if len(diffs) > 1:
        print(f"\n{'='*60}")
        print(f"  OVERALL RESULTS")
        print(f"{'='*60}")
        total_max = 0
        total_mean = 0
        for diff in diffs:
            s = all_results[diff]
            mx = max(s)
            mean = statistics.mean(s)
            total_max += mx
            total_mean += mean
            print(f"  {diff:8s}: max={mx:3d}  mean={mean:6.1f}  min={min(s):3d}")
        print(f"  {'TOTAL':8s}: max_sum={total_max}  mean_sum={total_mean:.1f}")
