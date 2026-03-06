"""DEPRECATED: Batch solve: all difficulties, multiple seeds.

This module is not part of the active production pipeline. Kept for reference.

Usage:
    python solve_all.py --beam-width 100 --seeds 7001-7040
    python solve_all.py --difficulty easy --beam-width 500
"""
import argparse
import json
import os
import time
from beam_search import beam_search
from ws_client import save_actions


def parse_seed_range(s):
    """Parse seed range like '7001-7040' or '7001,7005,7010'."""
    if '-' in s:
        start, end = s.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Batch Grocery Bot Solver')
    parser.add_argument('--difficulty', '-d', nargs='+',
                        default=['easy', 'medium', 'hard', 'expert'],
                        choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seeds', type=str, default='7001-7040',
                        help='Seed range (e.g., 7001-7040 or 7001,7005)')
    parser.add_argument('--beam-width', type=int, default=100)
    parser.add_argument('--max-per-bot', type=int, default=3)
    parser.add_argument('--max-joint', type=int, default=500)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    sol_dir = os.path.join(os.path.dirname(__file__), 'solutions')
    os.makedirs(sol_dir, exist_ok=True)

    results = {}
    for diff in args.difficulty:
        results[diff] = {'scores': [], 'times': []}
        print(f"\n{'='*60}")
        print(f"  {diff.upper()} - {len(seeds)} seeds, beam_width={args.beam_width}")
        print(f"{'='*60}")

        for seed in seeds:
            t0 = time.time()
            score, actions, stats = beam_search(
                seed, diff,
                beam_width=args.beam_width,
                max_per_bot=args.max_per_bot,
                max_joint=args.max_joint,
                verbose=not args.quiet,
            )
            dt = time.time() - t0
            results[diff]['scores'].append(score)
            results[diff]['times'].append(dt)

            # Save solution
            sol_path = os.path.join(sol_dir, f'{diff}_{seed}.json')
            save_actions(actions, sol_path)

            print(f"  {diff} seed={seed}: score={score} ({dt:.1f}s)")

        scores = results[diff]['scores']
        print(f"\n  {diff.upper()} Summary:")
        print(f"    Mean: {sum(scores)/len(scores):.1f}")
        print(f"    Max:  {max(scores)}")
        print(f"    Min:  {min(scores)}")
        print(f"    Total time: {sum(results[diff]['times']):.1f}s")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")
    total_max = 0
    for diff in args.difficulty:
        scores = results[diff]['scores']
        mx = max(scores)
        total_max += mx
        print(f"  {diff:8s}: mean={sum(scores)/len(scores):6.1f}  max={mx:3d}  min={min(scores):3d}")
    print(f"  {'TOTAL':8s}: max_sum={total_max}")

    # Save results summary
    summary_path = os.path.join(sol_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'beam_width': args.beam_width,
            'seeds': seeds,
            'results': {d: {'scores': results[d]['scores']} for d in results},
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
