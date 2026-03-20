"""Multi-dimensional parameter sweep for nightmare CascadeSolver (#6).

Sweeps combinations of PREFETCH_DEPTH, MAX_DELIVERY, STALL_LIMIT,
REASSIGN_COOLDOWN, and MAX_PREFETCH_CARRYING across multiple seeds.
Ranks by mean score.

Usage:
    python sweep_nightmare.py --seeds 1000-1009
    python sweep_nightmare.py --seeds 7005 --quick
    python sweep_nightmare.py --seeds 1000-1004 --param PREFETCH_DEPTH --values 4,6,8,10,12
"""
import argparse
import itertools
import sys
import time

import numpy as np

from configs import parse_seeds


# Default parameter grid (centered around known-optimal PD=4, MD=10, RC=3)
DEFAULT_GRID = {
    'PREFETCH_DEPTH': [3, 4, 5, 6],
    'MAX_DELIVERY': [8, 10, 12],
    'STALL_LIMIT': [3, 5],
    'REASSIGN_COOLDOWN': [2, 3, 5],
    'MAX_PREFETCH_CARRYING': [30, 40, 50],
}

QUICK_GRID = {
    'PREFETCH_DEPTH': [3, 4, 5],
    'MAX_DELIVERY': [8, 10],
    'STALL_LIMIT': [3],
    'REASSIGN_COOLDOWN': [3],
    'MAX_PREFETCH_CARRYING': [40],
}


def run_single(seed, params, live_map=None, verbose=False):
    """Run CascadeSolver with given params on a single seed."""
    from nightmare_cascade_solver import CascadeSolver
    score, _ = CascadeSolver.run_sim(
        seed, verbose=verbose, live_map=live_map, **params)
    return score


def run_sweep(seeds, param_grid, live_map=None, verbose=False, no_record=True):
    """Run full sweep and return sorted results."""
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combos = list(itertools.product(*param_values))

    total_runs = len(combos) * len(seeds)
    print(f"Sweeping {len(combos)} combos x {len(seeds)} seeds = {total_runs} runs",
          file=sys.stderr)

    results = []
    run_idx = 0

    for combo in combos:
        params = dict(zip(param_names, combo))
        seed_scores = {}
        t_combo = time.time()

        for seed in seeds:
            run_idx += 1
            param_str = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f"[{run_idx}/{total_runs}] seed={seed} | {param_str}",
                  file=sys.stderr)

            try:
                score = run_single(seed, params, live_map=live_map, verbose=verbose)
                seed_scores[seed] = score
                print(f"  -> {score}", file=sys.stderr)
            except Exception as e:
                print(f"  -> ERROR: {e}", file=sys.stderr)
                seed_scores[seed] = -1

        combo_time = time.time() - t_combo
        valid = [s for s in seed_scores.values() if s >= 0]
        mean_score = np.mean(valid) if valid else 0
        max_score = max(valid) if valid else 0
        min_score = min(valid) if valid else 0

        results.append((params, seed_scores, float(mean_score),
                        max_score, min_score, combo_time))

    # Sort by mean descending
    results.sort(key=lambda r: -r[2])
    return results


def print_results(results, seeds):
    """Print results table."""
    print(f"\n{'='*100}")
    print("NIGHTMARE PARAMETER SWEEP RESULTS")
    print(f"{'='*100}")

    # Header
    header = f"{'Parameters':55s}"
    for s in seeds:
        header += f" s{s:>5}"
    header += "   mean    max    min   time"
    print(header)
    print('-' * len(header))

    for params, seed_scores, mean, mx, mn, t in results:
        param_str = ', '.join(f'{k}={v}' for k, v in sorted(params.items()))
        if len(param_str) > 55:
            param_str = param_str[:52] + '...'
        line = f"{param_str:55s}"
        for s in seeds:
            sc = seed_scores.get(s, -1)
            line += f" {sc:>5}" if sc >= 0 else "   ERR"
        line += f" {mean:6.1f}  {mx:5}  {mn:5} {t:5.0f}s"
        print(line)

    if results:
        best = results[0]
        print(f"\nBest: mean={best[2]:.1f}, max={best[3]}, min={best[4]}")
        print(f"  Params: {best[0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for nightmare CascadeSolver')
    parser.add_argument('--seeds', default='1000-1004')
    parser.add_argument('--quick', action='store_true',
                        help='Quick sweep with fewer combos')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--no-live-map', action='store_true')
    parser.add_argument('--param', help='Single param to sweep')
    parser.add_argument('--values', help='Comma-separated values for --param')
    parser.add_argument('--no-record', action='store_true',
                        help='Skip PostgreSQL recording')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    # Load live map
    live_map = None
    if not args.no_live_map:
        try:
            from solution_store import load_capture
            from game_engine import build_map_from_capture
            cap = load_capture('nightmare')
            if cap and cap.get('grid'):
                live_map = build_map_from_capture(cap)
                print(f"Using live map: {live_map.width}x{live_map.height}",
                      file=sys.stderr)
        except Exception as e:
            print(f"Could not load live map: {e}", file=sys.stderr)

    # Build param grid
    if args.param and args.values:
        values = []
        for v in args.values.split(','):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                values.append(float(v))
        grid = {args.param: values}
    elif args.quick:
        grid = dict(QUICK_GRID)
    else:
        grid = dict(DEFAULT_GRID)

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"Grid: {grid}", file=sys.stderr)
    print(f"Total combos: {n_combos}", file=sys.stderr)

    t0 = time.time()
    results = run_sweep(seeds, grid, live_map=live_map, verbose=args.verbose)
    total = time.time() - t0

    print_results(results, seeds)
    print(f"\nTotal time: {total:.0f}s")


if __name__ == '__main__':
    main()
