"""Parameter sweep harness for GPU sequential solver.

Runs solve_sequential with different parameter combos on multiple seeds,
reports scores in a table for comparison.

Usage:
    python sweep_params.py hard --seeds 7001-7003
    python sweep_params.py expert --seeds 42,7001,7002
    python sweep_params.py hard --seeds 3           # seeds 7001-7003
    python sweep_params.py hard --baseline          # just run baseline defaults
    python sweep_params.py hard --param refine_iters --values 2,4,6,8
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_sequential_solver import solve_sequential


# Default parameter grid per difficulty
DEFAULT_GRIDS = {
    'easy': {
        'max_refine_iters': [0],
        'num_pass1_orderings': [1],
        'pipeline_fraction': [0.0],
        'max_states': [200_000],
    },
    'medium': {
        'max_refine_iters': [2, 4],
        'num_pass1_orderings': [1, 3],
        'pipeline_fraction': [0.0, 0.2, 0.4],
        'max_states': [200_000, 1_000_000],
    },
    'hard': {
        'max_refine_iters': [2, 5, 8, 12],
        'num_pass1_orderings': [1, 3, 5, 8],
        'pipeline_fraction': [0.0, 0.2, 0.4],
        'max_states': [1_000_000, 2_000_000, 5_000_000],
    },
    'expert': {
        'max_refine_iters': [2, 5, 8, 12],
        'num_pass1_orderings': [1, 3, 5, 8],
        'pipeline_fraction': [0.0, 0.2, 0.4],
        'max_states': [500_000, 1_000_000, 2_000_000],
    },
}


def parse_seeds(seeds_str):
    """Parse seed specification: '7001-7003', '42,7001', or '3' (count from 7001)."""
    if '-' in seeds_str and ',' not in seeds_str:
        parts = seeds_str.split('-')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if end < 100:  # range like 7001-3 means 7001 to 7003
                end = start + end - 1
            return list(range(start, end + 1))
    if ',' in seeds_str:
        return [int(s.strip()) for s in seeds_str.split(',')]
    n = int(seeds_str)
    if n < 100:
        return list(range(7001, 7001 + n))
    return [n]


def run_sweep(difficulty, seeds, param_grid, device='cuda', verbose_solve=False):
    """Run parameter sweep and collect results.

    Args:
        difficulty: Game difficulty.
        seeds: List of seed values.
        param_grid: Dict of param_name -> [values] to sweep.
        device: 'cuda' or 'cpu'.
        verbose_solve: If True, print full solve output.

    Returns:
        List of (params_dict, seed_scores_dict, mean_score, max_score, total_time).
    """
    # Generate all combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combos = list(itertools.product(*param_values))

    results = []
    total_combos = len(combos) * len(seeds)
    run_idx = 0

    for combo in combos:
        params = dict(zip(param_names, combo))
        seed_scores = {}
        t_combo = time.time()

        for seed in seeds:
            run_idx += 1
            param_str = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f"\n[{run_idx}/{total_combos}] seed={seed} | {param_str}",
                  file=sys.stderr)

            try:
                score, _ = solve_sequential(
                    seed=seed,
                    difficulty=difficulty,
                    device=device,
                    max_states=params.get('max_states', 1_000_000),
                    verbose=verbose_solve,
                    max_refine_iters=params.get('max_refine_iters', 2),
                    num_pass1_orderings=params.get('num_pass1_orderings', 1),
                    pipeline_fraction=params.get('pipeline_fraction', 0.4),
                    pass1_states=params.get('pass1_states'),
                )
                seed_scores[seed] = score
                print(f"  -> score={score}", file=sys.stderr)
            except Exception as e:
                print(f"  -> ERROR: {e}", file=sys.stderr)
                seed_scores[seed] = -1

        combo_time = time.time() - t_combo
        valid_scores = [s for s in seed_scores.values() if s >= 0]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        max_score = max(valid_scores) if valid_scores else 0

        results.append((params, seed_scores, mean_score, max_score, combo_time))

    return results


def print_results(results, seeds, difficulty):
    """Print results as a formatted table."""
    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS: {difficulty}")
    print(f"{'='*80}")

    # Header
    param_names = sorted(results[0][0].keys()) if results else []
    header_parts = [f"{'param_combo':50s}"]
    for s in seeds:
        header_parts.append(f"s{s:>6}")
    header_parts.extend(["  mean", "   max", "  time"])
    print(' | '.join(header_parts))
    print('-' * (55 + 9 * len(seeds) + 25))

    # Sort by mean score descending
    results_sorted = sorted(results, key=lambda r: -r[2])

    for params, seed_scores, mean, mx, t in results_sorted:
        param_str = ', '.join(f'{k}={v}' for k, v in sorted(params.items()))
        if len(param_str) > 50:
            param_str = param_str[:47] + '...'
        parts = [f"{param_str:50s}"]
        for s in seeds:
            sc = seed_scores.get(s, -1)
            parts.append(f"{sc:>6}" if sc >= 0 else "   ERR")
        parts.append(f"{mean:6.1f}")
        parts.append(f"{mx:6}")
        parts.append(f"{t:5.0f}s")
        print(' | '.join(parts))

    # Best combo
    if results_sorted:
        best = results_sorted[0]
        print(f"\nBest: mean={best[2]:.1f}, max={best[3]}")
        print(f"  Params: {best[0]}")


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for GPU sequential solver')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--seeds', default='7001-7003',
                        help='Seeds: "7001-7003", "42,7001", or count "3"')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true', help='Verbose solve output')
    parser.add_argument('--baseline', action='store_true',
                        help='Run only baseline defaults (no sweep)')
    parser.add_argument('--param', help='Single param to sweep (overrides grid)')
    parser.add_argument('--values', help='Comma-separated values for --param')
    parser.add_argument('--max-states', type=int, help='Fixed max_states (override grid)')
    parser.add_argument('--refine-iters', type=int, help='Fixed refine iters (override grid)')

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    if args.baseline:
        # Run with current defaults only
        grid = {
            'max_refine_iters': [2],
            'num_pass1_orderings': [1],
            'pipeline_fraction': [0.4],
            'max_states': [1_000_000],
        }
    elif args.param and args.values:
        # Single-param sweep
        values = []
        for v in args.values.split(','):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                values.append(float(v))
        grid = {args.param: values}
        # Fill in fixed defaults for other params
        defaults = {
            'max_refine_iters': 2,
            'num_pass1_orderings': 1,
            'pipeline_fraction': 0.4,
            'max_states': 1_000_000,
        }
        for k, v in defaults.items():
            if k not in grid:
                grid[k] = [v]
    else:
        grid = DEFAULT_GRIDS.get(args.difficulty, DEFAULT_GRIDS['hard'])

    # Apply overrides
    if args.max_states:
        grid['max_states'] = [args.max_states]
    if args.refine_iters is not None:
        grid['max_refine_iters'] = [args.refine_iters]

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    total_runs = n_combos * len(seeds)
    print(f"Sweeping {args.difficulty}: {n_combos} combos x {len(seeds)} seeds = {total_runs} runs",
          file=sys.stderr)
    print(f"Grid: {grid}", file=sys.stderr)

    t0 = time.time()
    results = run_sweep(args.difficulty, seeds, grid, device=args.device,
                        verbose_solve=args.verbose)
    total_time = time.time() - t0

    print_results(results, seeds, args.difficulty)
    print(f"\nTotal sweep time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
