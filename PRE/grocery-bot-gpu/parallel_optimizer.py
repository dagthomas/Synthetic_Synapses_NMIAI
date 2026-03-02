"""Parallel CPU optimizer: runs multiple planner-optimizers across CPU cores.

Uses ProcessPoolExecutor to run independent optimizers with different
max_active_bots values and random seeds simultaneously. Each worker uses
the proven planner-as-rest-policy approach.

For 16 cores: 8-12 parallel optimizers, each exploring different strategies.
"""
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from game_engine import init_game, MAX_ROUNDS
from configs import CONFIGS


def _worker_optimize(args):
    """Worker function: runs one optimizer instance. Must be top-level for pickling."""
    seed, difficulty, mab, time_limit, random_seed, worker_id = args[:6]
    capture_data = args[6] if len(args) > 6 else None
    pipeline_depth = args[7] if len(args) > 7 else 0

    from planner import solve as planner_solve
    from planner_optimizer import optimize_planner

    t0 = time.time()

    if capture_data is not None:
        from game_engine import init_game_from_capture
        gf = lambda: init_game_from_capture(capture_data)
    else:
        gf = lambda: init_game(seed, difficulty)

    # First get planner baseline
    try:
        planner_score, planner_actions = planner_solve(
            game_factory=gf, verbose=False, max_active_bots=mab,
            pipeline_depth=pipeline_depth
        )
    except Exception as e:
        return {'worker': worker_id, 'mab': mab, 'rseed': random_seed,
                'pipeline': pipeline_depth,
                'score': 0, 'actions': [], 'planner_score': 0, 'error': str(e)}

    if planner_score <= 0:
        return {'worker': worker_id, 'mab': mab, 'rseed': random_seed,
                'pipeline': pipeline_depth,
                'score': 0, 'actions': [], 'planner_score': 0}

    # Run optimizer with remaining time
    remaining = time_limit - (time.time() - t0)
    if remaining < 3:
        return {'worker': worker_id, 'mab': mab, 'rseed': random_seed,
                'pipeline': pipeline_depth,
                'score': planner_score, 'actions': planner_actions,
                'planner_score': planner_score}

    try:
        opt_score, opt_actions = optimize_planner(
            game_factory=gf,
            iterations=100000,
            time_limit=remaining,
            max_active_bots=mab,
            verbose=False,
            random_seed=random_seed,
        )

        final_score = max(planner_score, opt_score)
        final_actions = opt_actions if opt_score >= planner_score else planner_actions
    except Exception as e:
        final_score = planner_score
        final_actions = planner_actions

    elapsed = time.time() - t0
    return {
        'worker': worker_id, 'mab': mab, 'rseed': random_seed,
        'pipeline': pipeline_depth,
        'score': final_score, 'actions': final_actions,
        'planner_score': planner_score, 'elapsed': elapsed,
    }


def parallel_optimize(seed=None, difficulty=None, time_limit=120.0,
                       num_workers=None, verbose=True, capture_data=None):
    """Run multiple optimizers in parallel, return best result.

    Args:
        seed: game seed (for sim_server games)
        difficulty: game difficulty
        time_limit: total wall-clock budget
        num_workers: number of parallel workers (default: min(12, cpu_count))
        verbose: print progress
        capture_data: serializable capture dict for live games (replaces seed)

    Returns:
        (score, actions) tuple
    """
    t0 = time.time()

    if num_workers is None:
        num_workers = min(12, os.cpu_count() or 4)

    if difficulty is None and capture_data:
        difficulty = capture_data.get('difficulty', 'easy')

    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    # Design worker configurations based on difficulty
    configs = []
    worker_id = 0

    if num_bots == 1:
        for rseed in range(num_workers):
            configs.append((seed, difficulty, 1, time_limit - 2, rseed, worker_id, capture_data, 0))
            worker_id += 1

    elif num_bots <= 3:
        # More random seeds per mab value for diversity
        for mab in [1, 2, 3]:
            n_per = max(1, num_workers // 3)
            for rseed in range(n_per):
                configs.append((seed, difficulty, mab, time_limit - 2,
                                rseed * 10 + mab, worker_id, capture_data, 0))
                worker_id += 1

    elif num_bots <= 5:
        for mab, n in [(1, 2), (2, 3), (3, 3), (4, 2)]:
            for rseed in range(n):
                configs.append((seed, difficulty, mab, time_limit - 2,
                                rseed * 10 + mab, worker_id, capture_data, 0))
                worker_id += 1

    else:
        # Expert: mab=3 has highest optimizer ceiling, but mab=4 works on most seeds.
        for mab, n in [(1, 1), (2, 2), (3, 6), (4, 3)]:
            for rseed in range(n):
                configs.append((seed, difficulty, mab, time_limit - 2,
                                rseed * 10 + mab, worker_id, capture_data, 0))
                worker_id += 1

    # Trim to num_workers
    configs = configs[:num_workers]

    if verbose:
        mab_dist = {}
        for c in configs:
            m = c[2]
            mab_dist[m] = mab_dist.get(m, 0) + 1
        print(f"Parallel Optimizer: {difficulty} seed={seed} "
              f"{len(configs)} workers (mab dist: {mab_dist})")

    best_score = 0
    best_actions = []
    best_info = None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_optimize, cfg): cfg for cfg in configs}

        for future in as_completed(futures):
            try:
                result = future.result()
                if verbose:
                    err = result.get('error', '')
                    err_str = f" ERROR: {err}" if err else ""
                    pipe_str = f" pipe={result['pipeline']}" if result.get('pipeline', 0) > 0 else ""
                    print(f"  W{result['worker']} mab={result['mab']}{pipe_str}: "
                          f"planner={result.get('planner_score', 0)} "
                          f"final={result['score']} "
                          f"({result.get('elapsed', 0):.1f}s){err_str}")

                if result['score'] > best_score:
                    best_score = result['score']
                    best_actions = result['actions']
                    best_info = result

            except Exception as e:
                if verbose:
                    print(f"  Worker failed: {e}")

    elapsed = time.time() - t0
    if verbose:
        if best_info:
            print(f"  BEST: W{best_info['worker']} mab={best_info['mab']} "
                  f"score={best_score} ({elapsed:.1f}s)")
        else:
            print(f"  NO RESULTS ({elapsed:.1f}s)")

    return best_score, best_actions


def parallel_sweep(difficulty, seeds=range(7001, 7041), time_per_seed=60.0,
                   num_workers=None, verbose=True):
    """Sweep multiple seeds using parallel optimizer."""
    import statistics
    scores = []

    for seed in seeds:
        score, _ = parallel_optimize(
            seed, difficulty, time_limit=time_per_seed,
            num_workers=num_workers, verbose=verbose
        )
        scores.append(score)
        if verbose:
            print()

    mx = max(scores)
    mn = min(scores)
    mean = statistics.mean(scores)

    print(f"\n=== {difficulty.upper()} PARALLEL ===")
    print(f"Max={mx}  Mean={mean:.1f}  Min={mn}")
    print(f"Scores: {scores}")

    return scores


if __name__ == '__main__':
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7001
    time_lim = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    score, actions = parallel_optimize(seed, difficulty, time_limit=time_lim,
                                       num_workers=n_workers, verbose=True)
