"""Multi-strategy solver: try multiple approaches per seed, keep the best.

For each (seed, difficulty), tries:
1. Planner with different max_active_bots values
2. Beam search (for easy/medium)
3. Parallel optimizer workers on best strategies

Parallel mode (default): uses ProcessPoolExecutor for optimizer phase.
Sequential mode: runs optimizers one at a time (for constrained environments).

Returns the best score found.
"""
import time
import os
from planner import solve as planner_solve, solve_hybrid
from beam_search import beam_search
from configs import CONFIGS


def multi_solve(seed=None, difficulty=None, time_limit=60.0, verbose=True,
                game_factory=None, parallel=True, num_workers=None):
    """Try multiple strategies, return best (score, actions).

    Args:
        parallel: Use parallel workers for optimizer phase (default True).
        num_workers: Number of parallel workers (default: auto based on CPU count).
    """
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    # For parallel mode with seed (offline), delegate to parallel_optimizer
    if parallel and seed is not None and num_bots > 1:
        from parallel_optimizer import parallel_optimize
        if num_workers is None:
            num_workers = min(12, os.cpu_count() or 4)
        return parallel_optimize(
            seed=seed, difficulty=difficulty, time_limit=time_limit,
            num_workers=num_workers, verbose=verbose,
        )

    # Sequential mode (for live play, single-bot, or game_factory)
    t0 = time.time()
    results = []

    gf = game_factory
    if gf is None:
        from game_engine import init_game
        gf = lambda: init_game(seed, difficulty)

    def try_strategy(name, fn):
        elapsed = time.time() - t0
        if elapsed > time_limit - 2:
            return
        try:
            score, actions = fn()
            results.append((score, actions, name))
            if verbose:
                print(f"  {name}: score={score} ({time.time()-t0:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  {name}: FAILED ({e})")

    source = "capture" if game_factory else f"seed={seed}"
    if verbose:
        print(f"Multi-solve: {difficulty} {source}")

    # Strategy 1: Default planner
    try_strategy("planner-default", lambda: planner_solve(game_factory=gf, verbose=False))

    if num_bots == 1:
        for bw in [1, 10, 50, 100, 200]:
            if time.time() - t0 > time_limit - 10:
                break
            try_strategy(f"beam-w{bw}",
                lambda bw=bw: beam_search(game_factory=gf, beam_width=bw,
                                           max_per_bot=5, verbose=False)[:2])

    if num_bots <= 3:
        for mab in [1, 2, 3]:
            try_strategy(f"planner-mab{mab}",
                lambda mab=mab: planner_solve(game_factory=gf, verbose=False,
                                               max_active_bots=mab))

    if 3 < num_bots <= 5:
        for mab in [1, 2, 3, 4]:
            if time.time() - t0 > time_limit - 2:
                break
            try_strategy(f"planner-mab{mab}",
                lambda mab=mab: planner_solve(game_factory=gf, verbose=False,
                                               max_active_bots=mab))

    if num_bots > 5:
        for mab in [1, 2, 3]:
            try_strategy(f"planner-mab{mab}",
                lambda mab=mab: planner_solve(game_factory=gf, verbose=False,
                                               max_active_bots=mab))

    if not results:
        return 0, []

    results.sort(key=lambda x: -x[0])
    best_score, best_actions, best_name = results[0]

    if verbose:
        print(f"  Best planner: {best_name} score={best_score} ({time.time()-t0:.1f}s)")

    # Phase 2: Optimize planner results
    remaining = time_limit - (time.time() - t0)
    if remaining > 5 and best_actions:
        from planner_optimizer import optimize_planner

        # Filter meaningful results and deduplicate by mab value
        seen_mab = set()
        top_results = []
        for score, actions, name in sorted(results, key=lambda x: -x[0]):
            if 'mab' in name:
                mab_val = int(name.split('mab')[1])
            elif 'default' in name:
                mab_val = min(num_bots, 2)
            elif 'beam' in name:
                mab_val = f"beam-{name}"
            else:
                mab_val = name
            if mab_val not in seen_mab and score > 0:
                seen_mab.add(mab_val)
                top_results.append((score, actions, name))
            if len(top_results) >= 3:
                break

        n_restarts = len(top_results)
        min_time = max(10, remaining * 0.15)
        if n_restarts == 1:
            time_allocs = [(remaining - 1, top_results[0][0], top_results[0][1], top_results[0][2])]
        else:
            total_score = sum(s for s, _, _ in top_results) or 1
            time_allocs = []
            for s, a, n in top_results:
                alloc = max(min_time, s / total_score * remaining - 1)
                time_allocs.append((alloc, s, a, n))

        for alloc_time, planner_score, planner_actions, planner_name in time_allocs:
            if time.time() - t0 > time_limit - 3:
                break
            opt_remaining = min(alloc_time, time_limit - (time.time() - t0) - 1)
            if opt_remaining < 3:
                break

            opt_mab = None
            if 'mab' in planner_name:
                opt_mab = int(planner_name.split('mab')[1])

            try:
                opt_score, opt_actions = optimize_planner(
                    game_factory=gf,
                    iterations=20000,
                    time_limit=opt_remaining,
                    max_active_bots=opt_mab,
                    verbose=False,
                )
                if opt_score > best_score:
                    if verbose:
                        print(f"  Optimizer({planner_name}): {planner_score} -> {opt_score} (+{opt_score-planner_score}) ({time.time()-t0:.1f}s)")
                    best_score = opt_score
                    best_actions = opt_actions
                    best_name = f"optimized-{planner_name}"
            except Exception as e:
                if verbose:
                    print(f"  Optimizer({planner_name}): FAILED ({e})")

    if verbose:
        print(f"  FINAL: {best_name} score={best_score} ({time.time()-t0:.1f}s)")

    return best_score, best_actions


def sweep(difficulty, seeds=range(7001, 7041), time_per_seed=30.0, verbose=True,
          parallel=True, num_workers=None):
    """Sweep multiple seeds for one difficulty."""
    import statistics
    scores = []

    for seed in seeds:
        score, _ = multi_solve(seed, difficulty, time_limit=time_per_seed,
                               verbose=verbose, parallel=parallel,
                               num_workers=num_workers)
        scores.append(score)
        if verbose:
            print()

    mx = max(scores)
    mn = min(scores)
    mean = statistics.mean(scores)

    print(f"\n=== {difficulty.upper()} ===")
    print(f"Max={mx}  Mean={mean:.1f}  Min={mn}")
    print(f"Scores: {scores}")

    return scores


if __name__ == '__main__':
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'easy'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    time_lim = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    seeds = range(7001, 7001 + n_seeds)
    sweep(difficulty, seeds, time_per_seed=time_lim)
