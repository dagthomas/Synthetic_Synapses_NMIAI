"""CPU-parallel parameter search for brain.py tuning.

Searches over key brain parameters using multiprocessing to find the
combination that maximizes score on a given recording.

Usage:
    python param_search.py hard              # Grid search on hard
    python param_search.py hard --fine       # Fine-grained search around best
    python param_search.py medium            # Grid search on medium
"""

import itertools
import multiprocessing
import sys
import time

from recorder import list_recordings
from simulator import LocalSimulator, load_game_data

# Coarse grid for initial search
PARAM_GRID = {
    "spec_threshold": [10, 13, 15, 18, 22, 28, 35],
    "marginal_cost": [1, 2, 3, 4, 6, 8],
    "max_detour": [3, 4, 5, 6, 8, 10, 14],
    "max_delivering": [1, 2, 3, 5],
}


def evaluate_params(args):
    """Worker function: run simulation with given params, return score."""
    game_data_path, params = args

    # Each worker imports brain fresh (separate process)
    import brain
    brain._PARAMS = dict(params)

    game_data = load_game_data(game_data_path)
    sim = LocalSimulator(game_data)
    order_seq = game_data["order_sequence"]
    result = sim.run(
        lambda state: brain.decide_actions(state, game_plan=order_seq),
        verbose=False,
    )
    return dict(params), result["score"], result["orders_completed"], result["items_delivered"]


def grid_search(difficulty, param_grid=None, n_workers=None):
    """Search over parameter grid for best score."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return []

    recording_path = recordings[0]
    print(f"Recording: {recording_path}")

    game_data = load_game_data(recording_path)
    print(f"Bots: {len(game_data['bots'])}, Orders: {len(game_data['order_sequence'])}")

    if param_grid is None:
        param_grid = PARAM_GRID

    # Generate all parameter combinations
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))

    print(f"Testing {len(combos)} parameter combinations...")

    args = [(recording_path, dict(zip(keys, combo))) for combo in combos]

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 10)

    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(evaluate_params, args)
    elapsed = time.time() - t0

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCompleted in {elapsed:.1f}s ({len(combos) / elapsed:.0f} evals/sec)")
    print(f"\nTop 15 parameter combinations:")
    for params, score, orders, items in results[:15]:
        print(f"  Score={score:3d} (orders={orders:2d}, items={items:3d}) | "
              f"spec={params['spec_threshold']:2d} marg={params['marginal_cost']} "
              f"det={params['max_detour']:2d} del={params['max_delivering']}")

    # Find default baseline in results
    for params, score, orders, items in results:
        if (params.get("spec_threshold") == 15 and params.get("marginal_cost") == 3
                and params.get("max_detour") == 6 and params.get("max_delivering") == 5):
            print(f"\nDefault (spec=15,marg=3,det=6,del=5): Score={score}")
            break

    return results


def fine_search(difficulty, base_params, n_workers=None):
    """Fine-grained search around a known good parameter set."""
    # Create fine grid around base params
    fine_grid = {}
    for key, val in base_params.items():
        if isinstance(val, int):
            fine_grid[key] = sorted(set([
                max(1, val - 2), max(1, val - 1), val, val + 1, val + 2
            ]))
    print(f"\nFine search around {base_params}:")
    return grid_search(difficulty, fine_grid, n_workers)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python param_search.py <difficulty> [--fine]")
        sys.exit(1)

    difficulty = args[0]
    do_fine = "--fine" in args

    results = grid_search(difficulty)
    if not results:
        return

    best_params = results[0][0]
    best_score = results[0][1]
    print(f"\nBest: Score={best_score} with {best_params}")

    if do_fine and best_score > 0:
        fine_results = fine_search(difficulty, best_params)
        if fine_results:
            print(f"\nFine-tuned best: Score={fine_results[0][1]} with {fine_results[0][0]}")


if __name__ == "__main__":
    main()
