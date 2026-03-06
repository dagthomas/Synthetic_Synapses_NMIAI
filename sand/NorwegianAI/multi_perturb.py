"""Multi-perturbation search: try changing 2+ actions simultaneously.

After finding the best single overrides via iterative search, search for
additional overrides that improve when combined with existing ones.
Also tries random pairs to escape local optima.

Usage:
    python multi_perturb.py hard
"""

import json
import multiprocessing
import os
import random
import sys
import time

from recorder import list_recordings
from simulator import LocalSimulator, load_game_data

ALTERNATIVES = ["move_up", "move_down", "move_left", "move_right", "wait"]


def get_params(difficulty):
    from perturb_search import DIFFICULTY_PARAMS
    return dict(DIFFICULTY_PARAMS.get(difficulty, {}))


def evaluate_overrides(args):
    """Worker: evaluate a set of overrides."""
    game_data_path, params, overrides_dict = args

    import brain
    brain._PARAMS = dict(params)

    game_data = load_game_data(game_data_path)
    sim = LocalSimulator(game_data)
    order_seq = game_data["order_sequence"]

    def perturbed_brain(state):
        actions = brain.decide_actions(state, game_plan=order_seq)
        for a in actions:
            key = (state["round"], a["bot"])
            if key in overrides_dict:
                a["action"] = overrides_dict[key]
                a.pop("item_id", None)
        return actions

    result = sim.run(perturbed_brain, verbose=False)
    return result["score"]


def get_brain_trace(game_data_path, params, overrides=None):
    """Run brain and record all actions."""
    import brain
    brain._PARAMS = dict(params)

    game_data = load_game_data(game_data_path)
    sim = LocalSimulator(game_data)
    order_seq = game_data["order_sequence"]

    all_actions = []
    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()
        actions = brain.decide_actions(state, game_plan=order_seq)

        if overrides:
            for a in actions:
                key = (rnd, a["bot"])
                if key in overrides:
                    a["action"] = overrides[key]
                    a.pop("item_id", None)

        all_actions.append(actions)
        sim.apply_actions(actions)

    return all_actions, sim.score


def search_pairs_near_best(game_data_path, params, base_overrides, base_score, n_workers=None):
    """Search for pairs of overrides near existing best overrides."""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 10)

    # Get the brain trace with current overrides
    all_actions, _ = get_brain_trace(game_data_path, params, base_overrides)

    # Collect all action points
    targets = []
    for rnd, actions in enumerate(all_actions):
        for a in actions:
            targets.append((rnd, a["bot"], a["action"]))

    # Generate pair tasks: try each single override combined with base
    tasks = []
    for round_t, bot_id, original_action in targets:
        if (round_t, bot_id) in base_overrides:
            continue  # Skip already overridden
        for alt in ALTERNATIVES:
            if alt == original_action:
                continue
            test_overrides = dict(base_overrides)
            test_overrides[(round_t, bot_id)] = alt
            tasks.append((game_data_path, params, test_overrides))

    print(f"  Evaluating {len(tasks)} single additions to {len(base_overrides)} base overrides...")

    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        scores = pool.map(evaluate_overrides, tasks)
    elapsed = time.time() - t0

    # Find improvements
    results = []
    for i, score in enumerate(scores):
        if score > base_score:
            test_overrides = tasks[i][2]
            # Find the new override
            new_key = None
            for k in test_overrides:
                if k not in base_overrides:
                    new_key = k
                    break
            if new_key:
                results.append((new_key, test_overrides[new_key], score))

    results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Done in {elapsed:.1f}s ({len(tasks) / elapsed:.0f} evals/sec)")
    print(f"  Found {len(results)} improvements")

    return results


def random_multi_search(game_data_path, params, base_overrides, base_score,
                        n_random=1000, max_changes=3, n_workers=None):
    """Random multi-perturbation: try random sets of 2-3 overrides."""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 10)

    all_actions, _ = get_brain_trace(game_data_path, params, base_overrides)

    # Collect all action points
    all_points = []
    for rnd, actions in enumerate(all_actions):
        for a in actions:
            all_points.append((rnd, a["bot"], a["action"]))

    random.seed(42)
    tasks = []
    for _ in range(n_random):
        n_changes = random.randint(1, max_changes)
        test_overrides = dict(base_overrides)
        points = random.sample(all_points, min(n_changes, len(all_points)))
        for rnd, bid, orig in points:
            alts = [a for a in ALTERNATIVES if a != orig]
            test_overrides[(rnd, bid)] = random.choice(alts)
        tasks.append((game_data_path, params, test_overrides))

    print(f"  Random search: {len(tasks)} random override sets (1-{max_changes} changes each)...")

    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        scores = pool.map(evaluate_overrides, tasks)
    elapsed = time.time() - t0

    # Find best
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_overrides = tasks[best_idx][2]

    # Count improvements
    n_improvements = sum(1 for s in scores if s > base_score)
    print(f"  Done in {elapsed:.1f}s ({len(tasks) / elapsed:.0f} evals/sec)")
    print(f"  Found {n_improvements} improvements out of {len(tasks)}")
    print(f"  Best random: {best_score} (base was {base_score})")

    if best_score > base_score:
        # Show the overrides
        new_keys = [k for k in best_overrides if k not in base_overrides]
        for k in new_keys:
            print(f"    Round {k[0]}, Bot {k[1]}: {best_overrides[k]}")

    return best_overrides if best_score > base_score else None, best_score


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python multi_perturb.py <difficulty> [--random N]")
        sys.exit(1)

    difficulty = args[0]
    n_random = 2000
    if "--random" in args:
        idx = args.index("--random")
        if idx + 1 < len(args):
            n_random = int(args[idx + 1])

    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return

    game_data_path = recordings[0]
    params = get_params(difficulty)
    print(f"Recording: {game_data_path}")
    print(f"Params: {params}")

    # Load existing overrides as base
    from perturb_search import load_overrides, save_overrides
    data = load_overrides(difficulty)
    base_overrides = {}
    if data:
        for key_str, act in data["overrides"].items():
            parts = key_str.split(",")
            base_overrides[(int(parts[0]), int(parts[1]))] = act
        base_score = data["score"]
        print(f"Loaded {len(base_overrides)} base overrides, score={base_score}")
    else:
        # Get baseline score
        _, base_score = get_brain_trace(game_data_path, params)
        print(f"No base overrides, baseline score={base_score}")

    # Phase 1: Search for single additions to base overrides
    print(f"\n--- Phase 1: Single additions ---")
    results = search_pairs_near_best(game_data_path, params, base_overrides, base_score)

    if results:
        best_key, best_action, best_score = results[0]
        print(f"  Best addition: Round {best_key[0]}, Bot {best_key[1]}, '{best_action}' -> {best_score}")
        base_overrides[best_key] = best_action

        # Show top 5
        for key, action, score in results[:5]:
            print(f"    Round {key[0]:3d}, Bot {key[1]}, '{action}': {score}")

        # Iteratively add more
        for iteration in range(5):
            print(f"\n  Adding override #{len(base_overrides)}...")
            results = search_pairs_near_best(game_data_path, params, base_overrides, best_score)
            if not results:
                print("  No more improvements.")
                break
            best_key, best_action, new_score = results[0]
            if new_score <= best_score:
                print(f"  No improvement over {best_score}")
                break
            print(f"  +Override: Round {best_key[0]}, Bot {best_key[1]}, '{best_action}' -> {new_score} (+{new_score - best_score})")
            base_overrides[best_key] = best_action
            best_score = new_score

    current_score = best_score if results else base_score

    # Phase 2: Random multi-perturbation
    print(f"\n--- Phase 2: Random multi-perturbation ---")
    random_result, random_score = random_multi_search(
        game_data_path, params, base_overrides, current_score,
        n_random=n_random, max_changes=3
    )

    if random_result and random_score > current_score:
        print(f"  Random search improved: {current_score} -> {random_score}")
        base_overrides = random_result
        current_score = random_score

    # Save best result
    print(f"\nFinal score: {current_score}")
    print(f"Total overrides: {len(base_overrides)}")
    for (rnd, bid), act in sorted(base_overrides.items()):
        print(f"  Round {rnd}, Bot {bid}: {act}")

    save_overrides(base_overrides, params, difficulty, current_score)


if __name__ == "__main__":
    main()
