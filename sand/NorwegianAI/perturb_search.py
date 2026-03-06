"""Perturbation-based local search for action sequence optimization.

For each round where a bot waits or makes a suboptimal move,
try alternative actions and evaluate via full simulation.
Uses multiprocessing for parallel evaluation.

Usage:
    python perturb_search.py hard              # Search over waits
    python perturb_search.py hard --all        # Search ALL actions (slow)
    python perturb_search.py hard --iter 10    # Iterative hill climbing
"""

import json
import multiprocessing
import os
import sys
import time

from recorder import list_recordings
from simulator import LocalSimulator, load_game_data

# Per-difficulty optimal params (from param_search.py)
DIFFICULTY_PARAMS = {
    "easy": {},  # single bot, no tunable multi-bot params
    "medium": {"spec_threshold": 15, "marginal_cost": 1, "max_detour": 4, "max_delivering": 2},
    "hard": {"spec_threshold": 28, "marginal_cost": 3, "max_detour": 14, "max_delivering": 3, "home_aisle_penalty": 4},
    "expert": {"spec_threshold": 35, "marginal_cost": 3, "max_detour": 14},
}

ALTERNATIVES = ["move_up", "move_down", "move_left", "move_right", "wait"]


def get_params(difficulty):
    return dict(DIFFICULTY_PARAMS.get(difficulty, {}))


def record_brain_trace(game_data_path, params, overrides=None):
    """Run brain and record all actions + final score.

    overrides: dict of {(round, bot_id): action_str} to force at specific rounds
    """
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

        # Apply overrides
        if overrides:
            for a in actions:
                key = (rnd, a["bot"])
                if key in overrides:
                    a["action"] = overrides[key]
                    a.pop("item_id", None)

        all_actions.append(actions)
        sim.apply_actions(actions)

    return all_actions, sim.score, sim.orders_completed, sim.items_delivered


def evaluate_perturbation(args):
    """Worker: evaluate one perturbation (override one action at one round)."""
    game_data_path, params, round_t, bot_id, forced_action, base_overrides = args

    import brain
    brain._PARAMS = dict(params)

    game_data = load_game_data(game_data_path)
    sim = LocalSimulator(game_data)
    order_seq = game_data["order_sequence"]

    # Merge base overrides with this perturbation
    overrides = dict(base_overrides) if base_overrides else {}
    overrides[(round_t, bot_id)] = forced_action

    def perturbed_brain(state):
        actions = brain.decide_actions(state, game_plan=order_seq)
        for a in actions:
            key = (state["round"], a["bot"])
            if key in overrides:
                a["action"] = overrides[key]
                a.pop("item_id", None)
        return actions

    result = sim.run(perturbed_brain, verbose=False)
    return round_t, bot_id, forced_action, result["score"]


def find_wait_perturbations(all_actions):
    """Find all (round, bot_id, original_action) where bots wait."""
    targets = []
    for rnd, actions in enumerate(all_actions):
        for a in actions:
            if a["action"] == "wait":
                targets.append((rnd, a["bot"], "wait"))
    return targets


def find_all_perturbations(all_actions):
    """Find ALL action points for complete search."""
    targets = []
    for rnd, actions in enumerate(all_actions):
        for a in actions:
            targets.append((rnd, a["bot"], a["action"]))
    return targets


def search_perturbations(game_data_path, params, targets, base_overrides=None,
                          n_workers=None, label=""):
    """Search for score-improving perturbations."""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 10)

    # Generate evaluation tasks
    tasks = []
    for round_t, bot_id, original_action in targets:
        for alt in ALTERNATIVES:
            if alt == original_action:
                continue
            tasks.append((game_data_path, params, round_t, bot_id, alt, base_overrides))

    if not tasks:
        print(f"  No perturbations to try.")
        return []

    print(f"  {label}Evaluating {len(tasks)} perturbations ({len(targets)} targets × ~{len(ALTERNATIVES)-1} alternatives)...")

    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(evaluate_perturbation, tasks)
    elapsed = time.time() - t0

    results.sort(key=lambda x: x[3], reverse=True)
    print(f"  Completed in {elapsed:.1f}s ({len(tasks) / elapsed:.0f} evals/sec)")

    return results


def save_overrides(overrides, params, difficulty, score):
    """Save overrides to simulation/<difficulty>/overrides.json"""
    path = f"simulation/{difficulty}/overrides.json"
    # Convert tuple keys to string for JSON
    data = {
        "params": params,
        "score": score,
        "overrides": {f"{rnd},{bid}": act for (rnd, bid), act in overrides.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved overrides to {path}")


def load_overrides(difficulty):
    """Load overrides from simulation/<difficulty>/overrides.json"""
    path = f"simulation/{difficulty}/overrides.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def iterative_search(difficulty, search_all=False, max_iterations=3):
    """Iterative hill climbing: apply best perturbation, repeat."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return

    game_data_path = recordings[0]
    params = get_params(difficulty)
    print(f"Recording: {game_data_path}")
    print(f"Params: {params}")

    # Get baseline
    all_actions, baseline_score, orders, items = record_brain_trace(game_data_path, params)
    print(f"Baseline: Score={baseline_score} (orders={orders}, items={items})")

    overrides = {}
    current_score = baseline_score

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Record current trace (with accumulated overrides)
        all_actions, score, _, _ = record_brain_trace(game_data_path, params, overrides)
        if score != current_score:
            print(f"  Warning: score mismatch {score} vs {current_score}")
            current_score = score

        # Find targets
        if search_all:
            targets = find_all_perturbations(all_actions)
        else:
            targets = find_wait_perturbations(all_actions)

        if not targets:
            print("  No targets found.")
            break

        results = search_perturbations(
            game_data_path, params, targets, overrides,
            label=f"Iter {iteration + 1}: "
        )

        # Find best improvement
        best = results[0] if results else None
        if not best or best[3] <= current_score:
            print(f"  No improvement found. Stopping.")
            break

        round_t, bot_id, action, new_score = best
        improvement = new_score - current_score
        print(f"  Best: Round {round_t}, Bot {bot_id}, Action '{action}' -> Score {new_score} (+{improvement})")

        # Apply the best perturbation
        overrides[(round_t, bot_id)] = action
        current_score = new_score

        # Show top 5
        improving = [(r, b, a, s) for r, b, a, s in results if s > current_score - improvement]
        for r, b, a, s in improving[:5]:
            delta = s - (current_score - improvement)
            print(f"    Round {r:3d}, Bot {b}, '{a}': Score {s} ({'+' if delta > 0 else ''}{delta})")

    print(f"\nFinal: Score={current_score} (was {baseline_score}, improvement: +{current_score - baseline_score})")
    if overrides:
        print(f"Applied {len(overrides)} overrides:")
        for (rnd, bid), act in sorted(overrides.items()):
            print(f"  Round {rnd}, Bot {bid}: {act}")

    # Save results
    save_overrides(overrides, params, difficulty, current_score)

    return overrides, current_score


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python perturb_search.py <difficulty> [--all] [--iter N]")
        sys.exit(1)

    difficulty = args[0]
    search_all = "--all" in args
    max_iter = 10
    if "--iter" in args:
        idx = args.index("--iter")
        if idx + 1 < len(args):
            max_iter = int(args[idx + 1])

    iterative_search(difficulty, search_all=search_all, max_iterations=max_iter)


if __name__ == "__main__":
    main()
