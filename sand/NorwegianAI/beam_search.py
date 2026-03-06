"""Beam search perturbation optimizer.

Instead of greedily taking the single best perturbation each iteration,
maintain a beam of top-K candidates and explore from each. This explores
multiple paths through the override space and can escape local optima
that greedy search gets stuck in.

Usage:
    python beam_search.py hard                # Beam width 5, 5 iterations
    python beam_search.py hard --beam 10      # Wider beam
    python beam_search.py hard --iter 8       # More iterations
"""

import json
import multiprocessing
import os
import sys
import time

from recorder import list_recordings
from simulator import LocalSimulator, load_game_data

ALTERNATIVES = ["move_up", "move_down", "move_left", "move_right", "wait"]


def get_params(difficulty):
    from perturb_search import DIFFICULTY_PARAMS
    return dict(DIFFICULTY_PARAMS.get(difficulty, {}))


def record_brain_trace(game_data_path, params, overrides=None):
    """Run brain and record all actions + final score."""
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


def evaluate_perturbation(args):
    """Worker: evaluate one perturbation."""
    game_data_path, params, round_t, bot_id, forced_action, base_overrides = args

    import brain
    brain._PARAMS = dict(params)

    game_data = load_game_data(game_data_path)
    sim = LocalSimulator(game_data)
    order_seq = game_data["order_sequence"]

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


def search_from_candidate(game_data_path, params, base_overrides, n_workers=None):
    """Evaluate all single-action perturbations from a given override set.
    Returns list of (override_dict, score) sorted by score descending.
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 10)

    # Get brain trace with current overrides
    all_actions, base_score = record_brain_trace(game_data_path, params, base_overrides)

    # Find all action points
    targets = []
    for rnd, actions in enumerate(all_actions):
        for a in actions:
            targets.append((rnd, a["bot"], a["action"]))

    # Generate evaluation tasks
    tasks = []
    for round_t, bot_id, original_action in targets:
        if (round_t, bot_id) in base_overrides:
            continue  # Don't re-override existing overrides
        for alt in ALTERNATIVES:
            if alt == original_action:
                continue
            tasks.append((game_data_path, params, round_t, bot_id, alt, base_overrides))

    t0 = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(evaluate_perturbation, tasks)
    elapsed = time.time() - t0

    # Build candidate override sets for improvements
    candidates = []
    for round_t, bot_id, action, score in results:
        if score > base_score:
            new_overrides = dict(base_overrides)
            new_overrides[(round_t, bot_id)] = action
            candidates.append((new_overrides, score, round_t, bot_id, action))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates, base_score, len(tasks), elapsed


def beam_search(difficulty, beam_width=5, max_iterations=5):
    """Beam search over override space."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return

    game_data_path = recordings[0]
    params = get_params(difficulty)
    print(f"Recording: {game_data_path}")
    print(f"Params: {params}")
    print(f"Beam width: {beam_width}, Max iterations: {max_iterations}")

    # Get baseline
    _, baseline_score = record_brain_trace(game_data_path, params)
    print(f"Baseline: Score={baseline_score}")

    # Initialize beam: start with empty override set
    beam = [({}, baseline_score)]  # List of (overrides, score)

    # Seed beam with existing overrides if available (explore from known good state too)
    from perturb_search import load_overrides
    existing = load_overrides(difficulty)
    if existing and existing.get("overrides"):
        existing_overrides = {}
        for key_str, act in existing["overrides"].items():
            parts = key_str.split(",")
            existing_overrides[(int(parts[0]), int(parts[1]))] = act
        # Verify score
        _, existing_score = record_brain_trace(game_data_path, params, existing_overrides)
        print(f"Seeding beam with existing overrides: {len(existing_overrides)} overrides, score={existing_score}")
        beam.append((existing_overrides, existing_score))

        # Also seed with subsets of existing overrides (removal search)
        if len(existing_overrides) >= 2:
            for remove_key in list(existing_overrides.keys()):
                subset = {k: v for k, v in existing_overrides.items() if k != remove_key}
                _, subset_score = record_brain_trace(game_data_path, params, subset)
                beam.append((subset, subset_score))
                rnd, bid = remove_key
                print(f"  Without Round {rnd} Bot {bid}: score={subset_score}")

        # Deduplicate and keep best beam_width entries
        beam.sort(key=lambda x: x[1], reverse=True)
        beam = beam[:beam_width]
    best_overall = baseline_score
    best_overrides = {}

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"  Beam has {len(beam)} candidates (scores: {[s for _, s in beam]})")

        all_candidates = []
        total_evals = 0
        total_time = 0

        for beam_idx, (base_overrides, base_score) in enumerate(beam):
            print(f"  Expanding candidate {beam_idx + 1}/{len(beam)} (score={base_score}, {len(base_overrides)} overrides)...")
            candidates, verified_score, n_evals, elapsed = search_from_candidate(
                game_data_path, params, base_overrides
            )
            total_evals += n_evals
            total_time += elapsed

            if verified_score != base_score:
                print(f"    Score verification: {verified_score} (expected {base_score})")

            if candidates:
                print(f"    Found {len(candidates)} improvements (best: {candidates[0][1]})")
                # Keep top improvements from this branch
                for overrides, score, rnd, bid, action in candidates[:beam_width * 2]:
                    all_candidates.append((overrides, score, rnd, bid, action, beam_idx))
            else:
                print(f"    No improvements found")
                # Keep this candidate as-is (it may be a local optimum worth preserving)
                all_candidates.append((base_overrides, base_score, -1, -1, "", beam_idx))

        print(f"  Total: {total_evals} evals in {total_time:.1f}s ({total_evals / total_time:.0f} evals/sec)")

        if not all_candidates:
            print("  No candidates. Stopping.")
            break

        # Deduplicate by override set (convert to frozenset of items for hashing)
        seen = set()
        unique_candidates = []
        for item in all_candidates:
            key = frozenset(item[0].items())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(item)

        # Sort by score and keep top beam_width
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        new_beam = []
        for overrides, score, rnd, bid, action, parent_idx in unique_candidates[:beam_width]:
            new_beam.append((overrides, score))
            if rnd >= 0:
                print(f"    Beam slot: score={score} (+{score - baseline_score}) "
                      f"[parent={parent_idx}, +Round {rnd} Bot {bid} '{action}', "
                      f"{len(overrides)} overrides]")
            else:
                print(f"    Beam slot: score={score} (unchanged, {len(overrides)} overrides)")

        # Track best overall and save intermediate results
        if new_beam[0][1] > best_overall:
            best_overall = new_beam[0][1]
            best_overrides = dict(new_beam[0][0])
            # Save intermediate best
            from perturb_search import save_overrides as _save
            existing = load_overrides(difficulty)
            existing_score = existing["score"] if existing else 0
            if best_overall > existing_score:
                _save(best_overrides, params, difficulty, best_overall)
                print(f"  Intermediate save: score={best_overall}")

        # Check if beam improved over previous iteration
        prev_best = max(s for _, s in beam)
        curr_best = max(s for _, s in new_beam)
        if curr_best <= prev_best:
            print(f"  No improvement over previous iteration ({prev_best}). Stopping.")
            break

        beam = new_beam

    print(f"\n{'='*60}")
    print(f"Best score: {best_overall} (baseline: {baseline_score}, +{best_overall - baseline_score})")
    print(f"Overrides ({len(best_overrides)}):")
    for (rnd, bid), act in sorted(best_overrides.items()):
        print(f"  Round {rnd}, Bot {bid}: {act}")

    # Save if better than existing
    from perturb_search import load_overrides, save_overrides
    existing = load_overrides(difficulty)
    existing_score = existing["score"] if existing else 0
    if best_overall > existing_score:
        save_overrides(best_overrides, params, difficulty, best_overall)
        print(f"  Saved! (was {existing_score})")
    else:
        print(f"  Not saving (existing score {existing_score} >= {best_overall})")

    return best_overrides, best_overall


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python beam_search.py <difficulty> [--beam K] [--iter N]")
        sys.exit(1)

    difficulty = args[0]
    beam_width = 5
    max_iter = 5

    if "--beam" in args:
        idx = args.index("--beam")
        if idx + 1 < len(args):
            beam_width = int(args[idx + 1])
    if "--iter" in args:
        idx = args.index("--iter")
        if idx + 1 < len(args):
            max_iter = int(args[idx + 1])

    beam_search(difficulty, beam_width=beam_width, max_iterations=max_iter)


if __name__ == "__main__":
    main()
