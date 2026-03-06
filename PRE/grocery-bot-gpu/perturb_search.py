#!/usr/bin/env python3
"""Perturbation search: post-process GPU DP solutions by local action-space hill climbing.

For each (round, bot) action, try replacing with alternative moves/wait and
re-simulate the full game. Uses Zig FFI for ~0.3ms per evaluation and
multiprocessing for parallel evaluation across CPU cores.

This is complementary to GPU DP: DP finds globally optimal single-bot plans,
then perturbation search fixes multi-bot interaction issues (collisions,
timing, corridor congestion) that sequential DP can't see.

Usage:
    python perturb_search.py hard                    # 10 iterations
    python perturb_search.py hard --iter 20          # More iterations
    python perturb_search.py hard --all              # Try ALL actions (not just waits)
    python perturb_search.py hard --budget 60        # Time budget in seconds
    python perturb_search.py expert --workers 12     # Limit worker count
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import sys
import time
from typing import Any

import numpy as np

from configs import MAX_ROUNDS, DIFF_IDX
from game_engine import (
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, Order, init_game_from_capture,
)
from solution_store import load_solution, load_capture, save_solution, load_meta

# Movement alternatives to try for each perturbation
MOVE_ALTS = [ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


# ── Worker function (runs in separate process) ──

def _eval_perturbation(args):
    """Evaluate a single action perturbation via Zig FFI.

    Called in a worker process. Each worker loads the Zig DLL independently.
    Returns (round_idx, bot_id, new_action, score).
    """
    capture_data, orders_data, base_actions_flat, num_bots, round_idx, bot_id, new_action = args

    from zig_ffi import _get_lib, _capture_to_map_args, _orders_to_flat
    import ctypes

    lib = _get_lib()
    map_args, _keep = _capture_to_map_args(capture_data)
    order_types, order_lens = _orders_to_flat(orders_data)

    # Copy base actions and apply perturbation
    acts = base_actions_flat[0].copy()  # int8 array
    items = base_actions_flat[1].copy()  # int16 array

    idx = round_idx * num_bots + bot_id
    acts[idx] = new_action
    items[idx] = -1  # Movement actions don't need item

    score = lib.ffi_verify_live(
        *map_args,
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        items.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        order_types.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        order_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint16(len(orders_data)),
    )

    return round_idx, bot_id, new_action, score


def _eval_multi_perturbation(args):
    """Evaluate multiple simultaneous perturbations via Zig FFI.

    Returns (perturbation_list, score).
    """
    capture_data, orders_data, base_actions_flat, num_bots, perturbations = args

    from zig_ffi import _get_lib, _capture_to_map_args, _orders_to_flat
    import ctypes

    lib = _get_lib()
    map_args, _keep = _capture_to_map_args(capture_data)
    order_types, order_lens = _orders_to_flat(orders_data)

    acts = base_actions_flat[0].copy()
    items = base_actions_flat[1].copy()

    for round_idx, bot_id, new_action in perturbations:
        idx = round_idx * num_bots + bot_id
        acts[idx] = new_action
        items[idx] = -1

    score = lib.ffi_verify_live(
        *map_args,
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        items.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        order_types.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        order_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint16(len(orders_data)),
    )

    return perturbations, score


# ── Solution helpers ──

def solution_to_flat(solution, num_bots):
    """Convert solution list-of-lists to flat numpy arrays for Zig FFI."""
    acts = np.zeros(MAX_ROUNDS * num_bots, dtype=np.int8)
    items = np.full(MAX_ROUNDS * num_bots, -1, dtype=np.int16)
    for r, round_actions in enumerate(solution):
        for b, (act, item) in enumerate(round_actions):
            idx = r * num_bots + b
            acts[idx] = int(act)
            if item is not None and item >= 0:
                items[idx] = int(item)
    return acts, items


def flat_to_solution(acts, items, num_bots):
    """Convert flat arrays back to solution list-of-lists."""
    solution = []
    for r in range(MAX_ROUNDS):
        round_actions = []
        for b in range(num_bots):
            idx = r * num_bots + b
            round_actions.append((int(acts[idx]), int(items[idx])))
        solution.append(round_actions)
    return solution


def apply_overrides(acts, items, overrides, num_bots):
    """Apply a dict of overrides {(round, bot): action} to flat arrays."""
    acts = acts.copy()
    items = items.copy()
    for (r, b), action in overrides.items():
        idx = r * num_bots + b
        acts[idx] = action
        items[idx] = -1
    return acts, items


# ── Core search ──

def find_perturbation_targets(solution, num_bots, search_all=False):
    """Find (round, bot, current_action) targets for perturbation.

    If search_all=False, only targets where current action is wait (cheapest search).
    If search_all=True, targets all non-pickup/dropoff actions (more thorough).
    """
    targets = []
    for r, round_actions in enumerate(solution):
        for b in range(num_bots):
            act, item = round_actions[b]
            if search_all:
                # Try all movement/wait actions
                if act in (ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                    targets.append((r, b, act))
            else:
                # Only try replacing waits
                if act == ACT_WAIT:
                    targets.append((r, b, act))
    return targets


def run_perturbation_pass(capture_data, orders_data, base_flat, num_bots,
                          targets, overrides, n_workers, base_score):
    """Run one pass of perturbation search over all targets.

    Returns list of (round, bot, action, score) improvements sorted by score desc.
    """
    tasks = []
    for round_idx, bot_id, current_action in targets:
        if (round_idx, bot_id) in overrides:
            continue
        for alt in MOVE_ALTS:
            if alt == current_action:
                continue
            # Apply existing overrides to base
            if overrides:
                acts, items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
            else:
                acts, items = base_flat
            tasks.append((
                capture_data, orders_data, (acts, items), num_bots,
                round_idx, bot_id, alt,
            ))

    if not tasks:
        return []

    print(f"    Evaluating {len(tasks)} perturbations ({len(targets)} targets)...", end='', flush=True)
    t0 = time.perf_counter()

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_eval_perturbation, tasks, chunksize=max(1, len(tasks) // (n_workers * 4)))

    dt = time.perf_counter() - t0
    rate = len(tasks) / dt
    print(f" {dt:.1f}s ({rate:.0f} evals/s)")

    improvements = [(r, b, a, s) for r, b, a, s in results if s > base_score]
    improvements.sort(key=lambda x: x[3], reverse=True)
    return improvements


def neighborhood_search(capture_data, orders_data, base_flat, num_bots,
                        center_round, n_workers, overrides, current_score,
                        radius=10):
    """Search all actions in a neighborhood of rounds around a known improvement.

    When a perturbation at round R helps, nearby rounds often have further gains.
    """
    r_lo = max(0, center_round - radius)
    r_hi = min(MAX_ROUNDS, center_round + radius + 1)

    if overrides:
        cur_acts, cur_items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
        cur_solution = flat_to_solution(cur_acts, cur_items, num_bots)
    else:
        cur_solution = flat_to_solution(base_flat[0], base_flat[1], num_bots)

    targets = []
    for r in range(r_lo, r_hi):
        for b in range(num_bots):
            act, _ = cur_solution[r][b]
            if act in (ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                targets.append((r, b, act))

    if not targets:
        return []

    return run_perturbation_pass(
        capture_data, orders_data, base_flat, num_bots,
        targets, overrides, n_workers, current_score,
    )


def iterative_search(difficulty, search_all=False, max_iterations=10,
                     time_budget=None, n_workers=None):
    """Iterative perturbation hill climbing on GPU DP solution.

    Loads best solution from DB, tries action perturbations, saves improvements.
    Uses neighborhood search around found improvements for deeper exploration.
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 14)

    capture = load_capture(difficulty)
    if not capture:
        print(f"No capture data for {difficulty}")
        return None, 0

    solution = load_solution(difficulty)
    if not solution:
        print(f"No solution for {difficulty}")
        return None, 0

    meta = load_meta(difficulty)
    num_bots = capture['num_bots']

    # Build Order objects for Zig FFI
    _, all_orders = init_game_from_capture(capture)

    # Verify baseline
    from zig_ffi import zig_verify_live
    base_score = zig_verify_live(capture, all_orders, solution, num_bots)

    print(f"  Difficulty: {difficulty}")
    print(f"  Bots: {num_bots}, Orders: {len(capture['orders'])}")
    print(f"  DB score: {meta['score'] if meta else '?'}, Verified: {base_score}")
    print(f"  Workers: {n_workers}")
    print()

    # Convert to flat arrays
    base_flat = solution_to_flat(solution, num_bots)
    overrides = {}
    current_score = base_score
    start_time = time.perf_counter()

    for iteration in range(max_iterations):
        if time_budget and (time.perf_counter() - start_time) > time_budget:
            print(f"  Time budget ({time_budget}s) reached.")
            break

        print(f"  --- Iteration {iteration + 1}/{max_iterations} (score={current_score}) ---")

        # Re-build solution with current overrides for target detection
        if overrides:
            cur_acts, cur_items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
            current_solution = flat_to_solution(cur_acts, cur_items, num_bots)
        else:
            current_solution = solution

        targets = find_perturbation_targets(current_solution, num_bots, search_all=search_all)

        if not targets:
            print("    No targets found.")
            break

        improvements = run_perturbation_pass(
            capture, all_orders, base_flat, num_bots,
            targets, overrides, n_workers, current_score,
        )

        if not improvements:
            if not search_all:
                print("    No improvements from waits. Trying all actions...")
                targets = find_perturbation_targets(current_solution, num_bots, search_all=True)
                improvements = run_perturbation_pass(
                    capture, all_orders, base_flat, num_bots,
                    targets, overrides, n_workers, current_score,
                )
                if not improvements:
                    print("    No improvements found. Stopping.")
                    break
            else:
                print("    No improvements found. Stopping.")
                break

        # Apply best improvement
        best_r, best_b, best_a, best_score = improvements[0]
        delta = best_score - current_score
        act_names = ['wait', 'up', 'down', 'left', 'right', 'pickup', 'dropoff']
        print(f"    Best: R{best_r} B{best_b} -> {act_names[best_a]} = {best_score} (+{delta})")

        overrides[(best_r, best_b)] = best_a
        current_score = best_score

        # Show top 5
        for r, b, a, s in improvements[:5]:
            print(f"      R{r:3d} B{b} {act_names[a]:>5s} -> {s} (+{s - (current_score - delta)})")

        # Save if improved over DB
        if meta and current_score > meta['score']:
            cur_acts, cur_items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
            new_solution = flat_to_solution(cur_acts, cur_items, num_bots)
            saved = save_solution(difficulty, current_score, new_solution)
            if saved:
                print(f"    ** Saved to DB: {current_score} (was {meta['score']}) **")
                meta = load_meta(difficulty)

        # Neighborhood search: explore rounds near the improvement
        if time_budget and (time.perf_counter() - start_time) > time_budget:
            break

        print(f"    Neighborhood search around R{best_r}...", end='', flush=True)
        nb_improvements = neighborhood_search(
            capture, all_orders, base_flat, num_bots,
            best_r, n_workers, overrides, current_score,
        )
        if nb_improvements:
            nb_r, nb_b, nb_a, nb_score = nb_improvements[0]
            nb_delta = nb_score - current_score
            print(f"    Neighborhood: R{nb_r} B{nb_b} -> {act_names[nb_a]} = {nb_score} (+{nb_delta})")
            overrides[(nb_r, nb_b)] = nb_a
            current_score = nb_score

            if meta and current_score > meta['score']:
                cur_acts, cur_items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
                new_solution = flat_to_solution(cur_acts, cur_items, num_bots)
                saved = save_solution(difficulty, current_score, new_solution)
                if saved:
                    print(f"    ** Saved to DB: {current_score} (was {meta['score']}) **")
                    meta = load_meta(difficulty)

    elapsed = time.perf_counter() - start_time
    total_delta = current_score - base_score
    print()
    print(f"  Final: {current_score} (was {base_score}, {'+' if total_delta >= 0 else ''}{total_delta})")
    print(f"  Overrides: {len(overrides)}")
    print(f"  Time: {elapsed:.1f}s")

    return overrides, current_score


def random_multi_search(difficulty, n_random=2000, max_changes=3,
                        time_budget=None, n_workers=None):
    """Random multi-perturbation: try random sets of 2-3 simultaneous overrides.

    Explores beyond greedy single-action hill climbing to escape local optima.
    """
    import random as rng

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 14)

    capture = load_capture(difficulty)
    solution = load_solution(difficulty)
    if not capture or not solution:
        print(f"No capture/solution for {difficulty}")
        return None, 0

    meta = load_meta(difficulty)
    num_bots = capture['num_bots']
    _, all_orders = init_game_from_capture(capture)

    from zig_ffi import zig_verify_live
    base_score = zig_verify_live(capture, all_orders, solution, num_bots)
    print(f"  Random multi-perturbation: {n_random} trials, max {max_changes} changes")
    print(f"  Baseline: {base_score}")

    base_flat = solution_to_flat(solution, num_bots)

    # Collect all mutable action points
    mutable = []
    for r, round_actions in enumerate(solution):
        for b in range(num_bots):
            act, _ = round_actions[b]
            if act in (ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                mutable.append((r, b, act))

    rng.seed(42)
    tasks = []
    for _ in range(n_random):
        n_changes = rng.randint(1, max_changes)
        points = rng.sample(mutable, min(n_changes, len(mutable)))
        perturbations = []
        for r, b, orig in points:
            alts = [a for a in MOVE_ALTS if a != orig]
            perturbations.append((r, b, rng.choice(alts)))
        tasks.append((
            capture, all_orders, base_flat, num_bots, perturbations,
        ))

    print(f"  Evaluating {len(tasks)} random perturbation sets...", end='', flush=True)
    t0 = time.perf_counter()

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_eval_multi_perturbation, tasks,
                          chunksize=max(1, len(tasks) // (n_workers * 4)))

    dt = time.perf_counter() - t0
    print(f" {dt:.1f}s ({len(tasks)/dt:.0f} evals/s)")

    # Find best
    best_perturbations = None
    best_score = base_score
    n_improvements = 0
    for perturbations, score in results:
        if score > base_score:
            n_improvements += 1
        if score > best_score:
            best_score = score
            best_perturbations = perturbations

    print(f"  Improvements: {n_improvements}/{n_random}")
    print(f"  Best: {best_score} (delta: +{best_score - base_score})")

    if best_perturbations and best_score > base_score:
        act_names = ['wait', 'up', 'down', 'left', 'right', 'pickup', 'dropoff']
        for r, b, a in best_perturbations:
            print(f"    R{r} B{b} -> {act_names[a]}")

        # Save if improved
        if meta and best_score > meta['score']:
            overrides = {(r, b): a for r, b, a in best_perturbations}
            cur_acts, cur_items = apply_overrides(base_flat[0], base_flat[1], overrides, num_bots)
            new_solution = flat_to_solution(cur_acts, cur_items, num_bots)
            saved = save_solution(difficulty, best_score, new_solution)
            if saved:
                print(f"  ** Saved to DB: {best_score} (was {meta['score']}) **")

    return best_perturbations, best_score


def full_search(difficulty, max_iterations=10, n_random=2000,
                time_budget=None, n_workers=None, search_all=False):
    """Full search: iterative hill climbing + random multi-perturbation."""
    print(f"{'='*60}")
    print(f"  Perturbation Search: {difficulty.upper()}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    # Phase 1: Iterative single-action hill climbing
    print(f"\n  Phase 1: Iterative hill climbing")
    iter_budget = time_budget * 0.7 if time_budget else None
    overrides, score1 = iterative_search(
        difficulty, search_all=search_all, max_iterations=max_iterations,
        time_budget=iter_budget, n_workers=n_workers,
    )

    # Phase 2: Random multi-perturbation
    remaining = (time_budget - (time.perf_counter() - t0)) if time_budget else None
    if remaining is None or remaining > 5:
        print(f"\n  Phase 2: Random multi-perturbation")
        _, score2 = random_multi_search(
            difficulty, n_random=n_random, time_budget=remaining, n_workers=n_workers,
        )
    else:
        score2 = 0

    elapsed = time.perf_counter() - t0
    final_meta = load_meta(difficulty)
    final_score = final_meta['score'] if final_meta else 0

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Final DB score: {final_score}")
    print(f"{'='*60}")

    return final_score


def main():
    parser = argparse.ArgumentParser(description='Perturbation search on GPU DP solutions')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--iter', type=int, default=10, help='Max iterations for hill climbing')
    parser.add_argument('--random', type=int, default=2000, help='Random multi-perturbation trials')
    parser.add_argument('--budget', type=float, default=None, help='Time budget in seconds')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--all', action='store_true', help='Search all actions (not just waits)')
    args = parser.parse_args()

    full_search(
        args.difficulty,
        max_iterations=args.iter,
        n_random=args.random,
        time_budget=args.budget,
        n_workers=args.workers,
        search_all=args.all,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
