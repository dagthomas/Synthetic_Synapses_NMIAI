#!/usr/bin/env python3
"""Deep solver: exhaustive orderings + deep refinement + 3-bot joint DP.

Designed for long training sessions (hours). Uses the 5090's full 32GB VRAM
with larger state budgets and more thorough search than the 288s pipeline.

Phases:
  1. Exhaustive Pass 1: Try all N! bot orderings (or top-K screened subset)
  2. Deep Refinement: 50+ iterations with high state counts
  3. Joint 3-Bot Refinement: Replace weakest triple with joint-optimal plans
  4. Iterate phases 2-3 until convergence

Usage:
    python -m b200_solver.deep_solve hard --max-hours 3
    python -m b200_solver.deep_solve expert --max-hours 6 --states-3bot 100000
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Optional

# Add parent dir for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from game_engine import (
    init_game_from_capture, MAX_ROUNDS, ACT_WAIT,
    GameState, Order,
)
from gpu_sequential_solver import (
    solve_sequential, refine_from_solution,
    pre_simulate_locked, cpu_verify, _make_combined, _fresh_gs,
    compute_bot_contributions, SolveConfig,
)
from solution_store import load_capture, save_solution, load_meta, load_solution


def exhaustive_pass1(capture_data: dict, difficulty: str,
                     max_states: int = 100_000,
                     max_orderings: int = 120,
                     speed_bonus: float = 100.0,
                     max_time_s: Optional[float] = None,
                     verbose: bool = True) -> tuple[int, list]:
    """Try all (or top-K) bot orderings to find the best Pass 1 solution.

    For 5 bots: 120 orderings. For 3 bots: 6. For 10 bots: samples top-K.
    Each ordering runs sequential DP with the given state budget.
    """
    gs, all_orders = init_game_from_capture(
        capture_data, num_orders=len(capture_data['orders']))
    num_bots = len(gs.bot_positions)

    # Generate all permutations (or sample for large bot counts)
    all_perms = list(itertools.permutations(range(num_bots)))
    if len(all_perms) > max_orderings:
        import random
        rng = random.Random(42)
        # Always include forward and reverse
        selected = [list(range(num_bots)), list(range(num_bots - 1, -1, -1))]
        remaining = [p for p in all_perms if list(p) not in selected]
        rng.shuffle(remaining)
        selected.extend(list(p) for p in remaining[:max_orderings - 2])
        all_perms = [tuple(p) for p in selected]

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"EXHAUSTIVE PASS 1: {len(all_perms)} orderings, "
              f"{max_states//1000}K states", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    t0 = time.time()
    best_score = 0
    best_actions = None
    best_ordering = None

    for i, ordering in enumerate(all_perms):
        if max_time_s and (time.time() - t0) > max_time_s:
            if verbose:
                print(f"\n  Time limit reached after {i}/{len(all_perms)} orderings",
                      file=sys.stderr)
            break

        if verbose:
            order_str = ','.join(str(b) for b in ordering)
            print(f"\n  Ordering {i+1}/{len(all_perms)}: [{order_str}]",
                  file=sys.stderr)

        # Quick sequential DP with this ordering (no refinement)
        try:
            score, actions = solve_sequential(
                capture_data=capture_data,
                difficulty=difficulty,
                device='cuda',
                verbose=False,
                no_filler=True,
                max_states=max_states,
                max_refine_iters=0,       # No refinement — just Pass 1
                num_pass1_orderings=1,
                bot_order=list(ordering),
                speed_bonus=speed_bonus,
                max_time_s=60,            # Max 60s per ordering
            )
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}", file=sys.stderr)
            continue

        if verbose:
            elapsed = time.time() - t0
            marker = " ***" if score > best_score else ""
            print(f"    Score: {score}{marker} ({elapsed:.0f}s elapsed)",
                  file=sys.stderr)

        if score > best_score:
            best_score = score
            best_actions = actions
            best_ordering = ordering

    if verbose:
        elapsed = time.time() - t0
        order_str = ','.join(str(b) for b in best_ordering) if best_ordering else '?'
        print(f"\n  Best ordering: [{order_str}] → score={best_score} "
              f"({elapsed:.0f}s)", file=sys.stderr)

    return best_score, best_actions


def joint_3bot_refine(capture_data: dict, difficulty: str,
                      current_actions: list, current_score: int,
                      max_states: int = 50_000,
                      max_time_s: Optional[float] = None,
                      verbose: bool = True) -> tuple[int, list]:
    """Refine by replacing weakest 3-bot group with joint-optimal plan.

    Identifies the 3 weakest bots (by marginal contribution), plans them
    jointly with 3-bot DP while keeping other bots locked, and keeps the
    result only if it improves the total score.
    """
    from b200_solver.joint_3bot_dp import GPU3BotDP

    gs, all_orders = init_game_from_capture(
        capture_data, num_orders=len(capture_data['orders']))
    num_bots = len(gs.bot_positions)

    if num_bots < 3:
        return current_score, current_actions

    # Convert to per-bot format
    bot_actions = {}
    for bid in range(num_bots):
        bot_actions[bid] = [(r[bid][0], r[bid][1]) for r in current_actions]

    # Find weakest 3 bots
    dp_bots = list(range(min(num_bots, 7)))  # Only consider DP bots
    contribs = compute_bot_contributions(
        gs.copy(), all_orders, bot_actions, num_bots, dp_bots,
        capture_data=capture_data, no_filler=True)

    sorted_bots = sorted(dp_bots, key=lambda b: contribs.get(b, 0))
    triple = tuple(sorted_bots[:3])

    if verbose:
        contrib_str = ', '.join(f'b{b}:{contribs.get(b,0)}' for b in sorted_bots)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"JOINT 3-BOT REFINEMENT", file=sys.stderr)
        print(f"  Contributions: {contrib_str}", file=sys.stderr)
        print(f"  Target triple: bots {triple}", file=sys.stderr)
        print(f"  States: {max_states:,}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    t0 = time.time()

    # Lock all bots NOT in the triple
    locked_ids = sorted(b for b in range(num_bots) if b not in triple)
    locked = None
    if locked_ids:
        locked = pre_simulate_locked(
            gs.copy(), all_orders, bot_actions, locked_ids)

    # Run 3-bot joint DP
    solver = GPU3BotDP(
        gs.map_state, all_orders, bot_ids=triple,
        device='cuda', num_bots_total=num_bots,
        locked_trajectories=locked,
        speed_bonus=100.0)

    joint_score, joint_acts = solver.dp_search(
        gs.copy(), max_states=max_states, verbose=verbose)

    del solver
    import torch
    torch.cuda.empty_cache()

    # Apply joint actions and verify
    for bid in triple:
        bot_actions[bid] = joint_acts[bid]

    combined = _make_combined(bot_actions, num_bots)
    new_score = cpu_verify(gs.copy(), all_orders, combined, num_bots)

    elapsed = time.time() - t0

    if new_score > current_score:
        delta = new_score - current_score
        if verbose:
            print(f"\n  Joint 3-bot: {current_score} → {new_score} "
                  f"(+{delta}!) in {elapsed:.0f}s", file=sys.stderr)
        return new_score, combined
    else:
        if verbose:
            print(f"\n  Joint 3-bot: {new_score} (no improvement, keeping "
                  f"{current_score}) in {elapsed:.0f}s", file=sys.stderr)
        return current_score, current_actions


def deep_refine(capture_data: dict, difficulty: str,
                current_actions: list, current_score: int,
                max_states: int = 200_000,
                max_iters: int = 50,
                speed_bonus: float = 100.0,
                max_time_s: Optional[float] = None,
                verbose: bool = True) -> tuple[int, list]:
    """Deep refinement with high state count and many iterations."""
    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"DEEP REFINEMENT: {max_states//1000}K states, "
              f"{max_iters} iters", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    score, actions = refine_from_solution(
        current_actions,
        capture_data=capture_data,
        difficulty=difficulty,
        device='cuda',
        no_filler=True,
        max_states=max_states,
        max_refine_iters=max_iters,
        speed_bonus=speed_bonus,
        max_time_s=max_time_s,
    )
    return max(score, current_score), actions if score >= current_score else current_actions


def main():
    parser = argparse.ArgumentParser(
        description='Deep GPU solver: exhaustive orderings + 3-bot joint DP')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--max-hours', type=float, default=3.0,
                        help='Total time budget in hours (default: 3)')
    parser.add_argument('--states-1bot', type=int, default=200_000,
                        help='Max states for single-bot DP (default: 200K)')
    parser.add_argument('--states-3bot', type=int, default=50_000,
                        help='Max states for 3-bot joint DP (default: 50K)')
    parser.add_argument('--max-orderings', type=int, default=120,
                        help='Max orderings to try (default: 120 = 5!)')
    parser.add_argument('--refine-iters', type=int, default=50,
                        help='Deep refinement iterations (default: 50)')
    parser.add_argument('--speed-bonus', type=float, default=100.0)
    parser.add_argument('--skip-orderings', action='store_true',
                        help='Skip exhaustive orderings (use existing solution)')
    parser.add_argument('--skip-joint', action='store_true',
                        help='Skip 3-bot joint refinement')
    args = parser.parse_args()

    diff = args.difficulty
    max_time = args.max_hours * 3600
    t_start = time.time()

    capture = load_capture(diff)
    if not capture:
        print(f"No capture data for {diff}. Run production_run.py first.",
              file=sys.stderr)
        sys.exit(1)

    meta = load_meta(diff)
    prev_score = meta.get('score', 0) if meta else 0
    n_orders = len(capture.get('orders', []))

    print(f"\n{'#'*60}", file=sys.stderr)
    print(f"  DEEP SOLVER: {diff.upper()}", file=sys.stderr)
    print(f"  Orders: {n_orders}, Current best: {prev_score}", file=sys.stderr)
    print(f"  Budget: {args.max_hours:.1f} hours", file=sys.stderr)
    print(f"  1-bot states: {args.states_1bot:,}", file=sys.stderr)
    print(f"  3-bot states: {args.states_3bot:,}", file=sys.stderr)
    print(f"  Max orderings: {args.max_orderings}", file=sys.stderr)
    print(f"  Refine iters: {args.refine_iters}", file=sys.stderr)
    print(f"{'#'*60}\n", file=sys.stderr)

    best_score = prev_score
    best_actions = load_solution(diff)

    # Phase 1: Exhaustive orderings
    if not args.skip_orderings:
        remaining = max_time - (time.time() - t_start)
        # Allocate 40% of time to orderings
        ordering_budget = remaining * 0.4

        score, actions = exhaustive_pass1(
            capture, diff,
            max_states=args.states_1bot // 2,  # Lower states for speed
            max_orderings=args.max_orderings,
            speed_bonus=args.speed_bonus,
            max_time_s=ordering_budget)

        if score > best_score:
            best_score = score
            best_actions = actions
            saved = save_solution(diff, score, actions)
            print(f"\n  Orderings improved: {prev_score} → {score} "
                  f"(saved={saved})", file=sys.stderr)

    if not best_actions:
        print("No solution to refine. Run solve_sequential first.",
              file=sys.stderr)
        sys.exit(1)

    # Phase 2: Deep refinement
    remaining = max_time - (time.time() - t_start)
    if remaining > 60:
        refine_budget = remaining * 0.5 if not args.skip_joint else remaining * 0.8

        score, actions = deep_refine(
            capture, diff, best_actions, best_score,
            max_states=args.states_1bot,
            max_iters=args.refine_iters,
            speed_bonus=args.speed_bonus,
            max_time_s=refine_budget)

        if score > best_score:
            best_score = score
            best_actions = actions
            saved = save_solution(diff, score, actions)
            print(f"\n  Refinement improved: → {score} (saved={saved})",
                  file=sys.stderr)

    # Phase 3: Joint 3-bot refinement
    if not args.skip_joint:
        remaining = max_time - (time.time() - t_start)
        if remaining > 120:
            # Try joint 3-bot on multiple triples
            gs, all_orders = init_game_from_capture(
                capture, num_orders=len(capture['orders']))
            num_bots = len(gs.bot_positions)

            if num_bots >= 3:
                # Multiple rounds of joint refinement
                for joint_round in range(3):
                    remaining = max_time - (time.time() - t_start)
                    if remaining < 120:
                        break

                    print(f"\n--- Joint 3-bot round {joint_round+1}/3 ---",
                          file=sys.stderr)
                    score, actions = joint_3bot_refine(
                        capture, diff, best_actions, best_score,
                        max_states=args.states_3bot,
                        max_time_s=min(remaining * 0.4, 600))

                    if score > best_score:
                        best_score = score
                        best_actions = actions
                        saved = save_solution(diff, score, actions)
                        print(f"\n  Joint improved: → {score} (saved={saved})",
                              file=sys.stderr)
                    else:
                        break  # No improvement, stop joint rounds

    # Phase 4: Final deep refinement with remaining time
    remaining = max_time - (time.time() - t_start)
    if remaining > 120:
        print(f"\n--- Final deep refinement ({remaining:.0f}s remaining) ---",
              file=sys.stderr)
        score, actions = deep_refine(
            capture, diff, best_actions, best_score,
            max_states=args.states_1bot,
            max_iters=args.refine_iters,
            speed_bonus=args.speed_bonus * 0.5,  # Reduce speed bonus for final polish
            max_time_s=remaining - 10)

        if score > best_score:
            best_score = score
            best_actions = actions
            save_solution(diff, score, actions)

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'#'*60}", file=sys.stderr)
    print(f"  DEEP SOLVER COMPLETE", file=sys.stderr)
    print(f"  {diff}: {prev_score} → {best_score} (+{best_score - prev_score})",
          file=sys.stderr)
    print(f"  Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)",
          file=sys.stderr)
    print(f"{'#'*60}", file=sys.stderr)

    print(json.dumps({
        'type': 'deep_solve_complete',
        'difficulty': diff,
        'prev_score': prev_score,
        'final_score': best_score,
        'improvement': best_score - prev_score,
        'total_time': round(total_time, 1),
    }))


if __name__ == '__main__':
    main()
