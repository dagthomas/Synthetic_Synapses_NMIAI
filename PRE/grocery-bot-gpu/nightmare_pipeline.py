#!/usr/bin/env python3
"""Nightmare pipeline: V6 → Trajectory Search → GPU Refine → Iterate.

Combines three optimization approaches:
1. V6 heuristic (baseline ~330)
2. Trajectory perturbation (V6 re-planning, +~70)
3. GPU DP refinement (per-bot optimization, +~70)

Usage: python nightmare_pipeline.py --seed 7009 --max-time 1200
"""
from __future__ import annotations
import sys, time, argparse

from configs import DIFF_ROUNDS


def run_pipeline(seed: int, max_time: float = 1200, max_states: int = 50000,
                 verbose: bool = True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']

    # Phase 1: V6 baseline
    print(f'\n{"="*60}', file=sys.stderr)
    print(f'Phase 1: V6 heuristic (seed {seed})', file=sys.stderr)
    print(f'{"="*60}', file=sys.stderr)
    from nightmare_solver_v6 import NightmareSolverV6
    v6_score, v6_actions = NightmareSolverV6.run_sim(seed, verbose=False)
    print(f'  V6: {v6_score} ({time.time()-t_start:.1f}s)', file=sys.stderr)
    best_score = v6_score
    best_actions = v6_actions

    # Phase 2: Trajectory search (CPU-based, no GPU needed)
    remaining = max_time - (time.time() - t_start)
    traj_budget = min(remaining * 0.35, 300)  # 35% of budget, max 300s
    if traj_budget > 30:
        print(f'\n{"="*60}', file=sys.stderr)
        print(f'Phase 2: Trajectory search ({traj_budget:.0f}s budget)', file=sys.stderr)
        print(f'{"="*60}', file=sys.stderr)
        from nightmare_traj import run_trajectory_search
        traj_score, traj_actions = run_trajectory_search(
            seed, max_time=traj_budget, verbose=verbose)
        if traj_score > best_score:
            best_score = traj_score
            best_actions = traj_actions
            print(f'  Trajectory improved: {best_score}', file=sys.stderr)

    # Phase 3: GPU refine from best solution
    remaining = max_time - (time.time() - t_start)
    if remaining > 60:
        print(f'\n{"="*60}', file=sys.stderr)
        print(f'Phase 3: GPU DP refinement ({remaining:.0f}s remaining)', file=sys.stderr)
        print(f'{"="*60}', file=sys.stderr)

        from iterate_local import make_capture_from_seed
        from gpu_sequential_solver import refine_from_solution

        capture, ms, all_orders = make_capture_from_seed('nightmare', seed, 30)

        try:
            ref_score, ref_actions = refine_from_solution(
                best_actions,
                capture_data=capture,
                difficulty='nightmare',
                device='cuda',
                no_filler=True,
                max_states=max_states,
                max_refine_iters=15,
                max_time_s=remaining - 30,
                speed_bonus=100.0,
            )
            if ref_score > best_score:
                best_score = ref_score
                best_actions = ref_actions
                print(f'  GPU refined: {best_score}', file=sys.stderr)
            else:
                print(f'  GPU refine: {ref_score} (no improvement)', file=sys.stderr)
        except Exception as e:
            print(f'  GPU refine failed: {e}', file=sys.stderr)

    # Phase 4: Another trajectory search from GPU-refined solution
    remaining = max_time - (time.time() - t_start)
    if remaining > 60:
        print(f'\n{"="*60}', file=sys.stderr)
        print(f'Phase 4: Trajectory search round 2 ({remaining:.0f}s remaining)', file=sys.stderr)
        print(f'{"="*60}', file=sys.stderr)
        # TODO: trajectory search from GPU-refined solution
        # (requires modifying traj search to accept initial actions)

    elapsed = time.time() - t_start
    print(f'\n{"="*60}', file=sys.stderr)
    print(f'  FINAL: {best_score} (V6={v6_score}, elapsed={elapsed:.1f}s)', file=sys.stderr)
    print(f'{"="*60}', file=sys.stderr)

    # Save
    from solution_store import save_solution, save_capture
    from iterate_local import make_capture_from_seed
    capture, _, _ = make_capture_from_seed('nightmare', seed, 40)
    save_capture('nightmare', capture)
    saved = save_solution('nightmare', best_score, best_actions, seed=seed)
    print(f'  Saved: {saved}', file=sys.stderr)

    return best_score, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=1200)
    parser.add_argument('--max-states', type=int, default=50000)
    args = parser.parse_args()

    run_pipeline(args.seed, args.max_time, args.max_states)


if __name__ == '__main__':
    main()
