#!/usr/bin/env python3
"""Test nightmare hybrid: V6 heuristic → GPU DP refinement.

V6 gets ~320 with good collision avoidance.
GPU DP refines individual bots while preserving collision-free paths.
"""
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--max-states', type=int, default=50000)
    parser.add_argument('--max-time', type=int, default=600)
    parser.add_argument('--refine-iters', type=int, default=3)
    parser.add_argument('--speed-bonus', type=float, default=100.0)
    parser.add_argument('--num-orders', type=int, default=30)
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Generate V6 heuristic solution
    print(f"=== Step 1: V6 heuristic (seed {args.seed}) ===", file=sys.stderr)
    from nightmare_solver_v6 import NightmareSolverV6
    v6_score, v6_actions = NightmareSolverV6.run_sim(args.seed, verbose=True)
    print(f"  V6 score: {v6_score} ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Step 2: Build capture data with multi-dropoff
    from iterate_local import make_capture_from_seed
    capture, ms, all_orders = make_capture_from_seed('nightmare', args.seed, args.num_orders)
    print(f"  Capture: {len(capture['orders'])} orders, "
          f"drop_off_zones={capture.get('drop_off_zones')}", file=sys.stderr)

    # Step 3: GPU DP refinement from V6 solution
    print(f"\n=== Step 2: GPU DP refinement ===", file=sys.stderr)
    from gpu_sequential_solver import refine_from_solution
    ref_score, ref_actions = refine_from_solution(
        v6_actions,
        capture_data=capture,
        difficulty='nightmare',
        device='cuda',
        no_filler=True,
        max_states=args.max_states,
        max_refine_iters=args.refine_iters,
        max_time_s=args.max_time - (time.time() - t0),
        speed_bonus=args.speed_bonus,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  V6 heuristic: {v6_score}", file=sys.stderr)
    print(f"  GPU refined:  {ref_score}", file=sys.stderr)
    print(f"  Delta:        {ref_score - v6_score:+d}", file=sys.stderr)
    print(f"  Time:         {elapsed:.1f}s", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Save if improved
    if ref_score > 0:
        from solution_store import save_solution, save_capture
        save_capture('nightmare', capture)
        saved = save_solution('nightmare', ref_score, ref_actions, seed=args.seed)
        if saved:
            print(f"  Saved as best solution for nightmare", file=sys.stderr)

    return ref_score

if __name__ == '__main__':
    main()
