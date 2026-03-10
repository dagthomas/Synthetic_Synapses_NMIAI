#!/usr/bin/env python3
"""Local iterate pipeline: SOLVE → SIMULATE → DISCOVER ORDERS → REPEAT.

Two modes:
  --discover: Iterative order discovery (like live pipeline)
  (default): Full order foresight - give all orders, max solve time

Usage:
    python iterate_local.py hard --seed 42 --max-time 600 --max-states 100000
    python iterate_local.py hard --seed 42 --discover --max-time 300
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import numpy as np

from game_engine import (
    init_game, step as cpu_step, build_map,
    generate_all_orders, GameState, MapState, Order, CaptureData,
    MAX_ROUNDS, ACT_WAIT, CELL_WALL,
)
from configs import CONFIGS, DIFF_ROUNDS
from gpu_sequential_solver import solve_sequential, refine_from_solution, SolveConfig, solve_nightmare_zones


def make_capture_from_seed(difficulty: str, seed: int,
                           num_orders: int) -> tuple[CaptureData, MapState, list[Order]]:
    """Build capture_data dict from seed-based game engine (like server capture)."""
    ms = build_map(difficulty)
    all_orders = generate_all_orders(seed, ms, difficulty, count=max(num_orders, 100))

    # Convert wall positions from grid
    walls = []
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] == CELL_WALL:
                walls.append([x, y])

    # Items in capture format
    items = []
    for item in ms.items:
        items.append({
            'id': item['id'],
            'type': item['type'],
            'position': list(item['position']),
        })

    # Orders in capture format
    orders = []
    for o in all_orders[:num_orders]:
        req_names = [ms.item_type_names[int(tid)] for tid in o.required]
        orders.append({'items_required': req_names})

    capture = {
        'difficulty': difficulty,
        'num_bots': CONFIGS[difficulty]['bots'],
        'grid': {'width': ms.width, 'height': ms.height, 'walls': walls},
        'items': items,
        'drop_off': list(ms.drop_off),
        'spawn': list(ms.spawn),
        'orders': orders,
    }
    # Multi-dropoff support (nightmare has 3 zones)
    if hasattr(ms, 'drop_off_zones') and len(ms.drop_off_zones) > 1:
        capture['drop_off_zones'] = [list(z) for z in ms.drop_off_zones]
    return capture, ms, all_orders


def simulate_solution(difficulty: str, seed: int,
                      actions: list[list[tuple[int, int]]],
                      all_orders: list[Order]) -> tuple[int, int, int]:
    """Simulate a solution using game_engine, return (score, orders_completed, max_order_seen)."""
    gs, _ = init_game(seed, difficulty)
    score = 0
    max_order_seen = 2
    num_rounds = DIFF_ROUNDS.get(difficulty, 300)

    for r in range(min(len(actions), num_rounds)):
        round_actions = actions[r]
        delta = cpu_step(gs, round_actions, all_orders)
        score += delta
        max_order_seen = max(max_order_seen, gs.next_order_idx)

    return score, gs.orders_completed, max_order_seen


def run_full_foresight(args: argparse.Namespace) -> int:
    """Solve → Simulate → Discover Orders → Re-solve loop.

    Each solve produces a solution. Simulating it reveals how many orders
    the bots actually reach. Then we re-solve with exactly those orders
    (tighter focus = less state fragmentation = better beam search).
    Repeat until score stabilizes or time runs out.
    """
    t_start = time.time()

    print(f"=== Solve→Sim→Discover→Repeat: {args.difficulty} seed={args.seed} ===",
          file=sys.stderr)
    print(f"    Budget: {args.max_time}s, states: {args.max_states}, "
          f"refine: {args.refine_iters}, orderings: {args.orderings}", file=sys.stderr)

    # Start with a generous initial order set
    num_orders = args.num_orders
    ms = build_map(args.difficulty)
    all_orders = generate_all_orders(args.seed, ms, args.difficulty, count=100)

    best_score = 0
    best_actions = None

    for loop in range(args.max_loops):
        elapsed = time.time() - t_start
        remaining = args.max_time - elapsed
        if remaining < 30:
            print(f"\n  Time budget exhausted ({remaining:.0f}s left), stopping",
                  file=sys.stderr)
            break

        # Budget allocation: first loop gets more time, later loops refine
        if loop == 0:
            loop_time = min(remaining * 0.5, remaining - 30)
        else:
            loop_time = min(remaining * 0.7, remaining - 15)

        capture, _, _ = make_capture_from_seed(args.difficulty, args.seed, num_orders)
        loop_speed_bonus = args.speed_bonus * (args.speed_decay ** loop)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Loop {loop}: orders={num_orders}, budget={loop_time:.0f}s, "
              f"best={best_score}, speed_bonus={loop_speed_bonus:.1f}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        try:
            if args.difficulty == 'nightmare':
                # Nightmare: GPU DP refinement proven useless (0 improvement
                # on any of 20 bots). Use all time for heuristic training.
                from nightmare_offline import NightmareTrainer
                train_time = max(30, loop_time)
                print(f"  Nightmare: V3/V4 training ({train_time:.0f}s)...",
                      file=sys.stderr)
                trainer = NightmareTrainer(seed=args.seed, verbose=True)
                score, actions = trainer.train(max_time=train_time)
            elif best_actions and loop > 0:
                # Warm-start: refine from best solution with new order set
                score, actions = refine_from_solution(
                    best_actions, capture_data=capture,
                    difficulty=args.difficulty, device='cuda',
                    no_filler=True,
                    max_time_s=loop_time, max_states=args.max_states,
                    max_refine_iters=args.refine_iters,
                    speed_bonus=loop_speed_bonus)
            else:
                # Cold start: full sequential solve
                score, actions = solve_sequential(
                    capture_data=capture, difficulty=args.difficulty,
                    device='cuda', verbose=True, no_filler=True,
                    max_time_s=loop_time, max_states=args.max_states,
                    max_refine_iters=args.refine_iters,
                    num_pass1_orderings=args.orderings,
                    speed_bonus=loop_speed_bonus,
                    use_2bot_dp=args.two_bot)
        except Exception as e:
            print(f"  Solve failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            break

        # Simulate to verify and discover how many orders are actually reached
        sim_score, orders_completed, max_order_seen = simulate_solution(
            args.difficulty, args.seed, actions, all_orders)

        print(f"\n  Loop {loop}: GPU={score}, Sim={sim_score}, "
              f"completed={orders_completed}, seen={max_order_seen}", file=sys.stderr)

        if sim_score > best_score:
            best_score = sim_score
            best_actions = actions
            print(f"  *** NEW BEST: {best_score} ***", file=sys.stderr)

        # Discover: adjust order count based on what the simulation reached
        # More orders = more to score, but also more state fragmentation
        old_orders = num_orders
        # Give 3 extra orders beyond what was seen (headroom for improvements)
        new_orders = max_order_seen + 3
        if new_orders <= num_orders and loop > 0:
            # No new orders discovered, try with fewer (less fragmentation)
            new_orders = max(orders_completed + 3, num_orders - 2)
        num_orders = max(new_orders, orders_completed + 2)

        if num_orders == old_orders and loop > 0 and sim_score <= best_score:
            print(f"  No new orders and no improvement, continuing with refinement...",
                  file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  FINAL: {best_score} points in {elapsed:.1f}s", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Save
    if best_score > 0 and best_actions:
        from solution_store import save_solution, save_capture
        capture_full, _, _ = make_capture_from_seed(args.difficulty, args.seed, 40)
        save_capture(args.difficulty, capture_full)
        saved = save_solution(args.difficulty, best_score, best_actions, seed=args.seed)
        if saved:
            print(f"  Saved as best solution for {args.difficulty}", file=sys.stderr)
        else:
            print(f"  Not saved (existing solution is better)", file=sys.stderr)

    return best_score


def run_iterative(args: argparse.Namespace) -> int:
    """Iterative order discovery mode (simulates live pipeline)."""
    t_start = time.time()
    best_score = 0
    best_actions = None
    num_orders = 2

    print(f"=== Iterative Solve: {args.difficulty} seed={args.seed} ===", file=sys.stderr)
    print(f"    Budget: {args.max_time}s, states: {args.max_states}", file=sys.stderr)

    ms = build_map(args.difficulty)
    all_orders = generate_all_orders(args.seed, ms, args.difficulty, count=100)

    for iteration in range(args.max_iters):
        elapsed = time.time() - t_start
        remaining = args.max_time - elapsed
        if remaining < 20:
            break

        # Give more time to later iterations (they have more orders)
        iter_time = min(remaining * 0.6, max(30, remaining / 3))
        iter_time = min(iter_time, remaining - 10)

        iter_speed_bonus = args.speed_bonus * (args.speed_decay ** iteration)
        print(f"\n  [Iter {iteration}] Orders: {num_orders}, Budget: {iter_time:.0f}s, "
              f"Best: {best_score}, speed_bonus={iter_speed_bonus:.1f}", file=sys.stderr)

        capture, _, _ = make_capture_from_seed(args.difficulty, args.seed, num_orders)

        if args.difficulty == 'nightmare':
            # Nightmare: GPU DP refinement proven useless. Use heuristic training.
            from nightmare_offline import NightmareTrainer
            train_time = max(30, iter_time)
            print(f"  Nightmare: V3/V4 training ({train_time:.0f}s)...",
                  file=sys.stderr)
            trainer = NightmareTrainer(seed=args.seed, verbose=True)
            score, actions = trainer.train(max_time=train_time)
        elif best_actions and iteration > 0:
            try:
                score, actions = refine_from_solution(
                    best_actions, capture_data=capture,
                    difficulty=args.difficulty, device='cuda',
                    no_filler=True,
                    max_time_s=iter_time, max_states=args.max_states,
                    max_refine_iters=args.refine_iters,
                    speed_bonus=iter_speed_bonus)
            except Exception as e:
                print(f"  Refine failed: {e}, trying cold start", file=sys.stderr)
                score, actions = solve_sequential(
                    capture_data=capture, difficulty=args.difficulty,
                    device='cuda', verbose=True, no_filler=True,
                    max_time_s=iter_time, max_states=args.max_states,
                    max_refine_iters=args.refine_iters,
                    speed_bonus=iter_speed_bonus,
                    use_2bot_dp=args.two_bot)
        else:
            score, actions = solve_sequential(
                capture_data=capture, difficulty=args.difficulty,
                device='cuda', verbose=True, no_filler=True,
                max_time_s=iter_time, max_states=args.max_states,
                max_refine_iters=args.refine_iters,
                speed_bonus=iter_speed_bonus,
                use_2bot_dp=args.two_bot)

        sim_score, orders_completed, max_order_seen = simulate_solution(
            args.difficulty, args.seed, actions, all_orders)

        print(f"  [Iter {iteration}] Score: {sim_score}, Completed: {orders_completed}, "
              f"Seen: {max_order_seen}", file=sys.stderr)

        if sim_score > best_score:
            best_score = sim_score
            best_actions = actions
            print(f"  *** NEW BEST: {best_score} ***", file=sys.stderr)

        old_orders = num_orders
        num_orders = max(num_orders, max_order_seen + 2)
        if num_orders == old_orders:
            num_orders = min(num_orders + 3, 100)

    elapsed = time.time() - t_start
    print(f"\n  FINAL: {best_score} points in {elapsed:.1f}s", file=sys.stderr)

    if best_score > 0 and best_actions:
        from solution_store import save_solution, save_capture
        capture_full, _, _ = make_capture_from_seed(args.difficulty, args.seed, num_orders)
        save_capture(args.difficulty, capture_full)
        saved = save_solution(args.difficulty, best_score, best_actions, seed=args.seed)
        if saved:
            print(f"  Saved as best solution for {args.difficulty}", file=sys.stderr)

    return best_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-time', type=int, default=600, help='Total time budget')
    parser.add_argument('--max-states', type=int, default=50000)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--max-loops', type=int, default=5,
                        help='Max solve→sim→discover loops (default mode)')
    parser.add_argument('--num-orders', type=int, default=25,
                        help='Initial number of orders to give solver')
    parser.add_argument('--refine-iters', type=int, default=20)
    parser.add_argument('--orderings', type=int, default=3,
                        help='Number of pass1 bot orderings to try')
    parser.add_argument('--discover', action='store_true',
                        help='Iterative order discovery mode')
    parser.add_argument('--speed-bonus', type=float, default=100.0,
                        help='Speed bonus coefficient (higher = prefer faster solutions)')
    parser.add_argument('--speed-decay', type=float, default=0.5,
                        help='Per-loop multiplicative decay for speed bonus')
    parser.add_argument('--2bot', action='store_true', dest='two_bot',
                        help='Use joint 2-bot DP for pair planning')
    args = parser.parse_args()

    if args.discover:
        score = run_iterative(args)
    else:
        score = run_full_foresight(args)

    return score


if __name__ == '__main__':
    score = main()
    sys.exit(0 if score > 0 else 1)
