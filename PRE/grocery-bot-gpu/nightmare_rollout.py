#!/usr/bin/env python3
"""Nightmare rollout-guided solver: V6 + short rollout evaluation.

At each round, V6 suggests actions. Then we test N perturbations
(random bot actions) with K-round rollout using V6 re-planning.
Keep the perturbation that gives the best K-round score.

Usage: python nightmare_rollout.py --seed 7009
"""
from __future__ import annotations
import sys, time, copy, random
import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6


def get_valid_actions(state: GameState, bid: int, ms: MapState) -> list[tuple[int, int]]:
    bx = int(state.bot_positions[bid, 0])
    by = int(state.bot_positions[bid, 1])
    valid = [(ACT_WAIT, -1)]
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = bx + DX[act], by + DY[act]
        if 0 <= nx < ms.width and 0 <= ny < ms.height:
            if ms.grid[ny, nx] != CELL_WALL:
                valid.append((act, -1))
    if state.bot_inv_count(bid) < INV_CAP:
        for item_idx in range(ms.num_items):
            ix = int(ms.item_positions[item_idx, 0])
            iy = int(ms.item_positions[item_idx, 1])
            if abs(bx - ix) + abs(by - iy) == 1:
                valid.append((ACT_PICKUP, item_idx))
    if state.bot_inv_count(bid) > 0:
        if any(bx == dz[0] and by == dz[1] for dz in ms.drop_off_zones):
            valid.append((ACT_DROPOFF, -1))
    return valid


def rollout(state: GameState, all_orders: list[Order],
            solver: NightmareSolverV6, num_steps: int) -> int:
    """Run V6 for num_steps from state, return score delta."""
    start_score = state.score
    start_round = state.round
    for rnd in range(start_round, min(start_round + num_steps, 500)):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        step(state, actions, all_orders)
    return state.score - start_score


def run_rollout_guided(seed: int, num_perturbations: int = 10,
                       rollout_length: int = 30,
                       verbose: bool = True) -> tuple[int, list]:
    """Run V6 with rollout-guided perturbation at each round."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)

    # Main solver
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)

    action_log = []
    improvements = 0

    for rnd in range(num_rounds):
        state.round = rnd

        # Get V6 baseline actions
        v6_actions = solver.action(state, all_orders, rnd)

        # Only do rollout evaluation at key rounds (every 5th round to save time)
        best_actions = list(v6_actions)
        if rnd % 5 == 0 and rnd < num_rounds - rollout_length:
            # Evaluate baseline rollout
            state_copy = copy.deepcopy(state)
            solver_copy = NightmareSolverV6(ms, tables, future_orders=all_orders)
            step(state_copy, v6_actions, all_orders)
            baseline_delta = rollout(state_copy, all_orders, solver_copy, rollout_length - 1)
            baseline_delta += state_copy.score - state.score  # Include this round's delta
            best_delta = baseline_delta

            # Try perturbations
            for _ in range(num_perturbations):
                bid = random.randint(0, num_bots - 1)
                valid = get_valid_actions(state, bid, ms)
                if len(valid) <= 1:
                    continue
                alt = random.choice(valid)
                if alt == v6_actions[bid]:
                    continue

                perturbed = list(v6_actions)
                perturbed[bid] = alt

                state_copy = copy.deepcopy(state)
                solver_copy = NightmareSolverV6(ms, tables, future_orders=all_orders)
                step(state_copy, perturbed, all_orders)
                delta = rollout(state_copy, all_orders, solver_copy, rollout_length - 1)
                delta += state_copy.score - state.score

                if delta > best_delta:
                    best_delta = delta
                    best_actions = list(perturbed)
                    improvements += 1

        action_log.append(best_actions)
        step(state, best_actions, all_orders)

        if verbose and (rnd < 5 or rnd % 50 == 0):
            elapsed = time.time() - t_start
            print(f'R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d} '
                  f'({elapsed:.1f}s, imp={improvements})', file=sys.stderr)

    elapsed = time.time() - t_start
    if verbose:
        print(f'\nFinal: Score={state.score} Orders={state.orders_completed} '
              f'Improvements={improvements} Time={elapsed:.1f}s', file=sys.stderr)

    return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--perturbations', type=int, default=10)
    parser.add_argument('--rollout', type=int, default=30)
    args = parser.parse_args()

    score, actions = run_rollout_guided(
        args.seed, args.perturbations, args.rollout, verbose=True)

    # Compare to V6 baseline
    v6_score, _ = NightmareSolverV6.run_sim(args.seed, verbose=False)
    print(f'\nV6 baseline: {v6_score}', file=sys.stderr)
    print(f'Rollout:     {score} (delta={score-v6_score:+d})', file=sys.stderr)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
