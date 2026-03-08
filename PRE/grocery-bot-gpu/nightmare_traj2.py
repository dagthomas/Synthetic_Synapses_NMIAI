#!/usr/bin/env python3
"""Nightmare trajectory search V2: multi-bot perturbation + parallel V6 diversity.

Key improvements over V1:
1. Multi-bot perturbation: change 1-5 bots at perturbation point
2. Segment re-plan: replace R..R+seg with fresh V6 (different allocator randomness)
3. Faster iteration via smaller checkpoints
4. Score-proportional round weighting

Usage: python nightmare_traj2.py --seed 7009 --max-time 600
"""
from __future__ import annotations
import sys, time, random, copy
import numpy as np
from game_engine import (
    init_game, step, GameState, MapState, Order,
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


def run_v6_from_state(state: GameState, all_orders: list[Order],
                      from_round: int, num_rounds: int) -> tuple[int, list]:
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    action_log = []
    for rnd in range(from_round, num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def rebuild_checkpoints(seed: int, actions: list, num_rounds: int,
                        interval: int = 10) -> dict:
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def run_trajectory_search_v2(seed: int, max_time: float = 600,
                             verbose: bool = True,
                             initial_actions: list | None = None) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state
    tables = PrecomputedTables.get(ms)

    if initial_actions is not None:
        # Use provided initial solution
        best_actions = [list(a) for a in initial_actions]
        # Compute score
        state = copy.deepcopy(state0)
        for rnd in range(num_rounds):
            step(state, best_actions[rnd], all_orders)
        baseline_score = state.score
        best_score = baseline_score
        print(f'Initial solution: {baseline_score}', file=sys.stderr)
    else:
        # V6 baseline
        print(f'V6 baseline (seed {seed})...', file=sys.stderr)
        state = copy.deepcopy(state0)
        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
        baseline_actions = []
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            baseline_actions.append(list(actions))
            step(state, actions, all_orders)
        baseline_score = state.score
        best_score = baseline_score
        best_actions = baseline_actions
        print(f'V6: {baseline_score}', file=sys.stderr)

    # Build checkpoints every 10 rounds
    checkpoint_interval = 10
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

    iterations = 0
    improvements = 0

    # Weight toward early rounds (more cascade potential)
    round_options = list(range(0, num_rounds - 30, 1))
    round_weights = [max(0.05, 1.0 - r / num_rounds) for r in round_options]

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        # Pick perturbation point
        perturb_round = random.choices(round_options, weights=round_weights, k=1)[0]

        # Pick number of bots to perturb (1-5, biased toward 1-2)
        n_perturb = random.choices([1, 1, 1, 2, 2, 3], k=1)[0]

        # Find nearest checkpoint
        cp_round = (perturb_round // checkpoint_interval) * checkpoint_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints.keys() if r <= perturb_round)

        cp_state, all_orders_cp = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)

        for rnd in range(cp_round, perturb_round):
            step(state, best_actions[rnd], all_orders_cp)

        # Generate perturbation: change n_perturb bots' actions
        perturbed_actions = list(best_actions[perturb_round])
        bots_to_perturb = random.sample(range(num_bots), min(n_perturb, num_bots))
        changed = False

        for bid in bots_to_perturb:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) <= 1:
                continue
            original = best_actions[perturb_round][bid]
            alternatives = [a for a in valid if a != original]
            if alternatives:
                perturbed_actions[bid] = random.choice(alternatives)
                changed = True

        if not changed:
            iterations += 1
            continue

        # Apply perturbation and re-run V6
        perturbed_state = copy.deepcopy(state)
        step(perturbed_state, perturbed_actions, all_orders_cp)

        new_score, new_action_log = run_v6_from_state(
            perturbed_state, all_orders_cp, perturb_round + 1, num_rounds)

        if new_score > best_score:
            full_actions = best_actions[:perturb_round] + [perturbed_actions] + new_action_log
            best_score = new_score
            best_actions = full_actions
            improvements += 1

            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {best_score} '
                      f'(r={perturb_round}, bots={bots_to_perturb}, '
                      f'iter={iterations})', file=sys.stderr)

            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

        iterations += 1
        if iterations % 200 == 0 and verbose:
            elapsed2 = time.time() - t_start
            rate = iterations / elapsed2
            print(f'  [{elapsed2:.0f}s] iter={iterations} best={best_score} '
                  f'imp={improvements} rate={rate:.1f}/s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nTraj V2 done: {best_score} '
          f'(from {baseline_score}, delta={best_score-baseline_score:+d})',
          file=sys.stderr)
    print(f'  Iterations: {iterations}, improvements: {improvements}, '
          f'time: {elapsed:.1f}s', file=sys.stderr)
    return best_score, best_actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=600)
    args = parser.parse_args()

    score, actions = run_trajectory_search_v2(args.seed, args.max_time)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
