#!/usr/bin/env python3
"""Nightmare trajectory perturbation: modify V6 decision at critical points, re-plan rest.

At round R, inject a perturbation (random action for one bot), then
let V6 re-plan ALL bots from R+1. This preserves coordination while
exploring different game trajectories.

Usage: python nightmare_traj.py --seed 7009 --max-time 600
"""
from __future__ import annotations
import sys, time, random, argparse, copy
import numpy as np
from game_engine import (
    init_game, step, GameState, MapState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6


def run_v6_from_state(state: GameState, all_orders: list[Order],
                      from_round: int, num_rounds: int) -> tuple[int, list]:
    """Run V6 from a given state to end, return (score, action_log)."""
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


def run_v6_full(seed: int, num_rounds: int = 500) -> tuple[int, list, list]:
    """Run V6 for full game. Returns (score, action_log, checkpoint_states)."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)

    action_log = []
    # Save checkpoints every 25 rounds
    checkpoints = {}
    for rnd in range(num_rounds):
        if rnd % 25 == 0:
            checkpoints[rnd] = (copy.deepcopy(state), all_orders)
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log, checkpoints


def get_valid_actions(state: GameState, bid: int, ms: MapState) -> list[tuple[int, int]]:
    """Get valid actions for a bot."""
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


def rebuild_checkpoints(seed: int, actions: list, num_rounds: int,
                        interval: int = 25) -> dict:
    """Rebuild checkpoints by replaying actions."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def run_trajectory_search(seed: int, max_time: float = 600,
                          verbose: bool = True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    # Get V6 baseline
    print(f'V6 baseline (seed {seed})...', file=sys.stderr)
    baseline_score, baseline_actions, _ = run_v6_full(seed, num_rounds)
    print(f'V6: {baseline_score}', file=sys.stderr)

    best_score = baseline_score
    best_actions = [list(a) for a in baseline_actions]

    # Build checkpoints from best solution
    print(f'Building checkpoints...', file=sys.stderr)
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds)

    iterations = 0
    improvements = 0
    stale_count = 0
    checkpoint_interval = 25

    # Bias toward early rounds (more cascade effect)
    round_weights = [max(0.1, 1.0 - r / num_rounds) for r in range(0, num_rounds - 50, 5)]
    round_options = list(range(0, num_rounds - 50, 5))

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        # Pick perturbation point (biased toward early rounds)
        perturb_round = random.choices(round_options, weights=round_weights, k=1)[0]
        perturb_bot = random.randint(0, num_bots - 1)

        # Find nearest checkpoint
        cp_round = (perturb_round // checkpoint_interval) * checkpoint_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints.keys() if r <= perturb_round)

        cp_state, all_orders = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)

        for rnd in range(cp_round, perturb_round):
            step(state, best_actions[rnd], all_orders)

        # Get valid actions
        valid = get_valid_actions(state, perturb_bot, state.map_state)
        original_action = best_actions[perturb_round][perturb_bot]

        # Pick ONE random alternative action
        alternatives = [a for a in valid if a != original_action]
        if not alternatives:
            continue
        alt_action = random.choice(alternatives)

        # Apply perturbation
        perturbed_state = copy.deepcopy(state)
        perturbed_actions = list(best_actions[perturb_round])
        perturbed_actions[perturb_bot] = alt_action
        step(perturbed_state, perturbed_actions, all_orders)

        # Re-run V6 from perturb_round + 1
        new_score, new_action_log = run_v6_from_state(
            perturbed_state, all_orders, perturb_round + 1, num_rounds)

        if new_score > best_score:
            full_actions = best_actions[:perturb_round] + [perturbed_actions] + new_action_log
            best_score = new_score
            best_actions = full_actions
            improvements += 1
            stale_count = 0

            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {best_score} '
                      f'(bot={perturb_bot}, r={perturb_round}, '
                      f'act={alt_action}, iter={iterations})',
                      file=sys.stderr)

            # Rebuild checkpoints from new best
            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds)
        else:
            stale_count += 1

        iterations += 1
        if iterations % 200 == 0 and verbose:
            elapsed2 = time.time() - t_start
            rate = iterations / elapsed2
            print(f'  [{elapsed2:.0f}s] iter={iterations} best={best_score} '
                  f'imp={improvements} stale={stale_count} rate={rate:.1f}/s',
                  file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nTrajectory search done: {best_score} '
          f'(from V6 {baseline_score}, delta={best_score-baseline_score:+d})',
          file=sys.stderr)
    print(f'  Iterations: {iterations}, improvements: {improvements}, '
          f'time: {elapsed:.1f}s', file=sys.stderr)
    return best_score, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=600)
    args = parser.parse_args()

    score, actions = run_trajectory_search(args.seed, args.max_time)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
