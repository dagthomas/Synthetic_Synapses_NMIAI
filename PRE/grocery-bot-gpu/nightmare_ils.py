#!/usr/bin/env python3
"""Nightmare ILS: Iterated Local Search with large kicks.

Strategy:
1. Start from best solution
2. Apply large "kick" (replace 20-100 rounds with fresh V6 from perturbed state)
3. Run trajectory search (local search phase)
4. If improved, accept as new base
5. Repeat

Usage: python nightmare_ils.py --seed 7009 --max-time 3600
"""
from __future__ import annotations
import sys, time, random, copy, argparse
import numpy as np
from game_engine import (
    init_game, step, GameState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6


def get_valid_actions(state, bid, ms):
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


def run_v6_from_state(state, all_orders, from_round, num_rounds, drop_d_weight=0.8):
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    solver.allocator.drop_d_weight = drop_d_weight
    action_log = []
    for rnd in range(from_round, num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def kick(seed, actions, num_rounds, num_bots, ms):
    """Apply a large perturbation. Returns new full action sequence."""
    # Pick a random kick window
    kick_start = random.randint(0, num_rounds - 100)
    kick_len = random.randint(30, 80)
    kick_end = min(kick_start + kick_len, num_rounds)

    # Replay to kick_start
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(kick_start):
        step(state, actions[rnd], all_orders)

    # Perturb: change multiple bots' actions at kick_start
    n_perturb = random.randint(5, 15)
    bots = random.sample(range(num_bots), min(n_perturb, num_bots))
    perturbed_round = list(actions[kick_start])
    for bid in bots:
        valid = get_valid_actions(state, bid, ms)
        if valid:
            perturbed_round[bid] = random.choice(valid)

    step(state, perturbed_round, all_orders)

    # Run fresh V6 from kick_start+1 with randomized parameters
    ddw = random.choice([0.4, 0.6, 0.8, 0.8, 0.8, 0.9, 1.0])
    new_score, new_actions = run_v6_from_state(
        state, all_orders, kick_start + 1, num_rounds, drop_d_weight=ddw)

    # Full action sequence
    full = actions[:kick_start] + [perturbed_round] + new_actions
    return new_score, full, kick_start


def traj_search(seed, actions, score, max_time, ms, verbose=True):
    """Short trajectory search (local search phase)."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    best_score = score
    best_actions = [list(a) for a in actions]

    checkpoint_interval = 10
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

    round_options = list(range(0, num_rounds - 30, 1))
    round_weights = [max(0.05, 1.0 - r / num_rounds) for r in round_options]

    iterations = 0
    improvements = 0
    _, all_orders = init_game(seed, 'nightmare', num_orders=100)

    while time.time() - t_start < max_time:
        perturb_round = random.choices(round_options, weights=round_weights, k=1)[0]
        n_perturb = random.choices([1, 1, 2], k=1)[0]

        cp_round = (perturb_round // checkpoint_interval) * checkpoint_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints.keys() if r <= perturb_round)

        cp_state, all_orders_cp = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)

        for rnd in range(cp_round, perturb_round):
            step(state, best_actions[rnd], all_orders_cp)

        perturbed = list(best_actions[perturb_round])
        bots = random.sample(range(num_bots), min(n_perturb, num_bots))
        changed = False

        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) <= 1:
                continue
            original = best_actions[perturb_round][bid]
            alts = [a for a in valid if a != original]
            if alts:
                perturbed[bid] = random.choice(alts)
                changed = True

        if not changed:
            iterations += 1
            continue

        ps = copy.deepcopy(state)
        step(ps, perturbed, all_orders_cp)

        ddw = random.choice([0.4, 0.6, 0.8, 0.8, 0.8, 0.9])
        new_score, new_log = run_v6_from_state(
            ps, all_orders_cp, perturb_round + 1, num_rounds, drop_d_weight=ddw)

        if new_score > best_score:
            best_actions = best_actions[:perturb_round] + [perturbed] + new_log
            best_score = new_score
            improvements += 1
            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

        iterations += 1

    return best_score, best_actions, improvements


def run_ils(seed: int, max_time: float = 3600,
            initial_actions=None, verbose=True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    if initial_actions is not None:
        state = copy.deepcopy(state0)
        for rnd in range(num_rounds):
            step(state, initial_actions[rnd], all_orders)
        best_score = state.score
        best_actions = [list(a) for a in initial_actions]
        print(f'Starting from: {best_score}', file=sys.stderr)
    else:
        best_score, best_actions = NightmareSolverV6.run_sim(seed, verbose=False)
        print(f'V6 baseline: {best_score}', file=sys.stderr)

    global_best_score = best_score
    global_best_actions = [list(a) for a in best_actions]

    ils_iter = 0

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 30:
            break

        # Phase 1: Kick
        kick_score, kick_actions, kick_at = kick(
            seed, best_actions, num_rounds, num_bots, ms)

        # Phase 2: Local search (trajectory search) - shorter budget for more ILS iterations
        traj_budget = min(remaining * 0.5 / max(1, (remaining // 120)), 90)
        traj_score, traj_actions, traj_imp = traj_search(
            seed, kick_actions, kick_score, traj_budget, ms, verbose=False)

        # Accept if improvement
        if traj_score > best_score:
            best_score = traj_score
            best_actions = traj_actions
            tag = "ACCEPT"
        else:
            tag = "reject"

        if traj_score > global_best_score:
            global_best_score = traj_score
            global_best_actions = traj_actions
            tag = "GLOBAL BEST"

            from solution_store import save_solution
            save_solution('nightmare', global_best_score, global_best_actions, seed=seed)

        ils_iter += 1
        if verbose:
            print(f'  ILS {ils_iter} [{elapsed:.0f}s]: kick@{kick_at}={kick_score}, '
                  f'traj={traj_score} ({traj_imp} imp), '
                  f'best={best_score}, global={global_best_score} [{tag}]',
                  file=sys.stderr)

        # Random restart if stuck for 5 iterations
        if ils_iter % 5 == 0 and best_score <= global_best_score:
            # Restart from a fresh V6 with different noise
            from nightmare_multistart import NoisyV6Allocator
            state_r, ao_r = init_game(seed, 'nightmare', num_orders=100)
            tables = PrecomputedTables.get(ms)
            solver = NightmareSolverV6(ms, tables, future_orders=ao_r)
            noise = random.uniform(1.0, 4.0)
            solver.allocator = NoisyV6Allocator(
                ms, tables, solver.drop_zones,
                max_preview_pickers=99, drop_d_weight=0.8,
                noise_scale=noise)
            restart_actions = []
            for rnd in range(num_rounds):
                state_r.round = rnd
                actions = solver.action(state_r, ao_r, rnd)
                restart_actions.append(list(actions))
                step(state_r, actions, ao_r)
            restart_score = state_r.score
            if restart_score > 280:  # Only accept reasonable restarts
                best_score = restart_score
                best_actions = restart_actions
                print(f'  RESTART: {restart_score} (noise={noise:.1f})',
                      file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nILS done: {global_best_score} (elapsed={elapsed:.1f}s, '
          f'{ils_iter} iterations)', file=sys.stderr)

    return global_best_score, global_best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=3600)
    args = parser.parse_args()

    initial = None
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            initial = load_solution('nightmare')
            print(f'Loaded saved: {meta["score"]}', file=sys.stderr)
    except:
        pass

    score, actions = run_ils(args.seed, args.max_time, initial_actions=initial)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
