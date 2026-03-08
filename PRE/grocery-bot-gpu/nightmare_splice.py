#!/usr/bin/env python3
"""Nightmare segment splicing: mix segments from diverse V6 runs.

Strategy:
1. Generate many diverse V6 runs (with noise)
2. Pick a random segment from a random run
3. Splice it into current best at matching state
4. Re-run V6 from splice end
5. Keep if improvement found

This creates more diversity than single-action perturbation.

Usage: python nightmare_splice.py --seed 7009 --max-time 1800
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
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator
from nightmare_multistart import NoisyV6Allocator


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


def run_v6_from_state(state, all_orders, from_round, num_rounds):
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


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def generate_diverse_runs(seed, count=20, max_time=60):
    """Generate diverse V6 runs with noise."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    runs = []

    # Deterministic baseline first
    score, actions = NightmareSolverV6.run_sim(seed, verbose=False)
    runs.append((score, actions))
    print(f'  Base V6: {score}', file=sys.stderr)

    trial = 0
    while time.time() - t_start < max_time and trial < count:
        noise = random.uniform(0.5, 5.0)
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)

        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
        solver.allocator = NoisyV6Allocator(
            ms, tables, solver.drop_zones,
            max_preview_pickers=99, drop_d_weight=0.4,
            noise_scale=noise)

        action_log = []
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))
            step(state, actions, all_orders)

        runs.append((state.score, action_log))
        trial += 1

    runs.sort(key=lambda x: -x[0])
    print(f'  Generated {len(runs)} runs: '
          f'best={runs[0][0]}, worst={runs[-1][0]}, '
          f'mean={sum(r[0] for r in runs)/len(runs):.0f}', file=sys.stderr)
    return runs


def run_splice_search(seed: int, max_time: float = 1800,
                      initial_actions=None, verbose=True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Phase 1: Generate diverse V6 library
    print(f'Phase 1: Building V6 library...', file=sys.stderr)
    lib_budget = min(max_time * 0.1, 60)
    diverse_runs = generate_diverse_runs(seed, count=30, max_time=lib_budget)

    # Start from best available
    if initial_actions is not None:
        state = copy.deepcopy(state0)
        for rnd in range(num_rounds):
            step(state, initial_actions[rnd], all_orders)
        initial_score = state.score
        best_score = initial_score
        best_actions = [list(a) for a in initial_actions]
        print(f'  Starting from initial: {initial_score}', file=sys.stderr)
    else:
        best_score = diverse_runs[0][0]
        best_actions = diverse_runs[0][1]
        initial_score = best_score
        print(f'  Starting from best V6: {best_score}', file=sys.stderr)

    # Phase 2: Splice + trajectory search
    remaining = max_time - (time.time() - t_start)
    print(f'\nPhase 2: Splice search ({remaining:.0f}s)', file=sys.stderr)

    checkpoint_interval = 10
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

    iterations = 0
    improvements = 0
    splice_improvements = 0
    traj_improvements = 0

    round_options = list(range(0, num_rounds - 30, 1))
    round_weights = [max(0.05, 1.0 - r / num_rounds) for r in round_options]

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        # Alternate between splice and single-action perturbation
        use_splice = (random.random() < 0.3)

        if use_splice and len(diverse_runs) > 1:
            # Pick a random source run and segment
            src_idx = random.randint(0, len(diverse_runs) - 1)
            src_score, src_actions = diverse_runs[src_idx]

            seg_start = random.choices(round_options, weights=round_weights, k=1)[0]
            seg_len = random.choice([10, 20, 30, 50])
            seg_end = min(seg_start + seg_len, num_rounds)

            # Get state at seg_start from best solution
            cp_round = (seg_start // checkpoint_interval) * checkpoint_interval
            if cp_round not in checkpoints:
                cp_round = max(r for r in checkpoints.keys() if r <= seg_start)

            cp_state, all_orders_cp = checkpoints[cp_round]
            state = copy.deepcopy(cp_state)
            for rnd in range(cp_round, seg_start):
                step(state, best_actions[rnd], all_orders_cp)

            # Apply source segment
            for rnd in range(seg_start, seg_end):
                if rnd < len(src_actions):
                    step(state, src_actions[rnd], all_orders_cp)
                else:
                    step(state, [(ACT_WAIT, -1)] * num_bots, all_orders_cp)

            # Re-run V6 from seg_end
            new_score, new_action_log = run_v6_from_state(
                state, all_orders_cp, seg_end, num_rounds)

            if new_score > best_score:
                full_actions = (best_actions[:seg_start] +
                               src_actions[seg_start:seg_end] +
                               new_action_log)
                if len(full_actions) == num_rounds:
                    best_score = new_score
                    best_actions = full_actions
                    splice_improvements += 1
                    improvements += 1

                    if verbose:
                        print(f'  [{elapsed:.0f}s] SPLICE BEST: {best_score} '
                              f'(src={src_idx}, r={seg_start}-{seg_end}, '
                              f'iter={iterations})', file=sys.stderr)

                    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)
        else:
            # Standard single-action perturbation
            perturb_round = random.choices(round_options, weights=round_weights, k=1)[0]
            n_perturb = random.choices([1, 1, 1, 2, 2, 3], k=1)[0]

            cp_round = (perturb_round // checkpoint_interval) * checkpoint_interval
            if cp_round not in checkpoints:
                cp_round = max(r for r in checkpoints.keys() if r <= perturb_round)

            cp_state, all_orders_cp = checkpoints[cp_round]
            state = copy.deepcopy(cp_state)

            for rnd in range(cp_round, perturb_round):
                step(state, best_actions[rnd], all_orders_cp)

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

            perturbed_state = copy.deepcopy(state)
            step(perturbed_state, perturbed_actions, all_orders_cp)

            new_score, new_action_log = run_v6_from_state(
                perturbed_state, all_orders_cp, perturb_round + 1, num_rounds)

            if new_score > best_score:
                full_actions = best_actions[:perturb_round] + [perturbed_actions] + new_action_log
                best_score = new_score
                best_actions = full_actions
                traj_improvements += 1
                improvements += 1

                if verbose:
                    print(f'  [{elapsed:.0f}s] TRAJ BEST: {best_score} '
                          f'(r={perturb_round}, iter={iterations})', file=sys.stderr)

                checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

        iterations += 1
        if iterations % 500 == 0 and verbose:
            elapsed2 = time.time() - t_start
            rate = iterations / elapsed2
            print(f'  [{elapsed2:.0f}s] iter={iterations} best={best_score} '
                  f'imp={improvements} (splice={splice_improvements}, '
                  f'traj={traj_improvements}) rate={rate:.1f}/s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nSplice search done: {best_score} (from {initial_score}, '
          f'delta={best_score-initial_score:+d})', file=sys.stderr)
    print(f'  Iterations: {iterations}, improvements: {improvements} '
          f'(splice={splice_improvements}, traj={traj_improvements})', file=sys.stderr)

    return best_score, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=1800)
    args = parser.parse_args()

    # Try to load existing solution
    initial = None
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            initial = load_solution('nightmare')
            print(f'Starting from saved: {meta["score"]}', file=sys.stderr)
    except:
        pass

    score, actions = run_splice_search(args.seed, args.max_time,
                                       initial_actions=initial)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
