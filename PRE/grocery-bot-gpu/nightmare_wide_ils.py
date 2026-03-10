#!/usr/bin/env python3
"""Nightmare ILS with wide perturbation windows.

Key insight: small perturbations (1-10 rounds) get trapped because suffix
reconverges to the same local optimum. Wide perturbations (50-200 rounds)
create fundamentally different game states that lead to different suffix
trajectories.

Uses multiple suffix solvers (LMAPF + V6 variants) for each perturbation.

Usage: python nightmare_wide_ils.py --seed 7005 --max-time 3600
"""
from __future__ import annotations
import sys, time, random, copy, argparse
import numpy as np
from game_engine import (
    init_game, step,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables

sys.stdout.reconfigure(encoding='utf-8')

NUM_ROUNDS = DIFF_ROUNDS['nightmare']
NUM_BOTS = CONFIGS['nightmare']['bots']


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


_ils_suffix_counter = 0

def run_suffix(state, all_orders, from_round, solver_type='lmapf', solver_seed=0):
    """Run a suffix solver from given state."""
    if solver_type == 'lmapf':
        from nightmare_lmapf_solver import LMAPFSolver
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                             solver_seed=solver_seed)
        action_log = []
        for rnd in range(from_round, NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))
            step(state, actions, all_orders)
        return state.score, action_log
    else:
        from nightmare_solver_v6 import NightmareSolverV6
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
        action_log = []
        for rnd in range(from_round, NUM_ROUNDS):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(list(actions))
            step(state, actions, all_orders)
        return state.score, action_log


def best_suffix(state_template, all_orders, from_round, n_seeds=3):
    """Try multiple stochastic LMAPF seeds, return best."""
    global _ils_suffix_counter
    best_score = -1
    best_actions = None
    best_tag = None

    for i in range(n_seeds):
        _ils_suffix_counter += 1
        s = copy.deepcopy(state_template)
        score, actions = run_suffix(s, all_orders, from_round, 'lmapf',
                                     solver_seed=_ils_suffix_counter)
        if score > best_score:
            best_score, best_actions = score, actions
            best_tag = f'lmapf-s{_ils_suffix_counter}'

    return best_score, best_actions, best_tag


def build_checkpoints(seed, actions, interval=25):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(NUM_ROUNDS, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def wide_perturb(state, actions, ms, start, length, intensity):
    """Perturb a wide window of rounds."""
    perturbed = []
    s = copy.deepcopy(state)
    for rnd in range(start, min(start + length, NUM_ROUNDS)):
        if rnd >= len(actions):
            break
        prnd = list(actions[rnd])
        # Perturb some bots each round
        n_bots = max(1, int(NUM_BOTS * intensity))
        bots = random.sample(range(NUM_BOTS), min(n_bots, NUM_BOTS))
        for bid in bots:
            valid = get_valid_actions(s, bid, ms)
            if len(valid) > 1:
                alts = [a for a in valid if a != prnd[bid]]
                if alts:
                    prnd[bid] = random.choice(alts)
        perturbed.append(prnd)
        step(s, prnd, all_orders=None)  # don't need orders for state update
    return perturbed, s


def run_ils(seed, max_time=3600, verbose=True):
    t_start = time.time()

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state
    tables = PrecomputedTables.get(ms)

    # Phase 1: Multi-start
    print(f'Phase 1: Multi-start...', file=sys.stderr)
    from nightmare_lmapf_solver import LMAPFSolver
    score, action_log = LMAPFSolver.run_sim(seed, verbose=False)
    best_actions = [list(a) for a in action_log]
    best_score = score
    print(f'  LMAPF: {score}', file=sys.stderr)

    # Try V6 too
    from nightmare_solver_v6 import NightmareSolverV6
    score_v6, actions_v6 = NightmareSolverV6.run_sim(seed, verbose=False)
    if score_v6 > best_score:
        best_score = score_v6
        best_actions = [list(a) for a in actions_v6]
    print(f'  V6: {score_v6}', file=sys.stderr)

    # Try loading saved solution
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > best_score:
            saved = load_solution('nightmare')
            if saved:
                best_score = meta['score']
                best_actions = [list(a) for a in saved]
                print(f'  Loaded: {best_score}', file=sys.stderr)
    except:
        pass

    global_best = best_score
    global_actions = [list(a) for a in best_actions]

    print(f'\nPhase 2: Wide ILS (best={global_best})...', file=sys.stderr)

    # Build checkpoints
    cp_interval = 25
    checkpoints = build_checkpoints(seed, best_actions, cp_interval)

    iteration = 0
    no_improve = 0
    last_improve = time.time()

    # Perturbation schedule: mix of narrow and wide
    perturbation_configs = [
        # (window_length, intensity, weight)
        (1, 0.3, 2),     # narrow: 1 round, 30% bots
        (3, 0.3, 2),     # narrow: 3 rounds, 30% bots
        (5, 0.5, 2),     # medium: 5 rounds, 50% bots
        (10, 0.5, 2),    # medium: 10 rounds, 50% bots
        (25, 0.3, 3),    # wide: 25 rounds, 30% bots
        (50, 0.2, 3),    # wide: 50 rounds, 20% bots
        (100, 0.15, 2),  # very wide: 100 rounds, 15% bots
        (200, 0.1, 1),   # ultra wide: 200 rounds, 10% bots
    ]
    weights = [c[2] for c in perturbation_configs]

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 10:
            break

        # Choose perturbation config
        config = random.choices(perturbation_configs, weights=weights, k=1)[0]
        window_len, intensity, _ = config

        # Choose start point
        max_start = max(0, NUM_ROUNDS - window_len - 30)
        kick_start = random.randint(0, max_start)

        # Get checkpoint
        cp_round = (kick_start // cp_interval) * cp_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints if r <= kick_start)

        cp_state, cp_orders = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)
        for rnd in range(cp_round, kick_start):
            step(state, best_actions[rnd], cp_orders)

        # Perturb
        perturbed_rounds = []
        for k in range(window_len):
            rnd = kick_start + k
            if rnd >= NUM_ROUNDS:
                break
            n_bots = max(1, int(NUM_BOTS * intensity))
            bots = random.sample(range(NUM_BOTS), min(n_bots, NUM_BOTS))
            prnd = list(best_actions[rnd]) if rnd < len(best_actions) else [(ACT_WAIT, -1)] * NUM_BOTS
            for bid in bots:
                valid = get_valid_actions(state, bid, ms)
                if len(valid) > 1:
                    alts = [a for a in valid if a != prnd[bid]]
                    if alts:
                        prnd[bid] = random.choice(alts)
            perturbed_rounds.append(prnd)
            step(state, prnd, cp_orders)

        suffix_start = kick_start + len(perturbed_rounds)

        # Best suffix
        score, suffix, tag = best_suffix(state, cp_orders, suffix_start)

        full = best_actions[:kick_start] + perturbed_rounds + suffix

        if score > global_best:
            global_best = score
            global_actions = [list(a) for a in full]
            best_score = score
            best_actions = [list(a) for a in full]
            checkpoints = build_checkpoints(seed, best_actions, cp_interval)
            no_improve = 0
            last_improve = time.time()

            from solution_store import save_solution
            save_solution('nightmare', global_best, global_actions, seed=seed)

            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {global_best} '
                      f'(w={window_len}, i={intensity:.0%}, @{kick_start}, {tag})',
                      file=sys.stderr)
        else:
            no_improve += 1

        iteration += 1

        # Restart if stuck
        if no_improve >= 100:
            s = copy.deepcopy(state0)
            rs, ra, rt = best_suffix(s, all_orders, 0)
            if rs > 200:
                best_score = rs
                best_actions = ra
                checkpoints = build_checkpoints(seed, best_actions, cp_interval)
                no_improve = 0
                if verbose:
                    print(f'  [{elapsed:.0f}s] RESTART: {rs} [{rt}]', file=sys.stderr)

        if verbose and iteration % 50 == 0:
            stuck = time.time() - last_improve
            print(f'  [{elapsed:.0f}s] iter={iteration} best={global_best} '
                  f'no_imp={no_improve} stuck={stuck:.0f}s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nWide ILS done: {global_best} ({elapsed:.1f}s, {iteration} iter)',
          file=sys.stderr)
    return global_best, global_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--max-time', type=int, default=3600)
    args = parser.parse_args()

    score, actions = run_ils(args.seed, args.max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
