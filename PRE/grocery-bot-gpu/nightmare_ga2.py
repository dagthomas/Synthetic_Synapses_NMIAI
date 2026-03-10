#!/usr/bin/env python3
"""Nightmare GA v2: LMAPF-seeded population + multi-solver suffix.

Key improvements over v1:
1. LMAPF initial population (302 baseline vs V6's 254)
2. Best-of suffix: LMAPF + V6 variants, pick highest
3. Targeted perturbation: perturb weakest bots more
4. Per-bot trajectory search: optimize single-bot actions with others locked
5. Larger population with diversity preservation

Usage: python nightmare_ga2.py --seed 7005 --max-time 7200
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


def replay_to_round(seed, actions, target_round):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(target_round, len(actions))):
        step(state, actions[rnd], all_orders)
    return state, all_orders


def evaluate(seed, actions):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(NUM_ROUNDS, len(actions))):
        step(state, actions[rnd], all_orders)
    return state.score


def run_lmapf_suffix(state, all_orders, from_round, solver_seed=0):
    from nightmare_lmapf_solver import LMAPFSolver
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=solver_seed)
    action_log = []
    for rnd in range(from_round, NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def run_v6_suffix(state, all_orders, from_round, ddw=0.4, noise=0.0):
    from nightmare_solver_v6 import NightmareSolverV6
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    if noise > 0:
        from nightmare_multistart import NoisyV6Allocator
        solver.allocator = NoisyV6Allocator(
            ms, tables, solver.drop_zones,
            max_preview_pickers=99, drop_d_weight=ddw,
            noise_scale=noise)
    action_log = []
    for rnd in range(from_round, NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


_suffix_counter = 0

def best_suffix(state_template, all_orders, from_round, fast=False, n_lmapf=5):
    """Try multiple stochastic LMAPF seeds + V6 variants, return best."""
    global _suffix_counter
    best_score = -1
    best_actions = None
    best_tag = None

    # Multiple stochastic LMAPF seeds (the key diversity driver)
    n = 2 if fast else n_lmapf
    for i in range(n):
        _suffix_counter += 1
        s = copy.deepcopy(state_template)
        score, actions = run_lmapf_suffix(s, all_orders, from_round,
                                           solver_seed=_suffix_counter)
        if score > best_score:
            best_score, best_actions, best_tag = score, actions, f'lmapf-s{_suffix_counter}'

    if fast:
        return best_score, best_actions, best_tag

    # V6 variants for additional diversity
    for ddw, noise in [(0.4, 0.0), (0.3, 0.0), (0.4, 2.0)]:
        s = copy.deepcopy(state_template)
        score, actions = run_v6_suffix(s, all_orders, from_round, ddw=ddw, noise=noise)
        if score > best_score:
            best_score = score
            best_actions = actions
            best_tag = f'v6({ddw},{noise})'

    return best_score, best_actions, best_tag


def build_checkpoints(seed, actions, interval=10):
    """Build checkpoints for fast replay."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(NUM_ROUNDS, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def measure_bot_contributions(seed, actions):
    """Measure each bot's contribution (deliveries, pickups, stalls)."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    deliveries = [0] * NUM_BOTS
    pickups = [0] * NUM_BOTS
    prev_pos = {}

    for rnd in range(min(NUM_ROUNDS, len(actions))):
        for bid, act in enumerate(actions[rnd]):
            a, item = act
            if a == ACT_DROPOFF:
                deliveries[bid] += 1
            elif a == ACT_PICKUP:
                pickups[bid] += 1
        step(state, actions[rnd], all_orders)

    return deliveries, pickups


# ──────────────────────────────────────────────────
#  Phase 1: Population generation
# ──────────────────────────────────────────────────

def generate_population(seed, pop_size, max_time=60):
    t_start = time.time()
    population = []

    # LMAPF baseline (deterministic)
    from nightmare_lmapf_solver import LMAPFSolver
    score, actions = LMAPFSolver.run_sim(seed, verbose=False)
    population.append((score, [list(a) for a in actions]))
    print(f'  LMAPF: {score}', file=sys.stderr)

    # Stochastic LMAPF seeds (best diversity source, ~0.7s each)
    solver_seed = 1
    while time.time() - t_start < max_time and len(population) < pop_size:
        score, actions = LMAPFSolver.run_sim(seed, verbose=False,
                                              solver_seed=solver_seed)
        population.append((score, [list(a) for a in actions]))
        solver_seed += 1

    population.sort(key=lambda x: -x[0])
    scores = [s for s, _ in population]
    print(f'  Population: {len(population)}, '
          f'best={scores[0]}, worst={scores[-1]}, '
          f'mean={sum(scores)/len(scores):.1f}',
          file=sys.stderr)
    return population


# ──────────────────────────────────────────────────
#  Phase 2: GA operators
# ──────────────────────────────────────────────────

def crossover(seed, p1_actions, p2_actions, fast=False):
    """Segment crossover with multi-solver suffix."""
    cross_start = random.randint(10, NUM_ROUNDS - 50)
    seg_len = random.choice([10, 20, 30, 50, 80, 120])
    cross_end = min(cross_start + seg_len, NUM_ROUNDS)

    state, all_orders = replay_to_round(seed, p1_actions, cross_start)

    # Apply p2's segment
    for rnd in range(cross_start, cross_end):
        if rnd < len(p2_actions):
            step(state, p2_actions[rnd], all_orders)

    # Best suffix
    score, suffix, tag = best_suffix(state, all_orders, cross_end, fast=fast)

    child = p1_actions[:cross_start]
    for rnd in range(cross_start, cross_end):
        child.append(list(p2_actions[rnd]) if rnd < len(p2_actions)
                     else [(ACT_WAIT, -1)] * NUM_BOTS)
    child.extend(suffix)

    return score, child[:NUM_ROUNDS], f'CX@{cross_start}+{seg_len}[{tag}]'


def mutate(seed, actions, ms, fast=False):
    """Multi-round perturbation + multi-solver suffix."""
    mut_start = random.randint(0, NUM_ROUNDS - 30)
    mut_len = random.choices([1, 1, 2, 3, 5, 10], k=1)[0]
    mut_end = min(mut_start + mut_len, NUM_ROUNDS)

    state, all_orders = replay_to_round(seed, actions, mut_start)

    new_actions = list(actions[:mut_start])
    for rnd in range(mut_start, mut_end):
        perturbed = list(actions[rnd]) if rnd < len(actions) else [(ACT_WAIT, -1)] * NUM_BOTS
        n_bots = random.choices([1, 2, 3, 5, 8], weights=[3, 3, 2, 1, 1], k=1)[0]
        bots = random.sample(range(NUM_BOTS), min(n_bots, NUM_BOTS))
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) > 1:
                alts = [a for a in valid if a != perturbed[bid]]
                if alts:
                    perturbed[bid] = random.choice(alts)
        new_actions.append(perturbed)
        step(state, perturbed, all_orders)

    score, suffix, tag = best_suffix(state, all_orders, mut_end, fast=fast)
    new_actions.extend(suffix)

    return score, new_actions[:NUM_ROUNDS], f'MUT@{mut_start}+{mut_len}[{tag}]'


def targeted_mutate(seed, actions, ms, deliveries):
    """Perturb the weakest bots more aggressively."""
    # Find weakest bots
    weak_bots = sorted(range(NUM_BOTS), key=lambda b: deliveries[b])[:8]

    mut_start = random.randint(0, NUM_ROUNDS - 50)
    mut_len = random.randint(3, 15)
    mut_end = min(mut_start + mut_len, NUM_ROUNDS)

    state, all_orders = replay_to_round(seed, actions, mut_start)

    new_actions = list(actions[:mut_start])
    for rnd in range(mut_start, mut_end):
        perturbed = list(actions[rnd]) if rnd < len(actions) else [(ACT_WAIT, -1)] * NUM_BOTS
        # Perturb 3-6 of the weakest bots
        n_perturb = random.randint(3, 6)
        bots = random.sample(weak_bots, min(n_perturb, len(weak_bots)))
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) > 1:
                alts = [a for a in valid if a != perturbed[bid]]
                if alts:
                    perturbed[bid] = random.choice(alts)
        new_actions.append(perturbed)
        step(state, perturbed, all_orders)

    score, suffix, tag = best_suffix(state, all_orders, mut_end, fast=True)
    new_actions.extend(suffix)

    return score, new_actions[:NUM_ROUNDS], f'TMUT@{mut_start}+{mut_len}[{tag}]'


def traj_search(seed, actions, ms, checkpoints, time_budget=20):
    """Single-round single-bot perturbation search."""
    t0 = time.time()
    best_score = evaluate(seed, actions)
    best_actions = actions
    cp_interval = 10
    improved = False

    while time.time() - t0 < time_budget:
        tr = random.randint(0, NUM_ROUNDS - 30)
        cp_r = (tr // cp_interval) * cp_interval
        if cp_r not in checkpoints:
            cp_r = max(r for r in checkpoints if r <= tr)

        cp_s, cp_o = checkpoints[cp_r]
        ts = copy.deepcopy(cp_s)
        for rnd in range(cp_r, tr):
            step(ts, best_actions[rnd], cp_o)

        perturbed = list(best_actions[tr])
        bots = random.sample(range(NUM_BOTS), random.randint(1, 4))
        changed = False
        for bid in bots:
            valid = get_valid_actions(ts, bid, ms)
            if len(valid) <= 1:
                continue
            alts = [a for a in valid if a != best_actions[tr][bid]]
            if alts:
                perturbed[bid] = random.choice(alts)
                changed = True

        if not changed:
            continue

        step(ts, perturbed, cp_o)

        # Stochastic LMAPF suffix
        global _suffix_counter
        _suffix_counter += 1
        score, suffix = run_lmapf_suffix(ts, cp_o, tr + 1,
                                          solver_seed=_suffix_counter)

        if score > best_score:
            new_full = best_actions[:tr] + [perturbed] + suffix
            best_score = score
            best_actions = [list(a) for a in new_full]
            improved = True

    return best_score, best_actions, improved


# ──────────────────────────────────────────────────
#  Main GA loop
# ──────────────────────────────────────────────────

def run_ga(seed, max_time=7200, pop_size=20, initial_actions=None, verbose=True):
    t_start = time.time()

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Phase 1: Population
    print(f'Phase 1: Generating population...', file=sys.stderr)
    pop_budget = min(max_time * 0.03, 60)
    population = generate_population(seed, pop_size, max_time=pop_budget)

    if initial_actions is not None:
        init_score = evaluate(seed, initial_actions)
        population.append((init_score, [list(a) for a in initial_actions]))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        print(f'  Added initial: {init_score}', file=sys.stderr)

    global_best_score = population[0][0]
    global_best_actions = population[0][1]
    best_deliveries, _ = measure_bot_contributions(seed, global_best_actions)

    # Save initial best
    from solution_store import save_solution
    save_solution('nightmare', global_best_score, global_best_actions, seed=seed)

    print(f'\nPhase 2: GA + trajectory search (best={global_best_score})...',
          file=sys.stderr)

    gen = 0
    improvements = 0
    last_improve_time = time.time()

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 5:
            break

        # Adaptive: more trajectory search when stuck
        stuck_time = time.time() - last_improve_time

        if stuck_time > 120 and random.random() < 0.3:
            # Trajectory search on best solution
            checkpoints = build_checkpoints(seed, global_best_actions)
            traj_budget = min(remaining * 0.1, 30)
            ts, ta, improved = traj_search(
                seed, global_best_actions, ms, checkpoints, traj_budget)
            if improved and ts > global_best_score:
                global_best_score = ts
                global_best_actions = [list(a) for a in ta]
                best_deliveries, _ = measure_bot_contributions(seed, global_best_actions)
                # Update population
                worst_idx = min(range(len(population)), key=lambda i: population[i][0])
                population[worst_idx] = (ts, ta)
                save_solution('nightmare', global_best_score, global_best_actions, seed=seed)
                last_improve_time = time.time()
                improvements += 1
                if verbose:
                    print(f'  [{elapsed:.0f}s] TRAJ: {global_best_score}', file=sys.stderr)
            gen += 1
            continue

        # Tournament selection
        t_size = min(3, len(population))
        cands = random.sample(range(len(population)), t_size)
        p1_idx = max(cands, key=lambda i: population[i][0])
        cands = random.sample(range(len(population)), t_size)
        p2_idx = max(cands, key=lambda i: population[i][0])

        # Use fast suffix when time is tight
        fast = remaining < 300 or stuck_time < 30

        # Choose operator
        r = random.random()
        if r < 0.2 and p1_idx != p2_idx:
            # Crossover
            new_score, new_actions, op_tag = crossover(
                seed, population[p1_idx][1], population[p2_idx][1], fast=fast)
        elif r < 0.5:
            # Targeted mutation (perturb weak bots)
            new_score, new_actions, op_tag = targeted_mutate(
                seed, population[p1_idx][1], ms, best_deliveries)
        else:
            # Standard mutation
            new_score, new_actions, op_tag = mutate(
                seed, population[p1_idx][1], ms, fast=fast)

        # Replacement
        worst_idx = min(range(len(population)), key=lambda i: population[i][0])
        if new_score > population[worst_idx][0]:
            population[worst_idx] = (new_score, new_actions)
            improvements += 1

        if new_score > global_best_score:
            global_best_score = new_score
            global_best_actions = [list(a) for a in new_actions]
            best_deliveries, _ = measure_bot_contributions(seed, global_best_actions)
            last_improve_time = time.time()
            save_solution('nightmare', global_best_score, global_best_actions, seed=seed)
            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {global_best_score} ({op_tag})',
                      file=sys.stderr)

        gen += 1
        if gen % 50 == 0 and verbose:
            scores = sorted([s for s, _ in population], reverse=True)
            rate = gen / max(elapsed, 1)
            print(f'  [{elapsed:.0f}s] gen={gen} best={global_best_score} '
                  f'pop=[{scores[0]},{scores[len(scores)//2]},{scores[-1]}] '
                  f'imp={improvements} rate={rate:.1f}/s '
                  f'stuck={stuck_time:.0f}s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nGA2 done: {global_best_score} ({elapsed:.1f}s, '
          f'{gen} gen, {improvements} improvements)', file=sys.stderr)
    return global_best_score, global_best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--max-time', type=int, default=7200)
    parser.add_argument('--pop-size', type=int, default=20)
    args = parser.parse_args()

    initial = None
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            initial = load_solution('nightmare')
            print(f'Starting from saved: {meta["score"]}', file=sys.stderr)
    except:
        pass

    score, actions = run_ga(args.seed, args.max_time, args.pop_size,
                            initial_actions=initial)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
