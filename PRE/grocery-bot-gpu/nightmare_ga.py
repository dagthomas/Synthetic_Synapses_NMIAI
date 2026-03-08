#!/usr/bin/env python3
"""Nightmare GA: Genetic Algorithm with segment crossover.

Population of diverse solutions. Crossover = splice segments.
Mutation = multi-round perturbation + V6/NoisyV6 replay.
Selection = tournament.

Usage: python nightmare_ga.py --seed 7009 --max-time 7200 --pop-size 20
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


def run_v6_from_state(state, all_orders, from_round, num_rounds, noise=0.0):
    """Run V6 (optionally noisy) from a state to end. Returns score, action_log."""
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    if noise > 0:
        solver.allocator = NoisyV6Allocator(
            ms, tables, solver.drop_zones,
            max_preview_pickers=99, drop_d_weight=0.4,
            noise_scale=noise)
    action_log = []
    for rnd in range(from_round, num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def evaluate(seed, actions, num_rounds):
    """Evaluate a solution by replaying."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
    return state.score


def replay_to_round(seed, actions, target_round):
    """Replay to a specific round, return state and all_orders."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(target_round, len(actions))):
        step(state, actions[rnd], all_orders)
    return state, all_orders


def generate_initial_population(seed, pop_size, max_time=60):
    """Generate diverse V6 runs."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    population = []

    # V6 baseline
    score, actions = NightmareSolverV6.run_sim(seed, verbose=False)
    population.append((score, [list(a) for a in actions]))
    print(f'  V6 baseline: {score}', file=sys.stderr)

    # Noisy variants
    noise_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    trial = 0
    while time.time() - t_start < max_time and len(population) < pop_size:
        noise = noise_levels[trial % len(noise_levels)] if trial < len(noise_levels) else random.uniform(0.5, 6.0)
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
            a = solver.action(state, all_orders, rnd)
            action_log.append(list(a))
            step(state, a, all_orders)
        population.append((state.score, action_log))
        trial += 1

    population.sort(key=lambda x: -x[0])
    print(f'  Population: {len(population)} solutions, '
          f'best={population[0][0]}, worst={population[-1][0]}, '
          f'mean={sum(s for s,_ in population)/len(population):.0f}',
          file=sys.stderr)
    return population


def crossover(seed, parent1_actions, parent2_actions, num_rounds):
    """Segment crossover: take prefix from parent1, segment from parent2, V6 suffix."""
    # Random crossover point
    cross_start = random.randint(10, num_rounds - 50)
    seg_len = random.choice([10, 20, 30, 50, 80])
    cross_end = min(cross_start + seg_len, num_rounds)

    # Replay parent1 to cross_start
    state, all_orders = replay_to_round(seed, parent1_actions, cross_start)

    # Apply parent2's segment
    for rnd in range(cross_start, cross_end):
        if rnd < len(parent2_actions):
            step(state, parent2_actions[rnd], all_orders)

    # V6 suffix with optional noise
    noise = random.choice([0.0, 0.0, 0.0, 0.5, 1.0, 2.0])
    new_score, suffix = run_v6_from_state(state, all_orders, cross_end, num_rounds, noise=noise)

    child = parent1_actions[:cross_start]
    for rnd in range(cross_start, cross_end):
        if rnd < len(parent2_actions):
            child.append(list(parent2_actions[rnd]))
        else:
            child.append([(ACT_WAIT, -1)] * 20)
    child.extend(suffix)

    return new_score, child[:num_rounds]


def mutate(seed, actions, num_rounds, ms):
    """Multi-round perturbation + V6 replay."""
    num_bots = CONFIGS['nightmare']['bots']

    # Pick mutation window
    mut_start = random.randint(0, num_rounds - 30)
    mut_len = random.choices([1, 1, 2, 3, 5, 10], k=1)[0]
    mut_end = min(mut_start + mut_len, num_rounds)

    # Replay to mutation start
    state, all_orders = replay_to_round(seed, actions, mut_start)

    # Perturb each round in mutation window
    new_actions = list(actions[:mut_start])
    for rnd in range(mut_start, mut_end):
        perturbed = list(actions[rnd]) if rnd < len(actions) else [(ACT_WAIT, -1)] * num_bots
        n_bots = random.choices([1, 2, 3, 4, 5], weights=[4, 3, 2, 1, 1], k=1)[0]
        bots = random.sample(range(num_bots), min(n_bots, num_bots))
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) > 1:
                alts = [a for a in valid if a != perturbed[bid]]
                if alts:
                    perturbed[bid] = random.choice(alts)
        new_actions.append(perturbed)
        step(state, perturbed, all_orders)

    # V6 replay from mutation end
    noise = random.choice([0.0, 0.0, 0.0, 0.5, 1.0])
    new_score, suffix = run_v6_from_state(state, all_orders, mut_end, num_rounds, noise=noise)
    new_actions.extend(suffix)

    return new_score, new_actions[:num_rounds]


def run_ga(seed, max_time=7200, pop_size=20, initial_actions=None, verbose=True):
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Generate initial population
    print(f'Phase 1: Generating population...', file=sys.stderr)
    pop_budget = min(max_time * 0.05, 30)
    population = generate_initial_population(seed, pop_size, max_time=pop_budget)

    # If we have an initial (better) solution, add it
    if initial_actions is not None:
        init_score = evaluate(seed, initial_actions, num_rounds)
        population.append((init_score, [list(a) for a in initial_actions]))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        print(f'  Added initial: {init_score}', file=sys.stderr)

    global_best_score = population[0][0]
    global_best_actions = population[0][1]

    print(f'\nPhase 2: GA search...', file=sys.stderr)

    gen = 0
    improvements = 0
    cx_improvements = 0
    mut_improvements = 0

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        # Tournament selection (2 parents)
        tournament_size = 3
        candidates = random.sample(range(len(population)), min(tournament_size, len(population)))
        p1_idx = min(candidates, key=lambda i: -population[i][0])
        candidates = random.sample(range(len(population)), min(tournament_size, len(population)))
        p2_idx = min(candidates, key=lambda i: -population[i][0])

        # Crossover or mutation?
        if random.random() < 0.3 and p1_idx != p2_idx:
            # Crossover
            new_score, new_actions = crossover(
                seed, population[p1_idx][1], population[p2_idx][1], num_rounds)
            op = 'CX'
        else:
            # Mutation
            parent_idx = p1_idx
            new_score, new_actions = mutate(
                seed, population[parent_idx][1], num_rounds, ms)
            op = 'MUT'

        # Replacement: replace worst in population if better
        worst_idx = min(range(len(population)), key=lambda i: population[i][0])
        if new_score > population[worst_idx][0]:
            population[worst_idx] = (new_score, new_actions)
            improvements += 1
            if op == 'CX':
                cx_improvements += 1
            else:
                mut_improvements += 1

        if new_score > global_best_score:
            global_best_score = new_score
            global_best_actions = [list(a) for a in new_actions]
            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {global_best_score} ({op}, gen={gen})',
                      file=sys.stderr)
            # Save
            from solution_store import save_solution
            save_solution('nightmare', global_best_score, global_best_actions, seed=seed)

        gen += 1
        if gen % 200 == 0 and verbose:
            scores = sorted([s for s, _ in population], reverse=True)
            elapsed2 = time.time() - t_start
            rate = gen / elapsed2
            print(f'  [{elapsed2:.0f}s] gen={gen} best={scores[0]} '
                  f'pop=[{scores[0]},{scores[len(scores)//2]},{scores[-1]}] '
                  f'imp={improvements} (cx={cx_improvements},mut={mut_improvements}) '
                  f'rate={rate:.1f}/s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nGA done: {global_best_score} ({elapsed:.1f}s, '
          f'{gen} generations, {improvements} improvements)', file=sys.stderr)
    return global_best_score, global_best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=7200)
    parser.add_argument('--pop-size', type=int, default=20)
    args = parser.parse_args()

    # Try loading existing solution
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

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
