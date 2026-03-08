#!/usr/bin/env python3
"""Nightmare beam search: maintain N parallel game states, evolve the best.

At each round:
1. For each of N states, generate M action variants
2. Simulate one step for each variant
3. Score all N*M candidates
4. Keep the best N for next round

This explores a WIDER search space than trajectory perturbation.

Usage: python nightmare_beam.py --seed 7009 --beam-width 8 --variants 5
"""
from __future__ import annotations
import sys, time, random, copy, argparse
import numpy as np
from game_engine import (
    init_game, step, GameState, Order, MapState,
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


def state_heuristic(state, all_orders, ms, tables, drop_zones, num_bots):
    """Score a state: higher is better.

    Combines current score with forward-looking heuristics.
    """
    h = state.score * 100  # Weight actual score heavily

    active = state.get_active_order()
    preview = state.get_preview_order()

    if active:
        needs = active.needs()
        # Bonus for having active items in inventory near dropoff
        for bid in range(num_bots):
            inv = state.bot_inv_list(bid)
            bpos = (int(state.bot_positions[bid, 0]),
                    int(state.bot_positions[bid, 1]))
            drop_dist = min(tables.get_distance(bpos, dz) for dz in drop_zones)
            for t in inv:
                if t in needs:
                    h += max(0, 15 - drop_dist) * 5  # Closer to drop = better
                    # At dropoff = very good
                    if drop_dist == 0:
                        h += 20

        # Penalty for items still needed
        h -= len(needs) * 10

    if preview:
        preview_needs = preview.needs()
        # Bonus for preview items at dropoffs
        drop_set = set(tuple(dz) for dz in drop_zones)
        for bid in range(num_bots):
            bpos = (int(state.bot_positions[bid, 0]),
                    int(state.bot_positions[bid, 1]))
            if bpos in drop_set:
                inv = state.bot_inv_list(bid)
                for t in inv:
                    if t in preview_needs:
                        h += 30  # Preview items at dropoff = chain potential

    return h


def generate_action_variants(state, all_orders, ms, tables, drop_zones,
                             solver, num_variants, num_bots):
    """Generate diverse action sets for current state."""
    # V6 baseline action
    v6_actions = solver.action(state, all_orders, state.round)
    variants = [list(v6_actions)]

    # Generate perturbations of V6 actions
    for _ in range(num_variants - 1):
        perturbed = list(v6_actions)
        n_perturb = random.choices([1, 1, 2, 2, 3], k=1)[0]
        bots = random.sample(range(num_bots), min(n_perturb, num_bots))
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) > 1:
                alts = [a for a in valid if a != v6_actions[bid]]
                if alts:
                    perturbed[bid] = random.choice(alts)
        variants.append(perturbed)

    return variants


def run_beam_search(seed: int, beam_width: int = 8,
                    num_variants: int = 5,
                    verbose: bool = True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state
    tables = PrecomputedTables.get(ms)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

    # Each beam entry: (state, action_log, solver_instance)
    # Start with one beam (the initial state)
    beam = [(copy.deepcopy(state0), [], NightmareSolverV6(ms, tables, future_orders=all_orders))]

    for rnd in range(num_rounds):
        candidates = []

        for state, action_log, solver in beam:
            state.round = rnd

            # Generate action variants
            variants = generate_action_variants(
                state, all_orders, ms, tables, drop_zones,
                solver, num_variants, num_bots)

            for actions in variants:
                # Simulate one step
                new_state = copy.deepcopy(state)
                step(new_state, actions, all_orders)

                # Score
                h = state_heuristic(new_state, all_orders, ms, tables,
                                    drop_zones, num_bots)

                # Create fresh solver for this branch
                new_solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
                # Copy stall tracking from parent
                new_solver.stall_counts = dict(solver.stall_counts)
                new_solver.prev_positions = dict(solver.prev_positions)
                new_solver._pos_history = {k: list(v) for k, v in solver._pos_history.items()}
                new_solver.allocator._preview_bot_types = dict(solver.allocator._preview_bot_types)
                new_solver.allocator._last_preview_id = solver.allocator._last_preview_id
                new_solver.allocator._committed_stages = dict(solver.allocator._committed_stages)

                candidates.append((h, new_state, action_log + [list(actions)], new_solver))

        # Keep top beam_width
        candidates.sort(key=lambda x: -x[0])
        beam = [(s, al, sv) for _, s, al, sv in candidates[:beam_width]]

        if verbose and (rnd < 5 or rnd % 50 == 0 or rnd == num_rounds - 1):
            scores = [s.score for s, _, _ in beam]
            elapsed = time.time() - t_start
            print(f'R{rnd:3d}: scores={scores}, '
                  f'orders={[s.orders_completed for s, _, _ in beam]}, '
                  f't={elapsed:.1f}s', file=sys.stderr)

    # Return best
    best_state, best_actions, _ = beam[0]
    elapsed = time.time() - t_start
    print(f'\nBeam search done: score={best_state.score} '
          f'orders={best_state.orders_completed} '
          f'time={elapsed:.1f}s', file=sys.stderr)

    return best_state.score, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--beam-width', type=int, default=8)
    parser.add_argument('--variants', type=int, default=5)
    args = parser.parse_args()

    score, actions = run_beam_search(
        args.seed, args.beam_width, args.variants)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
