#!/usr/bin/env python3
"""Nightmare multi-start solver: randomized V6 + trajectory search.

Strategy:
1. Run V6 N times with randomized item assignment noise
2. Keep the best solution from each start
3. Run trajectory search from the best overall

Usage: python nightmare_multistart.py --seed 7009 --max-time 1800
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
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator


class NoisyV6Allocator(V6Allocator):
    """V6 allocator with random noise on item assignment costs."""

    def __init__(self, *args, noise_scale=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale

    def _assign_item(self, bot_pos, needed, assigned_counts, claimed,
                     strict=False, zone_filter=-1, type_bonus=None):
        best_idx = None
        best_adj = None
        best_cost = 9999
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            max_assign = need_count if strict else need_count + 1
            if assigned_counts.get(tid, 0) >= max_assign:
                continue
            bonus = type_bonus.get(tid, 0) if type_bonus else 0
            for item_idx, adj_cells, item_zone in self.type_items.get(tid, []):
                if item_idx in claimed:
                    continue
                if zone_filter >= 0 and item_zone != zone_filter:
                    continue
                for adj in adj_cells:
                    d = self.tables.get_distance(bot_pos, adj)
                    if zone_filter >= 0:
                        drop_d = self.tables.get_distance(
                            adj, self.zone_dropoff.get(zone_filter, self.drop_zones[0]))
                    else:
                        drop_d = self._drop_dist(adj)
                    noise = random.gauss(0, self.noise_scale)
                    cost = d + drop_d * self.drop_d_weight - bonus + noise
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj


def run_noisy_v6(seed: int, noise_scale: float = 2.0) -> tuple[int, list]:
    """Run V6 with noise on item assignment."""
    num_rounds = DIFF_ROUNDS['nightmare']
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)

    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    solver.allocator = NoisyV6Allocator(
        ms, tables, solver.drop_zones,
        max_preview_pickers=99, drop_d_weight=0.4,
        noise_scale=noise_scale)

    action_log = []
    for rnd in range(num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)

    return state.score, action_log


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


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def trajectory_search(seed, initial_actions, initial_score, max_time,
                      verbose=True):
    """Run trajectory search from a given solution."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    best_score = initial_score
    best_actions = [list(a) for a in initial_actions]

    checkpoint_interval = 10
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

    iterations = 0
    improvements = 0

    round_options = list(range(0, num_rounds - 30, 1))
    round_weights = [max(0.05, 1.0 - r / num_rounds) for r in round_options]

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

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
            improvements += 1

            if verbose:
                print(f'  [{elapsed:.0f}s] NEW BEST: {best_score} '
                      f'(r={perturb_round}, iter={iterations})', file=sys.stderr)

            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

        iterations += 1
        if iterations % 500 == 0 and verbose:
            elapsed2 = time.time() - t_start
            rate = iterations / elapsed2
            print(f'  [{elapsed2:.0f}s] iter={iterations} best={best_score} '
                  f'imp={improvements} rate={rate:.1f}/s', file=sys.stderr)

    return best_score, best_actions, improvements


def run_multistart(seed: int, max_time: float = 1800,
                   verbose: bool = True) -> tuple[int, list]:
    t_start = time.time()

    # Phase 1: Multi-start V6 with noise (20% of budget)
    phase1_budget = min(max_time * 0.15, 120)
    print(f'Phase 1: Multi-start V6 ({phase1_budget:.0f}s budget)', file=sys.stderr)

    # First, deterministic baseline
    from nightmare_solver_v6 import NightmareSolverV6
    baseline_score, baseline_actions = NightmareSolverV6.run_sim(seed, verbose=False)
    best_score = baseline_score
    best_actions = baseline_actions
    print(f'  V6 deterministic: {baseline_score}', file=sys.stderr)

    trial = 0
    while time.time() - t_start < phase1_budget:
        noise = random.uniform(0.5, 4.0)
        score, actions = run_noisy_v6(seed, noise_scale=noise)
        trial += 1
        if score > best_score:
            best_score = score
            best_actions = actions
            print(f'  Trial {trial}: {score} (noise={noise:.1f}) <<< NEW BEST',
                  file=sys.stderr)
        elif trial <= 5 or trial % 20 == 0:
            print(f'  Trial {trial}: {score} (noise={noise:.1f})', file=sys.stderr)

    print(f'  Phase 1 best: {best_score} ({trial} trials)', file=sys.stderr)

    # Phase 2: Trajectory search from best V6
    remaining = max_time - (time.time() - t_start)
    if remaining > 30:
        print(f'\nPhase 2: Trajectory search ({remaining:.0f}s)', file=sys.stderr)
        traj_score, traj_actions, imp = trajectory_search(
            seed, best_actions, best_score, remaining, verbose=verbose)
        if traj_score > best_score:
            best_score = traj_score
            best_actions = traj_actions

    elapsed = time.time() - t_start
    print(f'\nFinal: {best_score} (elapsed={elapsed:.1f}s)', file=sys.stderr)

    return best_score, best_actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=1800)
    args = parser.parse_args()

    score, actions = run_multistart(args.seed, args.max_time)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
