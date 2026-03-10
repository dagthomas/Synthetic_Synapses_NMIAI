#!/usr/bin/env python3
"""Nightmare Multi-Suffix ILS: Try N different V6 suffixes per perturbation.

Key insight: V6 with different parameters (ddw, over_assign) finds different
local optima from the same starting state. By trying 8-10 parameter combos
per kick, we maximize the chance of finding a good suffix.

Usage: python nightmare_multisuffix.py --seed 7009 --max-time 600
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
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator

sys.stdout.reconfigure(encoding='utf-8')


class OverAssignAllocator(V6Allocator):
    def __init__(self, *args, over_assign_bonus=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.over_assign_bonus = over_assign_bonus

    def _assign_item(self, bot_pos, needed, assigned_counts, claimed,
                     strict=False, zone_filter=-1, type_bonus=None):
        best_idx = None
        best_adj = None
        best_cost = 9999
        total_short = sum(needed.values())
        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            if strict:
                max_assign = need_count
            elif total_short <= 2:
                max_assign = need_count + self.over_assign_bonus
            else:
                max_assign = need_count + 1
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
                    drop_d = self._drop_dist(adj)
                    cost = d + drop_d * self.drop_d_weight - bonus
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj


SUFFIX_PARAMS = [
    (0.3, 2), (0.3, 3), (0.4, 2), (0.4, 3), (0.4, 4),
    (0.5, 3), (0.6, 3), (0.8, 3),
]


def run_v6_suffix(state, all_orders, from_round, num_rounds, ddw=0.4, oa=3):
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    solver.allocator = OverAssignAllocator(
        ms, tables, solver.drop_zones,
        max_preview_pickers=99, drop_d_weight=ddw, over_assign_bonus=oa)
    action_log = []
    for rnd in range(from_round, num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def best_suffix(state_template, all_orders, from_round, num_rounds, params=None):
    """Run V6 suffix with multiple parameter combos, return best."""
    if params is None:
        params = SUFFIX_PARAMS
    best_score = -1
    best_actions = None
    best_params = None
    for ddw, oa in params:
        s = copy.deepcopy(state_template)
        score, actions = run_v6_suffix(s, all_orders, from_round, num_rounds, ddw=ddw, oa=oa)
        if score > best_score:
            best_score = score
            best_actions = actions
            best_params = (ddw, oa)
    return best_score, best_actions, best_params


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


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def run_multisuffix_ils(seed, max_time=600, initial_actions=None, verbose=True):
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Phase 1: Multistart V6 with best-of-N
    best_score = 0
    best_actions = None

    if initial_actions:
        s = copy.deepcopy(state0)
        for rnd in range(num_rounds):
            step(s, initial_actions[rnd], all_orders)
        if s.score > best_score:
            best_score = s.score
            best_actions = [list(a) for a in initial_actions]
        print(f'Loaded: {s.score}', file=sys.stderr)

    # Fresh multistart
    s = copy.deepcopy(state0)
    ms_score, ms_actions, ms_params = best_suffix(
        s, all_orders, 0, num_rounds)
    if ms_score > best_score:
        best_score = ms_score
        best_actions = ms_actions
    print(f'Multistart best: {ms_score} {ms_params} ({time.time()-t_start:.1f}s)', file=sys.stderr)

    global_best = best_score
    global_actions = [list(a) for a in best_actions]

    # Phase 2: ILS with multi-suffix
    cp_interval = 10
    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, cp_interval)

    ils_iter = 0
    no_improve = 0

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 15:
            break

        # Kick: perturb 1-3 rounds
        kick_start = random.randint(0, num_rounds - 50)
        kick_len = random.randint(1, 3)

        # Replay to kick_start from checkpoint
        cp_round = (kick_start // cp_interval) * cp_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints.keys() if r <= kick_start)

        cp_state, cp_orders = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)
        for rnd in range(cp_round, kick_start):
            step(state, best_actions[rnd], cp_orders)

        # Perturb
        perturbed_rounds = []
        for k in range(kick_len):
            rnd = kick_start + k
            if rnd >= num_rounds:
                break
            n_perturb = random.randint(2, 8)
            bots = random.sample(range(num_bots), min(n_perturb, num_bots))
            prnd = list(best_actions[rnd])
            for bid in bots:
                valid = get_valid_actions(state, bid, ms)
                if valid:
                    prnd[bid] = random.choice(valid)
            perturbed_rounds.append(prnd)
            step(state, prnd, cp_orders)

        suffix_start = kick_start + len(perturbed_rounds)

        # Try N suffix parameter combos
        suffix_score, suffix_actions, suffix_params = best_suffix(
            state, cp_orders, suffix_start, num_rounds)

        full = best_actions[:kick_start] + perturbed_rounds + suffix_actions

        if suffix_score > global_best:
            global_best = suffix_score
            global_actions = [list(a) for a in full]
            best_score = suffix_score
            best_actions = [list(a) for a in full]
            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, cp_interval)
            no_improve = 0

            from solution_store import save_solution
            save_solution('nightmare', global_best, global_actions, seed=seed)

            if verbose:
                print(f'  ILS {ils_iter} [{elapsed:.0f}s]: BEST {global_best} '
                      f'(kick@{kick_start}, suffix={suffix_params})', file=sys.stderr)
        else:
            no_improve += 1

        # Trajectory search phase (cheaper, single-round perturbations)
        traj_budget = min(remaining * 0.2, 30)
        traj_t = time.time()
        while time.time() - traj_t < traj_budget:
            tr = random.randint(0, num_rounds - 30)
            cp_r = (tr // cp_interval) * cp_interval
            if cp_r not in checkpoints:
                cp_r = max(r for r in checkpoints.keys() if r <= tr)

            cp_s, cp_o = checkpoints[cp_r]
            ts = copy.deepcopy(cp_s)
            for rnd in range(cp_r, tr):
                step(ts, best_actions[rnd], cp_o)

            perturbed = list(best_actions[tr])
            bots = random.sample(range(num_bots), random.randint(1, 3))
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

            # Quick single-param suffix for traj search
            ddw = random.choice([0.3, 0.4, 0.4, 0.5, 0.6])
            oa = random.choice([2, 3, 3, 4])
            traj_score, traj_log = run_v6_suffix(
                ts, cp_o, tr + 1, num_rounds, ddw=ddw, oa=oa)

            if traj_score > global_best:
                new_full = best_actions[:tr] + [perturbed] + traj_log
                global_best = traj_score
                global_actions = [list(a) for a in new_full]
                best_score = traj_score
                best_actions = [list(a) for a in new_full]
                checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, cp_interval)
                no_improve = 0

                from solution_store import save_solution
                save_solution('nightmare', global_best, global_actions, seed=seed)

                if verbose:
                    print(f'  TRAJ [{time.time()-t_start:.0f}s]: BEST {global_best} '
                          f'(traj@{tr}, ddw={ddw})', file=sys.stderr)

        ils_iter += 1

        # Restart if stuck
        if no_improve >= 5:
            s = copy.deepcopy(state0)
            rs, ra, rp = best_suffix(s, all_orders, 0, num_rounds,
                                     params=[(random.uniform(0.1, 0.9), random.randint(1, 6))
                                             for _ in range(8)])
            if rs > 200:
                best_score = rs
                best_actions = ra
                checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, cp_interval)
                no_improve = 0
                if verbose:
                    print(f'  RESTART [{time.time()-t_start:.0f}s]: {rs} {rp}', file=sys.stderr)

        if verbose and ils_iter % 3 == 0:
            print(f'  ILS {ils_iter} [{elapsed:.0f}s]: best={global_best} no_imp={no_improve}',
                  file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nDone: {global_best} ({elapsed:.1f}s, {ils_iter} iter)', file=sys.stderr)
    return global_best, global_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=600)
    args = parser.parse_args()

    initial = None
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            initial = load_solution('nightmare')
            print(f'Loaded: {meta["score"]}', file=sys.stderr)
    except:
        pass

    score, actions = run_multisuffix_ils(args.seed, args.max_time, initial_actions=initial)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
