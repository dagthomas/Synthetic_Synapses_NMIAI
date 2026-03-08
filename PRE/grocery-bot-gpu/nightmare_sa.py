#!/usr/bin/env python3
"""Nightmare SA: Simulated Annealing with over-assign V6 suffix.

Combines:
1. Multistart V6 with randomized parameters
2. ILS-style kicks with V6 suffix
3. SA acceptance (accept worse solutions probabilistically)
4. Trajectory search with SA acceptance
5. Random restarts when stuck

Usage: python nightmare_sa.py --seed 7009 --max-time 3600
"""
from __future__ import annotations
import sys, time, random, copy, math, argparse
import numpy as np
from game_engine import (
    init_game, step, GameState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_WALL,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator

sys.stdout.reconfigure(encoding='utf-8')


class OverAssignAllocator(V6Allocator):
    """V6 allocator with over-assignment for last items."""

    def __init__(self, *args, over_assign_bonus=3, over_assign_threshold=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.over_assign_bonus = over_assign_bonus
        self.over_assign_threshold = over_assign_threshold

    def _assign_item(self, bot_pos, needed, assigned_counts, claimed,
                     strict=False, zone_filter=-1, type_bonus=None):
        """Override with dynamic over-assignment."""
        best_idx = None
        best_adj = None
        best_cost = 9999
        total_short = sum(needed.values())

        for tid, need_count in needed.items():
            if need_count <= 0:
                continue
            if strict:
                max_assign = need_count
            elif total_short <= self.over_assign_threshold:
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
                    if zone_filter >= 0:
                        drop_d = self.tables.get_distance(
                            adj, self.zone_dropoff.get(zone_filter, self.drop_zones[0]))
                    else:
                        drop_d = self._drop_dist(adj)
                    cost = d + drop_d * self.drop_d_weight - bonus
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = item_idx
                        best_adj = adj
        return best_idx, best_adj


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


def run_v6_suffix(state, all_orders, from_round, num_rounds,
                  drop_d_weight=0.4, over_assign=3):
    """Run over-assign V6 from a given state."""
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    solver.allocator = OverAssignAllocator(
        ms, tables, solver.drop_zones,
        max_preview_pickers=99, drop_d_weight=drop_d_weight,
        over_assign_bonus=over_assign, over_assign_threshold=2)
    action_log = []
    for rnd in range(from_round, num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def run_full_v6(seed, drop_d_weight=0.4, over_assign=3):
    """Run full V6 from scratch with parameters."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    return run_v6_suffix(state, all_orders, 0, DIFF_ROUNDS['nightmare'],
                         drop_d_weight=drop_d_weight, over_assign=over_assign)


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


def kick_and_suffix(seed, actions, num_rounds, num_bots, ms):
    """Kick + V6 suffix."""
    kick_start = random.randint(0, num_rounds - 80)
    kick_len = random.randint(1, 3)  # Perturb 1-3 rounds

    # Replay to kick_start
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(kick_start):
        step(state, actions[rnd], all_orders)

    # Perturb kick_len rounds
    perturbed_rounds = []
    for k in range(kick_len):
        rnd = kick_start + k
        if rnd >= num_rounds:
            break
        n_perturb = random.randint(2, 8)
        bots = random.sample(range(num_bots), min(n_perturb, num_bots))
        perturbed = list(actions[rnd]) if rnd < len(actions) else [(ACT_WAIT, -1)] * num_bots
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if valid:
                perturbed[bid] = random.choice(valid)
        perturbed_rounds.append(perturbed)
        step(state, perturbed, all_orders)

    # V6 suffix with random parameters
    ddw = random.choice([0.3, 0.4, 0.4, 0.5, 0.6, 0.8])
    oa = random.choice([2, 3, 3, 4, 5])
    suffix_start = kick_start + len(perturbed_rounds)
    new_score, suffix = run_v6_suffix(
        state, all_orders, suffix_start, num_rounds,
        drop_d_weight=ddw, over_assign=oa)

    full = actions[:kick_start] + perturbed_rounds + suffix
    return new_score, full, kick_start


def traj_search_sa(seed, actions, score, max_time, ms):
    """Trajectory search with SA acceptance."""
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    best_score = score
    current_score = score
    best_actions = [list(a) for a in actions]
    current_actions = [list(a) for a in actions]

    cp_interval = 10
    checkpoints = rebuild_checkpoints(seed, current_actions, num_rounds, cp_interval)

    round_options = list(range(0, num_rounds - 30))
    round_weights = [max(0.05, 1.0 - r / num_rounds) for r in round_options]

    iterations = 0
    improvements = 0
    temperature = 10.0

    while time.time() - t_start < max_time:
        perturb_round = random.choices(round_options, weights=round_weights, k=1)[0]
        n_perturb = random.choices([1, 1, 2, 2, 3], k=1)[0]

        cp_round = (perturb_round // cp_interval) * cp_interval
        if cp_round not in checkpoints:
            cp_round = max(r for r in checkpoints.keys() if r <= perturb_round)

        cp_state, all_orders_cp = checkpoints[cp_round]
        state = copy.deepcopy(cp_state)

        for rnd in range(cp_round, perturb_round):
            step(state, current_actions[rnd], all_orders_cp)

        perturbed = list(current_actions[perturb_round])
        bots = random.sample(range(num_bots), min(n_perturb, num_bots))
        changed = False
        for bid in bots:
            valid = get_valid_actions(state, bid, ms)
            if len(valid) <= 1:
                continue
            original = current_actions[perturb_round][bid]
            alts = [a for a in valid if a != original]
            if alts:
                perturbed[bid] = random.choice(alts)
                changed = True

        if not changed:
            iterations += 1
            continue

        ps = copy.deepcopy(state)
        step(ps, perturbed, all_orders_cp)

        ddw = random.choice([0.3, 0.4, 0.4, 0.5, 0.6, 0.8])
        oa = random.choice([2, 3, 3, 4, 5])
        new_score, new_log = run_v6_suffix(
            ps, all_orders_cp, perturb_round + 1, num_rounds,
            drop_d_weight=ddw, over_assign=oa)

        delta = new_score - current_score
        accept = delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1))

        if accept:
            new_full = current_actions[:perturb_round] + [perturbed] + new_log
            current_actions = new_full
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_actions = [list(a) for a in new_full]
                improvements += 1
            checkpoints = rebuild_checkpoints(seed, current_actions, num_rounds, cp_interval)

        temperature *= 0.999
        iterations += 1

    return best_score, best_actions, improvements


def run_sa(seed: int, max_time: float = 3600,
           initial_actions=None, verbose=True) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = CONFIGS['nightmare']['bots']

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Phase 1: Multistart V6
    best_score = 0
    best_actions = None

    if initial_actions is not None:
        state = copy.deepcopy(state0)
        for rnd in range(num_rounds):
            step(state, initial_actions[rnd], all_orders)
        if state.score > best_score:
            best_score = state.score
            best_actions = [list(a) for a in initial_actions]
        print(f'Loaded: {state.score}', file=sys.stderr)

    params = [
        (0.3, 2), (0.3, 3), (0.4, 2), (0.4, 3), (0.4, 4), (0.4, 5),
        (0.5, 3), (0.5, 4), (0.6, 2), (0.6, 3), (0.8, 3), (0.8, 4),
        (0.2, 3), (0.3, 5), (0.5, 5), (0.7, 3),
    ]
    for ddw, oa in params:
        if time.time() - t_start > 30:
            break
        score, actions = run_full_v6(seed, drop_d_weight=ddw, over_assign=oa)
        if score > best_score:
            best_score = score
            best_actions = actions
            print(f'  Multistart ddw={ddw} oa={oa}: {score}', file=sys.stderr)

    if best_actions is None:
        best_score, best_actions = run_full_v6(seed)
        print(f'  V6: {best_score}', file=sys.stderr)

    global_best = best_score
    global_actions = [list(a) for a in best_actions]
    current_score = best_score
    current_actions = [list(a) for a in best_actions]

    print(f'Phase 1 done: {best_score} ({time.time()-t_start:.1f}s)', file=sys.stderr)

    # Phase 2: SA with kicks + trajectory search
    sa_temp = 30.0
    sa_iter = 0
    no_improve = 0

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 15:
            break

        # Kick
        kick_score, kick_actions, kick_at = kick_and_suffix(
            seed, current_actions, num_rounds, num_bots, ms)

        # Trajectory search
        traj_budget = min(remaining * 0.3 / max(1, remaining // 90), 45)
        traj_score, traj_actions, traj_imp = traj_search_sa(
            seed, kick_actions, kick_score, traj_budget, ms)

        # SA acceptance for outer loop
        delta = traj_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(sa_temp, 0.1)):
            current_score = traj_score
            current_actions = traj_actions
            tag = "accept"
        else:
            tag = "reject"

        if traj_score > global_best:
            global_best = traj_score
            global_actions = [list(a) for a in traj_actions]
            tag = "BEST"
            no_improve = 0
            from solution_store import save_solution
            save_solution('nightmare', global_best, global_actions, seed=seed)
        else:
            no_improve += 1

        sa_temp *= 0.97
        sa_iter += 1

        if verbose:
            print(f'  SA {sa_iter} [{elapsed:.0f}s T={sa_temp:.1f}]: '
                  f'kick@{kick_at}={kick_score}, traj={traj_score}({traj_imp}), '
                  f'cur={current_score}, best={global_best} [{tag}]',
                  file=sys.stderr)

        # Restart if stuck
        if no_improve >= 8:
            ddw = random.uniform(0.2, 0.8)
            oa = random.randint(2, 5)
            score, actions = run_full_v6(seed, drop_d_weight=ddw, over_assign=oa)
            if score > 200:
                current_score = score
                current_actions = actions
                sa_temp = max(sa_temp, 15.0)
                no_improve = 0
                print(f'  RESTART: {score} (ddw={ddw:.2f}, oa={oa})', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nDone: {global_best} ({elapsed:.1f}s, {sa_iter} iter)', file=sys.stderr)
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

    score, actions = run_sa(args.seed, args.max_time, initial_actions=initial)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
