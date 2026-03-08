#!/usr/bin/env python3
"""Nightmare Crossover: Combine segments from different V6 runs.

Key insight: V6 suffix always converges to ~400-430. Instead of using V6 suffix,
directly combine action sequences from different V6 runs.

Approach:
1. Generate N diverse V6 solutions with different parameters
2. Crossover: take prefix from solution A, suffix from solution B
3. Simulate the hybrid (move actions still work; pickup/dropoff may fail)
4. Keep the best hybrid
5. Iterate with the best hybrid as a parent

Also: segment splicing — test each segment independently and combine the best.
"""
from __future__ import annotations
import sys, time, random, copy
import numpy as np
from game_engine import init_game, step
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


def run_v6(seed, drop_d_weight=0.4, over_assign=3):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    num_rounds = DIFF_ROUNDS['nightmare']
    solver = NightmareSolverV6(ms, tables, future_orders=all_orders)
    solver.allocator = OverAssignAllocator(
        ms, tables, solver.drop_zones,
        max_preview_pickers=99, drop_d_weight=drop_d_weight,
        over_assign_bonus=over_assign)
    action_log = []
    for rnd in range(num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(list(actions))
        step(state, actions, all_orders)
    return state.score, action_log


def simulate(seed, actions, num_rounds=None):
    """Simulate a complete action sequence."""
    if num_rounds is None:
        num_rounds = DIFF_ROUNDS['nightmare']
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
    return state.score


def blind_crossover(actions_a, actions_b, crossover_point):
    """Combine prefix of A with suffix of B."""
    return actions_a[:crossover_point] + actions_b[crossover_point:]


def multi_crossover(actions_a, actions_b, num_points=2):
    """Multiple crossover points."""
    n = len(actions_a)
    points = sorted(random.sample(range(1, n), min(num_points, n - 1)))
    result = []
    use_a = True
    prev = 0
    for p in points:
        if use_a:
            result.extend(actions_a[prev:p])
        else:
            result.extend(actions_b[prev:p])
        use_a = not use_a
        prev = p
    if use_a:
        result.extend(actions_a[prev:])
    else:
        result.extend(actions_b[prev:])
    return result


def segment_evaluate(seed, base_actions, donor_actions, seg_start, seg_len):
    """Replace a segment of base with donor and evaluate."""
    n = len(base_actions)
    seg_end = min(seg_start + seg_len, n)
    hybrid = base_actions[:seg_start] + donor_actions[seg_start:seg_end] + base_actions[seg_end:]
    return simulate(seed, hybrid)


def run_crossover_search(seed, max_time=600, verbose=True):
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']

    # Phase 1: Generate diverse V6 solutions
    print(f'Generating diverse V6 solutions...', file=sys.stderr)
    params = [
        (0.2, 2), (0.3, 2), (0.3, 3), (0.4, 2), (0.4, 3), (0.4, 4),
        (0.5, 3), (0.5, 4), (0.6, 2), (0.6, 3), (0.8, 3), (0.8, 4),
        (0.3, 5), (0.4, 5), (0.5, 5), (0.7, 3), (0.2, 4), (0.6, 5),
        (0.1, 3), (0.9, 3), (0.4, 6), (0.3, 1), (0.5, 1), (0.4, 0),
    ]

    solutions = []
    for ddw, oa in params:
        if time.time() - t_start > min(60, max_time * 0.15):
            break
        score, actions = run_v6(seed, drop_d_weight=ddw, over_assign=oa)
        solutions.append((score, actions, f'ddw={ddw},oa={oa}'))
        if verbose:
            print(f'  V6 ddw={ddw} oa={oa}: {score}', file=sys.stderr)

    solutions.sort(key=lambda x: -x[0])
    print(f'Top 5: {[s[0] for s in solutions[:5]]}', file=sys.stderr)

    best_score = solutions[0][0]
    best_actions = solutions[0][1]

    # Try loading existing solution
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > best_score:
            loaded = load_solution('nightmare')
            # Verify score
            check_score = simulate(seed, loaded)
            if check_score > best_score:
                best_score = check_score
                best_actions = loaded
                solutions.insert(0, (check_score, loaded, 'loaded'))
                print(f'  Loaded: {check_score}', file=sys.stderr)
    except:
        pass

    print(f'Phase 1 best: {best_score} ({time.time()-t_start:.1f}s)', file=sys.stderr)

    # Phase 2: Crossover search
    crossover_points = list(range(10, num_rounds - 10, 5))
    iteration = 0
    improvements = 0

    while time.time() - t_start < max_time * 0.8:
        # Pick two parents (bias toward top solutions)
        weights = [1.0 / (i + 1) for i in range(len(solutions))]
        idx_a = random.choices(range(len(solutions)), weights=weights)[0]
        idx_b = random.choices(range(len(solutions)), weights=weights)[0]
        while idx_b == idx_a and len(solutions) > 1:
            idx_b = random.choices(range(len(solutions)), weights=weights)[0]

        score_a, actions_a, name_a = solutions[idx_a]
        score_b, actions_b, name_b = solutions[idx_b]

        # Try different crossover strategies
        strategy = random.choice(['single', 'single', 'multi', 'segment'])

        if strategy == 'single':
            cp = random.choice(crossover_points)
            hybrid = blind_crossover(actions_a, actions_b, cp)
            hybrid_score = simulate(seed, hybrid)

            if hybrid_score > best_score:
                best_score = hybrid_score
                best_actions = hybrid
                solutions.append((hybrid_score, hybrid, f'cross({name_a},{name_b})@{cp}'))
                solutions.sort(key=lambda x: -x[0])
                improvements += 1
                if verbose:
                    print(f'  [{time.time()-t_start:.0f}s] BEST: {hybrid_score} '
                          f'(cross @{cp}, {name_a}+{name_b})', file=sys.stderr)

            # Also try reverse crossover
            hybrid_r = blind_crossover(actions_b, actions_a, cp)
            hybrid_r_score = simulate(seed, hybrid_r)
            if hybrid_r_score > best_score:
                best_score = hybrid_r_score
                best_actions = hybrid_r
                solutions.append((hybrid_r_score, hybrid_r, f'cross({name_b},{name_a})@{cp}'))
                solutions.sort(key=lambda x: -x[0])
                improvements += 1
                if verbose:
                    print(f'  [{time.time()-t_start:.0f}s] BEST: {hybrid_r_score} '
                          f'(reverse cross @{cp})', file=sys.stderr)

        elif strategy == 'multi':
            npts = random.choice([2, 3, 4])
            hybrid = multi_crossover(actions_a, actions_b, num_points=npts)
            hybrid_score = simulate(seed, hybrid)
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_actions = hybrid
                solutions.append((hybrid_score, hybrid, f'multi({npts})'))
                solutions.sort(key=lambda x: -x[0])
                improvements += 1
                if verbose:
                    print(f'  [{time.time()-t_start:.0f}s] BEST: {hybrid_score} '
                          f'(multi-{npts}pt)', file=sys.stderr)

        elif strategy == 'segment':
            seg_start = random.randint(0, num_rounds - 50)
            seg_len = random.randint(20, 100)
            # Replace segment of best with segment from donor
            donor_idx = random.choices(range(len(solutions)), weights=weights)[0]
            donor_actions = solutions[donor_idx][1]
            seg_end = min(seg_start + seg_len, num_rounds)
            hybrid = best_actions[:seg_start] + donor_actions[seg_start:seg_end] + best_actions[seg_end:]
            hybrid_score = simulate(seed, hybrid)
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_actions = hybrid
                solutions.append((hybrid_score, hybrid, f'seg({seg_start}-{seg_end})'))
                solutions.sort(key=lambda x: -x[0])
                improvements += 1
                if verbose:
                    print(f'  [{time.time()-t_start:.0f}s] BEST: {hybrid_score} '
                          f'(segment {seg_start}-{seg_end})', file=sys.stderr)

        iteration += 1
        if iteration % 500 == 0 and verbose:
            elapsed = time.time() - t_start
            print(f'  [{elapsed:.0f}s] iter={iteration} best={best_score} imp={improvements}',
                  file=sys.stderr)

        # Keep solution pool bounded
        if len(solutions) > 30:
            solutions = solutions[:20]

    # Phase 3: Fine-grained segment search on best
    print(f'\nPhase 3: Fine segment search (best={best_score})...', file=sys.stderr)
    for seg_len in [10, 20, 30, 50]:
        if time.time() - t_start > max_time - 10:
            break
        for seg_start in range(0, num_rounds - seg_len, seg_len // 2):
            if time.time() - t_start > max_time - 5:
                break
            for donor_score, donor_actions, _ in solutions[1:10]:
                seg_end = min(seg_start + seg_len, num_rounds)
                hybrid = best_actions[:seg_start] + donor_actions[seg_start:seg_end] + best_actions[seg_end:]
                hybrid_score = simulate(seed, hybrid)
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_actions = hybrid
                    improvements += 1
                    print(f'  Fine: {hybrid_score} (seg {seg_start}-{seg_end})', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nDone: {best_score} ({elapsed:.1f}s, {iteration} iter, {improvements} imp)',
          file=sys.stderr)
    return best_score, best_actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=600)
    args = parser.parse_args()

    score, actions = run_crossover_search(args.seed, args.max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
