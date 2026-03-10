#!/usr/bin/env python3
"""Sequential per-bot greedy optimization for nightmare.

1. Run LMAPF to get initial 20-bot solution
2. For each bot (weakest first), re-optimize its trajectory
   with all other bots' actions locked
3. Each bot's optimizer sees future locked bot positions → no collisions
4. Repeat until no improvement

Usage: python nightmare_seqopt.py --seed 7005 --max-time 600
"""
from __future__ import annotations
import sys, time, copy, argparse
import numpy as np
from game_engine import (
    init_game, step, GameState,
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


def evaluate(seed, actions):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    for rnd in range(min(NUM_ROUNDS, len(actions))):
        step(state, actions[rnd], all_orders)
    return state.score


def score_action_heuristic(state, bid, action, ms, tables, drop_zones):
    """Score a single action for a single bot. Higher = better."""
    a, item_idx = action
    bx = int(state.bot_positions[bid, 0])
    by = int(state.bot_positions[bid, 1])
    score = 0.0

    active = state.get_active_order()
    preview = state.get_preview_order()

    if a == ACT_DROPOFF:
        # Delivering is almost always good
        inv = state.bot_inv_list(bid)
        if active:
            matching = sum(1 for t in inv if active.needs_type(t))
            score += matching * 100  # huge bonus for delivering active items
            # Check if this completes the order
            remaining = len(active.needs())
            if matching >= remaining:
                score += 500  # order completion bonus
        else:
            score += len(inv) * 10  # deliver anything
        return score

    if a == ACT_PICKUP:
        tid = int(ms.item_types[item_idx])
        if active and active.needs_type(tid):
            # Check if we already carry this type
            inv = state.bot_inv_list(bid)
            if tid not in inv:
                score += 80  # good: unique active type
            else:
                score += 30  # ok: duplicate active type
        elif preview and preview.needs_type(tid):
            inv = state.bot_inv_list(bid)
            if tid not in inv:
                score += 40  # preview type
            else:
                score += 10
        else:
            score += -20  # picking up useless item = dead inventory risk
        return score

    if a == ACT_WAIT:
        # Waiting is neutral but slightly bad (wasted round)
        return -1

    # Movement: evaluate destination quality
    nx, ny = bx + DX[a], by + DY[a]

    # Distance to nearest needed item
    if active:
        needs = list(active.needs())
        if needs:
            min_item_dist = 999
            for item_idx2 in range(ms.num_items):
                tid = int(ms.item_types[item_idx2])
                if tid in needs:
                    for adj in ms.item_adjacencies.get(item_idx2, []):
                        d = tables.get_distance((nx, ny), adj)
                        if d < min_item_dist:
                            min_item_dist = d
            score += max(0, 20 - min_item_dist) * 2  # closer to needed item = better

    # Distance to nearest dropoff (good if carrying items)
    inv = state.bot_inv_list(bid)
    if inv:
        min_drop = min(tables.get_distance((nx, ny), dz) for dz in drop_zones)
        has_active = active and any(active.needs_type(t) for t in inv)
        if has_active:
            score += max(0, 20 - min_drop) * 3  # heading to dropoff with active items

    return score


def optimize_bot(seed, action_log, bot_id, ms, tables, drop_zones):
    """Re-optimize one bot's trajectory with all others locked."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    new_actions = [list(a) for a in action_log]

    for rnd in range(NUM_ROUNDS):
        # Get valid actions for this bot
        valid = get_valid_actions(state, bot_id, ms)

        if len(valid) <= 1:
            # Only wait available
            new_actions[rnd][bot_id] = valid[0]
            step(state, new_actions[rnd], all_orders)
            continue

        # Score each valid action
        best_act = valid[0]
        best_score = -9999

        for act in valid:
            s = score_action_heuristic(state, bot_id, act, ms, tables, drop_zones)

            # Collision check: would this move put us on a locked bot's position?
            a, item_idx = act
            if a in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                bx = int(state.bot_positions[bot_id, 0])
                by = int(state.bot_positions[bot_id, 1])
                nx, ny = bx + DX[a], by + DY[a]
                # Check if any other bot is at this position
                blocked = False
                for bid2 in range(NUM_BOTS):
                    if bid2 == bot_id:
                        continue
                    bx2 = int(state.bot_positions[bid2, 0])
                    by2 = int(state.bot_positions[bid2, 1])
                    if nx == bx2 and ny == by2 and (nx, ny) != ms.spawn:
                        blocked = True
                        break
                if blocked:
                    s -= 50  # strong penalty for collision

            if s > best_score:
                best_score = s
                best_act = act

        new_actions[rnd][bot_id] = best_act
        step(state, new_actions[rnd], all_orders)

    return state.score, new_actions


def run_seqopt(seed, max_time=600, verbose=True):
    t_start = time.time()

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state
    tables = PrecomputedTables.get(ms)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

    # Phase 1: LMAPF initial solution
    from nightmare_lmapf_solver import LMAPFSolver
    score, action_log = LMAPFSolver.run_sim(seed, verbose=False)
    action_log = [list(a) for a in action_log]
    if verbose:
        print(f'LMAPF initial: {score}', file=sys.stderr)

    global_best = score
    global_actions = [list(a) for a in action_log]

    # Measure per-bot contributions
    deliveries = [0] * NUM_BOTS
    for rnd_actions in action_log:
        for bid, (a, _) in enumerate(rnd_actions):
            if a == ACT_DROPOFF:
                deliveries[bid] += 1

    # Phase 2: Sequential per-bot optimization (weakest first)
    iteration = 0
    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time - 10:
            break

        bot_order = sorted(range(NUM_BOTS), key=lambda b: deliveries[b])
        improved = False

        for bot_id in bot_order:
            if time.time() - t_start > max_time - 5:
                break

            new_score, new_actions = optimize_bot(
                seed, global_actions, bot_id, ms, tables, drop_zones)

            if new_score > global_best:
                global_best = new_score
                global_actions = new_actions
                improved = True
                # Re-measure deliveries
                deliveries = [0] * NUM_BOTS
                for rnd_actions in global_actions:
                    for bid, (a, _) in enumerate(rnd_actions):
                        if a == ACT_DROPOFF:
                            deliveries[bid] += 1
                if verbose:
                    print(f'  Bot {bot_id}: {new_score} (+{new_score - score})',
                          file=sys.stderr)

        iteration += 1
        if verbose:
            print(f'Iter {iteration}: {global_best} ({time.time()-t_start:.0f}s)',
                  file=sys.stderr)
        if not improved:
            break

    elapsed = time.time() - t_start
    if verbose:
        print(f'\nSeqOpt done: {global_best} ({elapsed:.1f}s, {iteration} iters)',
              file=sys.stderr)
    return global_best, global_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--max-time', type=int, default=600)
    args = parser.parse_args()

    score, actions = run_seqopt(args.seed, args.max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
