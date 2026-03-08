#!/usr/bin/env python3
"""Parallel trajectory search: multiple workers exploring different trajectories.

Uses multiprocessing to run 4+ trajectory search workers in parallel.
Each worker explores different regions of the action space.
Workers share best solutions via a shared queue.

Usage: python nightmare_parallel_traj.py --seed 7009 --max-time 1800 --workers 4
"""
from __future__ import annotations
import sys, time, random, copy, argparse
import multiprocessing as mp
from multiprocessing import Queue
import numpy as np
from game_engine import (
    init_game, step, GameState, MapState, Order,
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


def rebuild_checkpoints(seed, actions, num_rounds, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(num_rounds, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints


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


def worker_fn(worker_id, seed, initial_actions, initial_score,
              result_queue, time_limit, update_queue):
    """Single trajectory search worker."""
    random.seed(worker_id * 1000 + int(time.time()))
    np.random.seed(worker_id * 1000 + int(time.time()) % (2**31))

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
        if elapsed > time_limit:
            break

        # Check for updates from other workers
        while not update_queue.empty():
            try:
                new_score, new_actions = update_queue.get_nowait()
                if new_score > best_score:
                    best_score = new_score
                    best_actions = new_actions
                    checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)
            except:
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

            result_queue.put((best_score, best_actions, worker_id))
            checkpoints = rebuild_checkpoints(seed, best_actions, num_rounds, checkpoint_interval)

            print(f'  W{worker_id} [{elapsed:.0f}s] NEW BEST: {best_score} '
                  f'(r={perturb_round}, iter={iterations})', file=sys.stderr)

        iterations += 1
        if iterations % 1000 == 0:
            elapsed2 = time.time() - t_start
            rate = iterations / elapsed2
            print(f'  W{worker_id} [{elapsed2:.0f}s] iter={iterations} best={best_score} '
                  f'imp={improvements} rate={rate:.1f}/s', file=sys.stderr)

    result_queue.put((best_score, best_actions, worker_id))
    return best_score


def run_parallel_traj(seed: int, max_time: float = 1800,
                      num_workers: int = 4,
                      initial_actions=None) -> tuple[int, list]:
    t_start = time.time()
    num_rounds = DIFF_ROUNDS['nightmare']

    # Get initial solution
    if initial_actions is not None:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        for rnd in range(num_rounds):
            step(state, initial_actions[rnd], all_orders)
        initial_score = state.score
        best_actions = [list(a) for a in initial_actions]
        best_score = initial_score
        print(f'Initial solution: {initial_score}', file=sys.stderr)
    else:
        # V6 baseline
        print(f'V6 baseline...', file=sys.stderr)
        baseline_score, baseline_actions = NightmareSolverV6.run_sim(seed, verbose=False)
        best_score = baseline_score
        best_actions = baseline_actions
        initial_score = baseline_score
        print(f'V6: {baseline_score}', file=sys.stderr)

    # Launch workers
    result_queue = Queue()
    update_queues = [Queue() for _ in range(num_workers)]
    workers = []

    for wid in range(num_workers):
        p = mp.Process(target=worker_fn,
                       args=(wid, seed, best_actions, best_score,
                             result_queue, max_time - 10,
                             update_queues[wid]))
        p.start()
        workers.append(p)

    print(f'Launched {num_workers} workers', file=sys.stderr)

    # Monitor results
    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            break

        alive = any(p.is_alive() for p in workers)
        if not alive:
            break

        try:
            while not result_queue.empty():
                score, actions, wid = result_queue.get_nowait()
                if score > best_score:
                    best_score = score
                    best_actions = actions
                    print(f'  GLOBAL BEST: {best_score} (from W{wid}, {elapsed:.0f}s)',
                          file=sys.stderr)
                    # Broadcast to other workers
                    for qid, q in enumerate(update_queues):
                        if qid != wid:
                            q.put((best_score, best_actions))
        except:
            pass

        time.sleep(1)

    # Drain remaining results
    try:
        while not result_queue.empty():
            score, actions, wid = result_queue.get_nowait()
            if score > best_score:
                best_score = score
                best_actions = actions
    except:
        pass

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    elapsed = time.time() - t_start
    print(f'\nParallel traj done: {best_score} (from {initial_score}, '
          f'elapsed={elapsed:.1f}s)', file=sys.stderr)

    return best_score, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7009)
    parser.add_argument('--max-time', type=int, default=1800)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # Try to load existing solution
    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            actions = load_solution('nightmare')
            print(f'Starting from saved solution: {meta["score"]}', file=sys.stderr)
            score, final_actions = run_parallel_traj(
                args.seed, args.max_time, args.workers, actions)
        else:
            score, final_actions = run_parallel_traj(
                args.seed, args.max_time, args.workers)
    except:
        score, final_actions = run_parallel_traj(
            args.seed, args.max_time, args.workers)

    if score > 0:
        from solution_store import save_solution
        saved = save_solution('nightmare', score, final_actions, seed=args.seed)
        print(f'Saved: {saved}', file=sys.stderr)


if __name__ == '__main__':
    main()
