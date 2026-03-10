#!/usr/bin/env python3
"""Nightmare SA: Simulated Annealing directly on action sequence.

Instead of ILS (perturb + suffix reconstruct), SA directly modifies individual
actions and evaluates via fast checkpoint-based replay.

Key advantage: makes fine-grained improvements that accumulate.
ILS needs suffix to beat current solution; SA can accept slightly worse
solutions to escape local optima.

Usage: python nightmare_sa2.py --seed 7005 --max-time 3600
"""
from __future__ import annotations
import sys, time, random, copy, math, argparse
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


def build_checkpoints(seed, actions, interval=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    checkpoints = {0: (copy.deepcopy(state), all_orders)}
    for rnd in range(min(NUM_ROUNDS, len(actions))):
        step(state, actions[rnd], all_orders)
        if (rnd + 1) % interval == 0:
            checkpoints[rnd + 1] = (copy.deepcopy(state), all_orders)
    return checkpoints, state.score


def evaluate_from_checkpoint(checkpoints, actions, change_round, cp_interval=10):
    """Evaluate by replaying from nearest checkpoint."""
    cp_round = (change_round // cp_interval) * cp_interval
    if cp_round not in checkpoints:
        cp_round = max(r for r in checkpoints if r <= change_round)

    cp_state, cp_orders = checkpoints[cp_round]
    state = copy.deepcopy(cp_state)
    for rnd in range(cp_round, NUM_ROUNDS):
        if rnd < len(actions):
            step(state, actions[rnd], cp_orders)
    return state.score


def run_sa(seed, max_time=3600, verbose=True):
    t_start = time.time()

    state0, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state0.map_state

    # Load best solution
    from nightmare_lmapf_solver import LMAPFSolver
    best_score = 0
    best_actions = None

    try:
        from solution_store import load_solution, load_meta
        meta = load_meta('nightmare')
        if meta and meta.get('score', 0) > 0:
            best_actions = load_solution('nightmare')
            best_score = meta['score']
            print(f'Loaded: {best_score}', file=sys.stderr)
    except:
        pass

    if best_actions is None or best_score < 200:
        score, actions = LMAPFSolver.run_sim(seed, verbose=False)
        best_score = score
        best_actions = [list(a) for a in actions]
        print(f'LMAPF: {best_score}', file=sys.stderr)

    best_actions = [list(a) for a in best_actions]

    # Build checkpoints
    cp_interval = 10
    checkpoints, verified_score = build_checkpoints(seed, best_actions, cp_interval)
    if verified_score != best_score:
        print(f'Score mismatch: expected {best_score}, got {verified_score}',
              file=sys.stderr)
        best_score = verified_score

    global_best = best_score
    global_actions = [list(a) for a in best_actions]

    current_score = best_score
    current_actions = [list(a) for a in best_actions]

    # SA parameters
    T_start = 20.0     # initial temperature
    T_end = 0.5        # final temperature
    T = T_start

    iteration = 0
    accepts = 0
    improves = 0
    last_improve_time = time.time()

    print(f'\nSA starting from {current_score}...', file=sys.stderr)

    while True:
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        if remaining < 3:
            break

        # Anneal temperature
        progress = elapsed / max_time
        T = T_start * (T_end / T_start) ** progress

        # Choose perturbation type
        r = random.random()
        if r < 0.5:
            # Single-bot single-round change
            n_changes = 1
        elif r < 0.8:
            # Multi-bot single-round change (2-5 bots)
            n_changes = random.randint(2, 5)
        else:
            # Multi-round change (2-3 consecutive rounds, 1-3 bots each)
            n_changes = -1  # special marker

        # Make a copy for the candidate
        candidate = [list(a) for a in current_actions]
        change_round = random.randint(0, NUM_ROUNDS - 10)

        if n_changes > 0:
            # Get state at this round for valid action check
            cp_round = (change_round // cp_interval) * cp_interval
            if cp_round not in checkpoints:
                cp_round = max(r2 for r2 in checkpoints if r2 <= change_round)
            cp_state, cp_orders = checkpoints[cp_round]
            state = copy.deepcopy(cp_state)
            for rnd in range(cp_round, change_round):
                step(state, candidate[rnd], cp_orders)

            bots = random.sample(range(NUM_BOTS), min(n_changes, NUM_BOTS))
            for bid in bots:
                valid = get_valid_actions(state, bid, ms)
                if len(valid) > 1:
                    alts = [a for a in valid if a != candidate[change_round][bid]]
                    if alts:
                        candidate[change_round][bid] = random.choice(alts)
        else:
            # Multi-round perturbation
            n_rounds = random.randint(2, 3)
            n_bots_per_round = random.randint(1, 3)
            cp_round = (change_round // cp_interval) * cp_interval
            if cp_round not in checkpoints:
                cp_round = max(r2 for r2 in checkpoints if r2 <= change_round)
            cp_state, cp_orders = checkpoints[cp_round]
            state = copy.deepcopy(cp_state)
            for rnd in range(cp_round, change_round):
                step(state, candidate[rnd], cp_orders)

            for dr in range(n_rounds):
                rnd = change_round + dr
                if rnd >= NUM_ROUNDS:
                    break
                bots = random.sample(range(NUM_BOTS), min(n_bots_per_round, NUM_BOTS))
                for bid in bots:
                    valid = get_valid_actions(state, bid, ms)
                    if len(valid) > 1:
                        alts = [a for a in valid if a != candidate[rnd][bid]]
                        if alts:
                            candidate[rnd][bid] = random.choice(alts)
                step(state, candidate[rnd], cp_orders)

        # Evaluate candidate
        new_score = evaluate_from_checkpoint(
            checkpoints, candidate, change_round, cp_interval)

        # SA acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(T, 0.01)):
            current_score = new_score
            current_actions = candidate
            accepts += 1

            if new_score > global_best:
                global_best = new_score
                global_actions = [list(a) for a in candidate]
                improves += 1
                last_improve_time = time.time()

                # Rebuild checkpoints for new best
                checkpoints, _ = build_checkpoints(seed, global_actions, cp_interval)

                from solution_store import save_solution
                save_solution('nightmare', global_best, global_actions, seed=seed)

                if verbose:
                    print(f'  [{elapsed:.0f}s] NEW BEST: {global_best} '
                          f'(T={T:.1f}, iter={iteration})', file=sys.stderr)

            # Periodically rebuild checkpoints for current solution
            if accepts % 100 == 0:
                checkpoints, _ = build_checkpoints(seed, current_actions, cp_interval)

        iteration += 1
        if verbose and iteration % 2000 == 0:
            stuck = time.time() - last_improve_time
            rate = iteration / max(elapsed, 1)
            print(f'  [{elapsed:.0f}s] iter={iteration} T={T:.1f} '
                  f'best={global_best} cur={current_score} '
                  f'acc={accepts} imp={improves} '
                  f'rate={rate:.0f}/s stuck={stuck:.0f}s', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\nSA done: {global_best} ({elapsed:.1f}s, {iteration} iter, '
          f'{improves} improvements)', file=sys.stderr)
    return global_best, global_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7005)
    parser.add_argument('--max-time', type=int, default=3600)
    args = parser.parse_args()

    score, actions = run_sa(args.seed, args.max_time)

    from solution_store import save_solution
    saved = save_solution('nightmare', score, actions, seed=args.seed)
    print(f'Final: {score} (saved={saved})', file=sys.stderr)


if __name__ == '__main__':
    main()
