"""Optimization-based solver using simulated game as evaluation.

Instead of hand-crafting a planner, this treats the problem as black-box
optimization. The game engine is used directly as the fitness function.

Approach:
  1. Parameterize bot strategies (assignment priorities, timing)
  2. Evaluate by running full game simulation
  3. Use evolutionary search (CMA-ES) to optimize parameters
  4. Works for any difficulty, but targets nightmare

This is the "ML/training" approach the user asked about: using
optimization to learn the best strategy parameters for each specific
game instance (seed + difficulty).

Usage:
    python optimize_search.py --seeds 7005 --difficulty nightmare
    python optimize_search.py --seeds 42 --difficulty hard --budget 60
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

from game_engine import (
    init_game, step as cpu_step,
    GameState, MapState, Order,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import CONFIGS, DIFF_ROUNDS, parse_seeds
from precompute import PrecomputedTables


class ParameterizedSolver:
    """Bot solver with tunable parameters for optimization.

    Parameters control:
    - Item-type preferences per bot (which types to prioritize)
    - Distance vs urgency trade-off
    - When to switch from pickup to deliver
    - DZ preference per bot (which DZ to use for nightmare)
    """

    def __init__(self, map_state: MapState, all_orders: list[Order],
                 num_bots: int, num_rounds: int = 300,
                 params: np.ndarray | None = None):
        self.ms = map_state
        self.all_orders = all_orders
        self.num_bots = num_bots
        self.num_rounds = num_rounds

        self.tables = PrecomputedTables.get(map_state)
        self.dz_list = [tuple(int(c) for c in dz) for dz in map_state.drop_off_zones]
        self.dz_set = set(self.dz_list)
        self.spawn = (int(map_state.spawn[0]), int(map_state.spawn[1]))

        # Item lookup
        self.type_to_items: dict[int, list[tuple[int, tuple, list]]] = defaultdict(list)
        for idx in range(map_state.num_items):
            tid = int(map_state.item_types[idx])
            pos = (int(map_state.item_positions[idx, 0]),
                   int(map_state.item_positions[idx, 1]))
            adj = [(int(a[0]), int(a[1]))
                   for a in map_state.item_adjacencies.get(idx, [])]
            self.type_to_items[tid].append((idx, pos, adj))

        # Walkable neighbors
        walkable = set()
        for y in range(map_state.height):
            for x in range(map_state.width):
                if map_state.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                    walkable.add((x, y))
        self._neighbors: dict[tuple, list] = {}
        for pos in walkable:
            nbrs = []
            for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                nx, ny = pos[0] + DX[act], pos[1] + DY[act]
                if (nx, ny) in walkable:
                    nbrs.append((act, (nx, ny)))
            self._neighbors[pos] = nbrs

        # Set parameters
        self._set_params(params)

    def _set_params(self, params: np.ndarray | None):
        """Decode parameter vector into solver settings."""
        if params is None:
            # Defaults
            self.deliver_threshold = 1  # deliver when this many items match active
            self.pickup_batch = INV_CAP  # try to batch this many pickups
            self.dz_preference = {}  # bot -> preferred DZ index
            self.type_weights = {}   # bot -> {type: weight}
            self.urgency_weight = 1.0
            return

        # Decode params:
        # [0]: deliver_threshold (0.5-3.0, clamped to int)
        # [1]: urgency_weight (0.1-5.0)
        # [2..2+num_bots-1]: DZ preference per bot (0-num_dz, float→int)
        # [2+num_bots..]: type weights (not used in basic version)
        idx = 0
        self.deliver_threshold = max(1, min(INV_CAP, int(np.clip(params[idx], 0.5, 3.5))))
        idx += 1
        self.urgency_weight = float(np.clip(params[idx], 0.1, 5.0))
        idx += 1
        n_dz = len(self.dz_list)
        self.dz_preference = {}
        for bid in range(self.num_bots):
            if idx < len(params):
                dz_idx = int(np.clip(params[idx], 0, n_dz - 0.01))
                self.dz_preference[bid] = dz_idx
                idx += 1

    @staticmethod
    def param_dim(num_bots: int) -> int:
        """Number of parameters for this solver."""
        return 2 + num_bots  # threshold + urgency + dz_pref per bot

    def _dist(self, a, b):
        ai = self.tables.pos_to_idx.get(a)
        bi = self.tables.pos_to_idx.get(b)
        if ai is None or bi is None:
            return 9999
        return int(self.tables.dist_matrix[ai, bi])

    def _first_step(self, a, b):
        ai = self.tables.pos_to_idx.get(a)
        bi = self.tables.pos_to_idx.get(b)
        if ai is None or bi is None:
            return ACT_WAIT
        return int(self.tables.next_step_matrix[ai, bi])

    def _preferred_dz(self, bid: int, pos: tuple) -> tuple:
        """Get preferred DZ for this bot."""
        dz_idx = self.dz_preference.get(bid, 0)
        if dz_idx < len(self.dz_list):
            return self.dz_list[dz_idx]
        # Fallback: nearest DZ
        best = self.dz_list[0]
        best_d = self._dist(pos, best)
        for dz in self.dz_list[1:]:
            d = self._dist(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best

    def _find_best_item(self, type_id: int, from_pos: tuple) -> tuple | None:
        candidates = self.type_to_items.get(type_id, [])
        if not candidates:
            return None
        best = None
        best_d = 9999
        for item_idx, shelf_pos, adj_list in candidates:
            for adj in adj_list:
                d = self._dist(from_pos, adj)
                if d < best_d:
                    best_d = d
                    best = (item_idx, shelf_pos, adj)
        return best

    def decide(self, state: GameState, all_orders: list[Order],
               round_num: int) -> list[tuple[int, int]]:
        """Decide actions for all bots."""
        active = state.get_active_order()
        active_needs = active.needs() if active else []

        actions = [(ACT_WAIT, -1)] * self.num_bots
        occupied = set()

        # Track occupied positions
        for bid in range(self.num_bots):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            occupied.add(pos)

        # Process bots in ID order (matching game engine)
        for bid in range(self.num_bots):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = [int(state.bot_inventories[bid, i]) for i in range(INV_CAP)
                   if state.bot_inventories[bid, i] >= 0]
            matching = [t for t in inv if t in active_needs]

            # Priority 1: At DZ with matching items → dropoff
            if pos in self.dz_set and matching:
                actions[bid] = (ACT_DROPOFF, -1)
                continue

            # Priority 2: Has enough matching items → go to DZ
            if len(matching) >= self.deliver_threshold or (
                    matching and len(inv) >= INV_CAP):
                target_dz = self._preferred_dz(bid, pos)
                act = self._move_toward(pos, target_dz, occupied, bid)
                actions[bid] = act
                if act[0] != ACT_WAIT:
                    new_pos = (pos[0] + DX[act[0]], pos[1] + DY[act[0]])
                    occupied.discard(pos)
                    occupied.add(new_pos)
                continue

            # Priority 3: Pick up needed item
            if len(inv) < INV_CAP and active_needs:
                # Find best type to pick up (urgency-weighted)
                best_type = None
                best_score = -1
                need_counts = defaultdict(int)
                for t in active_needs:
                    need_counts[t] += 1

                for tid, count in need_counts.items():
                    info = self._find_best_item(tid, pos)
                    if info:
                        _, _, adj = info
                        d = self._dist(pos, adj)
                        # Score: urgency / distance
                        score = count * self.urgency_weight / max(1, d)
                        if score > best_score:
                            best_score = score
                            best_type = tid

                if best_type is not None:
                    info = self._find_best_item(best_type, pos)
                    if info:
                        item_idx, shelf_pos, adj_pos = info
                        if pos == adj_pos:
                            # Adjacent → pickup
                            actions[bid] = (ACT_PICKUP, item_idx)
                            continue
                        # Move toward item
                        act = self._move_toward(pos, adj_pos, occupied, bid)
                        actions[bid] = act
                        if act[0] != ACT_WAIT:
                            new_pos = (pos[0] + DX[act[0]], pos[1] + DY[act[0]])
                            occupied.discard(pos)
                            occupied.add(new_pos)
                        continue

            # Priority 4: At DZ with no useful items → move away
            if pos in self.dz_set:
                for act, (nx, ny) in self._neighbors.get(pos, []):
                    if (nx, ny) not in occupied or (nx, ny) == self.spawn:
                        actions[bid] = (act, -1)
                        occupied.discard(pos)
                        occupied.add((nx, ny))
                        break
                continue

            # Priority 5: Wait
            pass

        return actions

    def _move_toward(self, pos: tuple, target: tuple, occupied: set, bid: int
                     ) -> tuple[int, int]:
        """Move one step toward target, avoiding occupied cells."""
        if pos == target:
            return (ACT_WAIT, -1)

        # Try optimal direction first
        step = self._first_step(pos, target)
        if step != ACT_WAIT:
            nx, ny = pos[0] + DX[step], pos[1] + DY[step]
            if (nx, ny) not in occupied or (nx, ny) == self.spawn:
                return (step, -1)

        # Try alternatives
        best_act = ACT_WAIT
        best_dist = self._dist(pos, target)
        for act, (nx, ny) in self._neighbors.get(pos, []):
            if (nx, ny) in occupied and (nx, ny) != self.spawn:
                continue
            d = self._dist((nx, ny), target)
            if d < best_dist:
                best_dist = d
                best_act = act

        return (best_act, -1)

    def run_game(self) -> tuple[int, int, list]:
        """Run full game and return (score, orders_completed, actions_log)."""
        gs = GameState(self.ms)
        gs.bot_positions = np.zeros((self.num_bots, 2), dtype=np.int16)
        gs.bot_inventories = np.full((self.num_bots, INV_CAP), -1, dtype=np.int8)
        for i in range(self.num_bots):
            gs.bot_positions[i] = [self.spawn[0], self.spawn[1]]

        orders_copy = [o.copy() for o in self.all_orders]
        gs.orders = [orders_copy[0].copy(), orders_copy[1].copy()]
        gs.orders[0].status = 'active'
        gs.orders[1].status = 'preview'
        gs.next_order_idx = 2
        gs.active_idx = 0

        actions_log = []
        for r in range(self.num_rounds):
            gs.round = r
            acts = self.decide(gs, orders_copy, r)
            actions_log.append(acts)
            cpu_step(gs, acts, orders_copy)

        return gs.score, gs.orders_completed, actions_log


def optimize_params(seed: int, difficulty: str, budget_s: float = 120.0,
                    pop_size: int = 20, verbose: bool = True
                    ) -> tuple[int, np.ndarray, list]:
    """Optimize solver parameters using evolutionary search.

    Uses CMA-ES-like random search to find best parameters.
    Returns (best_score, best_params, best_actions).
    """
    gs, all_orders = init_game(seed, difficulty)
    ms = gs.map_state
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']
    num_rounds = DIFF_ROUNDS.get(difficulty, 300)

    dim = ParameterizedSolver.param_dim(num_bots)

    # Initial parameter distribution
    mean = np.zeros(dim)
    mean[0] = 1.5  # deliver_threshold
    mean[1] = 1.0  # urgency_weight
    for i in range(num_bots):
        mean[2 + i] = i % len(ms.drop_off_zones)  # round-robin DZ

    sigma = np.ones(dim) * 0.5

    best_score = -1
    best_params = mean.copy()
    best_actions = []
    t0 = time.time()

    generation = 0
    while time.time() - t0 < budget_s:
        generation += 1
        # Generate population
        pop = [mean + sigma * np.random.randn(dim) for _ in range(pop_size)]
        pop[0] = mean  # elitism: always include current best

        scores = []
        for params in pop:
            solver = ParameterizedSolver(ms, all_orders, num_bots, num_rounds,
                                         params=params)
            score, orders, actions = solver.run_game()
            scores.append((score, params, actions))

        # Sort by score (descending)
        scores.sort(key=lambda x: -x[0])

        # Update best
        if scores[0][0] > best_score:
            best_score = scores[0][0]
            best_params = scores[0][1].copy()
            best_actions = scores[0][2]

        # Update mean and sigma (simple ES: mean of top half)
        top_half = scores[:pop_size // 2]
        new_mean = np.mean([s[1] for s in top_half], axis=0)

        # Adaptive sigma
        mean = new_mean
        if generation > 3:
            sigma *= 0.95  # shrink over time

        if verbose and generation % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {generation}: best={best_score}, "
                  f"gen_best={scores[0][0]}, elapsed={elapsed:.1f}s",
                  file=sys.stderr)

    if verbose:
        print(f"  Optimized in {time.time()-t0:.1f}s: best_score={best_score}",
              file=sys.stderr)

    return best_score, best_params, best_actions


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Optimization-based solver')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--difficulty', '-d', default='hard')
    parser.add_argument('--budget', type=float, default=60.0,
                        help='Time budget per seed (seconds)')
    parser.add_argument('--pop', type=int, default=20)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    diff = args.difficulty

    scores = []
    for seed in seeds:
        print(f"\n=== Seed {seed}, {diff} ===", file=sys.stderr)

        # First: baseline with default params
        gs, all_orders = init_game(seed, diff)
        ms = gs.map_state
        cfg = CONFIGS[diff]
        num_rounds = DIFF_ROUNDS.get(diff, 300)

        solver = ParameterizedSolver(ms, all_orders, cfg['bots'], num_rounds)
        baseline, _, _ = solver.run_game()
        print(f"  Baseline: {baseline}", file=sys.stderr)

        # Optimize
        best, params, actions = optimize_params(
            seed, diff, budget_s=args.budget, pop_size=args.pop,
            verbose=args.verbose)

        scores.append(best)
        print(f"  Best: {best} (baseline={baseline})", file=sys.stderr)

    if len(scores) > 1:
        print(f"\nSummary: mean={np.mean(scores):.1f}, scores={scores}",
              file=sys.stderr)


if __name__ == '__main__':
    main()
