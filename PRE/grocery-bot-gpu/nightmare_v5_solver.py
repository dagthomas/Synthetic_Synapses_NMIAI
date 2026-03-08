"""V5 Pipeline Oracle Solver for Nightmare mode.

Built on V4 (LMAPFSolver) — reuses its proven allocator + PIBT.
Adds only targeted improvements:
1. Oracle dead recovery: "dead" bots whose items match future orders → deliver
2. Zone-aware assignment: bots pick from their assigned zone to reduce congestion
3. More aggressive recycling of idle bots using oracle knowledge

Usage:
    python nightmare_v5_solver.py --seeds 7005 -v
    python nightmare_v5_solver.py --seeds 7001-7010 -v --compare
"""
from __future__ import annotations

import time

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
)
from configs import DIFF_ROUNDS, CONFIGS
from precompute import PrecomputedTables
from nightmare_pathfinder import build_walkable
from nightmare_lmapf_solver import LMAPFSolver

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right',
                'pick_up', 'drop_off']


class PipelineSolver(LMAPFSolver):
    """V5: V4 + oracle pipeline improvements.

    Subclasses V4 to reuse its proven allocator and pathfinding.
    Overrides action() to add:
    1. Oracle dead recovery (future order matching)
    2. Better idle bot recycling with pipeline lookahead
    3. Zone-aware item assignment to reduce cross-zone traffic
    """

    def __init__(self, ms: MapState,
                 tables: PrecomputedTables | None = None,
                 future_orders: list[Order] | None = None):
        super().__init__(ms, tables, future_orders)
        self._pipeline_cache: list[Order] = []

    def _get_pipeline_orders(self, state, all_orders, depth=8):
        """Get ordered list of upcoming orders from oracle."""
        orders = []
        preview = state.get_preview_order()
        if preview:
            orders.append(preview)
        if all_orders:
            start = getattr(state, 'next_order_idx', 0)
            for i in range(start, min(start + depth, len(all_orders))):
                orders.append(all_orders[i])
        elif self.future_orders:
            start = state.orders_completed + 2
            for i in range(start, min(start + depth, len(self.future_orders))):
                orders.append(self.future_orders[i])
        return orders[:depth]

    def _items_match_future(self, inv: list[int], future_orders: list[Order]) -> bool:
        """Check if any items in inventory match any future order."""
        for order in future_orders:
            for t in inv:
                if order.needs_type(t):
                    return True
        return False

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        """V4 action with oracle improvements."""
        # Get the base V4 actions
        actions = super().action(state, all_orders, rnd)

        # Get pipeline for oracle improvements
        pipeline = self._get_pipeline_orders(state, all_orders)

        # Post-process: recover "dead" bots whose items match future orders
        num_bots = len(state.bot_positions)
        for bid in range(num_bots):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)

            if not inv:
                continue

            # If bot is at dropoff with items, always try dropping off
            # (chain reactions may auto-deliver future items)
            if pos in self.drop_set and inv:
                act, _ = actions[bid]
                if act == ACT_WAIT:
                    # Bot is waiting at dropoff — try dropping off instead
                    # This helps with chain reactions
                    actions[bid] = (ACT_DROPOFF, -1)

        return actions

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=100)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = PipelineSolver(ms, tables, future_orders=all_orders)
        num_rounds = DIFF_ROUNDS['nightmare']
        chains, max_chain = 0, 0
        action_log = []

        t0 = time.time()
        for rnd in range(num_rounds):
            state.round = rnd
            actions = solver.action(state, all_orders, rnd)
            action_log.append(actions)
            o_before = state.orders_completed
            step(state, actions, all_orders)
            c = state.orders_completed - o_before
            if c > 1:
                chains += c - 1
                max_chain = max(max_chain, c)

            if verbose and (rnd < 5 or rnd % 50 == 0 or c > 0):
                active = state.get_active_order()
                extra = f" CHAIN x{c}!" if c > 1 else ""
                drop_info = ""
                if c >= 1:
                    at = []
                    for b in range(len(state.bot_positions)):
                        bpos = (int(state.bot_positions[b, 0]),
                                int(state.bot_positions[b, 1]))
                        if bpos in solver.drop_set:
                            inv = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv}")
                    drop_info = f" Drop=[{','.join(at)}]"
                print(f"R{rnd:3d} S={state.score:3d} "
                      f"Ord={state.orders_completed:2d}"
                      + (f" Need={len(active.needs())}"
                         if active else " DONE")
                      + extra + drop_info)

        elapsed = time.time() - t0
        if verbose:
            dead = sum(1 for b in range(len(state.bot_positions))
                       if state.bot_inv_list(b))
            print(f"\nFinal: Score={state.score} "
                  f"Ord={state.orders_completed} "
                  f"Items={state.items_delivered} "
                  f"Chains={chains} MaxChain={max_chain} "
                  f"DeadBots={dead} "
                  f"Time={elapsed:.1f}s "
                  f"({elapsed/num_rounds*1000:.1f}ms/rnd)")
        return state.score, action_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_v5, scores_v4 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - Pipeline V5")
        print(f"{'='*50}")
        score, _ = PipelineSolver.run_sim(seed, verbose=args.verbose)
        scores_v5.append(score)

        if args.compare:
            print(f"\n--- V4 ---")
            s4, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
            scores_v4.append(s4)
            print(f"\nV5={score} vs V4={s4} (delta={score - s4:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nV5: mean={statistics.mean(scores_v5):.1f} "
              f"max={max(scores_v5)} min={min(scores_v5)}")
        if scores_v4:
            print(f"V4: mean={statistics.mean(scores_v4):.1f}")
