"""Chain Reactor: V4 + more preview pickers only.

Tests whether more preview pickers (10 instead of 4) enables chain reactions
by getting preview items to dropoffs before the trigger fires.

Key insight: V4's preview pickers start after active is covered (same round),
but only 4 of them. With 10, more preview items reach dropoffs in time.

NO other changes from V4: no staging ACT_DROPOFF, no trigger hold.
"""
from __future__ import annotations

import time

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver

MOVES = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]


class ChainReactorSolver(LMAPFSolver):
    """V4 with max_preview_pickers=10 (from 4)."""

    def action(self, state: GameState, all_orders: list[Order],
               rnd: int) -> list[tuple[int, int]]:
        ms = self.ms
        num_bots = len(state.bot_positions)
        num_rounds = DIFF_ROUNDS.get('nightmare', 500)

        # === IDENTICAL TO V4 ===
        bot_positions: dict[int, tuple[int, int]] = {}
        bot_inventories: dict[int, list[int]] = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        self.congestion.update(list(bot_positions.values()))
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if self.prev_positions.get(bid) == pos:
                self.stall_counts[bid] = self.stall_counts.get(bid, 0) + 1
            else:
                self.stall_counts[bid] = 0
            self.prev_positions[bid] = pos

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()
        future = self._get_future_orders(state, all_orders)

        active_needs: dict[int, int] = {}
        carrying_active: dict[int, int] = {}
        if active_order:
            for t in active_order.needs():
                active_needs[t] = active_needs.get(t, 0) + 1
            for inv in bot_inventories.values():
                for t in inv:
                    if t in active_needs:
                        carrying_active[t] = carrying_active.get(t, 0) + 1
        active_short: dict[int, int] = {}
        for t, need in active_needs.items():
            s = need - carrying_active.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())

        chain_plan = self.chain_planner.plan_chain(
            active_order, future, bot_positions, bot_inventories)

        active_id = active_order.id if active_order else -1
        if active_id != self._last_active_id:
            self._trips.clear()
            self._last_active_id = active_id

        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            trip['age'] += 1
            if trip['age'] > 20:
                del self._trips[bid]

        # ===== ONLY CHANGE: max_preview_pickers_override=10 =====
        goals, goal_types, pickup_targets = self.allocator.allocate(
            bot_positions, bot_inventories,
            active_order, preview_order, rnd, num_rounds,
            future_orders=future, chain_plan=chain_plan,
            allow_preview_pickup=True,
            max_preview_pickers_override=10)

        # Apply persistent trips (same as V4)
        for bid in list(self._trips.keys()):
            trip = self._trips[bid]
            pos = bot_positions[bid]
            if pos == trip['goal'] or len(bot_inventories[bid]) > trip.get('inv_count', 0):
                del self._trips[bid]
                continue
            tid = trip.get('type_id', -1)
            if trip['goal_type'] == 'pickup':
                if not (active_order and active_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            elif trip['goal_type'] == 'preview':
                if not (preview_order and preview_order.needs_type(tid)):
                    del self._trips[bid]
                    continue
            goals[bid] = trip['goal']
            goal_types[bid] = trip['goal_type']
            if trip.get('item_idx') is not None:
                pickup_targets[bid] = trip['item_idx']

        # Record new trips (same as V4)
        for bid in range(num_bots):
            gt = goal_types.get(bid)
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                if bid not in self._trips:
                    item_idx = pickup_targets[bid]
                    tid = int(self.ms.item_types[item_idx]) if item_idx >= 0 else -1
                    self._trips[bid] = {
                        'goal': goals[bid],
                        'goal_type': gt,
                        'item_idx': item_idx,
                        'type_id': tid,
                        'inv_count': len(bot_inventories[bid]),
                        'age': 0,
                    }

        # POST-PROCESS: (same as V4)
        claimed_items = set(pickup_targets.values())
        for bid in range(num_bots):
            gt = goal_types.get(bid, 'park')
            if gt not in ('flee', 'park'):
                continue
            pos = bot_positions[bid]
            inv = bot_inventories[bid]
            free = INV_CAP - len(inv)
            if free <= 0:
                continue

            if active_short:
                idx, adj = self._find_best_item(pos, active_short, claimed_items)
                if idx is not None:
                    goals[bid] = adj
                    goal_types[bid] = 'pickup'
                    pickup_targets[bid] = idx
                    claimed_items.add(idx)
                    self._trips[bid] = {
                        'goal': adj, 'goal_type': 'pickup',
                        'item_idx': idx,
                        'type_id': int(ms.item_types[idx]),
                        'inv_count': len(inv), 'age': 0,
                    }
                    continue

            if preview_order:
                preview_n: dict[int, int] = {}
                for t in preview_order.needs():
                    preview_n[t] = preview_n.get(t, 0) + 1
                for t in inv:
                    if t in preview_n:
                        preview_n[t] -= 1
                        if preview_n[t] <= 0:
                            del preview_n[t]
                if preview_n:
                    idx, adj = self._find_best_item(pos, preview_n, claimed_items)
                    if idx is not None:
                        goals[bid] = adj
                        goal_types[bid] = 'preview'
                        pickup_targets[bid] = idx
                        claimed_items.add(idx)
                        self._trips[bid] = {
                            'goal': adj, 'goal_type': 'preview',
                            'item_idx': idx,
                            'type_id': int(ms.item_types[idx]),
                            'inv_count': len(inv), 'age': 0,
                        }

        # Urgency order (same as V4)
        def _urgency_key(bid):
            gt = goal_types.get(bid, 'park')
            dist = self.tables.get_distance(
                bot_positions[bid], goals.get(bid, self.spawn))
            if gt == 'deliver':
                return (0, dist)
            elif gt == 'flee':
                drop_dist = min(self.tables.get_distance(bot_positions[bid], dz)
                                for dz in self.drop_zones)
                return (1 if drop_dist < 5 else 4, dist)
            elif gt == 'pickup':
                return (2, dist)
            elif gt in ('stage', 'preview'):
                return (3, dist)
            else:
                return (5, dist)
        urgency_order = sorted(range(num_bots), key=_urgency_key)

        # PIBT pathfinding (same as V4)
        path_actions = self.pathfinder.plan_all(
            bot_positions, goals, urgency_order,
            goal_types=goal_types)

        # Build actions (same as V4 — NO staging ACT_DROPOFF)
        actions: list[tuple[int, int]] = [(ACT_WAIT, -1)] * num_bots

        for bid in range(num_bots):
            pos = bot_positions[bid]
            gt = goal_types.get(bid, 'park')
            goal = goals.get(bid, self.spawn)

            if self.stall_counts.get(bid, 0) >= 3:
                act = self._escape_action(bid, pos, rnd)
                actions[bid] = (act, -1)
                continue

            # AT DROPOFF: deliver only for 'deliver' bots
            if pos in self.drop_set:
                if gt == 'deliver' and bot_inventories[bid]:
                    actions[bid] = (ACT_DROPOFF, -1)
                    continue

            # AT PICKUP TARGET
            if gt in ('pickup', 'preview') and bid in pickup_targets:
                item_idx = pickup_targets[bid]
                if pos == goal:
                    actions[bid] = (ACT_PICKUP, item_idx)
                    continue

            # Opportunistic adjacent pickup
            if gt in ('pickup', 'preview', 'deliver') and len(bot_inventories[bid]) < INV_CAP:
                pickup_act = self._check_adjacent_pickup(
                    bid, pos, active_order, preview_order, gt,
                    bot_inventories[bid], active_short, chain_plan)
                if pickup_act is not None:
                    actions[bid] = pickup_act
                    continue

            # PIBT action
            act = path_actions.get(bid, ACT_WAIT)
            actions[bid] = (act, -1)

        return actions

    @staticmethod
    def run_sim(seed: int, verbose: bool = False) -> tuple[int, list]:
        state, all_orders = init_game(seed, 'nightmare', num_orders=200)
        ms = state.map_state
        tables = PrecomputedTables.get(ms)
        solver = ChainReactorSolver(ms, tables, future_orders=all_orders)
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
                solver.chain_events.append((rnd, c))

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
                            inv_b = state.bot_inv_list(b)
                            at.append(f"b{b}:{inv_b}")
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
    parser = argparse.ArgumentParser(description='Chain Reactor')
    parser.add_argument('--seeds', default='7005')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    scores_cr, scores_v4 = [], []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} - Chain Reactor (pp=10)")
        print(f"{'='*50}")
        score, _ = ChainReactorSolver.run_sim(seed, verbose=args.verbose)
        scores_cr.append(score)

        if args.compare:
            print(f"\n--- V4 ---")
            s4, _ = LMAPFSolver.run_sim(seed, verbose=args.verbose)
            scores_v4.append(s4)
            print(f"\nCR={score} vs V4={s4} (delta={score - s4:+d})")

    if len(seeds) > 1:
        import statistics
        print(f"\nCR: mean={statistics.mean(scores_cr):.1f} "
              f"max={max(scores_cr)} min={min(scores_cr)}")
        if scores_v4:
            print(f"V4: mean={statistics.mean(scores_v4):.1f}")
