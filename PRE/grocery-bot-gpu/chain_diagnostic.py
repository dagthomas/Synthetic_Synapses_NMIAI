#!/usr/bin/env python3
"""Diagnose why chain reactions aren't firing in the LMAPF solver."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_DROPOFF, ACT_PICKUP, INV_CAP,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver

NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def run_diagnostic(seed, solver_seed=1):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
    drop_set = set(drop_zones)

    solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=0.6)

    chain_opportunities = 0
    chain_successes = 0
    near_misses = []
    total_preview_at_drops = 0
    total_preview_needed = 0
    completions = 0

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        actions = solver.action(state, all_orders, rnd)

        # Before step: check if any bot is delivering at a dropoff
        delivering_bots = []
        for bid, (act, _) in enumerate(actions):
            if act == ACT_DROPOFF:
                bx, by = int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])
                if (bx, by) in drop_set:
                    delivering_bots.append(bid)

        # Check state before step
        if delivering_bots and active and preview:
            # Would this delivery complete the active order?
            active_remaining = list(active.needs())
            # What could the delivering bots contribute?
            can_deliver = {}
            for bid in delivering_bots:
                inv = state.bot_inv_list(bid)
                for t in inv:
                    if t in [int(r) for r in active_remaining]:
                        can_deliver[t] = can_deliver.get(t, 0) + 1

            # Simulate: would active complete?
            remaining_after = list(active_remaining)
            for bid in delivering_bots:
                inv = state.bot_inv_list(bid)
                for t in inv:
                    if t in remaining_after:
                        remaining_after.remove(t)

            would_complete = len(remaining_after) == 0

            if would_complete:
                chain_opportunities += 1

                # Check preview items at dropoffs
                preview_needs = list(preview.needs())
                preview_have = {}  # items at dropoffs from ALL bots

                for bid2 in range(len(state.bot_positions)):
                    bx2, by2 = int(state.bot_positions[bid2, 0]), int(state.bot_positions[bid2, 1])
                    if (bx2, by2) not in drop_set:
                        continue
                    inv2 = state.bot_inv_list(bid2)
                    # After delivery, which items remain?
                    remaining_inv = list(inv2)
                    if bid2 in delivering_bots:
                        # Remove items matching active order
                        for t in list(remaining_inv):
                            if active.needs_type(t):
                                remaining_inv.remove(t)
                    # Check remaining items against preview
                    for t in remaining_inv:
                        if t in [int(r) for r in preview_needs]:
                            preview_have[t] = preview_have.get(t, 0) + 1

                # Count preview matches
                preview_matched = 0
                preview_need_count = {}
                for t in preview_needs:
                    t = int(t)
                    preview_need_count[t] = preview_need_count.get(t, 0) + 1

                for t, need in preview_need_count.items():
                    have = preview_have.get(t, 0)
                    preview_matched += min(have, need)

                total_preview_at_drops += preview_matched
                total_preview_needed += len(preview_needs)

                if preview_matched >= len(preview_needs):
                    chain_successes += 1

                # Near miss: any preview items at all
                if preview_matched > 0 and preview_matched < len(preview_needs):
                    near_misses.append({
                        'rnd': rnd,
                        'matched': preview_matched,
                        'needed': len(preview_needs),
                        'pct': preview_matched / len(preview_needs) * 100,
                    })

                if rnd < 100 or preview_matched > 0:
                    # Count bots at each dropoff
                    bots_at_drops = {dz: [] for dz in drop_zones}
                    for b in range(len(state.bot_positions)):
                        pos = (int(state.bot_positions[b, 0]),
                               int(state.bot_positions[b, 1]))
                        if pos in drop_set:
                            bots_at_drops[pos].append(b)

                    if preview_matched > 0 or rnd < 30:
                        print(f"  R{rnd}: CHAIN OPP! active_rem=0 preview={preview_matched}/{len(preview_needs)}"
                              f" bots@drops={sum(len(v) for v in bots_at_drops.values())}")

        completed_before = state.orders_completed
        step(state, actions, all_orders)
        if state.orders_completed > completed_before:
            completions += 1
            if state.orders_completed > completed_before + 1:
                print(f"  R{rnd}: *** CHAIN FIRED! depth={state.orders_completed - completed_before - 1} ***")

    print(f"\n=== Seed {seed} Results ===")
    print(f"  Score: {state.score}, Orders: {state.orders_completed}")
    print(f"  Chain opportunities (active would complete): {chain_opportunities}")
    print(f"  Chain successes (preview fully matched): {chain_successes}")
    print(f"  Preview items at drops: {total_preview_at_drops}/{total_preview_needed}"
          f" ({total_preview_at_drops/max(1,total_preview_needed)*100:.1f}%)")
    print(f"  Near misses (some preview matched): {len(near_misses)}")
    for nm in near_misses[:10]:
        print(f"    R{nm['rnd']}: {nm['matched']}/{nm['needed']} ({nm['pct']:.0f}%)")

    return state.score


if __name__ == '__main__':
    seeds = [7005, 11, 42, 45]
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}:")
        print(f"{'='*60}")
        run_diagnostic(seed, solver_seed=1)
