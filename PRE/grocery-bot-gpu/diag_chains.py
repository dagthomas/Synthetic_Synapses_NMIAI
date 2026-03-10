#!/usr/bin/env python3
"""Diagnose why chain reactions never fire in nightmare mode."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import (
    init_game, step, ACT_DROPOFF, INV_CAP,
)
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def diagnose_seed(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=0)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

    chains = 0
    near_misses = 0
    completions = 0
    total_chain_potential = 0

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)

        # Snapshot before step
        completed_before = state.orders_completed

        # Get current active and preview
        active_order = state.get_active_order()
        preview_order = None
        for o in state.orders:
            if not o.complete and o.status == 'preview':
                preview_order = o
                break

        # Check active order remaining needs
        active_remaining = 0
        active_needs_set = set()
        if active_order:
            for t in active_order.needs():
                active_remaining += 1
                active_needs_set.add(t)

        # Check who's at dropoffs with what inventory (BEFORE step)
        bots_at_drop_before = []
        for bid in range(20):
            bx = int(state.bot_positions[bid, 0])
            by = int(state.bot_positions[bid, 1])
            if any(bx == dz[0] and by == dz[1] for dz in drop_zones):
                inv = state.bot_inv_list(bid)
                bots_at_drop_before.append((bid, (bx, by), list(inv)))

        # Check who's doing ACT_DROPOFF
        dropoff_bots = []
        for bid in range(20):
            if actions[bid][0] == ACT_DROPOFF:
                bx = int(state.bot_positions[bid, 0])
                by = int(state.bot_positions[bid, 1])
                inv = state.bot_inv_list(bid)
                active_items = [t for t in inv if t in active_needs_set]
                dropoff_bots.append((bid, (bx, by), list(inv), active_items))

        step(state, actions, all_orders)
        completed_after = state.orders_completed

        if completed_after > completed_before:
            new_completions = completed_after - completed_before
            completions += new_completions
            if new_completions > 1:
                chains += new_completions - 1

            # Was there chain potential?
            if preview_order and new_completions == 1:
                preview_types = set()
                for t in preview_order.needs():
                    preview_types.add(t)

                chain_pot = 0
                for bid, pos, inv in bots_at_drop_before:
                    for t in inv:
                        if t in preview_types and t not in active_needs_set:
                            chain_pot += 1

                total_chain_potential += chain_pot
                if chain_pot > 0:
                    near_misses += 1
                    print(f"  R{rnd}: COMPLETE! {len(bots_at_drop_before)} at drops, "
                          f"chain_pot={chain_pot}, active_rem_was={active_remaining}")
                    for bid, pos, inv in bots_at_drop_before:
                        matching = [t for t in inv if t in preview_types and t not in active_needs_set]
                        if matching:
                            print(f"    bot{bid} @ {pos}: inv={inv} preview_match={matching}")
                    for bid, pos, inv, active_items in dropoff_bots:
                        print(f"    DROPOFF bot{bid} @ {pos}: active_items={active_items} all_inv={inv}")

    return completions, chains, near_misses, total_chain_potential, state.score


for seed in [7005, 42, 300]:
    print(f"\nChain diagnostic (seed {seed}):")
    print("=" * 60)
    comps, chains, misses, potential, score = diagnose_seed(seed)
    print(f"\nScore: {score}")
    print(f"Orders completed: {comps}")
    print(f"Chain reactions: {chains}")
    print(f"Near misses: {misses} (total chain potential items: {potential})")
