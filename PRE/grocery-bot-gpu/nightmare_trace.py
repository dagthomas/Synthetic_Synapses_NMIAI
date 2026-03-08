"""Trace a single order: what does each bot do, round by round."""
import numpy as np
from game_engine import (
    init_game, step, ACT_WAIT, ACT_PICKUP, ACT_DROPOFF,
    INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_solver_v6 import NightmareSolverV6, V6Allocator


def trace_order(seed, target_order=3):
    """Trace the Nth order in detail."""
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = NightmareSolverV6(ms, tables)
    drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

    num_rounds = DIFF_ROUNDS['nightmare']
    num_bots = len(state.bot_positions)

    order_count = 0
    last_active_id = -1
    tracing = False
    trace_start = 0

    for rnd in range(num_rounds):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        active_id = active.id if active else -1
        if active_id != last_active_id and last_active_id != -1:
            order_count += 1
            if tracing:
                print(f"\n  ORDER COMPLETED at round {rnd} ({rnd - trace_start} rounds)")
                break
            if order_count == target_order:
                tracing = True
                trace_start = rnd
                print(f"\n=== TRACING ORDER {order_count} (start round {rnd}) ===")
                if active:
                    needs = {}
                    for t in active.needs():
                        needs[t] = needs.get(t, 0) + 1
                    print(f"  Active needs: {needs}")
                if preview:
                    pneeds = {}
                    for t in preview.needs():
                        pneeds[t] = pneeds.get(t, 0) + 1
                    print(f"  Preview needs: {pneeds}")
        last_active_id = active_id

        if tracing and (rnd - trace_start) % 5 == 0:
            active_needs = {}
            if active:
                for t in active.needs():
                    active_needs[t] = active_needs.get(t, 0) + 1
            preview_needs = {}
            if preview:
                for t in preview.needs():
                    preview_needs[t] = preview_needs.get(t, 0) + 1

            print(f"\n  Round {rnd} (T+{rnd - trace_start}):")
            print(f"    Active remaining: {active_needs}")

            # Bot status
            active_c = preview_c = dead_c = empty_c = 0
            for bid in range(num_bots):
                inv = state.bot_inv_list(bid)
                pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
                if not inv:
                    empty_c += 1
                elif any(t in active_needs for t in inv):
                    active_c += 1
                elif any(t in preview_needs for t in inv):
                    preview_c += 1
                elif len(inv) < INV_CAP:
                    empty_c += 1
                else:
                    dead_c += 1
            print(f"    Bots: active={active_c} preview={preview_c} dead={dead_c} empty={empty_c}")

            # Show which bots are at dropoffs
            at_drop = []
            for bid in range(num_bots):
                pos = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
                if pos in drop_set:
                    inv = state.bot_inv_list(bid)
                    at_drop.append((bid, pos, inv))
            if at_drop:
                print(f"    At dropoff: {[(b, p, i) for b, p, i in at_drop]}")

        actions = solver.action(state, all_orders, rnd)
        step(state, actions, all_orders)

    print(f"\nFinal score: {state.score}, orders: {state.orders_completed}")


if __name__ == '__main__':
    # Trace order 5 of seed 1000
    trace_order(1000, target_order=5)
