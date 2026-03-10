#!/usr/bin/env python3
"""Diagnose per-bot efficiency: how much time is productive vs wasted."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import (
    init_game, step, ACT_WAIT, ACT_DROPOFF, ACT_PICKUP,
    ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    INV_CAP, DX, DY,
)
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver

NUM_ROUNDS = DIFF_ROUNDS['nightmare']
MOVES = {ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT}


def run_efficiency(seed, solver_seed=1):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

    solver = LMAPFSolver(ms, tables, future_orders=all_orders,
                         solver_seed=solver_seed, drop_d_weight=0.6)

    # Per-bot stats
    stats = {bid: {
        'moves': 0, 'waits': 0, 'pickups': 0, 'deliveries': 0,
        'stalls': 0, 'items_delivered': 0, 'trips': 0,
        'empty_rounds': 0, 'dead_rounds': 0,
    } for bid in range(20)}

    prev_pos = {}
    active_types_set = set()

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        active = state.get_active_order()
        preview = state.get_preview_order()

        if active:
            active_types_set = set(int(t) for t in active.needs())
        else:
            active_types_set = set()

        actions = solver.action(state, all_orders, rnd)

        for bid, (act, item) in enumerate(actions):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)

            if act in MOVES:
                stats[bid]['moves'] += 1
                # Check if actually moved
                if prev_pos.get(bid) == pos:
                    stats[bid]['stalls'] += 1
            elif act == ACT_WAIT:
                stats[bid]['waits'] += 1
            elif act == ACT_PICKUP:
                stats[bid]['pickups'] += 1
            elif act == ACT_DROPOFF:
                stats[bid]['deliveries'] += 1
                stats[bid]['items_delivered'] += len(inv)
                stats[bid]['trips'] += 1

            # Track empty and dead time
            if not inv:
                stats[bid]['empty_rounds'] += 1
            elif not any(t in active_types_set for t in inv):
                if active:
                    stats[bid]['dead_rounds'] += 1

            prev_pos[bid] = pos

        step(state, actions, all_orders)

    # Summarize
    print(f"\nSeed {seed}: Score={state.score}, Orders={state.orders_completed}")
    print(f"{'Bot':>4} {'Moves':>6} {'Wait':>5} {'Stall':>5} {'Pick':>5} "
          f"{'Deliv':>5} {'Items':>5} {'Trips':>5} {'Empty%':>7} {'Dead%':>6}")
    print("-" * 75)

    total_moves = 0
    total_stalls = 0
    total_waits = 0
    total_items = 0
    total_trips = 0
    total_empty = 0
    total_dead = 0

    for bid in range(20):
        s = stats[bid]
        total = NUM_ROUNDS
        empty_pct = s['empty_rounds'] / total * 100
        dead_pct = s['dead_rounds'] / total * 100
        print(f"{bid:4d} {s['moves']:6d} {s['waits']:5d} {s['stalls']:5d} "
              f"{s['pickups']:5d} {s['deliveries']:5d} {s['items_delivered']:5d} "
              f"{s['trips']:5d} {empty_pct:6.1f}% {dead_pct:5.1f}%")
        total_moves += s['moves']
        total_stalls += s['stalls']
        total_waits += s['waits']
        total_items += s['items_delivered']
        total_trips += s['trips']
        total_empty += s['empty_rounds']
        total_dead += s['dead_rounds']

    print("-" * 75)
    total_bot_rounds = 20 * NUM_ROUNDS
    print(f"Total: moves={total_moves} stalls={total_stalls} waits={total_waits}")
    print(f"  Items delivered={total_items} trips={total_trips}")
    print(f"  Avg items/trip={total_items/max(1,total_trips):.1f}")
    print(f"  Empty time={total_empty/total_bot_rounds*100:.1f}% "
          f"Dead time={total_dead/total_bot_rounds*100:.1f}%")
    print(f"  Move efficiency={total_moves/(total_moves+total_stalls+total_waits)*100:.1f}% "
          f"(moves vs total)")
    print(f"  Stall rate={total_stalls/max(1,total_moves)*100:.1f}% (stalls vs moves)")

    return state.score


if __name__ == '__main__':
    for seed in [7005, 42]:
        run_efficiency(seed)
