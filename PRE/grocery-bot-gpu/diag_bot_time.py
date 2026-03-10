#!/usr/bin/env python3
"""Profile bot time usage in nightmare LMAPF."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game, step, ACT_DROPOFF, ACT_PICKUP, ACT_WAIT, INV_CAP
from nightmare_lmapf_solver import LMAPFSolver
from precompute import PrecomputedTables
from configs import DIFF_ROUNDS

NUM_ROUNDS = DIFF_ROUNDS['nightmare']


def profile_seed(seed):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders, solver_seed=0)
    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

    # Per-bot counters
    time_idle = [0] * 20        # at spawn or parked, no items
    time_dead = [0] * 20        # full inventory, nothing useful
    time_fetching = [0] * 20    # moving toward pickup
    time_delivering = [0] * 20  # moving toward dropoff
    time_at_dropoff = [0] * 20  # at dropoff doing delivery
    time_pickup = [0] * 20      # at pickup spot
    time_stalled = [0] * 20     # position unchanged from last round
    prev_pos = {}

    # Track dead inventory progression
    dead_bots_per_50 = []

    for rnd in range(NUM_ROUNDS):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)

        active_order = state.get_active_order()
        preview_order = state.get_preview_order()
        active_types = set()
        preview_types = set()
        if active_order:
            for t in active_order.needs():
                active_types.add(t)
        if preview_order:
            for t in preview_order.needs():
                preview_types.add(t)

        dead_count = 0
        for bid in range(20):
            pos = (int(state.bot_positions[bid, 0]),
                   int(state.bot_positions[bid, 1]))
            inv = state.bot_inv_list(bid)
            act = actions[bid][0]

            # Stall check
            if prev_pos.get(bid) == pos:
                time_stalled[bid] += 1
            prev_pos[bid] = pos

            if act == ACT_DROPOFF:
                time_at_dropoff[bid] += 1
            elif act == ACT_PICKUP:
                time_pickup[bid] += 1
            elif not inv:
                # Empty bot
                if act == ACT_WAIT:
                    time_idle[bid] += 1
                else:
                    time_fetching[bid] += 1
            elif len(inv) >= INV_CAP:
                has_active = any(t in active_types for t in inv)
                has_preview = any(t in preview_types for t in inv)
                if has_active:
                    time_delivering[bid] += 1
                elif has_preview:
                    time_delivering[bid] += 1  # delivering preview
                else:
                    time_dead[bid] += 1
                    dead_count += 1
            else:
                has_active = any(t in active_types for t in inv)
                if has_active:
                    time_delivering[bid] += 1
                else:
                    time_fetching[bid] += 1

        if (rnd + 1) % 50 == 0:
            dead_bots_per_50.append(dead_count)

        step(state, actions, all_orders)

    return (state.score, time_idle, time_dead, time_fetching,
            time_delivering, time_at_dropoff, time_pickup,
            time_stalled, dead_bots_per_50)


seed = 7005
(score, idle, dead, fetch, deliver, dropoff, pickup,
 stalled, dead_prog) = profile_seed(seed)

total_bot_rounds = 20 * NUM_ROUNDS
print(f"Seed {seed}: score={score}")
print(f"Total bot-rounds: {total_bot_rounds}")
print(f"\nTime breakdown (% of total {total_bot_rounds} bot-rounds):")
print(f"  Idle:       {sum(idle):5d} ({100*sum(idle)/total_bot_rounds:.1f}%)")
print(f"  Dead:       {sum(dead):5d} ({100*sum(dead)/total_bot_rounds:.1f}%)")
print(f"  Fetching:   {sum(fetch):5d} ({100*sum(fetch)/total_bot_rounds:.1f}%)")
print(f"  Delivering: {sum(deliver):5d} ({100*sum(deliver)/total_bot_rounds:.1f}%)")
print(f"  At dropoff: {sum(dropoff):5d} ({100*sum(dropoff)/total_bot_rounds:.1f}%)")
print(f"  Picking up: {sum(pickup):5d} ({100*sum(pickup)/total_bot_rounds:.1f}%)")
print(f"  Stalled:    {sum(stalled):5d} ({100*sum(stalled)/total_bot_rounds:.1f}%)")

print(f"\nDead bots over time (per 50-round window):")
for i, d in enumerate(dead_prog):
    print(f"  Round {(i+1)*50:3d}: {d} dead bots")

# Per-bot summary
print(f"\nPer-bot dead time (top 10):")
bot_dead = sorted(range(20), key=lambda b: dead[b], reverse=True)
for bid in bot_dead[:10]:
    print(f"  Bot {bid:2d}: dead={dead[bid]:4d} idle={idle[bid]:4d} "
          f"fetch={fetch[bid]:4d} deliver={deliver[bid]:4d} "
          f"stall={stalled[bid]:4d}")
