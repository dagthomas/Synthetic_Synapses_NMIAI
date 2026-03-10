#!/usr/bin/env python3
"""Analyze order sequence for chain potential."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import init_game

state, all_orders = init_game(7005, 'nightmare', num_orders=100)

print(f"Total orders: {len(all_orders)}")
print(f"Dropoff zones: {[tuple(dz) for dz in state.map_state.drop_off_zones]}")
print()

# Analyze consecutive order pairs for chain potential
print("Order sequence analysis:")
print("-" * 70)
for i in range(min(50, len(all_orders) - 1)):
    o1 = all_orders[i]
    o2 = all_orders[i + 1]
    types1 = set(int(t) for t in o1.required)
    types2 = set(int(t) for t in o2.required)

    # Items needed for each
    needs1 = [int(t) for t in o1.required]
    needs2 = [int(t) for t in o2.required]

    # Preview-only types (in o2 but not o1)
    preview_only = types2 - types1
    shared = types1 & types2

    print(f"Order {i:2d} ({len(needs1)} items, {len(types1)} types) -> "
          f"Order {i+1:2d} ({len(needs2)} items, {len(types2)} types)")
    print(f"  Shared types: {len(shared)}, Preview-only: {len(preview_only)}")
    print(f"  To chain: need {len(needs2)} items for o{i+1} at dropoffs")

    # How many items for o2 can fit in 3 bots (9 slots) alongside o1's items?
    # If 3 delivery bots each carry some o1 items + some o2 items:
    # They deliver o1 items first, then o2 items auto-deliver via chain
    # Items from o1 that match o2 DON'T carry over (consumed by o1)

    # Max auto-delivery: 3 bots * 3 items = 9 items at dropoffs
    # But some are o1 items. Each bot carries mix.
    # After o1 delivery, remaining items = preview-only types

    # Best case: 3 bots with [o1_item, o2_item, o2_item]
    # After delivery: each has [o2_item, o2_item] (if o2 items don't match o1)
    # = 6 preview items. Enough for most orders (4-7 items)

    if len(preview_only) >= len(needs2):
        feasibility = "FULL CHAIN POSSIBLE"
    elif len(preview_only) >= 3:
        feasibility = f"PARTIAL ({len(preview_only)}/{len(needs2)})"
    else:
        feasibility = f"DIFFICULT ({len(preview_only)}/{len(needs2)})"
    print(f"  Chain feasibility: {feasibility}")
    print()

# Deeper analysis: chain depth 2
print("\n" + "=" * 70)
print("Chain depth 2 analysis (order N completes, N+1 and N+2 auto-complete):")
print("-" * 70)
for i in range(min(30, len(all_orders) - 2)):
    o1 = all_orders[i]
    o2 = all_orders[i + 1]
    o3 = all_orders[i + 2]
    needs2 = [int(t) for t in o2.required]
    needs3 = [int(t) for t in o3.required]
    total_chain_items = len(needs2) + len(needs3)

    # Unique types across o2 and o3
    types2 = set(int(t) for t in o2.required)
    types3 = set(int(t) for t in o3.required)
    union_types = types2 | types3

    # Types that are NOT in o1 (safe to carry without being consumed)
    types1 = set(int(t) for t in o1.required)
    safe_types = union_types - types1

    print(f"Orders {i}->{i+1}->{i+2}: need {len(needs2)}+{len(needs3)}={total_chain_items} items, "
          f"{len(safe_types)} safe types (3 bots = 9 slots)")
    if total_chain_items <= 9 and len(safe_types) >= total_chain_items:
        print(f"  ** CHAIN DEPTH 2 FEASIBLE **")
