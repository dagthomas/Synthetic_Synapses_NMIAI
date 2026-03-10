#!/usr/bin/env python3
"""Force a chain reaction to prove the concept works."""
import sys, copy, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

from game_engine import (
    init_game, step, GameState, Order,
    ACT_WAIT, ACT_DROPOFF, ACT_PICKUP, ACT_MOVE_UP, ACT_MOVE_DOWN,
    ACT_MOVE_LEFT, ACT_MOVE_RIGHT, INV_CAP, DX, DY, CELL_WALL,
)

# Create a game
state, all_orders = init_game(7005, 'nightmare', num_orders=100)
ms = state.map_state
drop_zones = [tuple(dz) for dz in ms.drop_off_zones]

print(f"Drop zones: {drop_zones}")
print(f"Spawn: {ms.spawn}")

# Print first 3 orders
for i, o in enumerate(all_orders[:3]):
    types = [int(t) for t in o.required]
    print(f"Order {i}: types={types} ({len(types)} items)")

# Get active and preview orders
active = state.get_active_order()
preview = state.get_preview_order()
print(f"\nActive: types={[int(t) for t in active.required]}")
print(f"Preview: types={[int(t) for t in preview.required]}")

# Manually set up a chain scenario:
# 1. Fill active order to need just 1 more item
# 2. Put a bot at a dropoff with the last active item + preview items
# 3. Put another bot at a different dropoff with preview items
# 4. Have the first bot do ACT_DROPOFF → completes active → chain fires

# First, deliver most of the active items manually
active_needs = list(active.needs())
print(f"\nActive needs: {active_needs}")

# Deliver all but 1 item
for i in range(len(active_needs) - 1):
    active.deliver_type(active_needs[i])

remaining = active.needs()
print(f"After manual delivery, active still needs: {remaining}")
print(f"Active complete? {active.is_complete()}")

# Now set up bots:
# Bot 0: at dropoff 0 with [last_active_item, preview_item_1, preview_item_2]
# Bot 1: at dropoff 1 with [preview_item_3, preview_item_4, preview_item_5]

last_active_type = remaining[0]
preview_types = [int(t) for t in preview.required]
print(f"Preview needs: {preview_types}")

# Place bot 0 at dropoff 0
state.bot_positions[0] = [drop_zones[0][0], drop_zones[0][1]]
# Set inventory: last active type + 2 preview types
state.bot_inventories[0] = [-1, -1, -1]  # clear
state.bot_inv_add(0, last_active_type)
if len(preview_types) >= 1:
    state.bot_inv_add(0, preview_types[0])
if len(preview_types) >= 2:
    state.bot_inv_add(0, preview_types[1])

# Place bot 1 at dropoff 1
state.bot_positions[1] = [drop_zones[1][0], drop_zones[1][1]]
state.bot_inventories[1] = [-1, -1, -1]
if len(preview_types) >= 3:
    state.bot_inv_add(1, preview_types[2])
if len(preview_types) >= 4:
    state.bot_inv_add(1, preview_types[3])
if len(preview_types) >= 5:
    state.bot_inv_add(1, preview_types[4])

# Place bot 2 at dropoff 2 with remaining preview types
state.bot_positions[2] = [drop_zones[2][0], drop_zones[2][1]]
state.bot_inventories[2] = [-1, -1, -1]
for i in range(5, min(8, len(preview_types))):
    state.bot_inv_add(2, preview_types[i])

# Move all other bots to spawn
for bid in range(3, 20):
    state.bot_positions[bid] = [ms.spawn[0], ms.spawn[1]]

# Print state
print(f"\nBot 0 @ {tuple(state.bot_positions[0])}: inv={state.bot_inv_list(0)}")
print(f"Bot 1 @ {tuple(state.bot_positions[1])}: inv={state.bot_inv_list(1)}")
print(f"Bot 2 @ {tuple(state.bot_positions[2])}: inv={state.bot_inv_list(2)}")

# Now: bot 0 does ACT_DROPOFF, others wait
actions = [(ACT_WAIT, -1)] * 20
actions[0] = (ACT_DROPOFF, -1)

score_before = state.score
completed_before = state.orders_completed
print(f"\nBefore step: score={score_before}, completed={completed_before}")

step(state, actions, all_orders)

print(f"After step: score={state.score}, completed={state.orders_completed}")
print(f"Score delta: +{state.score - score_before}")
print(f"Orders completed delta: +{state.orders_completed - completed_before}")

if state.orders_completed > completed_before + 1:
    print(f"\n*** CHAIN REACTION FIRED! depth={state.orders_completed - completed_before - 1} ***")
elif state.orders_completed > completed_before:
    print(f"\n(Active order completed, but no chain)")
else:
    print(f"\n(No order completion)")

# Check remaining inventories
print(f"\nBot 0 inv after: {state.bot_inv_list(0)}")
print(f"Bot 1 inv after: {state.bot_inv_list(1)}")
print(f"Bot 2 inv after: {state.bot_inv_list(2)}")
