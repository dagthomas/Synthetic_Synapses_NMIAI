#!/usr/bin/env python
"""Probe nightmare drop zones: send bots to all 3 zones with items to test parallel orders."""
import asyncio, json, sys
from collections import defaultdict

import websockets

URL = sys.argv[1]

# Nightmare drop zones (from game state)
ZONES = [[1, 16], [15, 16], [27, 16]]
SPAWN = [28, 16]

async def run():
    async with websockets.connect(URL) as ws:
        # Round 0: get initial state
        state = json.loads(await ws.recv())
        r = state['round']
        bots = state['bots']
        items = state['items']
        orders = state['orders']
        drop_zones = state.get('drop_off_zones', [state['drop_off']])
        num_bots = len(bots)

        print(f"R{r}: {num_bots} bots, {len(items)} items, {len(orders)} orders")
        print(f"Drop zones: {drop_zones}")
        print(f"Orders: {json.dumps(orders, indent=2)}")

        # Build item position lookup
        item_by_pos = {}
        for it in items:
            pos = tuple(it['position'])
            item_by_pos[pos] = it

        # Strategy:
        # Assign 3 groups of bots, each targeting a different zone
        # Group 0 (bots 0-6): zone 0 [1,16]
        # Group 1 (bots 7-13): zone 1 [15,16]
        # Group 2 (bots 14-19): zone 2 [27,16]

        # We need items from the active order. Let's use simple BFS pathfinding.
        # For the probe, just use hardcoded paths to pick up ANY items and deliver to each zone.

        # Track what each bot is doing
        bot_targets = {}  # bid -> (phase, target_pos, zone_idx)
        bot_carrying = defaultdict(list)

        # Items needed for active order
        active = next((o for o in orders if o['status'] == 'active'), None)
        needed_types = list(active['items_required']) if active else []
        print(f"\nActive order needs: {needed_types}")

        # Find nearest items of each needed type
        type_positions = defaultdict(list)
        for it in items:
            type_positions[it['type']].append(tuple(it['position']))

        # Simple pathfinding: BFS on walkable grid
        width = state['grid']['width']
        height = state['grid']['height']
        walls = set(tuple(w) for w in state['grid']['walls'])

        def bfs_path(start, goal):
            """BFS shortest path, returns list of moves."""
            start = tuple(start)
            goal = tuple(goal)
            if start == goal:
                return []
            from collections import deque
            q = deque([(start, [])])
            visited = {start}
            while q:
                (x, y), path = q.popleft()
                for dx, dy, move in [(0,-1,'move_up'),(0,1,'move_down'),(-1,0,'move_left'),(1,0,'move_right')]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and (nx,ny) not in walls and (nx,ny) not in visited:
                        new_path = path + [move]
                        if (nx,ny) == goal:
                            return new_path
                        visited.add((nx,ny))
                        q.append(((nx,ny), new_path))
            return None  # unreachable

        def adjacent_walkable(item_pos):
            """Find walkable cell adjacent to shelf item."""
            ix, iy = item_pos
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx, ny = ix+dx, iy+dy
                if 0 <= nx < width and 0 <= ny < height and (nx,ny) not in walls:
                    return (nx, ny)
            return None

        # For each zone, find items to pick up (just grab anything from the active order)
        zone_assignments = [[], [], []]  # items assigned to each zone group
        used_items = set()

        for zone_idx in range(3):
            # Each zone group picks up items for the active order
            for t in needed_types:
                for pos in type_positions.get(t, []):
                    if pos not in used_items:
                        zone_assignments[zone_idx].append((t, pos))
                        used_items.add(pos)
                        break

        print(f"\nZone 0 items: {zone_assignments[0]}")
        print(f"Zone 1 items: {zone_assignments[1]}")
        print(f"Zone 2 items: {zone_assignments[2]}")

        # Simple bot plans: pick up 1 item each, deliver to assigned zone
        # Bot groups: 0-6 -> zone 0, 7-13 -> zone 1, 14-19 -> zone 2
        def get_zone(bid):
            if bid < 7: return 0
            if bid < 14: return 1
            return 2

        bot_plans = {}  # bid -> list of (action, item_pos_or_none)

        # Assign 1 bot per needed item per zone
        for zone_idx in range(3):
            zone_bots = [b for b in range(num_bots) if get_zone(b) == zone_idx]
            zone_target = ZONES[zone_idx]

            for i, (itype, ipos) in enumerate(zone_assignments[zone_idx]):
                if i >= len(zone_bots):
                    break
                bid = zone_bots[i]
                adj = adjacent_walkable(ipos)
                if adj is None:
                    continue

                # Plan: go to item adjacent -> pick_up -> go to zone -> drop_off
                bot_pos = tuple(bots[bid]['position'])
                path_to_item = bfs_path(bot_pos, adj) or []
                path_to_zone = bfs_path(adj, tuple(zone_target)) or []

                plan = path_to_item + ['pick_up'] + path_to_zone + ['drop_off']
                bot_plans[bid] = plan
                print(f"Bot {bid} -> zone {zone_idx}: pick {itype} at {ipos}, {len(plan)} steps")

        # Run game loop
        max_rounds = state['max_rounds']
        bot_plan_idx = defaultdict(int)

        for rnd in range(max_rounds):
            if rnd > 0:
                state = json.loads(await ws.recv())

            bots = state['bots']
            orders = state['orders']
            score = state['score']

            # Log order changes
            statuses = [(o['id'], o['status'], len(o.get('items_delivered',[])), len(o['items_required'])) for o in orders]

            if rnd < 10 or rnd % 25 == 0 or score != (state.get('_prev_score', 0)):
                active_count = sum(1 for o in orders if o['status'] == 'active')
                preview_count = sum(1 for o in orders if o['status'] == 'preview')
                print(f"R{rnd:>3} score={score} orders={len(orders)} "
                      f"(active={active_count} preview={preview_count})")
                for o in orders:
                    print(f"      {o['id']}: {o['status']:>7} "
                          f"del={len(o.get('items_delivered',[]))}/{len(o['items_required'])} "
                          f"items={o['items_required']}")

            # Generate actions
            actions = []
            for bid in range(num_bots):
                plan = bot_plans.get(bid, [])
                idx = bot_plan_idx[bid]
                if idx < len(plan):
                    act = plan[idx]
                    bot_plan_idx[bid] += 1
                else:
                    act = 'wait'
                actions.append({"bot": bid, "action": act})

            await ws.send(json.dumps({"actions": actions}))

        # Final state
        final = json.loads(await ws.recv())
        print(f"\n=== GAME OVER ===")
        print(f"Final score: {final.get('score', '?')}")
        print(f"Orders in final state: {len(final.get('orders', []))}")
        for o in final.get('orders', []):
            print(f"  {o['id']}: {o['status']} del={len(o.get('items_delivered',[]))}/{len(o['items_required'])}")

asyncio.run(run())
