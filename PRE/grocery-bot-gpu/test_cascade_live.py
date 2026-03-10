"""Diagnostic live game — test cascade mechanics.

Logs every delivery event to understand:
1. Do multiple DZ bots all deliver in one round?
2. Does cascade fire mid-round (between bot actions)?
3. Can multiple bots stand on the same DZ tile?

Usage:
    python test_cascade_live.py "wss://game.ainm.no/ws?token=..."
"""
import sys
import json
import asyncio
import websockets


async def run_diagnostic(url: str):
    async with websockets.connect(url) as ws:
        # Get initial state
        raw = await ws.recv()
        data = json.loads(raw)

        difficulty = data.get('difficulty', '?')
        max_rounds = data.get('max_rounds', 500)
        bots = data.get('bots', [])
        num_bots = len(bots)
        drop_zones = data.get('drop_off_zones', [])
        dz_set = set(tuple(dz) for dz in drop_zones)
        items_map = {}  # item_id -> type_name
        for item in data.get('items', []):
            items_map[item['id']] = item['type']

        print(f"=== Cascade Diagnostic ===")
        print(f"Difficulty: {difficulty}, Bots: {num_bots}, DZs: {drop_zones}")
        print(f"Max rounds: {max_rounds}")
        print()

        prev_score = 0
        prev_orders_completed = 0
        prev_items_delivered = 0

        for rnd in range(max_rounds):
            # Analyze state BEFORE acting
            score = data.get('score', 0)
            orders = data.get('orders', [])
            active = None
            preview = None
            for o in orders:
                if o.get('status') == 'active':
                    active = o
                elif o.get('status') == 'preview':
                    preview = o

            # Count orders completed
            orders_completed = sum(1 for o in orders if o.get('status') == 'completed')
            items_delivered = data.get('items_delivered', 0)

            # Detect score change from last round
            score_delta = score - prev_score
            orders_delta = orders_completed - prev_orders_completed
            items_delta = items_delivered - prev_items_delivered

            # Log bot positions at DZ
            dz_bots = []
            for bot in data.get('bots', []):
                pos = tuple(bot['position'])
                if pos in dz_set:
                    inv = bot.get('inventory', [])
                    dz_bots.append({
                        'id': bot['id'],
                        'pos': pos,
                        'inv': inv,
                    })

            # Log if something interesting happened
            if score_delta > 0 or orders_delta > 0:
                print(f"R{rnd:3d} DELIVERY: score={score} (+{score_delta}) "
                      f"orders={orders_completed} (+{orders_delta}) "
                      f"items_del={items_delivered} (+{items_delta})")
                if orders_delta > 1:
                    print(f"  >>> CASCADE DETECTED! {orders_delta} orders completed in 1 round!")
                if dz_bots:
                    for db in dz_bots:
                        print(f"  DZ bot {db['id']} at {db['pos']}: {db['inv']}")
                if active:
                    need = len(active.get('items_required', [])) - len(active.get('items_delivered', []))
                    print(f"  Active order: need={need} req={active.get('items_required', [])}")
                print()

            # Log DZ occupancy periodically
            if rnd % 50 == 0:
                active_need = 0
                if active:
                    active_need = len(active.get('items_required', [])) - len(active.get('items_delivered', []))
                dz_count = len(dz_bots)
                print(f"R{rnd:3d} STATUS: score={score} orders={orders_completed} "
                      f"DZ_bots={dz_count} active_need={active_need}")
                # Check for multi-bot DZ stacking
                dz_positions = {}
                for db in dz_bots:
                    p = db['pos']
                    if p not in dz_positions:
                        dz_positions[p] = []
                    dz_positions[p].append(db['id'])
                for p, bids in dz_positions.items():
                    if len(bids) > 1:
                        print(f"  >>> STACKING: {len(bids)} bots at DZ {p}: {bids}")
                print()

            prev_score = score
            prev_orders_completed = orders_completed
            prev_items_delivered = items_delivered

            # Simple action: use CascadeSolver if available, else basic greedy
            try:
                from nightmare_cascade_solver import CascadeSolver
                from game_engine import MapState, build_map_from_capture, Order
                from precompute import PrecomputedTables

                if rnd == 0:
                    # Build map state from first frame
                    ms = build_map_from_capture(data)
                    tables = PrecomputedTables.get(ms)
                    solver = CascadeSolver(ms, tables)

                ws_actions = solver.ws_action(data.get('bots', []), data, ms)
            except Exception as e:
                if rnd == 0:
                    print(f"Solver init failed: {e}, using greedy")
                # Fallback: simple greedy
                ws_actions = []
                for bot in data.get('bots', []):
                    bid = bot['id']
                    pos = tuple(bot['position'])
                    inv = bot.get('inventory', [])

                    # At DZ with items: deliver
                    if pos in dz_set and inv:
                        ws_actions.append({'bot': bid, 'action': 'drop_off'})
                        continue

                    # Simple: wait
                    ws_actions.append({'bot': bid, 'action': 'wait'})

            # Send actions
            msg = json.dumps({'actions': ws_actions})
            await ws.send(msg)

            # Receive next state
            try:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get('type') == 'game_over':
                    final_score = data.get('score', score)
                    print(f"\n=== GAME OVER ===")
                    print(f"Final score: {final_score}")
                    print(f"Orders completed: {orders_completed + (1 if score_delta > 0 else 0)}")
                    break
            except websockets.exceptions.ConnectionClosed:
                print(f"\nConnection closed at round {rnd}")
                break

        print(f"\n=== Summary ===")
        print(f"Final score: {prev_score}")
        print(f"Orders completed: {prev_orders_completed}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_cascade_live.py <ws_url>")
        sys.exit(1)
    asyncio.run(run_diagnostic(sys.argv[1]))
