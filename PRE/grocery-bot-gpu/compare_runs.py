#!/usr/bin/env python3
"""Compare two production runs to check if orders/items are identical.

Fetches two separate tokens (different sessions), connects to each,
captures the first few rounds of game state, then compares:
- Item placement on shelves
- Order sequences (active + preview)
- Map layout

Usage:
    python compare_runs.py hard
    python compare_runs.py hard --rounds 10
    python compare_runs.py hard --url1 "wss://..." --url2 "wss://..."
"""
import argparse
import asyncio
import json
import sys
import os
import time

import websockets

ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']


async def capture_game_data(ws_url, max_rounds=5, label="run"):
    """Connect to a game and capture the first max_rounds of data."""
    rounds_data = []
    print(f"  [{label}] Connecting to {ws_url[:80]}...", file=sys.stderr)

    async with websockets.connect(ws_url) as ws:
        round_num = 0
        async for msg in ws:
            data = json.loads(msg)

            if data.get('type') == 'game_over':
                print(f"  [{label}] Game over at round {round_num}", file=sys.stderr)
                break

            rounds_data.append(data)
            round_num = data.get('round', round_num)

            # Send wait actions for all bots
            bots = data.get('bots', [])
            actions = [{"bot_id": b["id"], "action": "wait"} for b in bots]
            await ws.send(json.dumps(actions))

            if round_num >= max_rounds:
                print(f"  [{label}] Captured {max_rounds} rounds, disconnecting",
                      file=sys.stderr)
                break

    return rounds_data


def extract_fingerprint(rounds_data):
    """Extract comparable data from captured rounds."""
    if not rounds_data:
        return None

    r0 = rounds_data[0]

    # Items on shelves (should be static)
    items = []
    for item in r0.get('items', []):
        items.append({
            'id': item.get('id'),
            'type': item.get('type'),
            'position': tuple(item.get('position', [])),
        })
    items.sort(key=lambda x: x['id'])

    # Orders seen across all rounds
    orders_by_id = {}
    for rd in rounds_data:
        for order in rd.get('orders', []):
            oid = order.get('id')
            if oid not in orders_by_id:
                orders_by_id[oid] = {
                    'id': oid,
                    'items_required': order.get('items_required', []),
                    'status': order.get('status'),
                    'first_seen_round': rd.get('round', 0),
                }

    # Map layout
    map_cells = r0.get('map', {}).get('cells', [])

    # Bot spawn positions
    bots = [(b['id'], tuple(b['position'])) for b in r0.get('bots', [])]
    bots.sort()

    return {
        'items': items,
        'orders': orders_by_id,
        'map_cells': map_cells,
        'bots': bots,
        'num_bots': len(bots),
    }


def compare_fingerprints(fp1, fp2):
    """Compare two game fingerprints and report differences."""
    if fp1 is None or fp2 is None:
        print("ERROR: One or both fingerprints are None")
        return False

    all_match = True

    # Compare map
    if fp1['map_cells'] == fp2['map_cells']:
        print("  MAP: IDENTICAL")
    else:
        print("  MAP: DIFFERENT!")
        all_match = False

    # Compare items
    if fp1['items'] == fp2['items']:
        print(f"  ITEMS: IDENTICAL ({len(fp1['items'])} items)")
    else:
        print(f"  ITEMS: DIFFERENT!")
        all_match = False
        # Show details
        items1 = {i['id']: i for i in fp1['items']}
        items2 = {i['id']: i for i in fp2['items']}
        for iid in sorted(set(items1.keys()) | set(items2.keys())):
            i1 = items1.get(iid)
            i2 = items2.get(iid)
            if i1 != i2:
                print(f"    Item {iid}: run1={i1} vs run2={i2}")

    # Compare bots
    if fp1['bots'] == fp2['bots']:
        print(f"  BOTS: IDENTICAL ({fp1['num_bots']} bots)")
    else:
        print(f"  BOTS: DIFFERENT!")
        all_match = False

    # Compare orders
    orders1 = fp1['orders']
    orders2 = fp2['orders']
    common_ids = sorted(set(orders1.keys()) & set(orders2.keys()))
    only1 = sorted(set(orders1.keys()) - set(orders2.keys()))
    only2 = sorted(set(orders2.keys()) - set(orders1.keys()))

    orders_match = True
    for oid in common_ids:
        o1 = orders1[oid]
        o2 = orders2[oid]
        if o1['items_required'] != o2['items_required']:
            print(f"  ORDER {oid}: DIFFERENT items!")
            print(f"    run1: {o1['items_required']}")
            print(f"    run2: {o2['items_required']}")
            orders_match = False
            all_match = False

    if orders_match and not only1 and not only2:
        print(f"  ORDERS: IDENTICAL ({len(common_ids)} orders compared)")
    elif orders_match:
        print(f"  ORDERS: Items match for {len(common_ids)} common orders")
        if only1:
            print(f"    Only in run1: {only1}")
        if only2:
            print(f"    Only in run2: {only2}")

    # Also compare by sequence (not just by ID)
    # Orders might have different IDs but same content
    seq1 = [orders1[oid]['items_required'] for oid in sorted(orders1.keys())]
    seq2 = [orders2[oid]['items_required'] for oid in sorted(orders2.keys())]
    min_len = min(len(seq1), len(seq2))
    if min_len > 0:
        seq_match = all(seq1[i] == seq2[i] for i in range(min_len))
        if seq_match:
            print(f"  ORDER SEQUENCE: IDENTICAL (first {min_len} orders)")
        else:
            print(f"  ORDER SEQUENCE: DIFFERENT!")
            for i in range(min_len):
                if seq1[i] != seq2[i]:
                    print(f"    Order #{i}: run1={seq1[i]} vs run2={seq2[i]}")

    return all_match


async def main():
    parser = argparse.ArgumentParser(
        description='Compare two production runs for determinism')
    parser.add_argument('difficulty', nargs='?', default='hard',
                        choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--rounds', type=int, default=5,
                        help='Rounds to capture per run (default: 5)')
    parser.add_argument('--url1', help='WS URL for run 1 (skip token fetch)')
    parser.add_argument('--url2', help='WS URL for run 2 (skip token fetch)')
    parser.add_argument('--delay', type=int, default=15,
                        help='Seconds between runs (default: 15, for 10s cooldown)')
    args = parser.parse_args()

    if args.url1 and args.url2:
        url1, url2 = args.url1, args.url2
    else:
        from fetch_token import fetch_token
        import concurrent.futures
        # Run sync Playwright in a thread to avoid asyncio conflict
        print(f"Fetching token 1 for {args.difficulty}...", file=sys.stderr)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            url1 = await asyncio.get_event_loop().run_in_executor(
                pool, lambda: fetch_token(args.difficulty))
        if not url1:
            print("ERROR: Could not fetch token 1", file=sys.stderr)
            sys.exit(1)
        print(f"  Token 1: {url1[:60]}...", file=sys.stderr)

    # Run 1
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RUN 1: Capturing {args.rounds} rounds", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    data1 = await capture_game_data(url1, max_rounds=args.rounds, label="run1")
    fp1 = extract_fingerprint(data1)

    if not args.url2:
        # Wait for cooldown then fetch second token
        print(f"\nWaiting {args.delay}s for cooldown...", file=sys.stderr)
        await asyncio.sleep(args.delay)

        print(f"Fetching token 2 for {args.difficulty}...", file=sys.stderr)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            url2 = await asyncio.get_event_loop().run_in_executor(
                pool, lambda: fetch_token(args.difficulty))
        if not url2:
            print("ERROR: Could not fetch token 2", file=sys.stderr)
            sys.exit(1)
        print(f"  Token 2: {url2[:60]}...", file=sys.stderr)

    # Run 2
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RUN 2: Capturing {args.rounds} rounds", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    data2 = await capture_game_data(url2, max_rounds=args.rounds, label="run2")
    fp2 = extract_fingerprint(data2)

    # Compare
    print(f"\n{'='*60}")
    print(f"COMPARISON: {args.difficulty}")
    print(f"{'='*60}")
    all_match = compare_fingerprints(fp1, fp2)

    if all_match:
        print(f"\nRESULT: Games are IDENTICAL across sessions")
        print(f"  -> Orders are deterministic (same seed today)")
        print(f"  -> Token only gates access, doesn't change the game")
    else:
        print(f"\nRESULT: Games DIFFER across sessions")
        print(f"  -> Each token/session may get a different seed")

    # Dump raw data for manual inspection
    print(f"\n--- Run 1 Round 0 Orders ---")
    if data1:
        for o in data1[0].get('orders', []):
            print(f"  {o.get('id')}: {o.get('items_required')} [{o.get('status')}]")

    print(f"\n--- Run 2 Round 0 Orders ---")
    if data2:
        for o in data2[0].get('orders', []):
            print(f"  {o.get('id')}: {o.get('items_required')} [{o.get('status')}]")


if __name__ == '__main__':
    asyncio.run(main())
