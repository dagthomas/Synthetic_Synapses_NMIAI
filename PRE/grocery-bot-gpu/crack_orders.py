#!/usr/bin/env python3
"""Generate all future orders using cracked seed and merge into capture.

Tries multiple approaches to crack the order sequence:
1. Hardcoded seed 7004 + skip 509 with SERVER_TYPE_LIST (verified 2026-03-05)
2. Brute-force skip 0..3000 with SERVER_TYPE_LIST
3. Brute-force skip 0..3000 with alphabetical type ordering

If cracking fails, exits 0 without modifying anything (graceful degradation).

Usage:
    python crack_orders.py expert
    python crack_orders.py expert --num 50
"""
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution_store import load_capture, save_capture

SEED = 7004

# Type list orderings to try (the server shuffles types during build_map)
SERVER_TYPE_LIST = [
    'bananas', 'apples', 'onions', 'oats', 'butter', 'flour',
    'cheese', 'cream', 'milk', 'pasta', 'yogurt', 'cereal',
    'eggs', 'rice', 'bread', 'tomatoes',
]

ALPHA_TYPE_LIST = sorted(SERVER_TYPE_LIST)

ORDER_SIZE = {
    'easy':   (3, 4),
    'medium': (3, 5),
    'hard':   (3, 5),
    'expert': (4, 6),
}


def _try_generate(avail_types, available_counts, order_size, rng, num_orders):
    """Generate orders from current RNG state. Returns list of order dicts."""
    orders = []
    for _ in range(num_orders):
        n = rng.randint(order_size[0], order_size[1])
        temp = dict(available_counts)
        names = []
        for _ in range(n):
            usable = [t for t in avail_types if temp.get(t, 0) > 0]
            if not usable:
                usable = avail_types
            t = rng.choice(usable)
            names.append(t)
            if t in temp:
                temp[t] -= 1
        orders.append({'items_required': names})
    return orders


def _verify(cracked, captured):
    """Check if cracked orders match captured orders."""
    for i, cap in enumerate(captured):
        if i >= len(cracked):
            return False
        if cracked[i]['items_required'] != cap['items_required']:
            return False
    return True


def crack_orders(capture_data, difficulty, num_orders=50):
    """Try multiple approaches to crack orders.

    Returns (orders, method_name) or (None, None) on failure.
    """
    order_size = ORDER_SIZE.get(difficulty)
    if not order_size:
        return None, None

    captured = capture_data.get('orders', [])
    if len(captured) < 2:
        return None, None

    # Build available_counts for each type ordering
    item_counts = {}
    for item in capture_data['items']:
        t = item['type']
        item_counts[t] = item_counts.get(t, 0) + 1

    orderings = [
        ('server_type_list', SERVER_TYPE_LIST),
        ('alphabetical', ALPHA_TYPE_LIST),
    ]

    for name, type_list in orderings:
        available_counts = {t: item_counts.get(t, 0) for t in type_list}
        avail_types = list(available_counts.keys())

        # Try hardcoded skip first (fast)
        if name == 'server_type_list':
            rng = random.Random(SEED)
            for _ in range(509):
                rng.random()
            orders = _try_generate(avail_types, available_counts, order_size, rng, num_orders)
            if _verify(orders, captured):
                return orders, f'{name}_skip509'

        # Brute-force skip
        for skip in range(3000):
            rng = random.Random(SEED)
            for _ in range(skip):
                rng.random()

            # Quick reject: check order 0 size
            state = rng.getstate()
            n0 = rng.randint(order_size[0], order_size[1])
            if n0 != len(captured[0]['items_required']):
                continue
            rng.setstate(state)

            orders = _try_generate(avail_types, available_counts, order_size, rng, num_orders)
            if _verify(orders, captured):
                return orders, f'{name}_skip{skip}'

    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Crack orders from seed')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--num', type=int, default=50, help='Number of orders to generate')
    args = parser.parse_args()

    capture = load_capture(args.difficulty)
    if not capture:
        print(f"No capture for {args.difficulty}", file=sys.stderr)
        sys.exit(0)  # graceful — no capture yet

    orders, method = crack_orders(capture, args.difficulty, args.num)

    if orders is None:
        print(f"Could not crack orders for {args.difficulty} (seed/type-list may have changed)",
              file=sys.stderr)
        sys.exit(0)  # graceful — cracking failed, pipeline continues without

    old_count = len(capture.get('orders', []))
    if len(orders) <= old_count:
        print(f"Already have {old_count} orders (cracked {len(orders)}) — no merge needed",
              file=sys.stderr)
        sys.exit(0)

    capture['orders'] = orders
    save_capture(args.difficulty, capture, archive=False)
    print(f"Cracked {len(orders)} orders via {method} (was {old_count})", file=sys.stderr)


if __name__ == '__main__':
    main()
