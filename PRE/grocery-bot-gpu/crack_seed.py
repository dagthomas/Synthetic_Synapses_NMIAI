#!/usr/bin/env python3
"""Crack the game seed by matching captured orders against generated orders.

Usage:
    python crack_seed.py hard              # crack from capture.json
    python crack_seed.py expert            # crack expert
    python crack_seed.py hard --range 0 1000000  # custom range
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from configs import CONFIGS, ALL_TYPES
from game_engine import build_map, generate_order_from_rng


def load_captured_orders(difficulty: str) -> list[list[str]]:
    """Load captured orders from DB."""
    from solution_store import load_capture
    data = load_capture(difficulty)
    if data is None:
        print(f"No capture found in DB for {difficulty}", file=sys.stderr)
        sys.exit(1)
    return [o['items_required'] for o in data['orders']]


def generate_orders_for_seed(seed: int, difficulty: str, count: int = 10) -> list[list[str]]:
    """Generate first `count` orders for a given seed."""
    cfg = CONFIGS[difficulty]
    order_size = cfg['order_size']
    item_type_names = ALL_TYPES[:cfg['types']]
    ms = build_map(difficulty)

    # Count available items per type
    available_counts = {}
    for item in ms.items:
        t = item['type']
        available_counts[t] = available_counts.get(t, 0) + 1

    rng = random.Random(seed)
    orders = []
    for i in range(count):
        status = 'active' if i == 0 else ('preview' if i == 1 else 'future')
        required = generate_order_from_rng(rng, i, item_type_names, order_size, status, available_counts)
        orders.append(required)
    return orders


def check_seed_range(difficulty: str, captured_orders: list[list[str]],
                     start: int, end: int, num_match: int = 3) -> list[int]:
    """Check seeds in [start, end) against captured orders. Returns matching seeds."""
    matches = []
    # Pre-build map once (same for all seeds of same difficulty)
    cfg = CONFIGS[difficulty]
    order_size = cfg['order_size']
    item_type_names = ALL_TYPES[:cfg['types']]
    ms = build_map(difficulty)
    available_counts = {}
    for item in ms.items:
        t = item['type']
        available_counts[t] = available_counts.get(t, 0) + 1

    target = captured_orders[:num_match]

    for seed in range(start, end):
        rng = random.Random(seed)
        match = True
        for i in range(num_match):
            status = 'active' if i == 0 else ('preview' if i == 1 else 'future')
            required = generate_order_from_rng(rng, i, item_type_names, order_size, status, available_counts)
            if required != target[i]:
                match = False
                break
        if match:
            matches.append(seed)
    return matches


def verify_seed(seed: int, difficulty: str, captured_orders: list[list[str]]) -> int:
    """Verify how many orders match for a given seed. Returns match count."""
    generated = generate_orders_for_seed(seed, difficulty, count=len(captured_orders))
    for i, (gen, cap) in enumerate(zip(generated, captured_orders)):
        if gen != cap:
            return i
    return len(captured_orders)


def main():
    parser = argparse.ArgumentParser(description='Crack game seed from captured orders')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--range', nargs=2, type=int, default=[0, 10_000_000],
                        help='Seed range to search (default: 0 to 10M)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--match', type=int, default=3,
                        help='Number of orders to match for initial screen')
    args = parser.parse_args()

    captured = load_captured_orders(args.difficulty)
    print(f"Loaded {len(captured)} captured orders for {args.difficulty}", file=sys.stderr)
    print(f"First 3 orders:", file=sys.stderr)
    for i, o in enumerate(captured[:3]):
        print(f"  {i}: {o}", file=sys.stderr)

    start, end = args.range
    total = end - start
    chunk_size = max(10000, total // (args.workers * 10))

    print(f"\nSearching seeds {start}-{end} ({total:,} seeds) "
          f"with {args.workers} workers, matching first {args.match} orders...",
          file=sys.stderr)

    t0 = time.time()
    candidates = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            f = pool.submit(check_seed_range, args.difficulty, captured,
                            chunk_start, chunk_end, args.match)
            futures[f] = (chunk_start, chunk_end)

        done = 0
        for f in as_completed(futures):
            chunk_start, chunk_end = futures[f]
            results = f.result()
            candidates.extend(results)
            done += chunk_end - chunk_start
            if done % 1_000_000 < chunk_size:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done:,}/{total:,} ({done*100//total}%) "
                      f"found={len(candidates)}, {rate:,.0f} seeds/s, "
                      f"ETA {eta:.0f}s", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nSearch complete: {total:,} seeds in {elapsed:.1f}s "
          f"({total/elapsed:,.0f} seeds/s)", file=sys.stderr)

    if not candidates:
        print(f"\nNo seed found in range {start}-{end}!", file=sys.stderr)
        print("Try a wider range: --range 0 100000000", file=sys.stderr)
        return

    # Verify candidates against ALL captured orders
    print(f"\n{len(candidates)} candidates found, verifying against all "
          f"{len(captured)} orders...", file=sys.stderr)

    best_seed = None
    best_match = 0
    for seed in candidates:
        match_count = verify_seed(seed, args.difficulty, captured)
        print(f"  Seed {seed}: {match_count}/{len(captured)} orders match", file=sys.stderr)
        if match_count > best_match:
            best_match = match_count
            best_seed = seed

    if best_match >= len(captured):
        print(f"\n*** SEED CRACKED: {best_seed} ***", file=sys.stderr)
        print(f"All {best_match} captured orders verified!", file=sys.stderr)

        # Print upcoming orders
        all_orders = generate_orders_for_seed(best_seed, args.difficulty, count=50)
        print(f"\nAll 50 orders for seed {best_seed}:", file=sys.stderr)
        for i, o in enumerate(all_orders):
            marker = " <-- captured" if i < len(captured) else " <-- PREDICTED"
            print(f"  {i:2d}: {o}{marker}", file=sys.stderr)

        # Output machine-readable result
        print(json.dumps({
            'type': 'seed_cracked',
            'difficulty': args.difficulty,
            'seed': best_seed,
            'verified_orders': best_match,
        }))
    else:
        print(f"\nBest match: seed {best_seed} with {best_match}/{len(captured)} orders",
              file=sys.stderr)
        if best_match >= 3:
            print("Partial match — seed may be correct but order generation differs",
                  file=sys.stderr)


if __name__ == '__main__':
    main()
