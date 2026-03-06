#!/usr/bin/env python3
"""Crack order sequence from map_id UUID.

Theory: map_seed controls physical layout, map_id controls order sequence.

Tries various PRNG algorithms and seed derivations from map_id to match
captured orders.

Usage:
    python crack_map_id.py hard --map-id 05ddc283-9097-4314-824c-90b3269a3d95
    python crack_map_id.py hard --token "eyJ..."
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import struct
import sys
import uuid

# Live server item types per difficulty (NOT the old ALL_TYPES list)
LIVE_TYPES = {
    "hard": ["bread", "milk", "cereal", "rice", "yogurt", "eggs",
             "cheese", "cream", "oats", "pasta", "butter", "flour"],
    "expert": ["bread", "flour", "yogurt", "rice", "pasta", "tomatoes",
               "milk", "onions", "bananas", "eggs", "oats", "apples",
               "cheese", "butter", "cereal", "cream"],
}

ORDER_SIZE = {
    "easy": (3, 4),
    "medium": (3, 5),
    "hard": (3, 5),
    "expert": (4, 6),
}


def load_captured_orders(difficulty: str) -> list[list[str]]:
    capture_path = os.path.join(
        os.path.dirname(__file__), 'solutions', difficulty, 'capture.json')
    with open(capture_path) as f:
        data = json.load(f)
    return [o['items_required'] for o in data['orders']]


def decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification."""
    parts = token.split('.')
    payload_b64 = parts[1]
    payload_b64 += '=' * (4 - len(payload_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def gen_orders_python(seed_val, types: list[str], order_size: tuple[int, int],
                      count: int, available_counts: dict | None = None) -> list[list[str]]:
    """Generate orders using Python's random.Random."""
    rng = random.Random(seed_val)
    orders = []
    for _ in range(count):
        n = rng.randint(order_size[0], order_size[1])
        if available_counts:
            temp = dict(available_counts)
            avail = [t for t in types if temp.get(t, 0) > 0]
            req = []
            for _ in range(n):
                usable = [t for t in avail if temp.get(t, 0) > 0]
                if not usable:
                    usable = avail
                t = rng.choice(usable)
                req.append(t)
                if t in temp:
                    temp[t] -= 1
            orders.append(req)
        else:
            orders.append([rng.choice(types) for _ in range(n)])
    return orders


def gen_orders_simple(seed_val, types: list[str], order_size: tuple[int, int],
                      count: int) -> list[list[str]]:
    """Simple random.choice without availability tracking."""
    rng = random.Random(seed_val)
    orders = []
    for _ in range(count):
        n = rng.randint(order_size[0], order_size[1])
        orders.append([rng.choice(types) for _ in range(n)])
    return orders


class JavaRandom:
    """Java's java.util.Random (48-bit LCG)."""
    def __init__(self, seed: int):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.seed >> (48 - bits)

    def nextInt(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError
        # Power of 2
        if (bound & (bound - 1)) == 0:
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            if bits - val + (bound - 1) >= 0:
                return val


def gen_orders_java(seed_val: int, types: list[str], order_size: tuple[int, int],
                    count: int) -> list[list[str]]:
    """Generate orders using Java LCG."""
    rng = JavaRandom(seed_val & 0xFFFFFFFFFFFF)
    orders = []
    min_sz, max_sz = order_size
    for _ in range(count):
        n = min_sz + rng.nextInt(max_sz - min_sz + 1)
        orders.append([types[rng.nextInt(len(types))] for _ in range(n)])
    return orders


class GoRand:
    """Go's math/rand (rng from seed)."""
    def __init__(self, seed: int):
        # Go uses a different algorithm. Simplified version with LCG.
        self.state = seed & 0x7FFFFFFFFFFFFFFF

    def _next(self) -> int:
        # Go's default source uses a Lagged Fibonacci generator
        # This is a simplified approximation
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 1

    def intn(self, n: int) -> int:
        return self._next() % n


def gen_orders_go(seed_val: int, types: list[str], order_size: tuple[int, int],
                  count: int) -> list[list[str]]:
    """Generate orders using Go-like LCG."""
    rng = GoRand(seed_val)
    orders = []
    min_sz, max_sz = order_size
    for _ in range(count):
        n = min_sz + rng.intn(max_sz - min_sz + 1)
        orders.append([types[rng.intn(len(types))] for _ in range(n)])
    return orders


def match_orders(generated: list[list[str]], captured: list[list[str]]) -> int:
    """Count how many consecutive orders match from the start."""
    for i, (gen, cap) in enumerate(zip(generated, captured)):
        if gen != cap:
            return i
    return min(len(generated), len(captured))


def get_available_counts(difficulty: str) -> dict[str, int]:
    """Get item counts from capture data."""
    capture_path = os.path.join(
        os.path.dirname(__file__), 'solutions', difficulty, 'capture.json')
    with open(capture_path) as f:
        data = json.load(f)
    counts: dict[str, int] = {}
    for item in data['items']:
        t = item['type']
        counts[t] = counts.get(t, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description='Crack orders from map_id')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--map-id', type=str, help='map_id UUID')
    parser.add_argument('--map-seed', type=int, help='map_seed from token')
    parser.add_argument('--token', type=str, help='JWT token (or full wss:// URL)')
    args = parser.parse_args()

    # Extract map_id from token if provided
    map_id_str = args.map_id
    map_seed = args.map_seed
    if args.token:
        tok = args.token
        if 'token=' in tok:
            tok = tok.split('token=')[1]
        payload = decode_jwt_payload(tok)
        map_id_str = map_id_str or payload.get('map_id')
        map_seed = map_seed or payload.get('map_seed')
        print(f"JWT payload: {json.dumps(payload, indent=2)}", file=sys.stderr)

    if not map_id_str:
        print("ERROR: provide --map-id or --token", file=sys.stderr)
        sys.exit(1)

    captured = load_captured_orders(args.difficulty)
    print(f"Captured {len(captured)} orders for {args.difficulty}", file=sys.stderr)
    for i, o in enumerate(captured[:5]):
        print(f"  {i}: {o}", file=sys.stderr)

    types = LIVE_TYPES.get(args.difficulty)
    if not types:
        from configs import ALL_TYPES, CONFIGS
        types = ALL_TYPES[:CONFIGS[args.difficulty]['types']]
    order_size = ORDER_SIZE[args.difficulty]
    avail = get_available_counts(args.difficulty)

    u = uuid.UUID(map_id_str)

    # Generate seed candidates from map_id
    seed_candidates = {
        'uuid_int': u.int,
        'uuid_int_lower64': u.int & 0xFFFFFFFFFFFFFFFF,
        'uuid_int_upper64': u.int >> 64,
        'uuid_int_lower32': u.int & 0xFFFFFFFF,
        'uuid_int_upper32': u.int >> 96,
        'uuid_int_mid32': (u.int >> 32) & 0xFFFFFFFF,
        'md5_int32': int(hashlib.md5(map_id_str.encode()).hexdigest()[:8], 16),
        'sha256_int32': int(hashlib.sha256(map_id_str.encode()).hexdigest()[:8], 16),
        'sha256_int64': int(hashlib.sha256(map_id_str.encode()).hexdigest()[:16], 16),
        'md5_bytes_int32': int(hashlib.md5(u.bytes).hexdigest()[:8], 16),
        'sha256_bytes_int32': int(hashlib.sha256(u.bytes).hexdigest()[:8], 16),
        # map_id without hyphens as hex -> int
        'hex_nohyphen': int(map_id_str.replace('-', ''), 16),
        # Fields from UUID
        'time_low': u.time_low,
        'time_mid': u.time_mid,
        'time_hi': u.time_hi_version,
        'clock_seq': u.clock_seq,
        'node': u.node,
    }

    if map_seed:
        # Also try combining map_seed with map_id
        seed_candidates['map_seed'] = map_seed
        seed_candidates['map_seed_xor_lower32'] = map_seed ^ (u.int & 0xFFFFFFFF)
        seed_candidates['map_seed_xor_lower64'] = map_seed ^ (u.int & 0xFFFFFFFFFFFFFFFF)
        seed_candidates['md5_seed_mapid'] = int(hashlib.md5(f"{map_seed}:{map_id_str}".encode()).hexdigest()[:8], 16)
        seed_candidates['sha256_seed_mapid'] = int(hashlib.sha256(f"{map_seed}:{map_id_str}".encode()).hexdigest()[:8], 16)
        seed_candidates['md5_mapid_seed'] = int(hashlib.md5(f"{map_id_str}:{map_seed}".encode()).hexdigest()[:8], 16)

    generators = {
        'python_simple': lambda s: gen_orders_simple(s, types, order_size, len(captured)),
        'python_avail': lambda s: gen_orders_python(s, types, order_size, len(captured), avail),
        'java': lambda s: gen_orders_java(s, types, order_size, len(captured)),
        'go_lcg': lambda s: gen_orders_go(s, types, order_size, len(captured)),
    }

    # Also try different type orderings
    types_sorted = sorted(types)
    types_reverse = list(reversed(types))
    alt_type_lists = {
        'live_order': types,
        'sorted': types_sorted,
        'reversed': types_reverse,
    }

    print(f"\nTesting {len(seed_candidates)} seeds x {len(generators)} generators x {len(alt_type_lists)} type orderings...",
          file=sys.stderr)

    best_match = 0
    best_info = None

    for seed_name, seed_val in seed_candidates.items():
        for gen_name, gen_fn in generators.items():
            for type_name, type_list in alt_type_lists.items():
                # Rebind generator with different types
                if type_name != 'live_order':
                    if gen_name == 'python_simple':
                        gen = lambda s, tl=type_list: gen_orders_simple(s, tl, order_size, len(captured))
                    elif gen_name == 'python_avail':
                        gen = lambda s, tl=type_list: gen_orders_python(s, tl, order_size, len(captured), avail)
                    elif gen_name == 'java':
                        gen = lambda s, tl=type_list: gen_orders_java(s, tl, order_size, len(captured))
                    elif gen_name == 'go_lcg':
                        gen = lambda s, tl=type_list: gen_orders_go(s, tl, order_size, len(captured))
                    else:
                        continue
                else:
                    gen = gen_fn

                try:
                    generated = gen(seed_val)
                    m = match_orders(generated, captured)
                    if m > 0:
                        print(f"  MATCH {m}: seed={seed_name} ({seed_val}), "
                              f"gen={gen_name}, types={type_name}", file=sys.stderr)
                        if m >= len(captured):
                            print(f"\n*** ORDERS CRACKED ***", file=sys.stderr)
                        if m > best_match:
                            best_match = m
                            best_info = (seed_name, seed_val, gen_name, type_name, generated)
                except Exception as e:
                    pass

    # Also try brute-force small seeds near map_seed
    if map_seed:
        print(f"\nBrute-forcing seeds near map_seed ({map_seed}) +/- 10000...", file=sys.stderr)
        for s in range(max(0, map_seed - 10000), map_seed + 10001):
            for gen_name in ['python_simple', 'java']:
                try:
                    if gen_name == 'python_simple':
                        generated = gen_orders_simple(s, types, order_size, len(captured))
                    else:
                        generated = gen_orders_java(s, types, order_size, len(captured))
                    m = match_orders(generated, captured)
                    if m >= 1:
                        print(f"  MATCH {m}: seed={s}, gen={gen_name}", file=sys.stderr)
                        if m > best_match:
                            best_match = m
                            best_info = (f'brute_{s}', s, gen_name, 'live_order', generated)
                except Exception:
                    pass

    print(f"\n{'='*60}", file=sys.stderr)
    if best_match > 0 and best_info:
        seed_name, seed_val, gen_name, type_name, generated = best_info
        print(f"Best: {best_match}/{len(captured)} orders matched", file=sys.stderr)
        print(f"  Seed: {seed_name} = {seed_val}", file=sys.stderr)
        print(f"  Generator: {gen_name}, Types: {type_name}", file=sys.stderr)
        print(f"\nGenerated orders:", file=sys.stderr)
        for i, o in enumerate(generated):
            marker = "OK" if i < best_match else "MISS"
            cap = captured[i] if i < len(captured) else "?"
            print(f"  {i}: {o}  [{marker}]  (captured: {cap})", file=sys.stderr)
    else:
        print("No matches found. Server likely uses a different PRNG or seed derivation.", file=sys.stderr)
        # Print first generated order from each method for comparison
        print(f"\nSample outputs (seed=map_seed={map_seed}):", file=sys.stderr)
        for gen_name in ['python_simple', 'java', 'go_lcg']:
            if gen_name == 'python_simple':
                o = gen_orders_simple(map_seed, types, order_size, 3)
            elif gen_name == 'java':
                o = gen_orders_java(map_seed, types, order_size, 3)
            else:
                o = gen_orders_go(map_seed, types, order_size, 3)
            print(f"  {gen_name}: {o[:3]}", file=sys.stderr)
        print(f"  captured:       {captured[:3]}", file=sys.stderr)


if __name__ == '__main__':
    main()
