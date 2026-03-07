"""RTX 3060 expert solver with cracked seed foresight.

Generates all future orders from cracked seed 7004, then runs
GPU sequential solver with VRAM-safe settings for 12GB RTX 3060.

Usage:
    python solve_expert_3060.py
    python solve_expert_3060.py --max-states 200000   # if 100K fits fine
    python solve_expert_3060.py --refine-iters 10     # more refinement
"""
import json
import random
import sys
import os
import time

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import Order, build_map_from_capture
import gpu_sequential_solver
gpu_sequential_solver._ZIG_AVAILABLE = False  # Zig FFI crashes on this machine
from gpu_sequential_solver import solve_sequential
from solution_store import save_solution, save_capture, load_meta, find_latest_capture
from configs import CONFIGS

# --- Configuration ---
SEED = 7004
RNG_SKIP = 509      # RNG calls consumed by server's build_map
NUM_ORDERS = 50      # enough for 300 rounds of expert

# Server's internal type list (shuffled by RNG during build_map).
# Cracked by brute-forcing skip count + verifying against captured orders.
# This ordering is specific to seed 7004 / expert / 2026-03-05.
SERVER_TYPE_LIST = [
    'bananas', 'apples', 'onions', 'oats', 'butter', 'flour',
    'cheese', 'cream', 'milk', 'pasta', 'yogurt', 'cereal',
    'eggs', 'rice', 'bread', 'tomatoes',
]


def generate_cracked_orders(capture_data, seed, rng_skip, num_orders):
    """Generate all orders using cracked seed with RNG skip.

    The real server's RNG is seeded once, then used for:
      1. build_map (rng_skip calls, including type-list shuffle)
      2. Order generation (sequential after that)

    We use the cracked type list (SERVER_TYPE_LIST) and skip the map-gen
    calls, then generate orders using the exact same RNG state + logic
    as the server's generate_order function.
    """
    ms = build_map_from_capture(capture_data)
    order_size = CONFIGS['expert']['order_size']  # (4, 6)

    # Build available_counts keyed by server's type list order
    item_counts = {}
    for item in capture_data['items']:
        t = item['type']
        item_counts[t] = item_counts.get(t, 0) + 1
    available_counts = {t: item_counts.get(t, 0) for t in SERVER_TYPE_LIST}
    avail_types = list(available_counts.keys())  # preserves SERVER_TYPE_LIST order

    # Create RNG, skip map-generation calls
    rng = random.Random(seed)
    for _ in range(rng_skip):
        rng.random()

    # Generate orders using the server's exact logic
    orders = []
    for i in range(num_orders):
        status = 'active' if i == 0 else ('preview' if i == 1 else 'future')
        n = rng.randint(order_size[0], order_size[1])
        temp_counts = dict(available_counts)
        required_names = []
        for _ in range(n):
            usable = [t for t in avail_types if temp_counts.get(t, 0) > 0]
            if not usable:
                usable = avail_types
            t = rng.choice(usable)
            required_names.append(t)
            if t in temp_counts:
                temp_counts[t] -= 1

        required_ids = [ms.type_name_to_id[name] for name in required_names]
        order = Order(i, required_ids, status)
        order._required_names = required_names
        orders.append(order)

    return orders, ms


def verify_orders(cracked_orders, capture_data):
    """Verify cracked orders match captured orders."""
    captured = capture_data['orders']
    for i, cap_order in enumerate(captured):
        cracked = cracked_orders[i]._required_names
        expected = cap_order['items_required']
        if cracked != expected:
            print(f"MISMATCH at order {i}!", file=sys.stderr)
            print(f"  Cracked:  {cracked}", file=sys.stderr)
            print(f"  Expected: {expected}", file=sys.stderr)
            return False
    print(f"Verified: {len(captured)} captured orders match cracked orders",
          file=sys.stderr)
    return True


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description='RTX 3060 expert solver')
    parser.add_argument('--capture', type=str, default=None,
                        help='Explicit capture file path (default: auto-detect)')
    parser.add_argument('--max-states', type=int, default=100_000,
                        help='Max states per bot DP (default: 100K, safe for 12GB)')
    parser.add_argument('--refine-iters', type=int, default=5,
                        help='Refinement iterations (default: 5)')
    parser.add_argument('--num-orders', type=int, default=NUM_ORDERS,
                        help='Number of orders to generate (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only generate and verify orders, skip solving')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", file=sys.stderr)
    if device == 'cuda':
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({mem:.1f} GB)", file=sys.stderr)

    # 1. Load capture
    capture_path = args.capture or find_latest_capture('expert')
    if not capture_path:
        print("ERROR: No capture file found. Run a game first or use --capture.", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoading capture: {capture_path}", file=sys.stderr)
    with open(capture_path) as f:
        capture = json.load(f)

    # 2. Generate cracked orders
    print(f"Generating {args.num_orders} orders with seed={SEED}, "
          f"rng_skip={RNG_SKIP}...", file=sys.stderr)
    cracked_orders, ms = generate_cracked_orders(
        capture, SEED, RNG_SKIP, args.num_orders)

    # Print first few orders for inspection
    for i, order in enumerate(cracked_orders[:5]):
        print(f"  Order {i}: {order._required_names}", file=sys.stderr)
    print(f"  ... ({len(cracked_orders)} total)", file=sys.stderr)

    # 3. Verify against captured orders
    if not verify_orders(cracked_orders, capture):
        print("ORDER VERIFICATION FAILED - seed/skip may be wrong", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. Orders verified.", file=sys.stderr)
        return

    # 4. Solve with GPU
    t0 = time.time()
    print(f"\nStarting GPU solve: max_states={args.max_states}, "
          f"refine_iters={args.refine_iters}", file=sys.stderr)

    score, actions = solve_sequential(
        capture_data=capture,
        all_orders_override=cracked_orders,
        max_states=args.max_states,
        max_refine_iters=args.refine_iters,
        no_compile=False,
        device=device,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nFinal score: {score} ({elapsed:.0f}s)", file=sys.stderr)

    # 5. Save solution
    meta = load_meta('expert')
    old_score = meta.get('score', 0) if meta else 0

    if score > old_score:
        save_capture('expert', capture)
        save_solution('expert', score, actions, seed=SEED, force=True)
        print(f"Saved! (improved from {old_score} to {score})")
    elif score == old_score:
        print(f"Score {score} ties existing best {old_score}, not saving")
    else:
        print(f"Score {score} < existing best {old_score}, not saving")


if __name__ == '__main__':
    main()
