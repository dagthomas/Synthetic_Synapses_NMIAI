"""Extract capture data from a Zig bot game_log_*.jsonl file.

Parses all game_state lines to accumulate the full order list revealed
across 300 rounds, then saves capture.json for GPU DP optimization.

Usage:
    python capture_from_game_log.py <game_log_path> <difficulty>
    python capture_from_game_log.py --latest <difficulty>
"""
import json
import sys
import os
import glob
import argparse

from solution_store import save_capture

DIFFICULTY_DIMS = {
    (12, 10, 1): 'easy',
    (16, 12, 3): 'medium',
    (22, 14, 5): 'hard',
    (28, 18, 10): 'expert',
}


def detect_difficulty(width, height, num_bots):
    return DIFFICULTY_DIMS.get((width, height, num_bots))


def find_latest_game_log(search_dirs=None):
    """Find the most recent game_log_*.jsonl across search directories."""
    if search_dirs is None:
        here = os.path.dirname(os.path.abspath(__file__))
        zig_dir = os.path.join(os.path.dirname(here), 'grocery-bot-zig')
        search_dirs = [here, zig_dir]

    best = None
    best_mtime = 0
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, 'game_log_*.jsonl')):
            mtime = os.path.getmtime(f)
            size = os.path.getsize(f)
            if size > 0 and mtime > best_mtime:
                best = f
                best_mtime = mtime
    return best


def extract_capture(game_log_path, difficulty=None):
    """Parse a game_log JSONL and extract capture data.

    Returns (capture_dict, final_score, difficulty) or raises on error.
    """
    grid = None
    items = None
    drop_off = None
    num_bots = 0
    width = height = 0
    seen_order_ids = set()
    orders_in_order = []
    final_score = 0

    with open(game_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip action lines
            if 'actions' in data and 'type' not in data:
                continue

            if data.get('type') == 'game_over':
                final_score = data.get('score', final_score)
                continue

            if data.get('type') != 'game_state':
                continue

            # Round 0: extract static map data
            if data['round'] == 0:
                grid = data['grid']
                items = data['items']
                drop_off = data['drop_off']
                num_bots = len(data['bots'])
                width = grid['width']
                height = grid['height']

            final_score = data.get('score', final_score)

            # Accumulate unique orders across all rounds
            for order in data.get('orders', []):
                oid = order['id']
                if oid not in seen_order_ids:
                    seen_order_ids.add(oid)
                    orders_in_order.append({
                        'items_required': list(order['items_required']),
                    })

    if grid is None:
        raise ValueError(f"No valid game_state found in {game_log_path}")

    # Auto-detect difficulty if not provided
    if difficulty is None:
        difficulty = detect_difficulty(width, height, num_bots)
        if difficulty is None:
            raise ValueError(
                f"Cannot auto-detect difficulty for {width}x{height} with {num_bots} bots")

    capture = {
        'grid': grid,
        'items': items,
        'drop_off': drop_off,
        'num_bots': num_bots,
        'difficulty': difficulty,
        'orders': orders_in_order,
    }

    return capture, final_score, difficulty


def main():
    parser = argparse.ArgumentParser(
        description='Extract capture from Zig bot game_log')
    parser.add_argument('game_log', nargs='?',
                        help='Path to game_log_*.jsonl (or --latest)')
    parser.add_argument('difficulty', nargs='?',
                        choices=['easy', 'medium', 'hard', 'expert'],
                        help='Difficulty (auto-detected if omitted)')
    parser.add_argument('--latest', action='store_true',
                        help='Use most recent game_log file')
    args = parser.parse_args()

    if args.latest or args.game_log is None:
        log_path = find_latest_game_log()
        if log_path is None:
            print("ERROR: No game_log_*.jsonl files found", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest: {log_path}", file=sys.stderr)
    else:
        log_path = args.game_log

    if not os.path.exists(log_path):
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    capture, score, difficulty = extract_capture(log_path, args.difficulty)

    save_path = save_capture(difficulty, capture)

    # Output summary as JSON for pipeline consumption
    result = {
        'type': 'capture_done',
        'difficulty': difficulty,
        'orders': len(capture['orders']),
        'items': len(capture['items']),
        'num_bots': capture['num_bots'],
        'grid': f"{capture['grid']['width']}x{capture['grid']['height']}",
        'probe_score': score,
        'path': save_path,
    }
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
