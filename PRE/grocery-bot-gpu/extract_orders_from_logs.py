"""Extract order sequences from game log JSONL files.

Scans all game_log_*.jsonl in grocery-bot-zig/, determines difficulty from grid size,
and extracts the full order sequence as orders become visible (active/preview).

Grid sizes: Easy=12x10, Medium=16x12, Hard=22x14, Expert=28x18
"""

import json
import glob
import os
from collections import defaultdict

GRID_TO_DIFF = {
    (12, 10): "easy",
    (16, 12): "medium",
    (22, 14): "hard",
    (28, 18): "expert",
}

def extract_orders_from_log(filepath):
    """Extract grid info, orders seen, and final score from a single log file."""
    width = height = None
    num_bots = 0
    orders_seen = {}  # order_id -> {items_required, first_seen_round, status_when_first_seen}
    final_score = 0
    max_round = 0
    drop_off = None
    items = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines are "N→{json}" format
            if '→' in line:
                _, json_str = line.split('→', 1)
            else:
                json_str = line

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            if data.get('type') != 'game_state':
                continue

            rnd = data.get('round', 0)
            max_round = max(max_round, rnd)

            if width is None:
                grid = data.get('grid', {})
                width = grid.get('width')
                height = grid.get('height')
                num_bots = len(data.get('bots', []))
                drop_off = data.get('drop_off')
                items = data.get('items')

            final_score = data.get('score', final_score)

            for order in data.get('orders', []):
                oid = order['id']
                if oid not in orders_seen:
                    orders_seen[oid] = {
                        'items_required': order['items_required'],
                        'first_seen_round': rnd,
                        'status_when_first_seen': order['status'],
                    }

    difficulty = GRID_TO_DIFF.get((width, height), f"unknown_{width}x{height}")

    return {
        'file': os.path.basename(filepath),
        'difficulty': difficulty,
        'width': width,
        'height': height,
        'num_bots': num_bots,
        'drop_off': drop_off,
        'final_score': final_score,
        'max_round': max_round,
        'orders_seen': orders_seen,
        'num_orders_seen': len(orders_seen),
        'items': items,
    }


def main():
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'grocery-bot-zig')
    log_files = sorted(glob.glob(os.path.join(log_dir, 'game_log_*.jsonl')))

    print(f"Found {len(log_files)} game log files\n")

    # Group by difficulty
    by_diff = defaultdict(list)

    for f in log_files:
        info = extract_orders_from_log(f)
        by_diff[info['difficulty']].append(info)

    for diff in ['easy', 'medium', 'hard', 'expert']:
        logs = by_diff.get(diff, [])
        if not logs:
            continue

        print(f"{'='*60}")
        print(f"  {diff.upper()} ({len(logs)} games)")
        print(f"{'='*60}")

        # Merge all orders across games for this difficulty
        all_orders = {}  # order_id -> items_required
        best_score = 0
        best_file = ""

        for log in logs:
            print(f"\n  {log['file']}: score={log['final_score']}, "
                  f"rounds={log['max_round']}, orders_seen={log['num_orders_seen']}")

            if log['final_score'] > best_score:
                best_score = log['final_score']
                best_file = log['file']

            for oid, odata in log['orders_seen'].items():
                if oid not in all_orders:
                    all_orders[oid] = odata['items_required']

        # Sort by order index
        sorted_orders = sorted(all_orders.items(), key=lambda x: int(x[0].split('_')[1]))

        print(f"\n  Best score: {best_score} ({best_file})")
        print(f"  Total unique orders discovered: {len(sorted_orders)}")
        print(f"\n  Order sequence:")
        for oid, items in sorted_orders:
            idx = int(oid.split('_')[1])
            print(f"    {idx:3d}: {items}")

        # Save to DB via merge_capture
        capture_data = {
            'difficulty': diff,
            'total_orders_discovered': len(sorted_orders),
            'orders': [{'id': oid, 'items_required': items} for oid, items in sorted_orders],
        }

        # Also grab map data from any log
        if logs:
            sample = logs[0]
            capture_data['grid'] = {'width': sample['width'], 'height': sample['height']}
            capture_data['drop_off'] = sample['drop_off']
            capture_data['num_bots'] = sample['num_bots']
            if sample['items']:
                capture_data['items'] = sample['items']

        try:
            from solution_store import merge_capture
            merged, num_new, total = merge_capture(diff, capture_data)
            print(f"\n  Saved to DB ({total} orders, +{num_new} new)")
        except Exception as e:
            print(f"\n  DB save error: {e}")
        print()

    # Handle unknown difficulties
    for diff, logs in by_diff.items():
        if diff in ['easy', 'medium', 'hard', 'expert']:
            continue
        print(f"\n  UNKNOWN DIFFICULTY: {diff} ({len(logs)} games)")
        for log in logs:
            print(f"    {log['file']}: {log['width']}x{log['height']}, "
                  f"bots={log['num_bots']}, score={log['final_score']}")


if __name__ == '__main__':
    main()
