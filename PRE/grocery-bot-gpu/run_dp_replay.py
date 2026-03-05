"""Run Zig bot with DP replay, then capture orders from the game log.

Usage:
    python run_dp_replay.py <wss://...> <difficulty>
    python run_dp_replay.py <wss://...> medium
"""
import subprocess
import sys
import os
import glob
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ZIG_DIR = os.path.join(os.path.dirname(HERE), 'grocery-bot-zig')
ZIG_EXE = os.path.join(ZIG_DIR, 'zig-out', 'bin', 'grocery-bot.exe')
SOLUTIONS_DIR = os.path.join(HERE, 'solutions')
ORDER_LISTS_DIR = os.path.join(HERE, 'order_lists')


def find_newest_log(directory, after_ts=0):
    logs = glob.glob(os.path.join(directory, 'game_log_*.jsonl'))
    logs = [(f, os.path.getmtime(f)) for f in logs]
    logs = [(f, t) for f, t in logs if t >= after_ts]
    logs.sort(key=lambda x: -x[1])
    return logs[0][0] if logs else None


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <wss://...> <difficulty>", file=sys.stderr)
        sys.exit(1)

    ws_url = sys.argv[1]
    difficulty = sys.argv[2]
    dp_plan = os.path.join(SOLUTIONS_DIR, difficulty, 'dp_plan.json')
    capture_json = os.path.join(SOLUTIONS_DIR, difficulty, 'capture.json')

    if not os.path.exists(dp_plan):
        print(f"No DP plan found at {dp_plan}", file=sys.stderr)
        print(f"Run: python export_plan_for_zig.py {difficulty}", file=sys.stderr)
        sys.exit(1)

    # Build args
    args = [ZIG_EXE, ws_url, '--dp-plan', dp_plan]
    if os.path.exists(capture_json):
        args += ['--precomputed', capture_json]

    print(f"Running Zig bot with DP replay ({difficulty})...", file=sys.stderr)
    before_ts = time.time()

    # Run Zig bot
    result = subprocess.run(args, cwd=ZIG_DIR)

    # Find the game log created by this run
    log_path = find_newest_log(ZIG_DIR, before_ts)
    if not log_path:
        print("No game log found after run", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nGame log: {os.path.basename(log_path)}", file=sys.stderr)

    # Copy log to GPU dir for capture
    dest_log = os.path.join(HERE, os.path.basename(log_path))
    if not os.path.exists(dest_log):
        import shutil
        shutil.copy2(log_path, dest_log)

    # Capture orders from game log
    print(f"Capturing orders...", file=sys.stderr)
    capture_script = os.path.join(HERE, 'capture_from_game_log.py')
    cap_result = subprocess.run(
        [sys.executable, capture_script, dest_log, difficulty],
        capture_output=True, text=True)

    if cap_result.stderr:
        for line in cap_result.stderr.strip().split('\n'):
            print(f"  {line}", file=sys.stderr)

    # Update order_lists from capture
    if os.path.exists(capture_json):
        try:
            capture = json.load(open(capture_json))
            orders = capture.get('orders', [])
            total = len(orders)

            os.makedirs(ORDER_LISTS_DIR, exist_ok=True)
            order_file = os.path.join(ORDER_LISTS_DIR, f'{difficulty}_orders.json')

            order_data = {
                'difficulty': difficulty,
                'date': time.strftime('%Y-%m-%d'),
                'total_orders': total,
                'orders': [{'index': i, 'items_required': o['items_required']}
                           for i, o in enumerate(orders)],
            }

            # Only update if more orders
            should_write = True
            if os.path.exists(order_file):
                existing = json.load(open(order_file))
                if len(existing.get('orders', [])) >= total:
                    should_write = False

            if should_write:
                with open(order_file, 'w') as f:
                    json.dump(order_data, f, indent=2)
                print(f"  Order list updated: {order_file} ({total} orders)", file=sys.stderr)
            else:
                print(f"  Order list unchanged ({total} orders)", file=sys.stderr)
        except Exception as e:
            print(f"  Order list update error: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
