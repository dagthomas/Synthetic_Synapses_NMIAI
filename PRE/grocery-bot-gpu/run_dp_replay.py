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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIG_DIR = os.path.join(os.path.dirname(HERE), 'grocery-bot-zig')
ZIG_EXE = os.path.join(ZIG_DIR, 'zig-out', 'bin', 'grocery-bot.exe')


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

    from solution_store import load_dp_plan, load_capture

    dp_plan = load_dp_plan(difficulty)
    if dp_plan is None:
        print(f"No DP plan found in DB for {difficulty}", file=sys.stderr)
        print(f"Run: python export_plan_for_zig.py {difficulty}", file=sys.stderr)
        sys.exit(1)

    # Write temp files for Zig bot (it reads from disk)
    dp_plan_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', prefix=f'dp_plan_{difficulty}_', delete=False)
    json.dump(dp_plan, dp_plan_file)
    dp_plan_file.close()

    capture = load_capture(difficulty)
    capture_file = None
    if capture:
        capture_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', prefix=f'capture_{difficulty}_', delete=False)
        json.dump(capture, capture_file)
        capture_file.close()

    # Build args
    args = [ZIG_EXE, ws_url, '--dp-plan', dp_plan_file.name]
    if capture_file:
        args += ['--precomputed', capture_file.name]

    print(f"Running Zig bot with DP replay ({difficulty})...", file=sys.stderr)
    before_ts = time.time()

    try:
        # Run Zig bot
        result = subprocess.run(args, cwd=ZIG_DIR)
    finally:
        # Clean up temp files
        os.unlink(dp_plan_file.name)
        if capture_file:
            os.unlink(capture_file.name)

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


if __name__ == '__main__':
    main()
