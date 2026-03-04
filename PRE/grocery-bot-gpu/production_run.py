#!/usr/bin/env python3
"""Iterative live game pipeline within 5-minute token window.

Strategy: play → capture orders → GPU optimize → replay → capture more → repeat
Each cycle discovers more orders and achieves higher scores.

Usage:
    python production_run.py hard                          # full iterative pipeline
    python production_run.py hard --iterations 5           # up to 5 optimize→replay cycles
    python production_run.py hard --time-budget 240        # 4min budget (leave margin)
    python production_run.py expert --max-states 2000000
    python production_run.py hard --ws-url "wss://..."     # skip token fetch
"""
import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LIVE_GPU_SCRIPT = os.path.join(SCRIPT_DIR, 'live_gpu_stream.py')
REPLAY_SCRIPT = os.path.join(SCRIPT_DIR, 'replay_solution.py')
CAPTURE_SCRIPT = os.path.join(SCRIPT_DIR, 'capture_from_game_log.py')
IMPORT_SCRIPT = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'grocery-bot-zig', 'replay', 'import_logs.py',
))


def fetch_token_cli(difficulty, headed=False):
    """Fetch token using fetch_token.py."""
    try:
        from fetch_token import fetch_token
        print(f"  Fetching {difficulty} token...", file=sys.stderr)
        url = fetch_token(difficulty, headed=headed)
        if url:
            print(f"  Got token: ...{url[-20:]}", file=sys.stderr)
        else:
            print(f"  ERROR: No token found. Check token_fetch_debug.png", file=sys.stderr)
        return url
    except Exception as e:
        print(f"  ERROR fetching token: {e}", file=sys.stderr)
        return None


def find_latest_log():
    """Find most recently created game_log_*.jsonl in SCRIPT_DIR."""
    logs = glob.glob(os.path.join(SCRIPT_DIR, 'game_log_*.jsonl'))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def run_live_game(ws_url, difficulty, post_optimize_time=0, max_states=None,
                  save=True, record=True):
    """Run live_gpu_stream.py. Returns (score, log_path, elapsed)."""
    cmd = [sys.executable, LIVE_GPU_SCRIPT, ws_url, '--json-stream']
    if save:
        cmd.append('--save')
    if record:
        cmd.append('--record')
    if post_optimize_time:
        cmd.extend(['--post-optimize-time', str(post_optimize_time)])
    if max_states:
        cmd.extend(['--max-states', str(max_states)])

    score = 0
    log_path = None
    t0 = time.time()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True)

        import threading
        stderr_lines = []

        def read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                stderr_lines.append(line)
                if any(k in line for k in ['GAME_OVER', 'Plan improved', 'Saved', 'ERROR', 'Log:']):
                    print(f"    {line}", file=sys.stderr)

        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                etype = event.get('type', '')
                if etype == 'game_over':
                    score = event.get('score', 0)
                elif etype == 'post_optimize_done':
                    score = max(score, event.get('final_score', 0))
                elif etype == 'pipeline_done':
                    score = max(score, event.get('plan_score', 0))
            except json.JSONDecodeError:
                pass

        proc.wait()
        t.join(timeout=5)

        for line in stderr_lines:
            if line.startswith('Log: '):
                log_path = line[5:].strip()

    except KeyboardInterrupt:
        proc.kill()

    return score, log_path, time.time() - t0


def gpu_optimize(difficulty, max_states=None, max_time_s=None):
    """Run GPU sequential solver on existing capture. Returns (score, elapsed)."""
    from gpu_sequential_solver import solve_sequential
    from solution_store import load_capture, save_solution, load_meta

    capture = load_capture(difficulty)
    if not capture:
        print(f"    No capture data for {difficulty}", file=sys.stderr)
        return 0, 0

    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0

    kwargs = {
        'capture_data': capture,
        'difficulty': difficulty,
        'device': 'cuda',
        'verbose': True,
        'no_filler': True,
    }
    if max_states:
        kwargs['max_states'] = max_states
    if max_time_s:
        kwargs['max_time_s'] = max_time_s

    t0 = time.time()
    score, actions = solve_sequential(**kwargs)
    elapsed = time.time() - t0

    if score > 0:
        saved = save_solution(difficulty, score, actions)
        if saved:
            print(f"    GPU optimize: {prev_score} → {score} (+{score - prev_score}) "
                  f"in {elapsed:.0f}s", file=sys.stderr)
        else:
            print(f"    GPU optimize: score={score} (not saved, existing={prev_score}) "
                  f"in {elapsed:.0f}s", file=sys.stderr)
    else:
        print(f"    GPU optimize: FAILED in {elapsed:.0f}s", file=sys.stderr)

    return score, elapsed


def replay_solution_ws(ws_url, difficulty):
    """Replay existing best solution via WS. Returns (score, log_path, elapsed)."""
    from solution_store import load_solution, load_meta

    actions = load_solution(difficulty)
    meta = load_meta(difficulty)
    if not actions or not meta:
        print(f"    No solution to replay for {difficulty}", file=sys.stderr)
        return 0, None, 0

    print(f"    Replaying solution (score={meta.get('score', '?')})...", file=sys.stderr)

    cmd = [sys.executable, REPLAY_SCRIPT, ws_url, '--difficulty', difficulty]
    t0 = time.time()
    score = 0
    log_path = None

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True)

        stderr_lines = []
        import threading

        def read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                stderr_lines.append(line)
                if any(k in line for k in ['Final', 'score', 'Score', 'ERROR', 'Log:', 'desync']):
                    print(f"      {line}", file=sys.stderr)

        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        proc.wait(timeout=180)
        t.join(timeout=5)

        # Extract score from stderr
        for line in stderr_lines:
            if 'Final score' in line or 'GAME_OVER' in line:
                import re
                m = re.search(r'(?:score|Score)[:\s=]+(\d+)', line)
                if m:
                    score = int(m.group(1))
            if line.startswith('Log: '):
                log_path = line[5:].strip()

    except Exception as e:
        print(f"      Replay error: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"    Replay score: {score} in {elapsed:.0f}s", file=sys.stderr)
    return score, log_path, elapsed


def capture_from_log(log_path, difficulty):
    """Extract orders from game log and merge with existing capture."""
    if not log_path or not os.path.exists(log_path):
        return 0

    cmd = [sys.executable, CAPTURE_SCRIPT, log_path, difficulty, '--merge']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Parse output for order count
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data.get('type') == 'capture_done':
                    n_orders = data.get('orders', 0)
                    print(f"    Captured {n_orders} orders from log", file=sys.stderr)
                    return n_orders
            except json.JSONDecodeError:
                pass
        # Fallback: check stderr
        for line in result.stderr.splitlines():
            if 'orders' in line.lower():
                print(f"    {line}", file=sys.stderr)
    except Exception as e:
        print(f"    Capture error: {e}", file=sys.stderr)

    return 0


def import_log_to_db(log_path, run_type='live'):
    """Import game log to PostgreSQL."""
    if not log_path or not os.path.exists(log_path):
        return
    if not os.path.exists(IMPORT_SCRIPT):
        return
    try:
        subprocess.Popen(
            [sys.executable, IMPORT_SCRIPT, log_path, '--run-type', run_type],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Iterative pipeline: play → optimize → replay → repeat (5min window)')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--iterations', type=int, default=10,
                        help='Max optimize→replay cycles (default: 10)')
    parser.add_argument('--time-budget', type=int, default=250,
                        help='Total time budget in seconds (default: 250, leaves 50s margin)')
    parser.add_argument('--post-optimize-time', type=int, default=30,
                        help='Post-game GPU time for initial live run (default: 30)')
    parser.add_argument('--gpu-time-per-iter', type=int, default=60,
                        help='GPU optimize time per iteration (default: 60)')
    parser.add_argument('--max-states', type=int, default=None,
                        help='Override max GPU states')
    parser.add_argument('--headed', action='store_true',
                        help='Visible browser for token fetch')
    parser.add_argument('--ws-url', type=str, default=None,
                        help='Use this WS URL for initial game (skip token fetch)')
    parser.add_argument('--skip-live', action='store_true',
                        help='Skip initial live game (use existing capture)')
    parser.add_argument('--no-record', action='store_true')
    args = parser.parse_args()

    diff = args.difficulty
    t_start = time.time()
    best_score = 0
    iteration_scores = []

    print(f"{'='*60}", file=sys.stderr)
    print(f"ITERATIVE PIPELINE: {diff}", file=sys.stderr)
    print(f"  Time budget: {args.time_budget}s ({args.time_budget//60}m {args.time_budget%60}s)",
          file=sys.stderr)
    print(f"  Max iterations: {args.iterations}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # ── Phase 1: Initial live game (captures orders) ──
    if not args.skip_live:
        print(f"\n--- Phase 1: Initial live game ---", file=sys.stderr)

        if args.ws_url:
            ws_url = args.ws_url
        else:
            ws_url = fetch_token_cli(diff, headed=args.headed)

        if not ws_url:
            print("FATAL: Could not get token", file=sys.stderr)
            return

        score, log_path, elapsed = run_live_game(
            ws_url, diff,
            post_optimize_time=args.post_optimize_time,
            max_states=args.max_states,
            save=True,
            record=not args.no_record,
        )
        best_score = score
        iteration_scores.append(('live', score, elapsed))
        print(f"  Live game: score={score} in {elapsed:.0f}s", file=sys.stderr)

        # Extract orders from log
        if log_path:
            capture_from_log(log_path, diff)
    else:
        print(f"\n--- Skipping live game (using existing capture) ---", file=sys.stderr)
        from solution_store import load_meta
        meta = load_meta(diff)
        if meta:
            best_score = meta.get('score', 0)
            print(f"  Existing best: {best_score}", file=sys.stderr)

    # ── Phase 2: Iterative optimize → replay ──
    for i in range(args.iterations):
        elapsed_total = time.time() - t_start
        remaining = args.time_budget - elapsed_total

        # Need at least 60s for an optimize + replay cycle
        if remaining < 60:
            print(f"\n  Time budget nearly exhausted ({remaining:.0f}s left), stopping",
                  file=sys.stderr)
            break

        print(f"\n--- Iteration {i+1}/{args.iterations} "
              f"(best={best_score}, {remaining:.0f}s left) ---", file=sys.stderr)

        # GPU optimize with time limit
        gpu_time = min(args.gpu_time_per_iter, remaining - 30)
        if gpu_time < 15:
            print(f"  Not enough time for GPU ({gpu_time:.0f}s), stopping", file=sys.stderr)
            break

        print(f"  GPU optimizing (budget={gpu_time:.0f}s)...", file=sys.stderr)
        opt_score, opt_elapsed = gpu_optimize(
            diff, max_states=args.max_states, max_time_s=gpu_time)

        if opt_score > best_score:
            best_score = opt_score
            print(f"  New best: {best_score}!", file=sys.stderr)

        # Check time for replay
        elapsed_total = time.time() - t_start
        remaining = args.time_budget - elapsed_total
        if remaining < 25:
            print(f"  No time for replay ({remaining:.0f}s left), stopping", file=sys.stderr)
            iteration_scores.append(('optimize', opt_score, opt_elapsed))
            break

        # Fetch new token and replay
        print(f"  Fetching token for replay...", file=sys.stderr)
        replay_url = fetch_token_cli(diff, headed=args.headed)
        if not replay_url:
            print(f"  Could not get replay token, trying next iteration", file=sys.stderr)
            iteration_scores.append(('optimize', opt_score, opt_elapsed))
            continue

        replay_score, replay_log, replay_elapsed = replay_solution_ws(replay_url, diff)
        iteration_scores.append(('replay', replay_score, opt_elapsed + replay_elapsed))

        if replay_score > best_score:
            best_score = replay_score

        # Import replay log and capture new orders
        if replay_log:
            if not args.no_record:
                import_log_to_db(replay_log, run_type='replay')
            n_new = capture_from_log(replay_log, diff)
            if n_new == 0:
                print(f"  No new orders discovered, capture converged", file=sys.stderr)
                # Could still benefit from more GPU time, but orders are maxed

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PIPELINE COMPLETE: {diff}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"{'Phase':>10} {'Score':>6} {'Time':>8}", file=sys.stderr)
    print(f"{'-'*28}", file=sys.stderr)
    for phase, sc, t in iteration_scores:
        print(f"{phase:>10} {sc:>6} {t:>7.0f}s", file=sys.stderr)
    print(f"\nBest score: {best_score}", file=sys.stderr)
    print(f"Total time: {total_time:.0f}s", file=sys.stderr)

    # Print machine-readable result
    print(json.dumps({
        'type': 'pipeline_complete',
        'difficulty': diff,
        'best_score': best_score,
        'iterations': len(iteration_scores),
        'total_time': round(total_time, 1),
    }))


if __name__ == '__main__':
    main()
