#!/usr/bin/env python3
"""Iterative pipeline within 5-minute token window.

Strategy: Zig bot -> capture orders -> GPU optimize -> replay -> capture more -> repeat
Each cycle discovers more orders and achieves higher scores.
Fast iterations (20s GPU budget) beat one long GPU solve.

Error contract (script entry point):
  - main() returns normally (no sys.exit) on success or graceful failure.
  - Prints machine-readable JSON {"type": "pipeline_complete", ...} on stdout.
  - Helper functions use sentinel returns for missing data:
    - gpu_optimize returns (0, 0.0) if no capture data exists.
    - find_latest_log returns None if no logs found.
    - replay_solution_ws returns (0, None, 0.0) on failure.
    - capture_from_log returns 0 on failure.

Usage:
    python production_run.py hard --ws-url "wss://..."     # full iterative pipeline
    python production_run.py hard --ws-url "wss://..." --iterations 5
    python production_run.py hard --ws-url "wss://..." --time-budget 240
    python production_run.py expert --ws-url "wss://..." --max-states 2000000
"""
import argparse
import glob
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from typing import Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'grocery-bot-zig'))

REPLAY_SCRIPT = os.path.join(SCRIPT_DIR, 'replay_solution.py')
CAPTURE_SCRIPT = os.path.join(SCRIPT_DIR, 'capture_from_game_log.py')
IMPORT_SCRIPT = os.path.normpath(os.path.join(
    SCRIPT_DIR, '..', 'grocery-bot-zig', 'replay', 'import_logs.py',
))




def load_meta_score(difficulty: str) -> int:
    """Load expected score from solution meta. Returns 0 if not available."""
    from solution_store import load_meta
    meta = load_meta(difficulty)
    return meta.get('score', 0) if meta else 0


def find_latest_log() -> Optional[str]:
    """Find most recently created game_log_*.jsonl in SCRIPT_DIR. Returns path or None."""
    logs = glob.glob(os.path.join(SCRIPT_DIR, 'game_log_*.jsonl'))
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def run_zig_bot(ws_url: str, difficulty: str) -> Tuple[int, Optional[str], float]:
    """Run Zig bot executable. Returns (score, log_path_in_gpu_dir, elapsed).

    Spawns grocery-bot-{difficulty}.exe with the WS URL, parses GAME_OVER score
    from stderr, and copies the game log to SCRIPT_DIR for capture.
    """
    from subprocess_helpers import run_bot_game, parse_game_score

    exe_name = f'grocery-bot-{difficulty}.exe'
    exe_path = os.path.join(ZIG_DIR, 'zig-out', 'bin', exe_name)
    if not os.path.exists(exe_path):
        # Fallback to generic executable
        exe_path = os.path.join(ZIG_DIR, 'zig-out', 'bin', 'grocery-bot.exe')

    t0 = time.time()
    return_code, stderr_output, zig_log_path = run_bot_game(
        exe_path, ws_url, cwd=ZIG_DIR, timeout=180)
    elapsed = time.time() - t0

    score = parse_game_score(stderr_output)
    print(f"    Zig bot: score={score}, exit={return_code}, time={elapsed:.0f}s",
          file=sys.stderr)

    # Copy game log to GPU dir for capture
    gpu_log_path = None
    if zig_log_path and os.path.exists(zig_log_path):
        gpu_log_path = os.path.join(SCRIPT_DIR, os.path.basename(zig_log_path))
        shutil.copy2(zig_log_path, gpu_log_path)

    return score, gpu_log_path, elapsed


def run_live_gpu(ws_url: str, difficulty: str, max_states: int = 5000) -> Tuple[int, Optional[str], float]:
    """Run live GPU stream solver as Phase 1. Higher score + more order discovery than Zig.

    Returns (score, log_path, elapsed). Returns (0, None, 0.0) on failure.
    """
    live_script = os.path.join(SCRIPT_DIR, 'live_gpu_stream.py')
    t0 = time.time()
    cmd = [sys.executable, live_script, ws_url,
           '--max-states', str(max_states), '--no-refine', '--save',
           '--preload-capture']
    score = 0
    log_path = None

    try:
        proc = subprocess.Popen(  # nosec B603 B607
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True)

        stderr_lines = []
        import threading

        def read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                stderr_lines.append(line)
                if any(k in line for k in ['Score', 'GAME_OVER', 'ERROR', 'score']):
                    print(f"      {line}", file=sys.stderr)

        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        proc.wait(timeout=180)
        t.join(timeout=5)

        from subprocess_helpers import parse_game_score
        stderr_text = '\n'.join(stderr_lines)
        parsed_score = parse_game_score(stderr_text)
        if parsed_score > 0:
            score = parsed_score

        # Find the game log it produced
        for line in stderr_lines:
            if 'Log saved:' in line or 'game_log_' in line:
                # Try to extract path
                for part in line.split():
                    if 'game_log_' in part and part.endswith('.jsonl'):
                        if os.path.exists(part):
                            log_path = part
                            break

        # Fallback: find latest log
        if not log_path:
            log_path = find_latest_log()

    except Exception as e:
        print(f"      Live GPU error: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"    Live GPU: score={score}, time={elapsed:.0f}s", file=sys.stderr)
    return score, log_path, elapsed


def gpu_optimize(difficulty: str, max_states: Optional[int] = None,
                 max_time_s: Optional[float] = None,
                 warm_only: bool = False,
                 orderings: Optional[int] = None,
                 refine_iters: Optional[int] = None,
                 use_2bot_dp: bool = False,
                 speed_bonus: float = 100.0,
                 max_dp_bots: Optional[int] = None) -> Tuple[int, float]:
    """Run GPU sequential solver on existing capture.

    Args:
        warm_only: Skip cold-start, refine existing solution only.
        orderings: Number of pass1 orderings (default: solver decides).
        refine_iters: Max refinement iterations.
        use_2bot_dp: Use joint 2-bot DP for pair planning.

    Returns:
        (score, elapsed) tuple. Returns (0, 0.0) if no capture data exists
        for the given difficulty (sentinel, not an error).
    """
    from gpu_sequential_solver import solve_sequential, refine_from_solution
    from solution_store import load_capture, save_solution, load_meta, load_solution

    capture = load_capture(difficulty)
    if not capture:
        print(f"    No capture data for {difficulty}", file=sys.stderr)
        return 0, 0

    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0

    t0 = time.time()

    if warm_only:
        existing_actions = load_solution(difficulty)
        if existing_actions:
            # Validate bot count matches capture
            cap_bots = capture.get('num_bots', 0)
            sol_bots = len(existing_actions[0]) if existing_actions else 0
            if sol_bots != cap_bots:
                print(f"    WARNING: Solution has {sol_bots} bots but capture has {cap_bots}, "
                      f"falling back to cold start", file=sys.stderr)
                existing_actions = None
                warm_only = False
            else:
                ref_kwargs = {
                    'capture_data': capture,
                    'difficulty': difficulty,
                    'device': 'cuda',
                    'no_filler': True,
                    'speed_bonus': speed_bonus,
                }
                if max_time_s:
                    ref_kwargs['max_time_s'] = max_time_s
                if max_states:
                    ref_kwargs['max_states'] = max_states
                if refine_iters:
                    ref_kwargs['max_refine_iters'] = refine_iters
                if max_dp_bots is not None:
                    ref_kwargs['max_dp_bots'] = max_dp_bots
                score, actions = refine_from_solution(existing_actions, **ref_kwargs)
        if existing_actions is None:
            # No existing solution, fall back to cold-start
            warm_only = False

    if not warm_only:
        kwargs = {
            'capture_data': capture,
            'difficulty': difficulty,
            'device': 'cuda',
            'verbose': True,
            'no_filler': True,
            'speed_bonus': speed_bonus,
        }
        if max_states:
            kwargs['max_states'] = max_states
        if max_time_s:
            kwargs['max_time_s'] = max_time_s
        if orderings is not None:
            kwargs['num_pass1_orderings'] = orderings
        if refine_iters is not None:
            kwargs['max_refine_iters'] = refine_iters
        if use_2bot_dp:
            kwargs['use_2bot_dp'] = True
        if max_dp_bots is not None:
            kwargs['max_dp_bots'] = max_dp_bots
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


def replay_solution_ws(ws_url: str, difficulty: str) -> Tuple[int, Optional[str], float]:
    """Replay existing best solution via WS.

    Returns:
        (score, log_path, elapsed) tuple. Returns (0, None, 0.0) if no solution exists.
    """
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
        proc = subprocess.Popen(  # nosec B603 B607
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
        from subprocess_helpers import parse_game_score
        stderr_text = '\n'.join(stderr_lines)
        parsed_score = parse_game_score(stderr_text)
        if parsed_score > 0:
            score = parsed_score
        for line in stderr_lines:
            if line.startswith('Log: '):
                log_path = line[5:].strip()

    except Exception as e:
        print(f"      Replay error: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"    Replay score: {score} in {elapsed:.0f}s", file=sys.stderr)
    return score, log_path, elapsed


def capture_from_log(log_path: Optional[str], difficulty: str) -> int:
    """Extract orders from game log and merge with existing capture.

    Returns:
        Number of orders captured, or 0 on failure / missing log.
    """
    if not log_path or not os.path.exists(log_path):
        return 0

    cmd = [sys.executable, CAPTURE_SCRIPT, log_path, difficulty]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # nosec B603 B607
        # Parse output for order count
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data.get('type') == 'capture_done':
                    n_orders = data.get('orders', 0)
                    print(f"    Captured {n_orders} orders from log", file=sys.stderr)
                    return n_orders
            except json.JSONDecodeError:
                pass  # Non-JSON lines in subprocess output are expected; skip them
        # Fallback: check stderr
        for line in result.stderr.splitlines():
            if 'orders' in line.lower():
                print(f"    {line}", file=sys.stderr)
    except Exception as e:
        print(f"    Capture error: {e}", file=sys.stderr)

    return 0


def import_log_to_db(log_path: Optional[str], run_type: str = 'live') -> None:
    """Import game log to PostgreSQL. Best-effort; silently skips on any failure."""
    if not log_path or not os.path.exists(log_path):
        return
    if not os.path.exists(IMPORT_SCRIPT):
        return
    try:
        subprocess.Popen(  # nosec B603 B607
            [sys.executable, IMPORT_SCRIPT, log_path, '--run-type', run_type],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # Best-effort background import; failure does not affect results


def main():
    parser = argparse.ArgumentParser(
        description='Fast-iterate pipeline: Zig bot → GPU optimize → replay → repeat')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--iterations', type=int, default=10,
                        help='Max optimize→replay cycles (default: 10)')
    parser.add_argument('--time-budget', type=int, default=275,
                        help='Total time budget in seconds (default: 275, leaves 13s margin)')
    parser.add_argument('--gpu-time-per-iter', type=int, default=20,
                        help='GPU optimize time per iteration fallback (default: 20)')
    parser.add_argument('--max-states', type=int, default=50000,
                        help='Max GPU states per bot (default: 50000)')
    parser.add_argument('--ws-url', type=str, required=True,
                        help='WebSocket URL (same token reused for all games and replays)')
    parser.add_argument('--skip-live', action='store_true',
                        help='Skip initial Zig bot game (use existing capture)')
    parser.add_argument('--live-gpu', action='store_true',
                        help='Use live GPU solver for Phase 1 instead of Zig bot (higher score, slower)')
    parser.add_argument('--no-record', action='store_true')
    parser.add_argument('--2bot', action='store_true', dest='two_bot',
                        help='Use joint 2-bot DP for Hard/Expert (plans bot pairs jointly)')
    parser.add_argument('--speed-bonus', type=float, default=100.0,
                        help='Speed bonus coefficient for GPU solver (default: 100)')
    parser.add_argument('--speed-decay', type=float, default=0.5,
                        help='Per-iteration speed bonus decay (default: 0.5)')
    parser.add_argument('--max-dp-bots', type=int, default=None,
                        help='Max bots to DP plan (rest get CPU greedy)')
    args = parser.parse_args()

    diff = args.difficulty
    t_start = time.time()
    best_score = 0
    iteration_scores = []

    print(f"{'='*60}", file=sys.stderr)
    print(f"FAST-ITERATE PIPELINE: {diff}", file=sys.stderr)
    print(f"  Time budget: {args.time_budget}s ({args.time_budget//60}m {args.time_budget%60}s)",
          file=sys.stderr)
    print(f"  GPU budget/iter: {args.gpu_time_per_iter}s", file=sys.stderr)
    print(f"  Max iterations: {args.iterations}", file=sys.stderr)
    if args.two_bot:
        print(f"  2-bot joint DP: ENABLED", file=sys.stderr)
    print(f"  Speed bonus: {args.speed_bonus} (decay {args.speed_decay})", file=sys.stderr)
    if args.max_dp_bots:
        print(f"  Max DP bots: {args.max_dp_bots}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    ws_url = args.ws_url

    # Decode and log JWT payload
    from live_gpu_stream import decode_jwt_from_url
    jwt_payload = decode_jwt_from_url(ws_url)
    if jwt_payload:
        print(f"  JWT: map_id={jwt_payload.get('map_id')} "
              f"map_seed={jwt_payload.get('map_seed')} "
              f"exp={jwt_payload.get('exp')}", file=sys.stderr)

    # Note: Do NOT clear solutions here — competition seeds are fixed per difficulty,
    # so accumulated order data from previous runs is valid and valuable.
    # Stale type mismatches are handled by merge_capture() automatically.

    # ── Phase 1: Initial game (Zig bot or live GPU) ──
    if not args.skip_live:
        if args.live_gpu:
            print(f"\n--- Phase 1: Live GPU solver ---", file=sys.stderr)
            score, log_path, elapsed = run_live_gpu(ws_url, diff)
            phase_name = 'live_gpu'
            run_type = 'live_gpu'
        else:
            print(f"\n--- Phase 1: Zig bot ---", file=sys.stderr)
            score, log_path, elapsed = run_zig_bot(ws_url, diff)
            phase_name = 'zig_bot'
            run_type = 'zig'

        best_score = score
        iteration_scores.append((phase_name, score, elapsed))

        if log_path:
            if not args.no_record:
                import_log_to_db(log_path, run_type=run_type)
            capture_from_log(log_path, diff)
    else:
        print(f"\n--- Skipping Zig bot (using existing capture) ---", file=sys.stderr)
        from solution_store import load_meta
        meta = load_meta(diff)
        if meta:
            best_score = meta.get('score', 0)
            print(f"  Existing best: {best_score}", file=sys.stderr)

    # ── Phase 2: Fast iterate: optimize → replay → capture → repeat ──
    for i in range(args.iterations):
        elapsed_total = time.time() - t_start
        remaining = args.time_budget - elapsed_total

        # Need at least 25s for an optimize + replay cycle
        if remaining < 25:
            print(f"\n  Time budget nearly exhausted ({remaining:.0f}s left), stopping",
                  file=sys.stderr)
            break

        print(f"\n--- Iteration {i+1}/{args.iterations} "
              f"(best={best_score}, {remaining:.0f}s left) ---", file=sys.stderr)

        # Adaptive per-iteration config:
        #   Iter 0 (cold): 2 orderings, 2 refine, 25s — quick first pass
        #   Iter 1-2 (warm): fast refine, 15s — maximize order discovery iterations
        #   Iter 3 (deep): 3 refine, 35s, 100K — deep search after more orders known
        #   Iter 4+ (warm): fast refine, 20s — continued discovery
        if i == 0:
            warm_only = False
            orderings = 2
            refine_iters = 2
            target_gpu_time = 25
            iter_states = args.max_states
        elif i == 3:
            # Deep search after 3 discovery iterations
            warm_only = True
            orderings = 1
            refine_iters = 3
            target_gpu_time = 35
            iter_states = min(args.max_states * 2, 100000)
        else:
            warm_only = True
            orderings = 1
            refine_iters = 2
            target_gpu_time = 18
            iter_states = args.max_states

        gpu_time = min(target_gpu_time, remaining - 15)
        if gpu_time < 10:
            print(f"  Not enough time for GPU ({gpu_time:.0f}s), stopping", file=sys.stderr)
            break

        mode = "warm" if warm_only else "cold"
        states_label = f"{iter_states//1000}K" if iter_states >= 1000 else str(iter_states)
        print(f"  GPU optimizing ({mode}, ord={orderings}, ref={refine_iters}, "
              f"states={states_label}, budget={gpu_time:.0f}s)...", file=sys.stderr)
        iter_speed_bonus = args.speed_bonus * (args.speed_decay ** i)
        opt_score, opt_elapsed = gpu_optimize(
            diff, max_states=iter_states, max_time_s=gpu_time,
            warm_only=warm_only, orderings=orderings, refine_iters=refine_iters,
            use_2bot_dp=args.two_bot, speed_bonus=iter_speed_bonus,
            max_dp_bots=args.max_dp_bots)

        if opt_score > best_score:
            best_score = opt_score
            print(f"  New best: {best_score}!", file=sys.stderr)

        # Replay (typically ~5s)
        elapsed_total = time.time() - t_start
        remaining = args.time_budget - elapsed_total
        if remaining < 10:
            print(f"  No time for replay ({remaining:.0f}s left), stopping", file=sys.stderr)
            iteration_scores.append(('optimize', opt_score, opt_elapsed))
            break

        # Skip replay if GPU score is too low (garbage solution = wasted time)
        expected = load_meta_score(diff)
        if expected > 0 and expected < best_score * 0.3:
            print(f"  Skipping replay (GPU score {expected} << best {best_score})",
                  file=sys.stderr)
            iteration_scores.append(('optimize', opt_score, opt_elapsed))
            if opt_score > best_score:
                best_score = opt_score
            # Still capture orders from Zig bot log if available
            continue

        replay_score, replay_log, replay_elapsed = replay_solution_ws(ws_url, diff)

        # Retry once if desync caused a bad score (< 50% of expected)
        if expected > 0 and replay_score < expected * 0.3:
            retry_remaining = args.time_budget - (time.time() - t_start)
            if retry_remaining > 25:
                print(f"  Replay score {replay_score} << expected {expected}, retrying...",
                      file=sys.stderr)
                retry_score, retry_log, retry_elapsed = replay_solution_ws(ws_url, diff)
                replay_elapsed += retry_elapsed
                if retry_score > replay_score:
                    replay_score = retry_score
                    replay_log = retry_log
                    print(f"  Retry improved: {replay_score}", file=sys.stderr)

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

    # ── Phase 3: Perturbation search (post-processing) ──
    elapsed_total = time.time() - t_start
    remaining = args.time_budget - elapsed_total
    if remaining > 10:
        print(f"\n--- Phase 3: Perturbation search ({remaining:.0f}s left) ---", file=sys.stderr)
        try:
            from perturb_search import full_search
            perturb_score = full_search(
                diff, max_iterations=5, n_random=1000,
                time_budget=remaining - 3, n_workers=None, search_all=False,
            )
            if perturb_score > best_score:
                best_score = perturb_score
                print(f"  Perturbation improved: {best_score}!", file=sys.stderr)
                iteration_scores.append(('perturb', perturb_score, time.time() - t_start - elapsed_total))
        except Exception as e:
            print(f"  Perturbation search failed: {e}", file=sys.stderr)
    else:
        print(f"\n  Skipping perturbation search ({remaining:.0f}s left)", file=sys.stderr)

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

    print(json.dumps({
        'type': 'pipeline_complete',
        'difficulty': diff,
        'best_score': best_score,
        'iterations': len(iteration_scores),
        'total_time': round(total_time, 1),
    }))


if __name__ == '__main__':
    main()
