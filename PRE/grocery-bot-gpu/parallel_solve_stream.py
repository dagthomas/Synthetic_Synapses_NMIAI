"""Parallel optimizer with JSON streaming output for SvelteKit dashboard.

Outputs JSON lines to stdout for SSE consumption.
Loads captured orders, runs parallel optimizer, outputs progress.

Usage:
    python parallel_solve_stream.py <difficulty> [--time <seconds>] [--workers <n>]
"""
import json
import sys
import os
import time

def emit(data):
    """Output a JSON line to stdout for SSE streaming."""
    print(json.dumps(data), flush=True)


def solve(difficulty, time_limit=240.0, num_workers=None):
    from solution_store import load_capture, load_meta, save_solution
    from parallel_optimizer import parallel_optimize
    from configs import CONFIGS

    t0 = time.time()

    if num_workers is None:
        num_workers = min(12, os.cpu_count() or 4)

    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    # Load capture
    capture = load_capture(difficulty)
    if not capture:
        emit({"type": "error", "msg": f"No capture found for {difficulty}. Run Zig bot first."})
        return

    emit({
        "type": "init",
        "difficulty": difficulty,
        "num_bots": num_bots,
        "num_workers": num_workers,
        "time_limit": time_limit,
        "num_orders": len(capture.get('orders', [])),
        "num_items": len(capture.get('items', [])),
        "probe_score": capture.get('probe_score', 0),
        "solver": "parallel_optimizer",
    })

    # Get previous best
    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    emit({"type": "prev_best", "score": prev_score})

    # Run parallel optimizer
    emit({"type": "solving", "msg": f"Running {num_workers} parallel workers ({time_limit:.0f}s budget)..."})

    try:
        score, actions = parallel_optimize(
            capture_data=capture,
            difficulty=difficulty,
            time_limit=time_limit,
            num_workers=num_workers,
            verbose=True,  # prints to stderr
        )
    except Exception as e:
        emit({"type": "error", "msg": f"Solver failed: {e}"})
        return

    elapsed = time.time() - t0

    emit({
        "type": "result",
        "score": score,
        "time": round(elapsed, 1),
        "num_bots": num_bots,
        "optimal": False,
    })

    # Save solution
    if score > 0:
        improved = score > prev_score
        try:
            save_solution(difficulty, score, actions, force=True)
        except Exception as e:
            emit({"type": "error", "msg": f"Failed to save solution: {e}"})

        if improved:
            emit({"type": "improved", "old_score": prev_score, "new_score": score, "delta": score - prev_score})
        else:
            emit({"type": "no_improvement", "score": score, "prev": prev_score})

    emit({"type": "done", "score": score, "time": round(elapsed, 1)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parallel solve with streaming output')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert', 'nightmare'])
    parser.add_argument('--time', type=float, default=240.0, help='Time budget in seconds')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    solve(args.difficulty, time_limit=args.time, num_workers=args.workers)
