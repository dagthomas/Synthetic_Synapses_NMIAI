"""Worker subprocess for reactive_replay.py — runs GPU DP and saves result to file.

Usage (called by reactive_replay.py, not directly):
    python _reactive_solve_worker.py <capture_path> <difficulty> <result_path>
"""
import json
import sys
import time

import torch

from gpu_sequential_solver import solve_sequential


def main():
    capture_path = sys.argv[1]
    difficulty = sys.argv[2]
    result_path = sys.argv[3]

    with open(capture_path) as f:
        capture = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dedicated log handle — avoids reassigning global stderr/stdout
    log_path = result_path + '.log'
    _log_fh = open(log_path, 'w', buffering=1)  # Line-buffered

    t0 = time.time()
    try:
        print(f"Re-solving {difficulty} with {len(capture['orders'])} orders on {device}...",
              file=_log_fh, flush=True)

        score, actions = solve_sequential(
            capture_data=capture,
            difficulty=difficulty,
            device=device,
            max_states=500000,
            verbose=False,  # Minimal output for speed
            max_refine_iters=0,  # Speed over quality for reactive re-solve
        )

        elapsed = time.time() - t0
        print(f"Re-solve done: score={score}, time={elapsed:.1f}s", file=_log_fh, flush=True)
    except Exception as e:
        import traceback
        print(f"WORKER ERROR: {e}", file=_log_fh, flush=True)
        traceback.print_exc(file=_log_fh)
        _log_fh.close()
        sys.exit(1)
    finally:
        if not _log_fh.closed:
            _log_fh.close()

    # Save result as JSON
    serializable = [[(int(a), int(i)) for a, i in round_acts] for round_acts in actions]
    with open(result_path, 'w') as f:
        json.dump(serializable, f)


if __name__ == '__main__':
    main()
