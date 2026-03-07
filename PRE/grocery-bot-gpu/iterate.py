"""Iterative capture-solve-replay loop for competition day.

Each iteration:
  1. Replay the best solution (which captures new orders from the server)
  2. Re-solve with GPU using the enriched capture data
  3. Save if improved
  4. Prompt for the next game URL

First iteration (no existing solution):
  1. Run a probe game to capture orders
  2. Solve with GPU
  3. Save

Usage:
    python iterate.py <difficulty> [--solver gpu|cpu] [--max-states N]
"""
import argparse
import asyncio
import sys
import time

from solution_store import (
    load_capture, load_solution, load_meta, save_capture, save_solution,
)


def have_existing_solution(difficulty):
    """Check if we have a useful solution + capture ready for replay."""
    sol = load_solution(difficulty)
    cap = load_capture(difficulty)
    if sol is None or cap is None:
        return False
    # A score-0 solution is all WAITs — not worth replaying
    meta = load_meta(difficulty)
    if meta and meta.get('score', 0) == 0:
        return False
    return True


def prompt_url(iteration):
    """Prompt user for a WebSocket URL. Returns URL or None to quit."""
    print()
    if iteration > 0:
        print(">> 60s cooldown before next game! Wait for the timer on the website. <<")
        print()
    label = f"[Iteration {iteration}] Paste wss://... URL (or 'q' to quit): "
    try:
        url = input(label).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not url or url.lower() == 'q':
        return None
    return url


async def do_capture(ws_url, difficulty):
    """Run a probe game to capture orders. Returns capture dict."""
    from capture_game import capture_and_play
    capture = await capture_and_play(ws_url, difficulty)
    # Also save to solution_store so replay_best can find it
    save_capture(difficulty, capture)
    return capture


async def do_replay(ws_url, difficulty):
    """Replay best solution (also captures new orders). Returns server score."""
    from replay_solution import replay_best
    score = await replay_best(ws_url, difficulty=difficulty)
    return score


def do_solve(difficulty, device, max_states, warm_start=False):
    """Run GPU/CPU solver. Returns (score, actions)."""
    from gpu_sequential_solver import solve_sequential, refine_from_solution

    capture = load_capture(difficulty)
    if capture is None:
        print("ERROR: No capture data found!", file=sys.stderr)
        return None, None

    if warm_start:
        existing = load_solution(difficulty)
        if existing is not None:
            print(f"  Warm-starting from existing solution...")
            score, actions = refine_from_solution(
                existing,
                capture_data=capture,
                device=device,
                max_states=max_states,
            )
            return score, actions

    # Fresh solve
    score, actions = solve_sequential(
        capture_data=capture,
        device=device,
        max_states=max_states,
    )
    return score, actions


def print_status(difficulty):
    """Print current solution status."""
    meta = load_meta(difficulty)
    capture = load_capture(difficulty)
    if meta:
        orders = len(capture['orders']) if capture else '?'
        print(f"  Current best: score={meta['score']}, orders={orders}, "
              f"optimizations={meta.get('optimizations_run', 0)}")
    else:
        print(f"  No existing solution for '{difficulty}'")


def main():
    parser = argparse.ArgumentParser(description='Iterative capture-solve-replay loop')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--solver', choices=['gpu', 'cpu'], default='gpu',
                        help='Solver device (default: gpu)')
    parser.add_argument('--max-states', type=int, default=500000,
                        help='Max DP states per bot (default: 500000)')
    args = parser.parse_args()

    difficulty = args.difficulty
    device = 'cuda' if args.solver == 'gpu' else 'cpu'

    print(f"=== Iterate: {difficulty} (device={device}) ===")
    print_status(difficulty)

    iteration = 0
    while True:
        has_solution = have_existing_solution(difficulty)

        if not has_solution and iteration > 0:
            print("ERROR: Lost solution data?", file=sys.stderr)
            break

        ws_url = prompt_url(iteration)
        if ws_url is None:
            break

        # Track orders before
        cap_before = load_capture(difficulty)
        orders_before = len(cap_before['orders']) if cap_before else 0
        meta_before = load_meta(difficulty)
        score_before = meta_before['score'] if meta_before else 0

        if not has_solution:
            # First run: capture probe game
            print(f"\n--- Iteration {iteration}: PROBE CAPTURE ---")
            t0 = time.time()
            asyncio.run(do_capture(ws_url, difficulty))
            elapsed = time.time() - t0
            print(f"  Capture done in {elapsed:.1f}s")

            cap_after = load_capture(difficulty)
            orders_after = len(cap_after['orders']) if cap_after else 0
            print(f"  Orders captured: {orders_after}")

            # Solve from scratch
            print(f"\n  Solving from scratch...")
            t0 = time.time()
            score, actions = do_solve(difficulty, device, args.max_states, warm_start=False)
            elapsed = time.time() - t0

            if score is not None:
                saved = save_solution(difficulty, score, actions, force=True)
                print(f"\n  Solve done in {elapsed:.1f}s")
                print(f"  Score: {score} ({'saved' if saved else 'not saved'})")
            else:
                print("  Solve failed!", file=sys.stderr)

        else:
            # Replay existing solution (captures new orders)
            print(f"\n--- Iteration {iteration}: REPLAY + RE-SOLVE ---")
            t0 = time.time()
            replay_score = asyncio.run(do_replay(ws_url, difficulty))
            elapsed = time.time() - t0
            print(f"\n  Replay done in {elapsed:.1f}s, server score: {replay_score}")

            # Check for new orders
            cap_after = load_capture(difficulty)
            orders_after = len(cap_after['orders']) if cap_after else 0
            new_orders = orders_after - orders_before
            print(f"  Orders: {orders_before} -> {orders_after} (+{new_orders} new)")

            # Re-solve with warm start
            print(f"\n  Re-solving with warm start...")
            t0 = time.time()
            score, actions = do_solve(difficulty, device, args.max_states, warm_start=True)
            elapsed = time.time() - t0

            if score is not None:
                saved = save_solution(difficulty, score, actions)
                print(f"\n  Solve done in {elapsed:.1f}s")
                print(f"  Score: {score_before} -> {score} "
                      f"({'improved! saved' if saved else 'no improvement'})")
            else:
                print("  Solve failed!", file=sys.stderr)

        # Status after this iteration
        print()
        print_status(difficulty)
        iteration += 1

    print(f"\nDone after {iteration} iterations.")
    print_status(difficulty)


if __name__ == '__main__':
    main()
