"""Competition-day orchestration: capture -> solve -> replay -> improve -> replay.

Full pipeline for maximizing score within the 5-minute key window:
1. Capture orders using probe game (120s)
2. Parallel solve with all CPU cores + extended time (240s budget)
3. Replay best solution
4. If time remains, continue optimizing and replay better solutions
5. Cache solutions by (difficulty, order_hash) for reuse

Usage:
    python competition_runner.py <ws_url> <difficulty> [--time <seconds>]
    python competition_runner.py <ws_url> all [--time <seconds>]
"""
import asyncio
import json
import sys
import os
import time
import hashlib
from pathlib import Path

SOLUTION_DIR = Path('solutions')
CAPTURE_DIR = Path('captures')


def order_hash(capture_data):
    """Hash the order sequence for caching."""
    orders = capture_data.get('orders', [])
    order_str = json.dumps(orders, sort_keys=True)
    return hashlib.md5(order_str.encode()).hexdigest()[:12]


def get_cached_solution(difficulty, ohash):
    """Check for a cached solution with this order hash."""
    sol_dir = SOLUTION_DIR / difficulty
    if not sol_dir.exists():
        return None

    meta_path = sol_dir / f'{ohash}_meta.json'
    actions_path = sol_dir / f'{ohash}_actions.json'

    if meta_path.exists() and actions_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        with open(actions_path) as f:
            actions = json.load(f)
        return {
            'score': meta['score'],
            'actions': [[(a, i) for a, i in ra] for ra in actions],
            'meta': meta,
        }
    return None


def save_solution(difficulty, ohash, score, actions, meta=None):
    """Save solution to cache."""
    sol_dir = SOLUTION_DIR / difficulty
    sol_dir.mkdir(parents=True, exist_ok=True)

    actions_path = sol_dir / f'{ohash}_actions.json'
    meta_path = sol_dir / f'{ohash}_meta.json'

    serializable = [[(int(a), int(i)) for a, i in ra] for ra in actions]
    with open(actions_path, 'w') as f:
        json.dump(serializable, f)

    meta_data = meta or {}
    meta_data.update({
        'score': score,
        'difficulty': difficulty,
        'order_hash': ohash,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

    # Also update "best" symlink
    best_meta = sol_dir / 'best_meta.json'
    best_actions = sol_dir / 'best_actions.json'

    update_best = True
    if best_meta.exists():
        with open(best_meta) as f:
            existing = json.load(f)
        if existing.get('order_hash') == ohash and existing.get('score', 0) >= score:
            update_best = False

    if update_best:
        with open(best_meta, 'w') as f:
            json.dump(meta_data, f, indent=2)
        with open(best_actions, 'w') as f:
            json.dump(serializable, f)


async def capture_orders(ws_url, difficulty, verbose=True):
    """Capture orders by running a probe game. Returns capture_data dict."""
    from capture_game import capture_game
    capture = await capture_game(ws_url, difficulty, verbose=verbose)
    return capture


def solve_offline(capture_data, time_limit=240.0, verbose=True, num_workers=None):
    """Solve using captured orders with parallel optimizer."""
    from game_engine import init_game_from_capture
    from multi_solve import multi_solve

    difficulty = capture_data['difficulty']

    def game_factory():
        return init_game_from_capture(capture_data)

    score, actions = multi_solve(
        difficulty=difficulty,
        time_limit=time_limit,
        verbose=verbose,
        game_factory=game_factory,
        parallel=True,
        num_workers=num_workers,
    )

    return score, actions


async def replay_solution(ws_url, actions, difficulty, capture_data=None, verbose=True):
    """Replay a pre-computed solution on the server."""
    from ws_client import replay
    if capture_data:
        from game_engine import build_map_from_capture
        ms = build_map_from_capture(capture_data)
    else:
        from game_engine import build_map
        ms = build_map(difficulty)

    score = await replay(ws_url, actions, ms, verbose=verbose)
    return score


async def run_competition(ws_url, difficulty, total_time=280.0, num_workers=None, verbose=True):
    """Run the full competition pipeline for one difficulty.

    Timeline (280s total, within 5-min key window):
    - Phase 1 (0-120s): Capture orders via probe game
    - Phase 2 (120-240s): Solve offline with parallel optimizer
    - Phase 3 (240-280s): Replay best solution

    If a cached solution exists for the same orders, skip to replay immediately.
    """
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  COMPETITION: {difficulty.upper()}")
    print(f"  Time budget: {total_time}s")
    print(f"{'='*60}\n")

    # Phase 1: Capture orders
    print("[Phase 1] Capturing orders...")
    try:
        capture = await capture_orders(ws_url, difficulty, verbose=verbose)
    except Exception as e:
        print(f"  CAPTURE FAILED: {e}")
        print("  Falling back to Zig bot replay")
        return 0

    capture_time = time.time() - t0
    print(f"  Captured {len(capture.get('orders', []))} orders in {capture_time:.1f}s")
    print(f"  Probe score: {capture.get('probe_score', 'N/A')}")

    # Check cache
    ohash = order_hash(capture)
    cached = get_cached_solution(difficulty, ohash)
    if cached:
        print(f"\n  CACHED solution found! score={cached['score']} (hash={ohash})")
        best_score = cached['score']
        best_actions = cached['actions']
    else:
        best_score = 0
        best_actions = None

    # Phase 2: Solve offline
    remaining = total_time - (time.time() - t0)
    solve_budget = max(30, remaining - 40)  # leave 40s for replay

    if solve_budget > 30:
        print(f"\n[Phase 2] Solving offline ({solve_budget:.0f}s budget)...")
        try:
            score, actions = solve_offline(
                capture, time_limit=solve_budget,
                verbose=verbose, num_workers=num_workers
            )
            if score > best_score:
                print(f"  New best: {score} (was {best_score})")
                best_score = score
                best_actions = actions
                save_solution(difficulty, ohash, score, actions, {
                    'probe_score': capture.get('probe_score', 0),
                    'num_orders': len(capture.get('orders', [])),
                })
            else:
                print(f"  Cached solution still better ({best_score} vs {score})")
        except Exception as e:
            print(f"  SOLVE FAILED: {e}")

    if best_actions is None:
        print("\n  NO SOLUTION FOUND")
        return 0

    # Phase 3: Replay
    remaining = total_time - (time.time() - t0)
    if remaining < 5:
        print(f"\n  NOT ENOUGH TIME to replay ({remaining:.0f}s left)")
        print(f"  Best offline score: {best_score}")
        return best_score

    print(f"\n[Phase 3] Replaying best solution (score={best_score})...")
    try:
        server_score = await replay_solution(
            ws_url, best_actions, difficulty, capture, verbose=verbose
        )
        print(f"\n  SERVER SCORE: {server_score}")
        elapsed = time.time() - t0
        print(f"  Total time: {elapsed:.1f}s")
        return server_score
    except Exception as e:
        print(f"  REPLAY FAILED: {e}")
        return best_score


async def run_all_difficulties(ws_url, total_time=1200.0, num_workers=None, verbose=True):
    """Run competition for all 4 difficulties sequentially."""
    per_diff_time = total_time / 4
    results = {}

    for diff in ['easy', 'medium', 'hard', 'expert']:
        score = await run_competition(
            ws_url, diff, total_time=per_diff_time,
            num_workers=num_workers, verbose=verbose
        )
        results[diff] = score

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    total = 0
    for diff, score in results.items():
        print(f"  {diff:8s}: {score}")
        total += score
    print(f"  {'TOTAL':8s}: {total}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Competition runner')
    parser.add_argument('ws_url', help='WebSocket URL (e.g., wss://game.ainm.no/ws?token=...)')
    parser.add_argument('difficulty', help='Difficulty (easy/medium/hard/expert/all)')
    parser.add_argument('--time', type=float, default=280.0, help='Time budget per difficulty (default: 280s)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    if args.difficulty == 'all':
        asyncio.run(run_all_difficulties(
            args.ws_url, total_time=args.time * 4,
            num_workers=args.workers, verbose=True
        ))
    else:
        asyncio.run(run_competition(
            args.ws_url, args.difficulty, total_time=args.time,
            num_workers=args.workers, verbose=True
        ))
