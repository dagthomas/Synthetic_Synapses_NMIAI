"""Run nightmare games in a loop, tracking best score.

Usage:
    python run_nightmare_loop.py           # Run games until stopped
    python run_nightmare_loop.py --count 5 # Run 5 games
"""
import json
import sys
import time
import subprocess


def fetch_token():
    """Fetch nightmare token via Playwright."""
    try:
        result = subprocess.run(
            ['python', 'fetch_token.py', 'nightmare', '--json'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            return data.get('url', '')
    except Exception as e:
        print(f"Token error: {e}", file=sys.stderr)
    return None


def run_game(ws_url):
    """Run one nightmare game, return score."""
    try:
        result = subprocess.run(
            ['python', 'nightmare_live.py', ws_url],
            capture_output=True, text=True, timeout=600
        )
        # Score from stdout JSON
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.strip())
                return data.get('score', 0)
            except json.JSONDecodeError:
                pass
        # Print stderr for diagnostics
        for line in result.stderr.split('\n'):
            if any(k in line for k in ['Score', 'SLOW', 'SKIP', 'Error']):
                print(f"  {line.strip()}", file=sys.stderr)
        return 0
    except subprocess.TimeoutExpired:
        print("  Game timed out!", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"  Game error: {e}", file=sys.stderr)
        return 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=999,
                        help='Number of games to run')
    args = parser.parse_args()

    best_score = 0
    scores = []

    for i in range(1, args.count + 1):
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Game {i}/{args.count} (best so far: {best_score})",
              file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        # Fetch token
        t0 = time.time()
        ws_url = fetch_token()
        if not ws_url:
            print("Failed to get token, waiting 65s...", file=sys.stderr)
            time.sleep(65)
            continue

        fetch_time = time.time() - t0
        print(f"Token fetched in {fetch_time:.0f}s", file=sys.stderr)

        # Run game
        game_start = time.time()
        score = run_game(ws_url)
        game_elapsed = time.time() - game_start
        scores.append(score)

        if score > best_score:
            best_score = score
            print(f"*** NEW BEST: {score} ***", file=sys.stderr)
        else:
            print(f"Score: {score} (best: {best_score})", file=sys.stderr)

        # Stats
        if len(scores) >= 2:
            import statistics
            print(f"Stats: mean={statistics.mean(scores):.0f} "
                  f"max={max(scores)} min={min(scores)} "
                  f"runs={len(scores)}", file=sys.stderr)

        # Wait for cooldown (65s from game END)
        if i < args.count:
            game_end = time.time()
            cooldown_wait = 65
            print(f"Cooldown: {cooldown_wait}s...", file=sys.stderr)
            time.sleep(cooldown_wait)

    print(json.dumps({
        'best': best_score,
        'scores': scores,
        'runs': len(scores),
    }))


if __name__ == '__main__':
    main()
