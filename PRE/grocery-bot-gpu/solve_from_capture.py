"""Full pipeline: load captured server state, solve offline, save for replay.

Usage:
    python solve_from_capture.py <difficulty> [--time <seconds>] [--date <YYYY-MM-DD>]
    python solve_from_capture.py <capture_file.json> [--time <seconds>]

If no capture file specified, looks for captures/<difficulty>_<today>.json.
"""
import sys
import time
import json
from game_engine import init_game_from_capture, build_map_from_capture
from multi_solve import multi_solve
from ws_client import save_actions
from capture_game import load_capture, get_capture_path


def solve_from_capture(capture_data, time_limit=300.0, verbose=True):
    """Solve using captured server state."""
    t0 = time.time()
    difficulty = capture_data['difficulty']

    def game_factory():
        return init_game_from_capture(capture_data)

    if verbose:
        print(f"Solving {difficulty} from capture")
        print(f"  Orders captured: {len(capture_data['orders'])}")
        print(f"  Items: {len(capture_data['items'])}")
        print(f"  Probe score: {capture_data.get('probe_score', 'N/A')}")
        print(f"  Time limit: {time_limit:.0f}s")
        print()

    score, actions = multi_solve(
        difficulty=difficulty,
        time_limit=time_limit,
        verbose=verbose,
        game_factory=game_factory,
    )

    # Save solution with capture metadata
    filename = f'solution_{difficulty}_captured.json'
    save_actions(actions, filename)

    # Also save the map state info needed for replay
    map_state = build_map_from_capture(capture_data)
    meta = {
        'difficulty': difficulty,
        'score': score,
        'capture_file': capture_data.get('captured_at', ''),
        'probe_score': capture_data.get('probe_score', 0),
        'num_items': len(capture_data['items']),
        'num_orders_captured': len(capture_data['orders']),
    }
    meta_file = f'solution_{difficulty}_captured_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        probe_score = capture_data.get('probe_score', 0)
        print(f"\nFinal score: {score}")
        print(f"Improvement over probe: {score - probe_score:+d} ({probe_score} -> {score})")
        print(f"Saved to: {filename}")
        print(f"Total time: {time.time()-t0:.1f}s")
        print(f"\nTo replay:")
        print(f"  python ws_client.py <ws_url> {filename} --capture captures/{difficulty}_<date>.json")

    return score, filename


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python solve_from_capture.py <difficulty> [--time <secs>] [--date <YYYY-MM-DD>]")
        print("       python solve_from_capture.py <capture_file.json> [--time <secs>]")
        sys.exit(1)

    arg1 = sys.argv[1]
    time_lim = 300.0
    date_str = None

    for i, arg in enumerate(sys.argv):
        if arg == '--time' and i + 1 < len(sys.argv):
            time_lim = float(sys.argv[i + 1])
        if arg == '--date' and i + 1 < len(sys.argv):
            date_str = sys.argv[i + 1]

    # Load capture
    if arg1.endswith('.json'):
        with open(arg1) as f:
            capture = json.load(f)
    else:
        difficulty = arg1
        capture = load_capture(difficulty, date_str)
        if capture is None:
            path = get_capture_path(difficulty, date_str)
            print(f"No capture found at {path}")
            print(f"Run a probe first: python capture_game.py <ws_url> {difficulty}")
            sys.exit(1)

    solve_from_capture(capture, time_limit=time_lim)
