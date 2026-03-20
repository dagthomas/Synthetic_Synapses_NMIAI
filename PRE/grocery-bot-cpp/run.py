"""Orchestrator: probe → solve → replay pipeline.

Usage:
    python run.py <ws_url> [--difficulty easy|medium|hard|expert|nightmare]
    python run.py solve capture.json          # offline solve only
    python run.py replay <ws_url> plan.txt    # replay only
"""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Import client module
sys.path.insert(0, str(Path(__file__).parent))
from client import capture_game, replay_game, probe_game

ZIG = r"C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe"
SOLVER_SRC = Path(__file__).parent / "solver.cpp"
SOLVER_EXE = Path(__file__).parent / "solver.exe"

CONFIGS = {
    "easy":      {"w": 12, "h": 10, "bots": 1,  "aisles": 2, "types": 4,  "rounds": 300},
    "medium":    {"w": 16, "h": 12, "bots": 3,  "aisles": 3, "types": 8,  "rounds": 300},
    "hard":      {"w": 22, "h": 14, "bots": 5,  "aisles": 4, "types": 12, "rounds": 300},
    "expert":    {"w": 28, "h": 18, "bots": 10, "aisles": 5, "types": 16, "rounds": 300},
    "nightmare": {"w": 30, "h": 18, "bots": 20, "aisles": 6, "types": 21, "rounds": 500, "dropoffs": 3},
}


def detect_difficulty(capture):
    """Detect difficulty from capture data."""
    nb = capture['num_bots']
    w = capture['width']
    for name, cfg in CONFIGS.items():
        if cfg['bots'] == nb and cfg['w'] == w:
            return name
    return 'hard'  # fallback


def compile_solver():
    """Compile solver.cpp using zig c++."""
    if SOLVER_EXE.exists():
        src_mtime = SOLVER_SRC.stat().st_mtime
        exe_mtime = SOLVER_EXE.stat().st_mtime
        if exe_mtime > src_mtime:
            print("Solver already compiled (up to date)", file=sys.stderr)
            return True

    print("Compiling solver.cpp...", file=sys.stderr)
    cmd = [ZIG, "c++", "-O2", "-std=c++20",
           str(SOLVER_SRC), "-o", str(SOLVER_EXE)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr}", file=sys.stderr)
        return False
    print("Compiled successfully", file=sys.stderr)
    return True


def capture_to_solver_input(capture):
    """Convert capture JSON to solver text input format."""
    w = capture['width']
    h = capture['height']
    nb = capture['num_bots']
    difficulty = detect_difficulty(capture)
    cfg = CONFIGS[difficulty]
    max_rounds = cfg['rounds']

    dz = capture.get('drop_off_zones', [capture['drop_off']])
    # Ensure list of lists
    if dz and not isinstance(dz[0], list):
        dz = [dz]

    spawn_x = w - 2
    spawn_y = h - 2

    lines = []
    lines.append(f"{w} {h} {nb} {max_rounds}")
    lines.append(str(len(dz)))
    for z in dz:
        lines.append(f"{z[0]} {z[1]}")
    lines.append(f"{spawn_x} {spawn_y}")

    # Build grid
    walls_set = set()
    for wx, wy in capture['walls']:
        walls_set.add((wx, wy))

    shelves_set = set()
    for item in capture['items']:
        pos = item['position']
        shelves_set.add((pos[0], pos[1]))

    dz_set = set()
    for z in dz:
        dz_set.add((z[0], z[1]))

    for y in range(h):
        row = ''
        for x in range(w):
            if (x, y) in dz_set:
                row += 'D'
            elif (x, y) in shelves_set:
                row += 'S'
            elif (x, y) in walls_set:
                row += '#'
            else:
                row += '.'
        lines.append(row)

    # Items (sorted by position, matching solver expectation)
    items_sorted = sorted(capture['items'],
                          key=lambda it: (it['position'][0], it['position'][1]))

    # Build type name → ID mapping
    type_names = sorted(set(it['type'] for it in items_sorted))
    type_to_id = {name: i for i, name in enumerate(type_names)}

    lines.append(str(len(items_sorted)))
    for item in items_sorted:
        tid = type_to_id[item['type']]
        x, y = item['position']
        lines.append(f"{tid} {x} {y}")

    lines.append(str(len(type_names)))

    # Orders
    order_list = capture.get('orders', [])
    lines.append(str(len(order_list)))
    for order in order_list:
        types = order['items_required']
        ids = [type_to_id.get(t, 0) for t in types]
        lines.append(f"{len(ids)} " + " ".join(str(i) for i in ids))

    # One-way row: enable for nightmare (bottom corridor)
    if difficulty == 'nightmare':
        lines.append(str(h - 2))  # one_way_row = bottom corridor
    else:
        lines.append("-1")

    return "\n".join(lines) + "\n", type_names


def run_solver(solver_input, timeout=60):
    """Run the C++ solver and return the plan text."""
    print("Running solver...", file=sys.stderr)
    t0 = time.time()

    result = subprocess.run(
        [str(SOLVER_EXE)],
        input=solver_input,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    elapsed = time.time() - t0
    print(f"Solver finished in {elapsed:.1f}s", file=sys.stderr)

    if result.returncode != 0:
        print(f"Solver error:\n{result.stderr}", file=sys.stderr)
        return None

    if result.stderr:
        print(f"Solver log:\n{result.stderr}", file=sys.stderr)

    return result.stdout


def solve_capture(capture_file, plan_file='plan.txt'):
    """Offline solve from capture file."""
    capture = json.loads(Path(capture_file).read_text())
    difficulty = detect_difficulty(capture)
    print(f"Difficulty: {difficulty}", file=sys.stderr)

    if not compile_solver():
        return False

    solver_input, type_names = capture_to_solver_input(capture)

    # Save solver input for debugging
    Path('solver_input.txt').write_text(solver_input)

    plan = run_solver(solver_input)
    if plan is None:
        return False

    Path(plan_file).write_text(plan)
    print(f"Plan saved to {plan_file}", file=sys.stderr)
    return True


async def full_pipeline(ws_url, difficulty=None, iterations=3):
    """Full pipeline: capture → solve → replay."""

    # Step 1: Compile solver
    if not compile_solver():
        print("Failed to compile solver", file=sys.stderr)
        return

    # Step 2: Capture game data (probe with greedy)
    capture_file = 'capture.json'
    print("\n=== PHASE 1: Capture (greedy probe) ===", file=sys.stderr)
    capture = await capture_game(ws_url, capture_file)
    if not capture:
        print("Capture failed", file=sys.stderr)
        return

    difficulty = detect_difficulty(capture)
    print(f"Detected difficulty: {difficulty}", file=sys.stderr)
    print(f"Orders discovered: {len(capture.get('orders', []))}", file=sys.stderr)

    # Step 3: Solve
    print("\n=== PHASE 2: Offline Solve ===", file=sys.stderr)
    solver_input, type_names = capture_to_solver_input(capture)
    Path('solver_input.txt').write_text(solver_input)

    plan = run_solver(solver_input)
    if plan is None:
        print("Solver failed", file=sys.stderr)
        return

    plan_file = 'plan.txt'
    Path(plan_file).write_text(plan)

    # Step 4: Replay
    # Need a new game token/URL for replay
    print("\n=== PHASE 3: Replay ===", file=sys.stderr)
    print("Provide a new WebSocket URL to replay the plan.", file=sys.stderr)
    print(f"Run: python client.py replay <new_ws_url> --plan {plan_file}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'solve':
        # Offline solve from capture file
        capture_file = sys.argv[2] if len(sys.argv) > 2 else 'capture.json'
        plan_file = sys.argv[3] if len(sys.argv) > 3 else 'plan.txt'
        solve_capture(capture_file, plan_file)

    elif cmd == 'replay':
        # Replay mode
        ws_url = sys.argv[2]
        plan_file = sys.argv[3] if len(sys.argv) > 3 else 'plan.txt'
        asyncio.run(replay_game(ws_url, plan_file))

    elif cmd == 'compile':
        compile_solver()

    elif cmd.startswith('wss://') or cmd.startswith('ws://'):
        # Full pipeline
        ws_url = cmd
        asyncio.run(full_pipeline(ws_url))

    else:
        print(__doc__)


if __name__ == '__main__':
    main()
