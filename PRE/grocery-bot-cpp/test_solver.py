"""Generate solver test input from game_engine and verify output."""
import sys, json, subprocess, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'grocery-bot-gpu'))
from game_engine import *
from configs import CONFIGS, DIFF_ROUNDS

def gen_solver_input(difficulty='easy', seed=42, num_orders=50):
    """Generate solver input text from game_engine simulation."""
    state, all_orders = init_game(seed, difficulty)
    ms = state.map_state
    cfg = CONFIGS[difficulty]
    max_rounds = DIFF_ROUNDS[difficulty]

    lines = []
    lines.append(f"{ms.width} {ms.height} {cfg['bots']} {max_rounds}")
    lines.append(str(len(ms.drop_off_zones)))
    for dz in ms.drop_off_zones:
        lines.append(f"{dz[0]} {dz[1]}")
    lines.append(f"{ms.spawn[0]} {ms.spawn[1]}")

    # Grid
    for y in range(ms.height):
        row = ''
        for x in range(ms.width):
            c = ms.grid[y, x]
            if c == CELL_FLOOR: row += '.'
            elif c == CELL_WALL: row += '#'
            elif c == CELL_SHELF: row += 'S'
            elif c == CELL_DROPOFF: row += 'D'
            else: row += '.'
        lines.append(row)

    # Items (sorted by x,y)
    items_sorted = sorted(range(ms.num_items),
                          key=lambda i: (int(ms.item_positions[i,0]), int(ms.item_positions[i,1])))
    lines.append(str(ms.num_items))
    for idx in items_sorted:
        tid = int(ms.item_types[idx])
        x, y = int(ms.item_positions[idx, 0]), int(ms.item_positions[idx, 1])
        lines.append(f"{tid} {x} {y}")

    lines.append(str(ms.num_types))

    # Orders
    n_orders = min(num_orders, len(all_orders))
    lines.append(str(n_orders))
    for i in range(n_orders):
        o = all_orders[i]
        ids = [int(r) for r in o.required]
        lines.append(f"{len(ids)} " + " ".join(str(x) for x in ids))

    # One-way row
    if difficulty == 'nightmare':
        lines.append(str(ms.height - 2))
    else:
        lines.append("-1")

    return "\n".join(lines) + "\n", state, all_orders


def verify_plan(plan_text, state, all_orders, difficulty):
    """Verify the plan using game_engine simulation."""
    lines = plan_text.strip().split('\n')
    header = lines[0].split()
    num_rounds = int(header[0])
    num_bots = int(header[1])

    max_rounds = DIFF_ROUNDS[difficulty]

    # Reset state
    from game_engine import init_game
    state, all_orders = init_game(42, difficulty, 100)

    for r in range(min(num_rounds, max_rounds)):
        if r + 1 >= len(lines):
            break
        vals = list(map(int, lines[r + 1].split()))
        actions = []
        for b in range(num_bots):
            act = vals[b * 2]
            arg = vals[b * 2 + 1]
            actions.append((act, arg))
        step(state, actions, all_orders)

    return state.score, state.orders_completed, state.items_delivered


def main():
    difficulties = ['easy', 'medium', 'hard']
    if len(sys.argv) > 1:
        difficulties = sys.argv[1:]

    solver_exe = Path(__file__).parent / 'solver.exe'
    if not solver_exe.exists():
        print("solver.exe not found, compile first")
        return

    for diff in difficulties:
        print(f"\n=== Testing {diff} ===")
        inp, state, all_orders = gen_solver_input(diff, seed=42, num_orders=50)

        # Save input for debugging
        Path(f'test_input_{diff}.txt').write_text(inp)

        # Run solver
        t0 = time.time()
        result = subprocess.run(
            [str(solver_exe)],
            input=inp, capture_output=True, text=True, timeout=120
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:200]}")
            continue

        print(f"  Solver time: {elapsed:.1f}s")
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                print(f"  [solver] {line}")

        # Verify with game engine
        score, orders_done, items = verify_plan(result.stdout, state, all_orders, diff)
        print(f"  Verified: score={score}, orders={orders_done}, items={items}")

        Path(f'test_plan_{diff}.txt').write_text(result.stdout)


if __name__ == '__main__':
    main()
