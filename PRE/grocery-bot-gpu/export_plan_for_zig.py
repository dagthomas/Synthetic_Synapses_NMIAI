"""Export GPU DP solution to a format the Zig bot can replay.

Saves dp_plan to PostgreSQL (dp_plans table). Also writes to disk
for Zig bot consumption if --output is specified.

Usage:
    python export_plan_for_zig.py <difficulty> [--output dp_plan.json]
"""
import json
import sys

from solution_store import load_solution, load_capture, save_dp_plan
from game_engine import build_map_from_capture, init_game_from_capture, step, ACT_PICKUP
from configs import DIFF_ROUNDS

ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else 'medium'
    output = None
    if len(sys.argv) > 2 and sys.argv[2] != '--output':
        output = sys.argv[2]
    elif '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output = sys.argv[idx + 1]

    actions = load_solution(difficulty)
    capture = load_capture(difficulty)

    if actions is None:
        print(f"No solution found for {difficulty}", file=sys.stderr)
        sys.exit(1)
    if capture is None:
        print(f"No capture found for {difficulty}", file=sys.stderr)
        sys.exit(1)

    map_state = build_map_from_capture(capture)
    num_bots = len(actions[0])
    num_rounds = min(len(actions), DIFF_ROUNDS.get(difficulty, 300))

    # Simulate to get expected positions per round
    gs, all_orders = init_game_from_capture(capture)
    expected_positions = []
    for rnd in range(num_rounds):
        pos = []
        for bid in range(num_bots):
            pos.append([int(gs.bot_positions[bid, 0]), int(gs.bot_positions[bid, 1])])
        expected_positions.append(pos)
        step(gs, actions[rnd], all_orders)

    # Build per-round action data
    rounds = []
    for rnd in range(num_rounds):
        bot_actions = []
        for bid in range(num_bots):
            act_type, item_idx = actions[rnd][bid]
            entry = {
                'action': ACTION_NAMES[act_type],
            }
            if act_type == ACT_PICKUP and 0 <= item_idx < len(map_state.items):
                entry['item_id'] = map_state.items[item_idx]['id']
            bot_actions.append(entry)
        rounds.append({
            'positions': expected_positions[rnd],
            'actions': bot_actions,
        })

    plan = {
        'difficulty': difficulty,
        'num_bots': num_bots,
        'num_rounds': num_rounds,
        'score': gs.score,
        'rounds': rounds,
    }

    # Save to DB
    save_dp_plan(difficulty, plan)
    print(f"Saved DP plan to DB: {difficulty}, {num_rounds} rounds, {num_bots} bots", file=sys.stderr)

    # Optionally write to disk
    if output:
        with open(output, 'w') as f:
            json.dump(plan, f)
        import os
        print(f"Also written to {output} ({os.path.getsize(output) / 1024:.1f} KB)", file=sys.stderr)

    print(f"Expected score: {gs.score}", file=sys.stderr)


if __name__ == '__main__':
    main()
