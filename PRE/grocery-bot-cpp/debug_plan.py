"""Debug: simulate plan step-by-step and show what's happening."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'grocery-bot-gpu'))
from game_engine import *
from configs import CONFIGS, DIFF_ROUNDS

def debug_plan(difficulty='medium', seed=42):
    state, all_orders = init_game(seed, difficulty, 100)
    ms = state.map_state
    max_rounds = DIFF_ROUNDS[difficulty]

    # Load plan
    plan_file = Path(f'test_plan_{difficulty}.txt')
    lines = plan_file.read_text().strip().split('\n')
    header = lines[0].split()
    num_rounds = int(header[0])
    num_bots = int(header[1])

    action_names = ['wait', 'up', 'down', 'left', 'right', 'pickup', 'dropoff']

    for r in range(min(30, num_rounds)):
        vals = list(map(int, lines[r + 1].split()))
        actions = []
        act_strs = []
        for b in range(num_bots):
            act = vals[b * 2]
            arg = vals[b * 2 + 1]
            actions.append((act, arg))
            s = action_names[act]
            if act == 5: s += f'({arg})'
            act_strs.append(s)

        # Pre-step info
        bot_info = []
        for b in range(num_bots):
            bx, by = int(state.bot_positions[b, 0]), int(state.bot_positions[b, 1])
            inv = state.bot_inv_list(b)
            bot_info.append(f"({bx},{by}){inv}")

        active = state.get_active_order()
        active_needs = active.needs() if active else []

        score_before = state.score
        delta = step(state, actions, all_orders)

        if delta > 0 or r < 25:
            print(f"R{r:3d}: {' | '.join(act_strs):40s} bots={bot_info} "
                  f"score={state.score}(+{delta}) needs={active_needs}")

    print(f"\nFinal: score={state.score}, orders={state.orders_completed}, items={state.items_delivered}")


if __name__ == '__main__':
    diff = sys.argv[1] if len(sys.argv) > 1 else 'medium'
    debug_plan(diff)
