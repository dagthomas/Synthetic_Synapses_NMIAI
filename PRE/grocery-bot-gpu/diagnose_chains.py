"""Diagnostic: analyze V4's chain reaction potential.

For each order completion, shows:
- What the new active (was preview) needs
- What bots at dropoffs are carrying (INCLUDING deliver bot leftovers)
- How many items auto-delivered vs missing
- Gap to chain completion
"""
from game_engine import init_game, step, Order, ACT_DROPOFF
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver


def run_diagnostic(seed: int):
    state, all_orders = init_game(seed, 'nightmare', num_orders=200)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders)
    num_rounds = DIFF_ROUNDS['nightmare']
    drop_set = set(tuple(dz) for dz in ms.drop_off_zones)

    type_names = {}
    for name, tid in ms.type_name_to_id.items():
        type_names[tid] = name

    print(f"Dropoff zones: {sorted(drop_set)}")

    total_gap = 0
    chain_events = 0
    completions = 0

    for rnd in range(num_rounds):
        state.round = rnd
        active_before = state.get_active_order()
        preview_before = state.get_preview_order()
        o_before = state.orders_completed

        actions = solver.action(state, all_orders, rnd)

        # Find delivering bots and their actions
        delivering = []
        for bid, (act, arg) in enumerate(actions):
            if act == ACT_DROPOFF:
                pos = (int(state.bot_positions[bid, 0]),
                       int(state.bot_positions[bid, 1]))
                inv = state.bot_inv_list(bid)
                delivering.append((bid, pos, list(inv)))

        # Check if this delivery will complete the active order
        if delivering and active_before and preview_before:
            # What does active still need?
            active_needs_list = list(active_before.needs())
            active_needs = {}
            for t in active_needs_list:
                active_needs[t] = active_needs.get(t, 0) + 1

            # What will be delivered this round?
            for bid, pos, inv in delivering:
                for t in inv:
                    if t in active_needs and active_needs[t] > 0:
                        active_needs[t] -= 1

            remaining = sum(max(0, v) for v in active_needs.values())

            if remaining == 0:
                # Order WILL complete! Analyze chain potential
                completions += 1
                preview_needs_list = list(preview_before.needs())
                preview_needs = {}
                for t in preview_needs_list:
                    preview_needs[t] = preview_needs.get(t, 0) + 1

                # Count what's at dropoffs for the chain
                # Include: staging bots + deliver bot LEFTOVERS
                chain_items = {}  # type -> count
                delivering_bids = {b[0] for b in delivering}

                # Staging bots at dropoffs (not delivering)
                staging_bots = []
                for b in range(len(state.bot_positions)):
                    if b in delivering_bids:
                        continue
                    bp = (int(state.bot_positions[b, 0]),
                          int(state.bot_positions[b, 1]))
                    inv = state.bot_inv_list(b)
                    if bp in drop_set and inv:
                        staging_bots.append((b, bp, list(inv)))
                        for t in inv:
                            chain_items[t] = chain_items.get(t, 0) + 1

                # Deliver bot leftovers (items not matching active)
                for bid, pos, inv in delivering:
                    active_copy = list(active_before.needs())
                    leftover = []
                    for t in inv:
                        if t in active_copy:
                            active_copy.remove(t)
                        else:
                            leftover.append(t)
                    for t in leftover:
                        chain_items[t] = chain_items.get(t, 0) + 1

                # How many preview items would auto-deliver?
                matched = 0
                for t, need in preview_needs.items():
                    matched += min(need, chain_items.get(t, 0))

                gap = len(preview_needs_list) - matched
                total_gap += gap

                # Bots with preview items NOT at dropoff
                near_drop = []
                for b in range(len(state.bot_positions)):
                    if b in delivering_bids:
                        continue
                    bp = (int(state.bot_positions[b, 0]),
                          int(state.bot_positions[b, 1]))
                    if bp in drop_set:
                        continue
                    inv = state.bot_inv_list(b)
                    matching = [t for t in inv if t in preview_needs]
                    if matching:
                        d = min(tables.get_distance(bp, dz) for dz in drop_set)
                        near_drop.append((b, d, [type_names.get(t, f'?{t}') for t in matching]))

                need_str = ', '.join(f"{type_names.get(t, f'?{t}')}×{n}"
                                     for t, n in sorted(preview_needs.items()))
                chain_str = ', '.join(f"{type_names.get(t, f'?{t}')}×{n}"
                                      for t, n in sorted(chain_items.items()) if t in preview_needs)

                marker = " *** CHAIN! ***" if gap == 0 else ""
                print(f"\nR{rnd:3d} ORDER #{o_before+1} COMPLETING{marker}")
                print(f"  Preview needs ({len(preview_needs_list)}): {need_str}")
                print(f"  At dropoffs: {chain_str}  [matched {matched}/{len(preview_needs_list)}, gap={gap}]")
                if staging_bots:
                    for b, bp, inv in staging_bots:
                        print(f"    staging b{b}@{bp}: {[type_names.get(t,'?') for t in inv]}")
                if near_drop:
                    near_drop.sort(key=lambda x: x[1])
                    for b, d, mtypes in near_drop[:5]:
                        print(f"    NEAR b{b} dist={d}: {mtypes}")
                if gap == 0:
                    chain_events += 1

        step(state, actions, all_orders)
        c = state.orders_completed - o_before
        if c > 1:
            print(f"R{rnd} ACTUAL CHAIN x{c}!")

    print(f"\n{'='*60}")
    print(f"Score: {state.score}  Orders: {state.orders_completed}")
    print(f"Completions analyzed: {completions}")
    print(f"Chain possible: {chain_events}/{completions}")
    print(f"Average gap: {total_gap/max(completions,1):.1f} items")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='7005')
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC: Seed {seed} (V4 baseline)")
        print(f"{'='*60}")
        run_diagnostic(seed)
