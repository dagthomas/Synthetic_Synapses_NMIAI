"""Chain reaction analysis for nightmare mode LMAPF solver.

Runs the LMAPF solver on seed 7005 and tracks at every round:
- Bots at each dropoff zone and their inventories
- Active and preview order needs
- Whether a chain reaction COULD have fired (active completed + preview items staged)
- Whether a chain DID fire

Key question: at the moment each active order completes, how close were we
to having the preview order's items at dropoff zones?

Usage:
    python analyze_chains.py
"""
from __future__ import annotations

import time

from game_engine import init_game, step, GameState, Order, MapState, INV_CAP
from configs import DIFF_ROUNDS
from precompute import PrecomputedTables
from nightmare_lmapf_solver import LMAPFSolver


def analyze_chains(seed: int = 7005):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    ms = state.map_state
    tables = PrecomputedTables.get(ms)
    solver = LMAPFSolver(ms, tables, future_orders=all_orders)
    num_rounds = DIFF_ROUNDS['nightmare']

    drop_zones = [tuple(dz) for dz in ms.drop_off_zones]
    drop_set = set(drop_zones)

    # --- Tracking data ---
    # Per-round snapshots
    rounds_with_chain_potential = 0  # active completed AND preview had items
    total_chains_fired = 0
    total_chain_orders = 0  # extra orders from chains (beyond the trigger)

    # Per-order completion record
    order_completions: list[dict] = []

    # Running stats
    prev_orders_completed = 0
    prev_active_id = -1

    t0 = time.time()

    for rnd in range(num_rounds):
        state.round = rnd

        # --- BEFORE step: snapshot state ---
        active_order = state.get_active_order()
        preview_order = state.get_preview_order()

        # Bot positions and inventories
        num_bots = len(state.bot_positions)
        bot_positions = {}
        bot_inventories = {}
        for bid in range(num_bots):
            bot_positions[bid] = (int(state.bot_positions[bid, 0]),
                                  int(state.bot_positions[bid, 1]))
            bot_inventories[bid] = state.bot_inv_list(bid)

        # Bots at dropoff zones
        bots_at_drop: dict[tuple[int, int], list[tuple[int, list[int]]]] = {
            dz: [] for dz in drop_zones
        }
        for bid in range(num_bots):
            pos = bot_positions[bid]
            if pos in drop_set:
                bots_at_drop[pos].append((bid, bot_inventories[bid]))

        # Active order needs
        active_needs: list[int] = []
        active_needs_dict: dict[int, int] = {}
        if active_order:
            active_needs = active_order.needs()
            for t in active_needs:
                active_needs_dict[t] = active_needs_dict.get(t, 0) + 1

        # Preview order needs
        preview_needs: list[int] = []
        preview_needs_dict: dict[int, int] = {}
        if preview_order:
            preview_needs = preview_order.needs()
            for t in preview_needs:
                preview_needs_dict[t] = preview_needs_dict.get(t, 0) + 1

        # Items at dropoffs matching preview needs
        preview_at_drop: dict[int, int] = {}  # type_id -> count at dropoffs
        if preview_order:
            for dz, bots in bots_at_drop.items():
                for bid, inv in bots:
                    for t in inv:
                        if t in preview_needs_dict:
                            preview_at_drop[t] = preview_at_drop.get(t, 0) + 1

        # How close is the active order to completion?
        # Count items at dropoffs + items being carried TO dropoffs
        active_at_drop_count = 0
        if active_order:
            remaining_active = dict(active_needs_dict)
            for dz, bots in bots_at_drop.items():
                for bid, inv in bots:
                    for t in inv:
                        if remaining_active.get(t, 0) > 0:
                            remaining_active[t] -= 1
                            active_at_drop_count += 1

        # --- Get solver actions and step ---
        actions = solver.action(state, all_orders, rnd)
        orders_before = state.orders_completed
        score_before = state.score
        step(state, actions, all_orders)
        orders_after = state.orders_completed
        score_after = state.score

        completions_this_round = orders_after - orders_before

        # --- AFTER step: detect chain reactions ---
        if completions_this_round >= 1:
            chain_fired = completions_this_round > 1
            chain_extra = completions_this_round - 1 if chain_fired else 0

            if chain_fired:
                total_chains_fired += 1
                total_chain_orders += chain_extra

            # Compute what preview items WERE at dropoffs when active completed
            # (using the BEFORE-step snapshot)
            preview_staged = 0
            preview_total = len(preview_needs)
            preview_missing_types: dict[int, int] = {}

            if preview_order and preview_needs:
                remaining_preview = dict(preview_needs_dict)
                for dz, bots in bots_at_drop.items():
                    for bid, inv in bots:
                        for t in inv:
                            if remaining_preview.get(t, 0) > 0:
                                remaining_preview[t] -= 1
                                preview_staged += 1
                for t, count in remaining_preview.items():
                    if count > 0:
                        preview_missing_types[t] = count
                preview_missing = sum(preview_missing_types.values())
            else:
                preview_missing = preview_total

            # Could chain have fired if we had all preview items staged?
            could_chain = preview_order is not None and preview_total > 0
            if could_chain:
                rounds_with_chain_potential += 1

            # Nearby preview items (within 3 steps of any dropoff)
            preview_nearby = 0
            preview_nearby_detail: list[tuple[int, int, int, int]] = []
            if preview_order and preview_needs:
                rem_nearby = dict(preview_needs_dict)
                for bid in range(num_bots):
                    pos = bot_positions[bid]
                    if pos in drop_set:
                        continue  # already counted in staged
                    d_drop = min(tables.get_distance(pos, dz) for dz in drop_zones)
                    if d_drop <= 3:
                        for t in bot_inventories[bid]:
                            if rem_nearby.get(t, 0) > 0:
                                rem_nearby[t] -= 1
                                preview_nearby += 1
                                preview_nearby_detail.append((bid, t, d_drop, rnd))

            completion_record = {
                'round': rnd,
                'active_order_id': active_order.id if active_order else -1,
                'preview_order_id': preview_order.id if preview_order else -1,
                'active_needs_before': list(active_needs),
                'preview_needs': list(preview_needs),
                'preview_total': preview_total,
                'preview_staged_at_drop': preview_staged,
                'preview_nearby': preview_nearby,
                'preview_missing': sum(preview_missing_types.values()) if preview_missing_types else (preview_total - preview_staged),
                'preview_missing_types': dict(preview_missing_types),
                'chain_fired': chain_fired,
                'chain_extra': chain_extra,
                'completions': completions_this_round,
                'score_delta': score_after - score_before,
                'bots_at_drop_snapshot': {
                    str(dz): [(bid, inv) for bid, inv in bots]
                    for dz, bots in bots_at_drop.items()
                },
            }
            order_completions.append(completion_record)

    elapsed = time.time() - t0

    # --- SUMMARY ---
    print("=" * 70)
    print(f"CHAIN REACTION ANALYSIS — Seed {seed}, {num_rounds} rounds")
    print(f"Final score: {state.score}, Orders completed: {state.orders_completed}")
    print(f"Simulation time: {elapsed:.1f}s")
    print("=" * 70)

    print(f"\nTotal order completions: {len(order_completions)}")
    print(f"Chain reactions fired: {total_chains_fired} "
          f"(extra orders from chains: {total_chain_orders})")
    print(f"Rounds where chain COULD have fired "
          f"(active completed + preview existed): {rounds_with_chain_potential}")

    # Overall chain-readiness stats
    total_staged = sum(c['preview_staged_at_drop'] for c in order_completions)
    total_needed = sum(c['preview_total'] for c in order_completions)
    total_nearby = sum(c['preview_nearby'] for c in order_completions)
    total_missing = sum(c['preview_missing'] for c in order_completions)

    print(f"\nAcross all {len(order_completions)} completion events:")
    print(f"  Preview items AT dropoff when active completed: {total_staged}/{total_needed}")
    print(f"  Preview items NEAR dropoff (within 3 steps):    {total_nearby}")
    print(f"  Preview items MISSING entirely:                 {total_missing}")
    if total_needed > 0:
        print(f"  Staging rate: {total_staged / total_needed * 100:.1f}%")
        print(f"  Staging+nearby rate: {(total_staged + total_nearby) / total_needed * 100:.1f}%")

    # Histogram of staged counts at completion
    staged_hist: dict[int, int] = {}
    for c in order_completions:
        n = c['preview_staged_at_drop']
        staged_hist[n] = staged_hist.get(n, 0) + 1
    print(f"\n  Staged items at completion (histogram):")
    for n in sorted(staged_hist.keys()):
        bar = "#" * staged_hist[n]
        print(f"    {n} items staged: {staged_hist[n]:3d}x {bar}")

    # Missing items breakdown by type
    all_missing_types: dict[int, int] = {}
    for c in order_completions:
        for t, count in c['preview_missing_types'].items():
            all_missing_types[t] = all_missing_types.get(t, 0) + count
    if all_missing_types:
        print(f"\n  Missing preview items by type (across all completions):")
        type_names = ms.item_type_names
        for t in sorted(all_missing_types.keys(), key=lambda x: -all_missing_types[x]):
            name = type_names[t] if t < len(type_names) else f"type_{t}"
            print(f"    {name:12s}: {all_missing_types[t]:3d} missing")

    # Per-order detail
    print(f"\n{'='*70}")
    print("PER-ORDER COMPLETION DETAIL")
    print(f"{'='*70}")
    print(f"{'Rnd':>4s} {'ActOrd':>6s} {'PrvOrd':>6s} {'ActNeed':>7s} "
          f"{'PrvNeed':>7s} {'Staged':>6s} {'Near':>4s} {'Miss':>4s} "
          f"{'Chain':>5s} {'ScoreDelta':>10s}")
    print("-" * 70)

    for c in order_completions:
        chain_str = f"x{c['completions']}" if c['chain_fired'] else "-"
        print(f"{c['round']:4d} "
              f"{c['active_order_id']:6d} "
              f"{c['preview_order_id']:6d} "
              f"{len(c['active_needs_before']):7d} "
              f"{c['preview_total']:7d} "
              f"{c['preview_staged_at_drop']:6d} "
              f"{c['preview_nearby']:4d} "
              f"{c['preview_missing']:4d} "
              f"{chain_str:>5s} "
              f"{c['score_delta']:10d}")

        # Show what was missing
        if c['preview_missing_types']:
            type_names = ms.item_type_names
            missing_str = ", ".join(
                f"{type_names[t] if t < len(type_names) else f'type_{t}'}x{n}"
                for t, n in sorted(c['preview_missing_types'].items())
            )
            print(f"      Missing: {missing_str}")

        # Show what was staged at dropoffs
        has_bots = False
        for dz_str, bots in c['bots_at_drop_snapshot'].items():
            if bots:
                has_bots = True
                type_names = ms.item_type_names
                bot_strs = []
                for bid, inv in bots:
                    inv_names = [type_names[t] if t < len(type_names) else f"t{t}"
                                 for t in inv]
                    bot_strs.append(f"b{bid}:[{','.join(inv_names)}]")
                print(f"      At {dz_str}: {' '.join(bot_strs)}")

    # Identify "near miss" chains — where 1-2 more items would have triggered
    print(f"\n{'='*70}")
    print("NEAR-MISS CHAINS (1-2 items away from full chain)")
    print(f"{'='*70}")
    near_misses = [c for c in order_completions
                   if not c['chain_fired'] and 0 < c['preview_missing'] <= 2
                   and c['preview_total'] > 0]
    if near_misses:
        for c in near_misses:
            type_names = ms.item_type_names
            missing_str = ", ".join(
                f"{type_names[t] if t < len(type_names) else f'type_{t}'}x{n}"
                for t, n in sorted(c['preview_missing_types'].items())
            )
            print(f"  R{c['round']:3d}: Order {c['active_order_id']} completed, "
                  f"preview {c['preview_order_id']} needed {c['preview_total']} items, "
                  f"had {c['preview_staged_at_drop']} staged + {c['preview_nearby']} nearby, "
                  f"missing: {missing_str}")
    else:
        print("  (none)")

    # Identify successful chains
    print(f"\n{'='*70}")
    print("SUCCESSFUL CHAINS")
    print(f"{'='*70}")
    successes = [c for c in order_completions if c['chain_fired']]
    if successes:
        for c in successes:
            print(f"  R{c['round']:3d}: Order {c['active_order_id']} triggered chain x{c['completions']}, "
                  f"score delta: +{c['score_delta']}")
    else:
        print("  (none)")

    # Score opportunity analysis
    print(f"\n{'='*70}")
    print("SCORE OPPORTUNITY ANALYSIS")
    print(f"{'='*70}")
    # Each chain completion = +5 bonus + items delivered
    # If we could have chained every completion with its preview:
    potential_chain_completions = sum(
        1 for c in order_completions
        if not c['chain_fired'] and c['preview_total'] > 0
    )
    # Conservative estimate: each chain saves ~10 rounds of delivery time
    # and gives +5 bonus per chained order
    print(f"  Completions without chain: {potential_chain_completions}")
    print(f"  If ALL had chained: +{potential_chain_completions * 5} bonus points "
          f"(+5 per chained order)")
    print(f"  Plus saved delivery rounds -> more orders possible")

    # Average preview order size
    sizes = [c['preview_total'] for c in order_completions if c['preview_total'] > 0]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        print(f"  Average preview order size: {avg_size:.1f} items")
        print(f"  Average items staged at completion: "
              f"{total_staged / len(sizes):.1f}")


if __name__ == '__main__':
    analyze_chains()
