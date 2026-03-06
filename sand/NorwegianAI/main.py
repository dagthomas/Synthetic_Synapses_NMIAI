"""Grocery Bot — NM i AI 2026

Usage:
    python main.py easy   <URL>       Run live game (smart mode)
    python main.py scout  easy <URL>  Scout run (discover orders)
    python main.py plan   easy        Build optimal plan from scout data
    python main.py play   easy <URL>  Execute with pre-computed plan
    python main.py sim    easy        Simulate full game locally (real scores!)
    python main.py local  easy        Replay and compare actions
    python main.py optimize hard      Full-game beam search optimizer
    python main.py replay  hard <URL> Play live with optimized plan
"""

import asyncio
import json
import sys
import time

import websockets

from brain import decide_actions
from pathfinding import reset_shelf_cache
from recorder import GameRecorder, list_recordings
from simulate import run_local

VALID_DIFFICULTIES = {"easy", "medium", "hard", "expert"}


async def run_game(difficulty, url, game_plan=None, mode="smart"):
    """Core game loop shared by all live modes."""
    recorder = GameRecorder(difficulty)
    reset_shelf_cache()

    label = {"smart": "SMART", "scout": "SCOUT", "plan": "PLAN"}[mode]
    print(f"\n  Difficulty: {difficulty.upper()} | Mode: {label}")
    print(f"  Connecting to game server...")

    async with websockets.connect(url) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)

            if data["type"] == "game_over":
                recorder.record_result(data)
                filepath = recorder.save()
                print(f"\n{'=' * 40}")
                print(f"  GAME OVER")
                print(f"  Score:            {data['score']}")
                print(f"  Rounds used:      {data['rounds_used']}")
                print(f"  Items delivered:  {data['items_delivered']}")
                print(f"  Orders completed: {data['orders_completed']}")
                print(f"{'=' * 40}")
                return filepath

            if data["type"] == "game_state":
                t0 = time.monotonic()
                actions = decide_actions(data, game_plan=game_plan)
                dt_ms = (time.monotonic() - t0) * 1000
                # Send actions FIRST, then do recording/printing (minimize latency)
                payload = json.dumps({"actions": actions})
                await ws.send(payload)

                rnd = data["round"]
                score = data["score"]
                recorder.record_round(data, actions)

                n_bots = len(data["bots"])
                if rnd % 10 == 0:
                    active = next(
                        (o for o in data["orders"] if o.get("status") == "active"),
                        None,
                    )
                    delivered = len(active["items_delivered"]) if active else "?"
                    required = len(active["items_required"]) if active else "?"
                    # Detect stuck bots
                    positions = " ".join(
                        f"B{b['id']}({b['position'][0]},{b['position'][1]})"
                        for b in data["bots"]
                    )
                    acts = " ".join(
                        f"{a['action'][:4]}" for a in actions
                    )
                    slow = f" SLOW:{dt_ms:.0f}ms" if dt_ms > 50 else ""
                    print(
                        f"  R{rnd:3d} S{score:3d} D{delivered}/{required} "
                        f"| {positions} | {acts}{slow}"
                    )


def cmd_scout(difficulty, url):
    """Scout run: play normally, discover orders for later planning."""
    print("  Phase 1/2: Scouting (discovering order sequence)...")
    filepath = asyncio.run(run_game(difficulty, url, mode="scout"))
    if filepath:
        from planner import extract_orders_from_recording
        orders = extract_orders_from_recording(filepath)
        print(f"\n  Orders discovered: {len(orders)}")
        for i, o in enumerate(orders):
            print(f"    {i}: {o['items_required']}")
        print(f"\n  Next: run 'python main.py plan {difficulty}' to build optimal plan")
        print(f"  Then: run 'python main.py play {difficulty} <URL>' to execute")


def cmd_plan(difficulty):
    """Build optimal plan from the latest scout recording."""
    from distance import DistanceMatrix
    from planner import (
        build_game_plan,
        extract_initial_items,
        extract_orders_from_recording,
        save_game_plan,
    )

    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"  No recordings for '{difficulty}'. Run scout first.")
        sys.exit(1)

    recording_path = recordings[0]
    print(f"  Using recording: {recording_path}")

    orders = extract_orders_from_recording(recording_path)
    items = extract_initial_items(recording_path)
    print(f"  Orders found: {len(orders)}")
    print(f"  Items on map: {len(items)}")

    # Build distance matrix from round 0 state
    import json as _json
    with open(recording_path) as f:
        data = _json.load(f)
    state0 = data["rounds"][0]["state"]
    dist_matrix = DistanceMatrix(state0)
    drop_off = tuple(state0["drop_off"])

    print(f"  Computing optimal plan...")
    plan = build_game_plan(dist_matrix, items, orders, drop_off)
    save_game_plan(plan, difficulty)

    print(f"\n  Plan details:")
    for i, op in enumerate(plan):
        items_list = [it["type"] for it in op["assigned_items"]]
        n_trips = len(op["trips"])
        total_cost = sum(t[1] for t in op["trips"])
        print(f"    Order {i}: {items_list} | {n_trips} trip(s) | ~{total_cost} rounds")


def cmd_play(difficulty, url):
    """Execute game with pre-computed plan."""
    from planner import load_game_plan
    plan = load_game_plan(difficulty)
    if not plan:
        print(f"  No plan found for '{difficulty}'. Run plan first.")
        print(f"  Steps: scout → plan → play")
        sys.exit(1)
    print(f"  Loaded plan: {len(plan)} orders pre-planned")
    asyncio.run(run_game(difficulty, url, game_plan=plan, mode="plan"))


def usage():
    print(__doc__)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if not args:
        usage()

    cmd = args[0].lower()

    # --- LOCAL SIMULATION (full game engine, real scores) ---
    if cmd == "sim":
        if len(args) < 2 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py sim <{'|'.join(sorted(VALID_DIFFICULTIES))}>")
            sys.exit(1)
        from simulator import run_simulation
        run_simulation(args[1])
        return

    # --- LOCAL REPLAY (compare actions against recording) ---
    if cmd == "local":
        if len(args) < 2 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py local <{'|'.join(sorted(VALID_DIFFICULTIES))}>")
            sys.exit(1)
        run_local(args[1])
        return

    # --- SCOUT ---
    if cmd == "scout":
        if len(args) < 3 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py scout <difficulty> <URL>")
            sys.exit(1)
        cmd_scout(args[1], args[2])
        return

    # --- PLAN ---
    if cmd == "plan":
        if len(args) < 2 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py plan <difficulty>")
            sys.exit(1)
        cmd_plan(args[1])
        return

    # --- PLAY (with plan) ---
    if cmd == "play":
        if len(args) < 3 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py play <difficulty> <URL>")
            sys.exit(1)
        cmd_play(args[1], args[2])
        return

    # --- PERTURB (perturbation search) ---
    if cmd == "perturb":
        if len(args) < 2 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py perturb <difficulty> [--iter N]")
            sys.exit(1)
        from perturb_search import iterative_search
        max_iter = 10
        if "--iter" in args:
            idx = args.index("--iter")
            if idx + 1 < len(args):
                max_iter = int(args[idx + 1])
        iterative_search(args[1], search_all=True, max_iterations=max_iter)
        return

    # --- PERTURB-PLAY (play live with perturbation overrides) ---
    if cmd == "perturb-play":
        if len(args) < 3 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py perturb-play <difficulty> <URL>")
            sys.exit(1)
        from perturb_search import load_overrides, get_params
        data = load_overrides(args[1])
        if not data:
            print(f"  No overrides for '{args[1]}'. Run 'perturb' first.")
            sys.exit(1)
        from brain import _PARAMS
        import brain
        params = get_params(args[1])
        params["overrides"] = data["overrides"]
        brain._PARAMS = params
        print(f"  Loaded overrides: {len(data['overrides'])} overrides, expected score {data['score']}")
        from recorder import list_recordings
        order_seq = None
        recs = list_recordings(args[1])
        if recs:
            with open(recs[0]) as f:
                rec_data = json.load(f)
            seen = set()
            order_seq = []
            for rnd in rec_data["rounds"]:
                for o in rnd["state"]["orders"]:
                    if o["id"] not in seen:
                        seen.add(o["id"])
                        order_seq.append({"id": o["id"], "items_required": list(o["items_required"])})
        asyncio.run(run_game(args[1], args[2], game_plan=order_seq, mode="plan"))
        return

    # --- OPTIMIZE (full-game beam search) ---
    if cmd == "optimize":
        if len(args) < 2 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py optimize <difficulty> [beam_width] [passes]")
            sys.exit(1)
        from optimizer import run_optimization
        beam = int(args[2]) if len(args) > 2 else 1000
        passes = int(args[3]) if len(args) > 3 else 3
        run_optimization(args[1], beam_width=beam, passes=passes)
        return

    # --- REPLAY (with optimized plan) ---
    if cmd == "replay":
        if len(args) < 3 or args[1] not in VALID_DIFFICULTIES:
            print(f"Usage: python main.py replay <difficulty> <URL>")
            sys.exit(1)
        from optimizer import load_optimized_plan
        plan = load_optimized_plan(args[1])
        if not plan:
            print(f"  No optimized plan for '{args[1]}'. Run optimize first.")
            sys.exit(1)
        print(f"  Loaded optimized plan for {plan['n_bots']} bots")
        asyncio.run(run_game(args[1], args[2], game_plan=plan, mode="plan"))
        return

    # --- SMART MODE (default live game) ---
    difficulty = cmd
    if difficulty not in VALID_DIFFICULTIES:
        print(f"Unknown command or difficulty: '{cmd}'")
        usage()

    if len(args) < 2:
        print(f"Usage: python main.py {difficulty} <URL>")
        sys.exit(1)

    url = args[1]

    # Load best params for this difficulty (from perturb_search)
    from perturb_search import get_params, load_overrides
    import brain
    params = get_params(difficulty)
    if params:
        brain._PARAMS = params
        print(f"  Loaded params: {params}")

    asyncio.run(run_game(difficulty, url, mode="smart"))


if __name__ == "__main__":
    main()
