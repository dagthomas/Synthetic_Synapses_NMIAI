"""All-in-one Python pipeline: capture orders → solve → save solution.

Outputs JSON lines to stdout for SSE consumption by SvelteKit dashboard.
Supports both GPU (sequential per-bot DP) and CPU (parallel optimizer) solvers.

Usage:
    python capture_and_solve_stream.py <ws_url> <difficulty> [--solver gpu|cpu] [--time <seconds>] [--workers <n>]
"""
import asyncio
import json
import sys
import os
import time


def emit(data):
    """Output a JSON line to stdout for SSE streaming."""
    print(json.dumps(data), flush=True)


async def capture_phase(ws_url, difficulty):
    """Phase 1: Play a probe game to capture all orders."""
    import websockets
    from capture_game import decide_action

    captured = {
        'difficulty': difficulty,
        'grid': None,
        'items': None,
        'drop_off': None,
        'num_bots': 0,
        'orders': [],
    }

    seen_order_ids = set()
    walls_set = None
    width = height = 0

    emit({"type": "phase", "phase": "capture"})
    emit({"type": "status", "message": f"Phase 1: Python probe game ({difficulty})"})

    try:
        async with websockets.connect(ws_url) as ws:
            async for message in ws:
                data = json.loads(message)

                if data["type"] == "game_over":
                    captured['probe_score'] = data['score']
                    emit({
                        "type": "final_score",
                        "score": data['score'],
                        "phase": "capture",
                    })
                    emit({
                        "type": "capture_done",
                        "score": data['score'],
                        "orders": len(captured['orders']),
                        "message": f"Probe complete: score={data['score']}, {len(captured['orders'])} orders captured",
                    })
                    break

                if data["type"] != "game_state":
                    continue

                rnd = data["round"]

                if rnd == 0:
                    width = data["grid"]["width"]
                    height = data["grid"]["height"]
                    captured['grid'] = data['grid']
                    captured['items'] = data['items']
                    captured['drop_off'] = data['drop_off']
                    captured['num_bots'] = len(data['bots'])

                    walls_set = set()
                    for w_pos in data["grid"]["walls"]:
                        walls_set.add((w_pos[0], w_pos[1]))
                    for item in data["items"]:
                        ix, iy = item["position"]
                        walls_set.add((ix, iy))

                # Capture new orders
                for order in data["orders"]:
                    oid = order["id"]
                    if oid not in seen_order_ids:
                        seen_order_ids.add(oid)
                        captured['orders'].append({
                            'id': oid,
                            'items_required': list(order['items_required']),
                        })

                # Progress
                if rnd % 10 == 0 or rnd <= 2 or rnd >= 295:
                    emit({
                        "type": "progress",
                        "round": rnd,
                        "max_rounds": data.get("max_rounds", 300),
                        "score": data.get("score", 0),
                    })

                # Play greedy actions
                actions = []
                for bot in data["bots"]:
                    action = decide_action(bot, data, walls_set, width, height)
                    actions.append(action)

                await ws.send(json.dumps({"actions": actions}))

    except Exception as e:
        emit({"type": "error", "msg": f"Capture failed: {e}"})
        return None

    return captured


def solve_phase_gpu(captured):
    """Phase 2 (GPU): Solve using sequential per-bot GPU DP with refinement."""
    from solution_store import save_capture, load_meta, save_solution
    from gpu_sequential_solver import solve_sequential
    from configs import CONFIGS
    import torch

    difficulty = captured['difficulty']
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    # Save capture for future re-use
    save_capture(difficulty, captured)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    emit({"type": "phase", "phase": "solver"})
    emit({
        "type": "solver_init",
        "difficulty": difficulty,
        "num_bots": num_bots,
        "num_orders": len(captured.get('orders', [])),
        "num_items": len(captured.get('items', [])),
        "probe_score": captured.get('probe_score', 0),
        "solver": "gpu_sequential",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A',
    })

    # Previous best
    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    emit({"type": "solver_prev_best", "score": prev_score})

    emit({"type": "solver_solving", "msg": f"GPU sequential DP ({num_bots} bots, {device})..."})

    # Streaming callbacks
    def on_bot_progress(bot_id, total_bots, score, elapsed):
        emit({
            "type": "bot_done",
            "bot_id": bot_id,
            "total_bots": total_bots,
            "score": score,
            "time": round(elapsed, 3),
        })

    def on_round(bot_id, rnd, score, unique, expanded, elapsed):
        if rnd == 0:
            emit({
                "type": "bot_start",
                "bot_id": bot_id,
                "total_bots": num_bots,
            })
        # Emit every 50 rounds to avoid flooding stdout
        if rnd % 50 == 0 or rnd >= 299:
            emit({
                "type": "round",
                "bot_id": bot_id,
                "r": rnd,
                "score": score,
                "unique": unique,
                "expanded": expanded,
                "time": round(elapsed, 3),
            })

    def on_phase(phase_name, iteration, cpu_score):
        emit({
            "type": "gpu_phase",
            "phase": phase_name,
            "iteration": iteration,
            "cpu_score": cpu_score,
        })

    t0 = time.time()
    try:
        score, actions = solve_sequential(
            capture_data=captured,
            difficulty=difficulty,
            device=device,
            max_states=500000,
            verbose=True,
            on_bot_progress=on_bot_progress,
            on_round=on_round,
            on_phase=on_phase,
        )
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"type": "solver_error", "msg": f"GPU solver failed: {e}"})
        return 0, []

    elapsed = time.time() - t0

    emit({
        "type": "solver_result",
        "score": score,
        "time": round(elapsed, 1),
        "num_bots": num_bots,
        "solver": "gpu_sequential",
    })

    # Save solution
    if score > 0:
        try:
            save_solution(difficulty, score, actions, force=True)
        except Exception as e:
            emit({"type": "log", "text": f"Failed to save: {e}"})

        if score > prev_score:
            emit({"type": "solver_improved", "old_score": prev_score, "new_score": score, "delta": score - prev_score})
        else:
            emit({"type": "solver_no_improvement", "score": score, "prev": prev_score})

    emit({"type": "solver_done", "score": score, "time": round(elapsed, 1)})
    return score, actions


def solve_phase_cpu(captured, time_limit=240.0, num_workers=None):
    """Phase 2 (CPU): Solve offline using parallel optimizer."""
    from solution_store import save_capture, load_meta, save_solution
    from parallel_optimizer import parallel_optimize
    from configs import CONFIGS

    difficulty = captured['difficulty']
    cfg = CONFIGS[difficulty]
    num_bots = cfg['bots']

    if num_workers is None:
        num_workers = min(12, os.cpu_count() or 4)

    # Save capture for future re-use
    save_capture(difficulty, captured)

    emit({"type": "phase", "phase": "solver"})
    emit({
        "type": "solver_init",
        "difficulty": difficulty,
        "num_bots": num_bots,
        "num_workers": num_workers,
        "time_limit": time_limit,
        "num_orders": len(captured.get('orders', [])),
        "num_items": len(captured.get('items', [])),
        "probe_score": captured.get('probe_score', 0),
        "solver": "parallel_optimizer",
    })

    # Previous best
    meta = load_meta(difficulty)
    prev_score = meta.get('score', 0) if meta else 0
    emit({"type": "solver_prev_best", "score": prev_score})

    emit({"type": "solver_solving", "msg": f"Running {num_workers} parallel workers ({time_limit:.0f}s budget)..."})

    t0 = time.time()
    try:
        score, actions = parallel_optimize(
            capture_data=captured,
            difficulty=difficulty,
            time_limit=time_limit,
            num_workers=num_workers,
            verbose=True,  # prints worker progress to stderr
        )
    except Exception as e:
        emit({"type": "solver_error", "msg": f"Solver failed: {e}"})
        return 0, []

    elapsed = time.time() - t0

    emit({
        "type": "solver_result",
        "score": score,
        "time": round(elapsed, 1),
        "num_bots": num_bots,
    })

    # Save solution
    if score > 0:
        try:
            save_solution(difficulty, score, actions, force=True)
        except Exception as e:
            emit({"type": "log", "text": f"Failed to save: {e}"})

        if score > prev_score:
            emit({"type": "solver_improved", "old_score": prev_score, "new_score": score, "delta": score - prev_score})
        else:
            emit({"type": "solver_no_improvement", "score": score, "prev": prev_score})

    emit({"type": "solver_done", "score": score, "time": round(elapsed, 1)})
    return score, actions


async def run_pipeline(ws_url, difficulty, solver='gpu', time_limit=240.0, num_workers=None):
    """Full pipeline: capture → solve → done."""
    t0 = time.time()

    emit({"type": "status", "message": f"Starting pipeline: {difficulty} (solver={solver})"})

    # Phase 1: Capture
    captured = await capture_phase(ws_url, difficulty)
    if not captured:
        emit({"type": "done", "code": 1, "message": "Capture failed", "capture_score": 0})
        return

    capture_time = time.time() - t0
    emit({"type": "status", "message": f"Capture done in {capture_time:.0f}s. Starting {solver} solver..."})

    # Phase 2: Solve
    if solver == 'gpu':
        score, actions = solve_phase_gpu(captured)
    else:
        remaining = max(30, time_limit - capture_time)
        score, actions = solve_phase_cpu(captured, time_limit=remaining, num_workers=num_workers)

    total_time = time.time() - t0
    emit({
        "type": "done",
        "code": 0,
        "message": "Full pipeline complete",
        "capture_score": captured.get('probe_score', 0),
        "solver_score": score,
        "total_time": round(total_time, 1),
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Capture & solve pipeline')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('difficulty', choices=['easy', 'medium', 'hard', 'expert'])
    parser.add_argument('--solver', choices=['gpu', 'cpu'], default='gpu',
                        help='Solver backend (default: gpu)')
    parser.add_argument('--time', type=float, default=240.0, help='Total time budget (CPU only)')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (CPU only)')
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.ws_url, args.difficulty,
                             solver=args.solver,
                             time_limit=args.time, num_workers=args.workers))
