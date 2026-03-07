#!/usr/bin/env python3
"""Remote GPU optimization server — run on B200/cloud machine.

Minimal HTTP API that accepts capture data, runs GPU solver, returns solution.
No PostgreSQL needed. No game connection needed. Just Python + PyTorch + CUDA.

Usage on RunPod:
    python3 gpu_server.py                    # port 5555, auto-detect GPU
    python3 gpu_server.py --port 8080        # custom port
    python3 gpu_server.py --gpu b200         # force GPU profile

Connect from local machine via SSH tunnel:
    ssh -p 30618 -i ~/.ssh/id_ed25519 root@<IP> -L 5555:localhost:5555
    Then GUI at localhost:5173 can reach GPU at localhost:5555
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import torch

# ── GPU info ────────────────────────────────────────────────────────────

def gpu_info():
    if not torch.cuda.is_available():
        return {"available": False, "name": "none", "vram_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / (1024**3), 1),
    }


# ── Solver wrapper ──────────────────────────────────────────────────────

def run_optimize(capture_data: dict, params: dict) -> dict:
    """Run GPU optimization on capture data. Returns solution dict."""
    from gpu_sequential_solver import solve_sequential, refine_from_solution

    difficulty = params.get("difficulty", "expert")
    max_states = params.get("max_states", 200000)
    max_time = params.get("max_time", 120)
    refine_iters = params.get("refine_iters", 20)
    speed_bonus = params.get("speed_bonus", 50.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    events = []

    def on_phase(phase_name, iteration, cpu_score):
        events.append({"type": "gpu_phase", "phase": phase_name,
                       "iteration": iteration, "score": cpu_score})
        print(f"  [opt] {phase_name} iter={iteration} score={cpu_score}", flush=True)

    # Pass 1: sequential solve
    score, actions = solve_sequential(
        capture_data=capture_data,
        difficulty=difficulty,
        device=device,
        max_states=max_states,
        max_refine_iters=refine_iters,
        on_phase=on_phase,
        verbose=True,
    )

    # Pass 2+: refine
    elapsed = time.time() - t0
    remaining = max_time - elapsed
    if remaining > 10 and score > 0:
        ref_score, ref_actions = refine_from_solution(
            actions,
            capture_data=capture_data,
            difficulty=difficulty,
            device=device,
            max_states=max_states,
            max_refine_iters=refine_iters,
            on_phase=on_phase,
            verbose=True,
        )
        if ref_score > score:
            score, actions = ref_score, ref_actions

    total_time = time.time() - t0

    return {
        "score": score,
        "actions": actions,
        "elapsed": round(total_time, 1),
        "events": events,
    }


def run_deep_optimize(capture_data: dict, params: dict) -> dict:
    """Run deep multi-phase optimization (B200 mode)."""
    try:
        sys.path.insert(0, "../grocery-bot-b200")
        from squad_solver import solve_squads, SquadConfig
        from b200_config import get_params
    except ImportError:
        return run_optimize(capture_data, params)

    difficulty = params.get("difficulty", "expert")
    gpu = params.get("gpu", "auto")
    budget = params.get("budget", 1800)
    bp = get_params(difficulty, gpu)

    max_states = params.get("max_states", bp.max_states)

    t0 = time.time()
    events = []

    # Phase 1: Exploration
    phase1_budget = budget * 0.3
    config = SquadConfig(
        difficulty=difficulty,
        max_states=bp.explore_states,
        joint_states=0,
        joint_squad_size=1,
        explore_states=bp.explore_states,
        pass1_orderings=bp.pass1_orderings,
        refine_iters=0,
        lns_rounds=0,
        max_dp_bots=bp.max_dp_bots,
        max_time_s=phase1_budget,
        speed_bonus=bp.speed_bonus,
        device="cuda",
    )
    best_score, best_actions = solve_squads(capture_data, config)
    events.append({"type": "deep_phase", "phase": 1,
                   "name": "exploration", "score": best_score,
                   "elapsed": round(time.time() - t0, 1)})

    # Phase 2: Intensification
    remaining = budget - (time.time() - t0)
    phase2_budget = min(budget * 0.5, remaining * 0.7)
    if phase2_budget > 30:
        config2 = SquadConfig(
            difficulty=difficulty,
            max_states=max_states,
            joint_states=bp.joint_states,
            joint_squad_size=bp.joint_squad_size,
            explore_states=bp.explore_states,
            pass1_orderings=1,
            refine_iters=bp.refine_iters,
            lns_rounds=0,
            max_dp_bots=bp.max_dp_bots,
            max_time_s=phase2_budget,
            speed_bonus=bp.speed_bonus,
            device="cuda",
        )
        score2, actions2 = solve_squads(capture_data, config2)
        if score2 > best_score:
            best_score, best_actions = score2, actions2
        events.append({"type": "deep_phase", "phase": 2,
                       "name": "intensification", "score": best_score,
                       "elapsed": round(time.time() - t0, 1)})

    # Phase 3: LNS
    remaining = budget - (time.time() - t0)
    if remaining > 30:
        config3 = SquadConfig(
            difficulty=difficulty,
            max_states=max_states,
            joint_states=bp.joint_states,
            joint_squad_size=bp.joint_squad_size,
            explore_states=bp.explore_states,
            pass1_orderings=1,
            refine_iters=0,
            lns_rounds=bp.lns_rounds,
            max_dp_bots=bp.max_dp_bots,
            max_time_s=remaining * 0.9,
            speed_bonus=bp.speed_bonus,
            device="cuda",
        )
        score3, actions3 = solve_squads(capture_data, config3)
        if score3 > best_score:
            best_score, best_actions = score3, actions3
        events.append({"type": "deep_phase", "phase": 3,
                       "name": "lns", "score": best_score,
                       "elapsed": round(time.time() - t0, 1)})

    return {
        "score": best_score,
        "actions": best_actions,
        "elapsed": round(time.time() - t0, 1),
        "events": events,
    }


# ── HTTP Server ─────────────────────────────────────────────────────────

class GPUHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Health check / GPU info."""
        info = gpu_info()
        self._respond(200, info)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/optimize":
            self._handle_optimize(body)
        elif self.path == "/deep":
            self._handle_deep(body)
        else:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})

    def _handle_optimize(self, body):
        capture = body.get("capture")
        params = body.get("params", {})
        if not capture:
            self._respond(400, {"error": "Missing 'capture' in body"})
            return
        try:
            print(f"[optimize] difficulty={params.get('difficulty')} "
                  f"max_states={params.get('max_states', 200000)} "
                  f"max_time={params.get('max_time', 120)}s", flush=True)
            result = run_optimize(capture, params)
            print(f"[optimize] done: score={result['score']} "
                  f"in {result['elapsed']}s", flush=True)
            self._respond(200, result)
        except Exception as e:
            traceback.print_exc()
            self._respond(500, {"error": str(e)})

    def _handle_deep(self, body):
        capture = body.get("capture")
        params = body.get("params", {})
        if not capture:
            self._respond(400, {"error": "Missing 'capture' in body"})
            return
        try:
            print(f"[deep] difficulty={params.get('difficulty')} "
                  f"budget={params.get('budget', 1800)}s "
                  f"gpu={params.get('gpu', 'auto')}", flush=True)
            result = run_deep_optimize(capture, params)
            print(f"[deep] done: score={result['score']} "
                  f"in {result['elapsed']}s", flush=True)
            self._respond(200, result)
        except Exception as e:
            traceback.print_exc()
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quieter logging
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU optimization server")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--gpu", default="auto", choices=["auto", "b200", "5090"])
    args = parser.parse_args()

    info = gpu_info()
    print(f"GPU Server starting on port {args.port}", flush=True)
    print(f"  GPU: {info['name']} ({info['vram_gb']} GB)", flush=True)
    print(f"  Endpoints:", flush=True)
    print(f"    GET  /          — health check + GPU info", flush=True)
    print(f"    POST /optimize  — quick GPU optimize (capture + params)", flush=True)
    print(f"    POST /deep      — deep B200 optimize (capture + params)", flush=True)
    print(f"", flush=True)

    server = HTTPServer(("0.0.0.0", args.port), GPUHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
