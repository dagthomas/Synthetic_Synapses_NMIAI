"""Anytime online GPU stream solver for Grocery Bot.

Provides GPU-computed actions at every round of the live game using a
tiered anytime architecture:

  Tier 0: Immediate greedy BFS  — always available from round 0 (<1ms)
  Tier 1: MAPF planner (1-10s)  — upgrade when better
  Tier 2+: GPU DP passes with increasing state budgets — keep upgrading

Background threads compute better plans continuously. The live game always
uses the best currently-available plan, falling back to greedy if no plan
is ready. Plan upgrades happen instantly when a better one finishes.

Usage:
    python live_gpu_stream.py "wss://..."
    python live_gpu_stream.py "wss://..." --save
    python live_gpu_stream.py "wss://..." --max-states 50000
    python live_gpu_stream.py "wss://..." --no-refine
    python live_gpu_stream.py "wss://..." --cpu
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

import websockets

import numpy as np

from game_engine import (
    init_game_from_capture, step, build_map_from_capture,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, MAX_ROUNDS, MAX_ORDER_SIZE,
    CELL_FLOOR, CELL_WALL, CELL_SHELF, CELL_DROPOFF, DX, DY,
    MapState, CaptureData, Order,
)
from replay_solution import predict_full_sim, extract_goals, bfs_next_action, build_walkable
from configs import detect_difficulty, DIFF_ROUNDS
from live_solver import ws_to_capture
from gpu_sequential_solver import solve_sequential, refine_from_solution, solve_multi_restart
from solution_store import save_solution, merge_capture, save_capture as store_capture, load_meta
from precompute import PrecomputedTables

# ── GPU pass state budgets per difficulty ──────────────────────────────────────
PASSES = {
    'easy':      [50_000, 500_000, 2_000_000],
    'medium':    [20_000, 200_000, 1_000_000],
    'hard':      [10_000, 100_000, 500_000],
    'expert':    [5_000,  50_000,  200_000],
    'nightmare': [5_000,  25_000,  100_000],
}

# ── Per-round GPU DP parameters per difficulty ─────────────────────────────────
# max_states: state budget per bot per round, horizon: rounds to look ahead
PR_PARAMS = {
    'easy':      {'max_states': 50_000, 'horizon': 80},
    'medium':    {'max_states': 25_000, 'horizon': 55},
    'hard':      {'max_states': 40_000, 'horizon': 60},
    'expert':    {'max_states': 20_000, 'horizon': 50},
    'nightmare': {'max_states': 15_000, 'horizon': 40},
}

ACTION_NAMES = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']


def decode_jwt_from_url(ws_url: str) -> dict | None:
    """Extract and decode JWT payload from a wss:// game URL."""
    try:
        parsed = urlparse(ws_url)
        qs = parse_qs(parsed.query)
        token = qs.get('token', [None])[0]
        if not token:
            return None
        payload_b64 = token.split('.')[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None


@dataclass
class PlanState:
    score: int = 0
    actions: Optional[list] = None       # 300 × [(act, item_idx)] × num_bots
    expected_pos: Optional[list] = None  # positions[rnd][bid] = (x, y) before action
    goals: Optional[dict] = None         # bid → [(rnd, pos, act, item_idx), ...]
    source: str = 'none'


class AnytimeGPUStream:
    def __init__(self, ws_url: str, save: bool = False,
                 max_states: int | None = None, no_refine: bool = False,
                 device: str = 'cuda', post_optimize_time: int = 0,
                 json_stream: bool = False, preload_capture: bool = False,
                 record: bool = False, pipeline_mode: bool = False) -> None:
        self.ws_url = ws_url
        self.do_save = save
        self.pipeline_mode = pipeline_mode  # skip heavy offline GPU (pipeline handles it separately)
        self.max_states_override = max_states
        self.no_refine = no_refine
        self.device = device
        self.post_optimize_time = post_optimize_time  # seconds to continue GPU after game_over
        self.json_stream = json_stream  # emit JSON events to stdout for SSE dashboard
        self.preload_capture = preload_capture  # preload existing capture orders at round 0
        self.do_record = record  # import game log to PostgreSQL after game

        # ── shared state (protected by _lock) ─────────────────────────────────
        self._lock = threading.Lock()
        self._plan = PlanState()
        self._capture = None       # capture dict, extended as orders appear
        self._map_state = None     # MapState (static, set once on round 0)
        self._walkable = None      # set of walkable (x,y) cells
        self._tables = None        # PrecomputedTables for O(1) BFS lookup
        self._round_offset = 0     # cumulative missed-round offset; reset on plan upgrade

        # ── per-round GPU DP state ────────────────────────────────────────────
        self._pr_lock = threading.Lock()
        self._pr_searcher = None   # GPUBeamSearcher (no locked bots), rebuilt on new orders
        self._pr_all_orders = []   # all_orders list matching _pr_searcher
        self._pr_searcher_gen = -1 # gen when searcher was built
        self._pr_actions = {}      # round → (ws_actions, score_est) from per-round GPU
        self._pr_event = threading.Event()  # signals new round data for pr worker
        self._pr_rnd = -1          # most recent round number for pr worker
        self._pr_ws_data = None    # most recent ws_data for pr worker
        self._order_id_to_idx = {}  # order_id → index in capture['orders']

        # ── bookkeeping ───────────────────────────────────────────────────────
        self._difficulty = None
        self._num_bots = 0
        self._num_rounds = 300  # updated from WS data in _init_round0
        self._solve_gen = 0        # bumped when new orders arrive
        self._seen_order_ids = set()
        self._data_ready = threading.Event()  # set after round 0

        # ── anti-congestion state (populated after round 0) ────────────────────
        self._aisle_cols = set()         # x-coords of narrow aisle columns
        self._corridor_rows = set()      # y-coords of wide horizontal corridors
        self._bottom_corridor_y = 0      # max y in corridor rows
        self._bot_pos_history = {}       # bot_id → list of last 6 positions
        self._bot_stall_count = {}       # bot_id → rounds stuck at same position

        # ── multi-dropoff zone state (nightmare: 3 zones) ─────────────────────
        self._drop_off_zones = []        # list of (x,y) dropoff positions
        self._zone_for_bot = {}          # bot_id → zone_idx
        self._zone_dropoff = {}          # zone_idx → (x,y)
        self._zone_shelves = {}          # zone_idx → set of item indices


    # ── JSON streaming (for SvelteKit SSE dashboard) ─────────────────────────

    def _emit(self, event: dict) -> None:
        """Emit a JSON event to stdout for SSE consumption (only in json_stream mode)."""
        if self.json_stream:
            import json as _json
            print(_json.dumps(event), flush=True)

    # ── plan upgrade ──────────────────────────────────────────────────────────

    def _update_plan(self, score: int, actions: list, expected_pos: list | None,
                     goals: dict | None, source: str, gen: int) -> bool:
        """Upgrade plan if score improved.

        Called from background threads. Resets round_offset on upgrade.

        Stale-gen plans are still accepted if they score better — improvement A
        ensures that once a bot exhausts its plan goals it falls back to greedy,
        so a plan covering fewer orders is still useful for the orders it knows.
        """
        with self._lock:
            current_gen = self._solve_gen
            staleness = f"(stale gen={gen}/{current_gen})" if gen < current_gen else ""
            if score > self._plan.score:
                old = self._plan.source
                self._plan = PlanState(
                    score=score, actions=actions,
                    expected_pos=expected_pos, goals=goals, source=source,
                )
                self._round_offset = 0  # reset on plan change
                print(f"  UPGRADE {old}→{source}: score={score} {staleness}", file=sys.stderr)
                self._emit({"type": "plan_upgrade", "from_source": old, "to_source": source,
                            "score_estimate": score})
                return True
            print(f"  [{source}] score={score} not better than {self._plan.score} {staleness}",
                  file=sys.stderr)
            return False

    def _capture_snapshot(self) -> CaptureData | None:
        """Thread-safe deep copy of current capture."""
        with self._lock:
            if self._capture is None:
                return None
            return copy.deepcopy(self._capture)

    def _map_ref(self) -> MapState | None:
        """MapState reference (read-only, safe without lock)."""
        return self._map_state

    # ── solution preload ─────────────────────────────────────────────────────

    def _preload_solution(self) -> None:
        """Load best existing solution from DB and set as initial plan.

        This gives the live solver immediate plan-following from round 0
        instead of relying on greedy until MAPF/GPU background threads finish.
        The gpu_refine_worker will warm-start refinement from this plan.
        """
        try:
            from solution_store import load_solution as db_load_solution, load_meta as db_load_meta

            diff = self._difficulty
            meta = db_load_meta(diff)
            if not meta or meta.get('score', 0) <= 0:
                print(f"  [preload_sol] No existing solution for {diff}", file=sys.stderr)
                return

            actions = db_load_solution(diff)
            if not actions:
                print(f"  [preload_sol] Solution load failed for {diff}", file=sys.stderr)
                return

            score = meta['score']
            ms = self._map_ref()
            cap = self._capture_snapshot()

            # Simulate to get expected positions and goals
            exp_pos = None
            goals = None
            try:
                exp_pos = predict_full_sim(actions, cap, ms)
                goals = extract_goals(actions, ms, exp_pos)
            except Exception as e:
                print(f"  [preload_sol] Sim failed (plan still usable): {e}",
                      file=sys.stderr)

            with self._lock:
                self._plan = PlanState(
                    score=score, actions=actions,
                    expected_pos=exp_pos, goals=goals,
                    source='preloaded')
            print(f"  [preload_sol] Loaded solution: score={score} "
                  f"rounds={len(actions)}", file=sys.stderr)
            self._emit({"type": "plan_upgrade", "from_source": "none",
                        "to_source": "preloaded", "score_estimate": score})

        except Exception as e:
            print(f"  [preload_sol] Error: {e}", file=sys.stderr)

    # ── MAPF background tier ──────────────────────────────────────────────────

    def _mapf_worker(self, gen):
        """Run MAPF planner in background. Upgrade plan if better score."""
        try:
            from planner import solve as planner_solve
            from configs import CONFIGS

            capture = self._capture_snapshot()
            if capture is None:
                return

            diff = capture.get('difficulty', 'easy')
            nb = CONFIGS[diff]['bots']

            if nb == 1:
                # Single-bot: MAPF adds no value over greedy (no coordination needed)
                print("  [mapf] Skipping MAPF for single-bot (greedy is superior)",
                      file=sys.stderr)
                return
            elif nb <= 3:
                mab_values = [1, 2, 3]
            elif nb <= 5:
                mab_values = [1, 2, 3, 4, 5]
            else:
                mab_values = [1, 2, 3, 4, 5]

            n_orders = len(capture['orders'])
            def game_factory():
                # no_filler: only plan for real captured orders, not random filler
                # (filler orders have wrong items → dead inventory in live game)
                return init_game_from_capture(capture, num_orders=n_orders)

            best_score = 0
            best_actions = None
            t0 = time.time()

            # Enable pipeline pre-fetching when we have enough orders and multiple bots
            pl_depth = 1 if (len(capture['orders']) >= 10 and nb > 1) else 0

            for mab in mab_values:
                if self._solve_gen != gen:
                    break
                try:
                    sc, acts = planner_solve(
                        game_factory=game_factory, verbose=False, max_active_bots=mab,
                        pipeline_depth=pl_depth)
                    if sc > best_score:
                        best_score = sc
                        best_actions = acts
                except Exception as e:
                    print(f"  [mapf] mab={mab} failed: {e}", file=sys.stderr)

            if best_actions:
                elapsed = time.time() - t0
                print(f"  [mapf] Done: score={best_score} ({elapsed:.1f}s)", file=sys.stderr)
                ms = self._map_ref()
                cap = self._capture_snapshot()
                try:
                    exp_pos = predict_full_sim(best_actions, cap, ms)
                    goals = extract_goals(best_actions, ms, exp_pos)
                except Exception as e:
                    print(f"  [mapf] sim failed: {e}", file=sys.stderr)
                    exp_pos = None
                    goals = None
                self._update_plan(best_score, best_actions, exp_pos, goals, 'mapf', gen)

        except Exception as e:
            print(f"  [mapf] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    def _start_mapf(self, gen):
        t = threading.Thread(target=self._mapf_worker, args=(gen,), daemon=True)
        t.start()

    # ── GPU background tier ────────────────────────────────────────────────────

    def _gpu_worker(self):
        """Continuously run GPU DP passes with increasing budgets.

        Restarts from pass 0 when new orders arrive (solve_gen changes).
        Waits for new orders after all passes complete.
        """
        try:
            import torch
            dev = self.device if torch.cuda.is_available() else 'cpu'
        except ImportError:
            dev = 'cpu'

        while True:
            self._data_ready.wait(timeout=10.0)

            capture = self._capture_snapshot()
            if capture is None:
                continue

            diff = capture.get('difficulty', 'easy')
            passes = list(PASSES.get(diff, PASSES['easy']))
            if self.max_states_override:
                passes = [self.max_states_override]

            for pass_idx, max_states in enumerate(passes):
                # Refresh capture and snapshot gen at the start of each pass so that
                # newly-arrived orders are used immediately.  We no longer break early
                # on gen change — instead _update_plan discards stale results.
                fresh = self._capture_snapshot()
                if fresh:
                    capture = fresh
                current_gen = self._solve_gen

                pass_name = f'gpu_pass{pass_idx + 1}'
                max_refine = 0 if self.no_refine else (1 if pass_idx == 0 else 2)
                # Use multi-start orderings for last pass to improve quality
                n_p1_ord = 1 if pass_idx < len(passes) - 1 else 2

                print(f"  [{pass_name}] max_states={max_states:,} refine={max_refine} "
                      f"orderings={n_p1_ord} gen={current_gen}", file=sys.stderr)

                t0 = time.time()
                try:
                    score, actions = solve_sequential(
                        capture_data=capture,
                        device=dev,
                        max_states=max_states,
                        verbose=True,
                        max_refine_iters=max_refine,
                        no_filler=True,
                        num_pass1_orderings=n_p1_ord,
                        no_compile=True,
                    )
                except Exception as e:
                    print(f"  [{pass_name}] Error: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    time.sleep(2.0)
                    break

                elapsed = time.time() - t0
                print(f"  [{pass_name}] score={score} elapsed={elapsed:.1f}s", file=sys.stderr)

                if score > 0 and actions:
                    ms = self._map_ref()
                    cap = self._capture_snapshot()
                    try:
                        exp_pos = predict_full_sim(actions, cap, ms)
                        goals = extract_goals(actions, ms, exp_pos)
                    except Exception as e:
                        print(f"  [{pass_name}] sim failed: {e}", file=sys.stderr)
                        exp_pos = None
                        goals = None
                    self._update_plan(score, actions, exp_pos, goals, pass_name, current_gen)

            # All passes done — wait for new orders (solve_gen change)
            final_gen = self._solve_gen
            print(f"  [gpu] All passes done gen={final_gen}, waiting for new orders...",
                  file=sys.stderr)
            while self._solve_gen == final_gen:
                time.sleep(1.0)

    # ── GPU sequential DP solver (multi-bot) ─────────────────────────────────

    def _gpu_refine_worker(self):
        """GPU sequential DP solver for multi-bot live games.

        Runs solve_sequential with progressively larger state budgets,
        then refines the best solution. Does NOT wait for MAPF — starts
        immediately with a small budget for a fast first plan.

        Budget schedule:
          Pass 0: solve_sequential (cold start, small budget) → fast first plan
          Pass 1+: refine_from_solution (warm start, growing budget) → improve

        Restarts from scratch when new orders arrive (solve_gen changes).
        """
        try:
            import torch
            dev = self.device if torch.cuda.is_available() else 'cpu'
        except ImportError:
            dev = 'cpu'

        # Progressive state budgets (cold start → warm refinements)
        # Sized so cold start finishes in ~15-30s on RTX 5090 for Hard
        _budgets = {
            'medium':    [50_000, 200_000, 1_000_000, 5_000_000],
            'hard':      [50_000, 200_000, 1_000_000, 5_000_000],
            'expert':    [20_000, 100_000,   500_000, 2_000_000],
            'nightmare': [10_000,  50_000,   200_000, 1_000_000],
        }
        _extended_budget = {
            'medium': 5_000_000,
            'hard': 5_000_000,
            'expert': 2_000_000,
            'nightmare': 1_000_000,
        }

        while True:
            self._data_ready.wait(timeout=10.0)

            capture = self._capture_snapshot()
            if capture is None:
                time.sleep(1.0)
                continue

            diff = capture.get('difficulty', 'easy')

            with self._lock:
                current_gen = self._solve_gen

            budgets = _budgets.get(diff, [8_000, 20_000, 60_000])

            # When preloaded solution exists, skip small budgets (can't improve 180+ plans)
            # and start at higher budgets for more effective warm refinement.
            with self._lock:
                preloaded_score = self._plan.score if self._plan else 0
            if preloaded_score >= 100 and len(budgets) > 2:
                # Skip first budget (50K too small for warm refine of strong plans)
                budgets = budgets[1:]
                print(f"  [gpu_seq] Preloaded score={preloaded_score}, "
                      f"skipping small budgets", file=sys.stderr)

            print(f"  [gpu_seq] Starting: diff={diff}, orders={len(capture['orders'])}, "
                  f"gen={current_gen}, budgets={budgets}", file=sys.stderr)
            t_refine_start = time.time()

            best_score = 0
            best_actions = None

            # Check if an existing plan (MAPF or previous GPU pass) can seed us
            with self._lock:
                existing_plan = self._plan
            if (existing_plan and existing_plan.actions and existing_plan.score > 0
                    and existing_plan.source not in ('none', 'greedy')):
                best_score = existing_plan.score
                best_actions = existing_plan.actions
                print(f"  [gpu_seq] Warm-seeding from {existing_plan.source} "
                      f"score={best_score}", file=sys.stderr)

            gen_changed_during_passes = False
            for pass_idx, max_states in enumerate(budgets):
                # Abort this budget series if new orders arrived
                if self._solve_gen != current_gen:
                    gen_changed_during_passes = True
                    print(f"  [gpu_seq] Gen changed at pass {pass_idx}, restarting",
                          file=sys.stderr)
                    break

                cap_snap = self._capture_snapshot()
                with self._lock:
                    snap_gen = self._solve_gen

                try:
                    # Scale refinement iters with pass index: more time per pass → more iters
                    # Pass 0 (cold): 2 refine, Pass 1: 3, Pass 2: 4, Pass 3+: 6
                    pass_refine_iters = min(pass_idx + 2, 6)

                    if best_actions is None:
                        # Cold start: multi-restart sequential DP.
                        # For multi-bot difficulties, screen 50 orderings (cheap GPU
                        # greedy rollout) and run the top 3 in full DP to find the
                        # best bot ordering. For single-bot, falls back to standard DP.
                        num_bots = cap_snap.get('num_bots', 1)
                        n_restarts = 5 if num_bots >= 10 else (3 if num_bots > 1 else 1)
                        num_screen = 200 if num_bots >= 5 else 60
                        print(f"  [gpu_seq] Pass {pass_idx}: multi-restart "
                              f"max_states={max_states:,} restarts={n_restarts} "
                              f"screen={num_screen} refine={pass_refine_iters}",
                              file=sys.stderr)
                        score, actions = solve_multi_restart(
                            capture_data=cap_snap,
                            device=dev,
                            max_states=max_states,
                            max_refine_iters=pass_refine_iters,
                            num_restarts=n_restarts,
                            num_screen=num_screen,
                            n_screen_steps=20,
                            no_filler=True,
                            verbose=True,
                            all_orders_override=None,
                            no_compile=True,
                        )
                    else:
                        # Warm start: refine from best available plan.
                        # Check if MAPF or another external plan improved since last pass.
                        with self._lock:
                            ext = self._plan
                        if (ext and ext.actions and ext.score > best_score
                                and ext.source not in ('none', 'greedy')):
                            best_score = ext.score
                            best_actions = ext.actions
                            print(f"  [gpu_seq] Upgraded base to {ext.source} "
                                  f"score={best_score}", file=sys.stderr)
                        print(f"  [gpu_seq] Pass {pass_idx}: refine "
                              f"max_states={max_states:,} refine={pass_refine_iters} "
                              f"(from score={best_score})",
                              file=sys.stderr)
                        score, actions = refine_from_solution(
                            combined_actions=best_actions,
                            capture_data=cap_snap,
                            device=dev,
                            max_states=max_states,
                            max_refine_iters=pass_refine_iters,
                            no_filler=True,
                            verbose=True,
                            all_orders_override=None,
                            no_compile=True,
                        )

                except Exception as e:
                    print(f"  [gpu_seq] Pass {pass_idx} error: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    time.sleep(1.0)
                    break

                # Check gen again after possibly long computation
                if self._solve_gen != current_gen:
                    print(f"  [gpu_seq] Gen changed after pass {pass_idx}, discarding",
                          file=sys.stderr)
                    break

                self._emit({"type": "gpu_pass_done",
                            "pass": pass_idx,
                            "max_states": max_states,
                            "score": score,
                            "elapsed": round(time.time() - t_refine_start, 1)})

                if score > best_score and actions:
                    best_score = score
                    best_actions = actions
                    # Publish plan upgrade
                    ms_r = self._map_ref()
                    cap_r = self._capture_snapshot()
                    try:
                        exp_pos = predict_full_sim(actions, cap_r, ms_r)
                        goals = extract_goals(actions, ms_r, exp_pos)
                    except Exception as e:
                        print(f"  [gpu_seq] sim failed: {e}", file=sys.stderr)
                        exp_pos = None
                        goals = None
                    self._update_plan(score, actions, exp_pos, goals,
                                      f'gpu_seq_p{pass_idx}', snap_gen)

            # If the pass loop broke early due to a gen change, skip the extended
            # refinement loop and restart immediately from the top with the new gen.
            if gen_changed_during_passes:
                continue

            # Extended refinement while waiting for new orders.
            # Keep polishing the plan with the largest budget until new orders arrive.
            final_gen = self._solve_gen
            ext_budget = _extended_budget.get(diff, 2_000_000)
            print(f"  [gpu_seq] Budget cycle done gen={final_gen} best={best_score}, "
                  f"entering extended refinement loop (budget={ext_budget:,})", file=sys.stderr)
            while self._solve_gen == final_gen and best_actions is not None:
                # Check if external plan improved
                with self._lock:
                    ext = self._plan
                if (ext and ext.actions and ext.score > best_score
                        and ext.source not in ('none', 'greedy')):
                    best_score = ext.score
                    best_actions = ext.actions
                    print(f"  [gpu_seq] Extended: upgraded from {ext.source} "
                          f"score={best_score}", file=sys.stderr)

                cap_snap = self._capture_snapshot()
                try:
                    score, actions = refine_from_solution(
                        combined_actions=best_actions,
                        capture_data=cap_snap,
                        device=dev,
                        max_states=ext_budget,
                        max_refine_iters=3,
                        no_filler=True,
                        verbose=True,
                        all_orders_override=None,
                        no_compile=True,
                    )
                except Exception as e:
                    print(f"  [gpu_seq] Extended refine error: {e}", file=sys.stderr)
                    break

                if self._solve_gen != final_gen:
                    break

                if score > best_score and actions:
                    best_score = score
                    best_actions = actions
                    ms_r = self._map_ref()
                    cap_r = self._capture_snapshot()
                    try:
                        exp_pos = predict_full_sim(actions, cap_r, ms_r)
                        goals = extract_goals(actions, ms_r, exp_pos)
                    except Exception as e:
                        print(f"  [gpu_seq] Extended sim failed: {e}", file=sys.stderr)
                        exp_pos = None
                        goals = None
                    self._update_plan(score, actions, exp_pos, goals,
                                      'gpu_seq_ext', final_gen)
                    print(f"  [gpu_seq] Extended refinement improved: {best_score}",
                          file=sys.stderr)
                    continue  # Immediately try another pass after improvement

            print(f"  [gpu_seq] Waiting for new orders (gen={final_gen})...",
                  file=sys.stderr)
            while self._solve_gen == final_gen:
                time.sleep(0.5)

    # ── round 0 initialization ────────────────────────────────────────────────

    def _init_round0(self, data: dict) -> None:
        """Initialize from round 0 WebSocket data. Starts background threads."""
        capture = ws_to_capture(data)
        self._difficulty = capture['difficulty']
        self._num_bots = capture['num_bots']
        self._num_rounds = data.get('max_rounds', DIFF_ROUNDS.get(self._difficulty, 300))

        # Decode and store JWT payload (map_id, map_seed, etc.)
        jwt_payload = decode_jwt_from_url(self.ws_url)
        if jwt_payload:
            capture['jwt'] = jwt_payload
            print(f"  [jwt] map_id={jwt_payload.get('map_id')} "
                  f"map_seed={jwt_payload.get('map_seed')} "
                  f"exp={jwt_payload.get('exp')}", file=sys.stderr)

        # Track order IDs seen so far (ws_to_capture doesn't store ids)
        for i, order in enumerate(data.get('orders', [])):
            oid = order.get('id', f'order_{i}')
            self._seen_order_ids.add(oid)
            self._order_id_to_idx[oid] = i

        # Pre-load existing capture orders so GPU starts solving with full order list.
        # Orders already in seen_order_ids (from live round 0) are skipped.
        # When those orders appear later in the live game, gen won't be bumped.
        if self.preload_capture:
            diff = self._difficulty
            preloaded_cap = None
            try:
                from solution_store import load_capture as db_load_capture
                preloaded_cap = db_load_capture(diff)
                if preloaded_cap:
                    print(f"  [preload] Loaded capture from DB: "
                          f"{len(preloaded_cap.get('orders', []))} orders",
                          file=sys.stderr)
                else:
                    print(f"  [preload] No capture in DB for {diff}", file=sys.stderr)
            except Exception as e:
                print(f"  [preload] DB load failed: {e}", file=sys.stderr)

            if preloaded_cap:
                injected = 0
                for order in preloaded_cap.get('orders', []):
                    oid = order.get('id', f'order_{len(self._order_id_to_idx)}')
                    if oid not in self._seen_order_ids:
                        self._seen_order_ids.add(oid)
                        idx = len(self._order_id_to_idx)
                        self._order_id_to_idx[oid] = idx
                        capture['orders'].append({
                            'id': oid,
                            'items_required': list(order['items_required']),
                            'items_delivered': list(order.get('items_delivered', [])),
                            'status': 'future',
                        })
                        injected += 1
                print(f"  [preload] Injected {injected} orders (total={len(capture['orders'])})",
                      file=sys.stderr)

        with self._lock:
            self._capture = capture
            self._map_state = build_map_from_capture(capture)
            self._walkable = build_walkable(self._map_state)

        # ── Aisle / corridor detection for anti-congestion ───────────────
        ms = self._map_state
        walkable = self._walkable
        # Aisle columns: x where walkable cells are flanked by shelf/wall on both sides
        self._aisle_cols = set()
        for x in range(ms.width):
            col_walkable = [(x, y) for y in range(ms.height) if (x, y) in walkable]
            if len(col_walkable) < 2:
                continue
            flanked = 0
            for (cx, cy) in col_walkable:
                left = ms.grid[cy, cx - 1] if cx > 0 else CELL_WALL
                right = ms.grid[cy, cx + 1] if cx < ms.width - 1 else CELL_WALL
                if left in (CELL_WALL, CELL_SHELF) and right in (CELL_WALL, CELL_SHELF):
                    flanked += 1
            if flanked >= len(col_walkable) * 0.6:
                self._aisle_cols.add(x)
        # Corridor rows: y where >60% of cells are walkable (horizontal cross-aisles)
        self._corridor_rows = set()
        for y in range(ms.height):
            row_walk = sum(1 for x in range(ms.width) if (x, y) in walkable)
            if row_walk > ms.width * 0.6:
                self._corridor_rows.add(y)
        # Bottom corridor: max y in corridor rows (typically the main highway)
        self._bottom_corridor_y = max(self._corridor_rows) if self._corridor_rows else ms.height - 2
        # Per-bot position history and stall tracking
        self._bot_pos_history = {}   # bot_id → list of last 6 positions
        self._bot_stall_count = {}   # bot_id → rounds stuck at same position

        if self._difficulty == 'nightmare':
            # Nightmare: defer PrecomputedTables + V3 to background thread
            # so round 0 responds in <100ms (greedy fallback until ready).
            self._tables = None
            self._nm_solver_v3 = None
            def _bg_init_nightmare():
                try:
                    tables = PrecomputedTables.get(self._map_state)
                    self._tables = tables
                    print(f"  [bg] PrecomputedTables ready", file=sys.stderr)
                except Exception as e:
                    print(f"  [bg] PrecomputedTables failed: {e}", file=sys.stderr)
                    return
                try:
                    from nightmare_cascade_solver import CascadeSolver
                    future_orders = self._load_nightmare_future_orders()
                    self._nm_solver_v3 = CascadeSolver(
                        self._map_state, tables,
                        future_orders=future_orders)
                    print(f"  [bg] CascadeSolver ready ({len(future_orders)} orders)", file=sys.stderr)
                except Exception as e:
                    print(f"  [bg] CascadeSolver init failed: {e}", file=sys.stderr)
                    # Fallback to V3
                    try:
                        from nightmare_solver_v2 import NightmareSolverV3
                        self._nm_solver_v3 = NightmareSolverV3(
                            self._map_state, tables,
                            future_orders=future_orders)
                        print(f"  [bg] Fallback to V3", file=sys.stderr)
                    except Exception as e2:
                        print(f"  [bg] V3 fallback failed: {e2}", file=sys.stderr)
            threading.Thread(target=_bg_init_nightmare, daemon=True).start()
        else:
            try:
                self._tables = PrecomputedTables.get(self._map_state)
            except Exception as e:
                print(f"  [init] PrecomputedTables unavailable: {e}", file=sys.stderr)
                self._tables = None

        # ── Multi-dropoff zone setup (nightmare: 3 zones) ──────────────────
        self._drop_off_zones = list(ms.drop_off_zones)
        num_zones = len(self._drop_off_zones)
        for zi, dz in enumerate(self._drop_off_zones):
            self._zone_dropoff[zi] = tuple(dz)

        if num_zones > 1:
            # Assign items to zones by shelf x-coordinate proximity to zone dropoffs
            zone_x_bounds = sorted(self._drop_off_zones, key=lambda z: z[0])
            for zi in range(num_zones):
                self._zone_shelves[zi] = set()
            for item_idx, item in enumerate(ms.items):
                ix = item['position'][0]
                # Assign to zone whose dropoff x is closest
                best_zi = 0
                best_dx = abs(ix - zone_x_bounds[0][0])
                for zi, dz in enumerate(zone_x_bounds):
                    dx = abs(ix - dz[0])
                    if dx < best_dx:
                        best_dx = dx
                        best_zi = zi
                self._zone_shelves[best_zi].add(item_idx)

            # Assign bots to zones: distribute evenly by bot ID
            bots_per_zone = self._num_bots // num_zones
            remainder = self._num_bots % num_zones
            zi = 0
            count = 0
            zone_cap = bots_per_zone + (1 if zi < remainder else 0)
            for bid in range(self._num_bots):
                self._zone_for_bot[bid] = zi
                count += 1
                if count >= zone_cap and zi < num_zones - 1:
                    zi += 1
                    count = 0
                    zone_cap = bots_per_zone + (1 if zi < remainder else 0)

            print(f"  [zones] {num_zones} dropoff zones: {[tuple(dz) for dz in self._drop_off_zones]}",
                  file=sys.stderr)
            zone_bot_counts = {}
            for bid, zi in self._zone_for_bot.items():
                zone_bot_counts[zi] = zone_bot_counts.get(zi, 0) + 1
            print(f"  [zones] Bot assignment: {zone_bot_counts}", file=sys.stderr)
            for zi, s in self._zone_shelves.items():
                print(f"  [zones] Zone {zi} ({tuple(self._zone_dropoff[zi])}): {len(s)} items",
                      file=sys.stderr)
        else:
            # Single dropoff — all bots to zone 0
            self._zone_shelves[0] = set(range(len(ms.items)))
            for bid in range(self._num_bots):
                self._zone_for_bot[bid] = 0

        self._data_ready.set()

        print(f"  [init] {self._difficulty} {self._num_bots}bots "
              f"orders={len(capture['orders'])} walkable={len(self._walkable)}",
              file=sys.stderr)

        # Pre-load best existing solution as initial plan warm seed.
        # This gives the live solver a strong starting point — all background
        # threads try to improve from this plan instead of starting from zero.
        if self.preload_capture:
            self._preload_solution()

        # Build per-round searcher (no locked bots) for immediate fast actions
        gen = self._solve_gen

        # Nightmare (20 bots): skip ALL background GPU/MAPF threads.
        # They block the GIL for 2+ seconds, causing round timeouts.
        # Pure greedy with zone-aware routing is the only viable approach.
        if self._difficulty == 'nightmare':
            print(f"  [init] Nightmare: greedy-only mode (no MAPF/GPU background)",
                  file=sys.stderr)
        else:
            self._rebuild_pr_searcher()
            self._start_mapf(gen)

            # Single-bot: full sequential GPU DP (completes in ~3s, optimal).
            # Multi-bot: GPU warm-start refinement from MAPF plan (much faster than full DP).
            # Pipeline mode: skip offline GPU passes — they block the GIL for 3-4s per pass,
            # causing the asyncio event loop to miss the server's 2s round timeout.
            # The pipeline runs optimize_and_save.py in a separate process instead.
            if self.pipeline_mode:
                print(f"  [init] Pipeline mode: offline GPU disabled (greedy + per-round GPU only)",
                      file=sys.stderr)
            else:
                if self._num_bots == 1:
                    gpu_t = threading.Thread(target=self._gpu_worker, daemon=True)
                    gpu_t.start()
                else:
                    gpu_r = threading.Thread(target=self._gpu_refine_worker, daemon=True)
                    gpu_r.start()
            pr_t = threading.Thread(target=self._per_round_gpu_worker, daemon=True)
            pr_t.start()

    # ── order tracking ────────────────────────────────────────────────────────

    def _check_new_orders(self, data: dict) -> None:
        """Detect new orders in incoming data. Update capture and bump gen."""
        new_count = 0
        for order in data.get('orders', []):
            oid = order.get('id', f'order_{len(self._seen_order_ids)}')
            if oid not in self._seen_order_ids:
                self._seen_order_ids.add(oid)
                idx = len(self._order_id_to_idx)
                self._order_id_to_idx[oid] = idx
                with self._lock:
                    if self._capture is not None:
                        self._capture['orders'].append({
                            'id': oid,
                            'items_required': order['items_required'],
                            'items_delivered': list(order.get('items_delivered', [])),
                            'status': order['status'],
                        })
                new_count += 1

        if new_count > 0:
            total = len(self._seen_order_ids)
            with self._lock:
                self._solve_gen += 1
                gen = self._solve_gen
            print(f"  [orders] +{new_count} new (total={total}), gen→{gen}", file=sys.stderr)
            if self._difficulty == 'nightmare':
                # Feed discovered orders into V3's future_orders for chain planning
                if hasattr(self, '_nm_solver_v3') and self._nm_solver_v3 is not None:
                    ms = self._map_state
                    for order in data.get('orders', []):
                        oid = order.get('id', '')
                        req_names = order.get('items_required', [])
                        req_ids = [ms.type_name_to_id.get(n, 0) for n in req_names]
                        from game_engine import Order as GEOrder
                        new_order = GEOrder(total, req_ids, 'future')
                        # Avoid duplicates by checking required types
                        req_key = tuple(sorted(req_ids))
                        existing_keys = {tuple(sorted(o.required))
                                         for o in self._nm_solver_v3.future_orders}
                        if req_key not in existing_keys:
                            self._nm_solver_v3.future_orders.append(new_order)
                    print(f"  [nightmare] V3 future_orders: {len(self._nm_solver_v3.future_orders)}",
                          file=sys.stderr)
            else:
                self._start_mapf(gen)
                # Rebuild per-round searcher with updated order list
                threading.Thread(target=self._rebuild_pr_searcher, daemon=True).start()

    # ── per-round GPU DP ──────────────────────────────────────────────────────

    def _rebuild_pr_searcher(self) -> None:
        """Build a fresh GPUBeamSearcher (no locked bots) from current capture.

        Called at round 0 and whenever new orders arrive. Fast because
        PrecomputedTables are cached; only order tensor needs rebuilding.
        """
        try:
            from gpu_beam_search import GPUBeamSearcher
            import torch

            capture = self._capture_snapshot()
            if capture is None:
                return

            with self._lock:
                gen = self._solve_gen
                ms = self._map_state

            if ms is None:
                return

            gs, all_orders = init_game_from_capture(
                capture, num_orders=len(capture['orders']))

            dev = self.device
            try:
                import torch as _torch
                if not _torch.cuda.is_available():
                    dev = 'cpu'
            except ImportError:
                dev = 'cpu'

            searcher = GPUBeamSearcher(
                ms, all_orders, device=dev, num_bots=self._num_bots,
                no_compile=True, num_rounds=self._num_rounds)

            with self._pr_lock:
                self._pr_searcher = searcher
                self._pr_all_orders = all_orders
                self._pr_searcher_gen = gen
                self._pr_actions.clear()

            print(f"  [pr] Searcher built: {len(all_orders)} orders, gen={gen}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [pr] Searcher build failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    def _build_bot_gpu_state(self, bot: dict, ws_data: dict,
                             device: str | None = None,
                             locked_positions: list[tuple[int, int]] | None = None,
                             locked_inventories: list[list[int]] | None = None
                             ) -> dict[str, Any] | None:
        """Convert WS bot data to GPU initial state dict for per-round DP.

        Args:
            bot: WS bot dict.
            ws_data: Full WS round data.
            device: Target device (default: from _pr_searcher).
            locked_positions: Optional list of (x, y) for locked bots'
                              ACTUAL current positions. If provided, adds
                              locked_bx/locked_by/locked_inv to the state
                              so the DP starts locked bots at their real
                              positions (not spawn).
            locked_inventories: Optional list of [INV_CAP] int8 arrays (type IDs)
                              for locked bots' ACTUAL current inventories.
                              If None, locked bots start with empty inventories.

        Returns state dict with keys: bot_x, bot_y, bot_inv, active_idx,
        active_del, score, orders_comp — all [1]-shaped tensors on device.
        Returns None on failure.
        """
        try:
            import torch

            ms = self._map_state
            if ms is None:
                return None

            if device is None:
                with self._pr_lock:
                    searcher = self._pr_searcher
                if searcher is None:
                    return None
                dev = searcher.device
            else:
                dev = device

            bx, by = bot['position']

            # Inventory: type name strings → type IDs
            inv = [-1] * INV_CAP
            for s, type_name in enumerate(bot.get('inventory', [])):
                if s < INV_CAP:
                    inv[s] = ms.type_name_to_id.get(type_name, -1)

            # Find active order index and delivery mask
            active_idx = 0
            active_del = np.zeros(MAX_ORDER_SIZE, dtype=np.int8)

            for order in ws_data.get('orders', []):
                if order.get('status') != 'active':
                    continue
                oid = order.get('id', '')
                active_idx = self._order_id_to_idx.get(oid, 0)

                # Reconstruct delivery mask via greedy slot matching
                required = order.get('items_required', [])
                delivered_names = list(order.get('items_delivered', []))
                # Count delivered items by type
                from collections import Counter
                del_counts = Counter(delivered_names)
                for slot, type_name in enumerate(required):
                    if slot >= MAX_ORDER_SIZE:
                        break
                    if del_counts.get(type_name, 0) > 0:
                        active_del[slot] = 1
                        del_counts[type_name] -= 1
                break  # only one active order

            state = {
                'bot_x': torch.tensor([bx], dtype=torch.int16, device=dev),
                'bot_y': torch.tensor([by], dtype=torch.int16, device=dev),
                'bot_inv': torch.tensor([inv], dtype=torch.int8, device=dev),
                'active_idx': torch.tensor([active_idx], dtype=torch.int32, device=dev),
                'active_del': torch.tensor([active_del], dtype=torch.int8, device=dev),
                'score': torch.tensor([0], dtype=torch.int32, device=dev),
                'orders_comp': torch.tensor([0], dtype=torch.int32, device=dev),
            }

            # Locked bot initial positions and inventories (actual current state)
            if locked_positions:
                num_locked = len(locked_positions)
                if locked_inventories is not None:
                    # Use actual inventories from WS data
                    linv_arr = np.array(locked_inventories, dtype=np.int8)  # [num_locked, INV_CAP]
                else:
                    linv_arr = np.full((num_locked, INV_CAP), -1, dtype=np.int8)
                state['locked_inv'] = torch.tensor(
                    linv_arr[np.newaxis], dtype=torch.int8, device=dev)  # [1, L, INV_CAP]
                state['locked_bx'] = torch.tensor(
                    [[x for x, y in locked_positions]], dtype=torch.int16, device=dev)
                state['locked_by'] = torch.tensor(
                    [[y for x, y in locked_positions]], dtype=torch.int16, device=dev)

            return state
        except Exception as e:
            print(f"  [pr] build_bot_gpu_state failed: {e}", file=sys.stderr)
            return None

    def _simulate_single_bot(self, start_pos, planned_acts, start_rnd, horizon):
        """Simulate one bot's trajectory from start_pos using planned_acts.

        Returns list of (x, y) positions for rounds start_rnd..start_rnd+horizon-1,
        representing the bot's position AFTER each action.
        """
        walkable = self._walkable
        bx, by = start_pos
        positions = []
        for i in range(min(horizon, len(planned_acts))):
            act, _item = planned_acts[i]
            if ACT_MOVE_UP <= act <= ACT_MOVE_RIGHT:
                nx = bx + DX[act]
                ny = by + DY[act]
                if (nx, ny) in walkable:
                    bx, by = nx, ny
            positions.append((bx, by))
        # Pad with final position if planned_acts is shorter than horizon
        while len(positions) < horizon:
            positions.append((bx, by))
        return positions

    def _build_seq_locked_trajs(self, planned_bots, start_rnd):
        """Build locked_trajectories from sequentially planned bots.

        Args:
            planned_bots: list of (real_bot_id, planned_acts, positions) where
                          planned_acts: list of (act, item) for horizon rounds
                          positions: list of (x, y) for horizon rounds (after action)
            start_rnd: first round index for planned_acts[0]

        Returns locked_trajectories dict for GPUBeamSearcher.__init__.
        """
        num_locked = len(planned_bots)
        locked_actions = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int8)
        locked_action_items = np.full((num_locked, MAX_ROUNDS), -1, dtype=np.int16)
        locked_pos_x = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
        locked_pos_y = np.zeros((num_locked, MAX_ROUNDS), dtype=np.int16)
        locked_bot_ids = []

        for li, (real_bid, planned_acts, positions) in enumerate(planned_bots):
            locked_bot_ids.append(real_bid)
            for i, ((act, item), (px, py)) in enumerate(zip(planned_acts, positions)):
                r = start_rnd + i
                if r >= MAX_ROUNDS:
                    break
                locked_actions[li, r] = act
                locked_action_items[li, r] = item if item is not None and item >= 0 else -1
                locked_pos_x[li, r] = px
                locked_pos_y[li, r] = py
            # Fill rounds before start_rnd with the bot's starting position
            if positions:
                first_x, first_y = positions[0]
            else:
                first_x, first_y = 0, 0
            for r in range(start_rnd):
                locked_pos_x[li, r] = first_x
                locked_pos_y[li, r] = first_y
            # Fill rounds after planned horizon with the last position
            if positions:
                last_x, last_y = positions[-1]
                last_r = start_rnd + len(positions)
                for r in range(last_r, MAX_ROUNDS):
                    locked_pos_x[li, r] = last_x
                    locked_pos_y[li, r] = last_y

        return {
            'locked_actions': locked_actions,
            'locked_action_items': locked_action_items,
            'locked_pos_x': locked_pos_x,
            'locked_pos_y': locked_pos_y,
            'locked_bot_ids': locked_bot_ids,
        }

    @staticmethod
    def _pr_priority_key(bot, drop_pos, tables):
        """Sort key for per-round GPU worker priority ordering.

        Delivery bots (carrying active-order items) first, by distance to dropoff.
        Others last.
        """
        bpos = tuple(bot['position'])
        inv = bot.get('inventory', [])
        # Heuristic: any non-empty inventory bot is "potentially delivering"
        # (we don't have active order data here, use non-empty as proxy)
        if inv:
            if tables is not None:
                try:
                    dist = int(tables.get_distance(bpos, drop_pos))
                except Exception:
                    # Table lookup can fail if position is outside precomputed grid; fall back to Manhattan
                    dist = abs(bpos[0] - drop_pos[0]) + abs(bpos[1] - drop_pos[1])
            else:
                dist = abs(bpos[0] - drop_pos[0]) + abs(bpos[1] - drop_pos[1])
            return (0, dist, bot['id'])
        return (1, 0, bot['id'])

    def _per_round_gpu_worker(self):
        """Background thread: runs GPU DP from actual state for each round.

        For single-bot: standard per-round DP (fast path).
        For multi-bot: SEQUENTIAL locked-bot planning. Each bot is planned
        with all previously-planned bots' trajectories locked in, giving
        proper multi-bot coordination from actual live positions.

        Bots are processed in priority order (delivery bots first).
        Results used by _get_actions() as fallback when plan is desynced.
        """
        while True:
            self._pr_event.wait(timeout=1.0)
            self._pr_event.clear()

            with self._pr_lock:
                rnd = self._pr_rnd
                ws_data = self._pr_ws_data
                searcher = self._pr_searcher
                all_orders = self._pr_all_orders

            if rnd < 0 or ws_data is None or searcher is None:
                continue

            diff = self._difficulty or 'easy'
            params = PR_PARAMS.get(diff, PR_PARAMS['easy'])
            max_states = params['max_states']
            remaining = MAX_ROUNDS - rnd
            horizon = min(params['horizon'], remaining)

            if horizon <= 0:
                continue

            t0 = time.time()
            live_bots = ws_data.get('bots', [])
            num_bots = len(live_bots)

            try:
                ms = self._map_state
                drop_pos = tuple(ms.drop_off) if ms else (0, 0)

                # Sort bots by priority: delivery bots first, then by dist to dropoff
                priority_sorted = sorted(
                    live_bots,
                    key=lambda b: self._pr_priority_key(b, drop_pos, self._tables))

                bot_acts_by_id = {}   # bot_id -> (act, item_idx)
                score_sum = 0

                if num_bots == 1:
                    # ── Single-bot fast path (no locked bots needed) ──────────────
                    bot = priority_sorted[0]
                    init_state = self._build_bot_gpu_state(
                        bot, ws_data, device=searcher.device)
                    if init_state is not None:
                        score, acts = searcher.dp_search(
                            game_state=None,
                            max_states=max_states,
                            verbose=False,
                            start_rnd=rnd,
                            max_rounds=horizon,
                            init_state=init_state,
                        )
                        bot_acts_by_id[bot['id']] = acts[0] if acts else (ACT_WAIT, -1)
                        score_sum = score
                    else:
                        bot_acts_by_id[bot['id']] = (ACT_WAIT, -1)

                else:
                    # ── Multi-bot: sequential locked-bot DP ──────────────────────
                    # Plan bots in priority order. Each bot is planned with all
                    # previously-planned bots locked to their actual trajectories,
                    # preventing conflicts and enabling coordination.
                    # A new GPUBeamSearcher is created per bot (cheap: BFS tables cached).
                    from gpu_beam_search import GPUBeamSearcher

                    dev = searcher.device
                    planned_bots = []  # (real_bid, planned_acts, positions)
                    locked_positions_cur = []   # current (x,y) of locked bots
                    locked_inventories_cur = [] # current inventories of locked bots [INV_CAP]
                    # Map WS bot id → game engine index (0..N-1)
                    bot_id_to_idx = {b['id']: i for i, b in enumerate(live_bots)}

                    n_bots_total = len(priority_sorted)
                    for rank, bot in enumerate(priority_sorted):
                        real_bid = bot_id_to_idx.get(bot['id'], len(planned_bots))
                        bot_pos = tuple(bot['position'])

                        # P1: Adaptive time budget — hard cutoff when running low
                        remaining_time = 1.5 - (time.time() - t0)
                        if remaining_time < 0.3 and rank > 0:
                            bot_acts_by_id[bot['id']] = (ACT_WAIT, -1)
                            planned_bots.append((real_bid, [(ACT_WAIT, -1)] * horizon,
                                                 [bot_pos] * horizon))
                            locked_positions_cur.append(bot_pos)
                            locked_inventories_cur.append([-1] * INV_CAP)
                            continue

                        # Priority-scale state budget and horizon
                        scale = 1.5 - rank * (1.0 / max(1, n_bots_total - 1))
                        scaled_states = max(5000, int(max_states * scale))
                        bot_horizon = min(max(horizon // 2, 20) if rank >= 3 else horizon, remaining)

                        # Build locked trajectories from previously planned bots
                        locked_trajs = None
                        if planned_bots:
                            locked_trajs = self._build_seq_locked_trajs(planned_bots, rnd)

                        # Build init_state with locked bot positions AND inventories
                        init_state = self._build_bot_gpu_state(
                            bot, ws_data, device=dev,
                            locked_positions=locked_positions_cur if locked_positions_cur else None,
                            locked_inventories=locked_inventories_cur if locked_inventories_cur else None)
                        if init_state is None:
                            bot_acts_by_id[bot['id']] = (ACT_WAIT, -1)
                            # Treat as stationary locked bot with empty inv
                            planned_bots.append((real_bid, [(ACT_WAIT, -1)] * bot_horizon,
                                                 [bot_pos] * bot_horizon))
                            locked_positions_cur.append(bot_pos)
                            locked_inventories_cur.append([-1] * INV_CAP)
                            continue

                        # Create searcher with locked trajectories for this bot
                        if locked_trajs:
                            cur_searcher = GPUBeamSearcher(
                                ms, all_orders, device=dev,
                                num_bots=num_bots,
                                locked_trajectories=locked_trajs,
                                no_compile=True,
                                num_rounds=self._num_rounds)
                        else:
                            cur_searcher = searcher  # first bot: reuse base searcher

                        score, acts = cur_searcher.dp_search(
                            game_state=None,
                            max_states=scaled_states,
                            verbose=False,
                            start_rnd=rnd,
                            max_rounds=bot_horizon,
                            init_state=init_state,
                            bot_id=real_bid,
                        )

                        if locked_trajs:
                            del cur_searcher  # free immediately to save GPU memory

                        full_acts = acts if acts else [(ACT_WAIT, -1)] * bot_horizon
                        positions = self._simulate_single_bot(bot_pos, full_acts, rnd, bot_horizon)
                        planned_bots.append((real_bid, full_acts, positions))
                        locked_positions_cur.append(bot_pos)
                        # Record actual inventory for this bot (for subsequent locked bots)
                        bot_inv_ids = []
                        for type_name in bot.get('inventory', []):
                            bot_inv_ids.append(ms.type_name_to_id.get(type_name, -1))
                        while len(bot_inv_ids) < INV_CAP:
                            bot_inv_ids.append(-1)
                        locked_inventories_cur.append(bot_inv_ids[:INV_CAP])

                        first_act = full_acts[0] if full_acts else (ACT_WAIT, -1)
                        bot_acts_by_id[bot['id']] = first_act
                        score_sum += score

                elapsed = time.time() - t0

                # Format as WS actions in priority order for conflict resolution
                ws_actions_by_priority = []
                for bot in priority_sorted:
                    act, item_idx = bot_acts_by_id.get(bot['id'], (ACT_WAIT, -1))
                    a = {'bot': bot['id'], 'action': ACTION_NAMES[act]}
                    if act == ACT_PICKUP and ms and 0 <= item_idx < len(ms.items):
                        a['item_id'] = ms.items[item_idx]['id']
                    ws_actions_by_priority.append((a, bot))

                # Sequential locked-bot already avoids conflicts; this is a safety net
                self._resolve_pr_conflicts_ordered(ws_actions_by_priority)

                # Rebuild ws_actions in original bot list order
                act_map_by_id = {a['bot']: a for a, _ in ws_actions_by_priority}
                ws_actions = [act_map_by_id.get(bot['id'],
                              {'bot': bot['id'], 'action': 'wait'})
                              for bot in live_bots]

                with self._pr_lock:
                    self._pr_actions[rnd] = (ws_actions, score_sum)
                    old_keys = [k for k in self._pr_actions if k < rnd - 10]
                    for k in old_keys:
                        del self._pr_actions[k]

                print(f"  [pr] R{rnd} score_est={score_sum} elapsed={elapsed:.2f}s "
                      f"horizon={horizon} bots={num_bots} max_states={max_states}",
                      file=sys.stderr)

            except Exception as e:
                print(f"  [pr] Error R{rnd}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    def _resolve_pr_conflicts_ordered(self, ws_actions_with_bots):
        """Priority-ordered conflict resolution for per-round GPU actions.

        ws_actions_with_bots: list of (action_dict, bot) in PRIORITY order.
        Higher-priority bots (earlier in list) win cell conflicts.
        Modifies action dicts in-place.
        """
        _act_map = {
            'move_up': ACT_MOVE_UP, 'move_down': ACT_MOVE_DOWN,
            'move_left': ACT_MOVE_LEFT, 'move_right': ACT_MOVE_RIGHT,
        }
        claimed_cells = set()

        for ws_act, bot in ws_actions_with_bots:
            action_name = ws_act.get('action', 'wait')
            act_int = _act_map.get(action_name)
            if act_int is not None:
                bx, by = bot['position']
                target = (bx + DX[act_int], by + DY[act_int])
                if target in claimed_cells:
                    ws_act['action'] = 'wait'
                else:
                    claimed_cells.add(target)
            # dropoff: multiple bots can be at dropoff, no conflict

    def _resolve_pr_conflicts(self, ws_actions, live_bots):
        """Basic conflict resolution for per-round GPU actions (legacy).

        If multiple bots plan to move to the same cell, later bots wait instead.
        Modifies ws_actions in-place.
        """
        self._resolve_pr_conflicts_ordered(list(zip(ws_actions, live_bots)))

    # ── greedy fallback ───────────────────────────────────────────────────────

    def _dist(self, a, b):
        """O(1) distance lookup via PrecomputedTables, falling back to Manhattan."""
        if self._tables is not None:
            try:
                return int(self._tables.get_distance(a, b))
            except Exception:
                pass  # Table lookup failed; fall back to Manhattan distance below
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_dropoff(self, pos):
        """Return the nearest dropoff zone (x,y) to pos using BFS distance."""
        if len(self._drop_off_zones) <= 1:
            return tuple(self._map_state.drop_off)
        best_dz = tuple(self._drop_off_zones[0])
        best_d = self._dist(pos, best_dz)
        for dz in self._drop_off_zones[1:]:
            d = self._dist(pos, tuple(dz))
            if d < best_d:
                best_d = d
                best_dz = tuple(dz)
        return best_dz

    def _bot_dropoff(self, bot_id, bot_pos):
        """Return the dropoff zone for this bot.

        For multi-zone maps, uses the bot's assigned zone.
        Falls back to nearest zone if assignment doesn't exist.
        """
        if len(self._drop_off_zones) <= 1:
            return tuple(self._map_state.drop_off)
        zi = self._zone_for_bot.get(bot_id)
        if zi is not None and zi in self._zone_dropoff:
            return self._zone_dropoff[zi]
        return self._nearest_dropoff(bot_pos)

    def _at_any_dropoff(self, pos):
        """Check if pos is at ANY dropoff zone."""
        for dz in self._drop_off_zones:
            if pos == tuple(dz):
                return True
        return False

    def _build_aisle_blocked(self, my_pos, occupied):
        """Block non-corridor cells in aisle columns that have other bots.

        For each aisle column containing at least one bot (other than the bot at
        my_pos), block walkable cells in that column that are NOT in corridor rows.
        Corridor cells stay open so horizontal movement and dropoff access work.

        This prevents bots from entering occupied aisle segments (the main cause
        of 10-bot deadlocks) while preserving horizontal highway traffic.

        Returns a new set: occupied | aisle_blocked_cells.
        """
        if not self._aisle_cols or self._num_bots < 5:
            return occupied
        blocked = set(occupied)
        my_col = my_pos[0]
        for ax in self._aisle_cols:
            # Don't block our own aisle (we need to exit it)
            if ax == my_col:
                continue
            # Check if any OTHER bot is in a non-corridor cell of this aisle
            has_other_in_aisle = any(
                ox == ax and oy not in self._corridor_rows and (ox, oy) != my_pos
                for (ox, oy) in occupied
            )
            if has_other_in_aisle:
                # Block non-corridor walkable cells in this aisle column
                for y in range(self._map_state.height):
                    if y not in self._corridor_rows and (ax, y) in self._walkable:
                        blocked.add((ax, y))
        return blocked

    @staticmethod
    def _bfs_strict(start, goal, walkable, blocked):
        """BFS avoiding `blocked` cells at EVERY step (not just first).

        Unlike bfs_next_action, this finds paths that don't pass through any
        currently occupied cell — except the goal itself (delivery bots can step
        into an occupied dropoff to queue for delivery).

        Returns action constant, or ACT_WAIT if no clear path exists.
        """
        from collections import deque
        if start == goal:
            return ACT_WAIT
        queue = deque()
        visited = {start: None}
        for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            nx, ny = start[0] + DX[act], start[1] + DY[act]
            if (nx, ny) not in walkable:
                continue
            if (nx, ny) in blocked and (nx, ny) != goal:
                continue
            if (nx, ny) not in visited:
                visited[(nx, ny)] = act
                if (nx, ny) == goal:
                    return act
                queue.append((nx, ny))
        while queue:
            cx, cy = queue.popleft()
            first_act = visited[(cx, cy)]
            for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                nx, ny = cx + DX[act], cy + DY[act]
                if (nx, ny) not in walkable:
                    continue
                if (nx, ny) in blocked and (nx, ny) != goal:
                    continue
                if (nx, ny) not in visited:
                    visited[(nx, ny)] = first_act
                    if (nx, ny) == goal:
                        return first_act
                    queue.append((nx, ny))
        return ACT_WAIT  # no clear path

    def _greedy_action(self, bot, data, occupied, target_types=None, allow_preview=True):
        """Greedy BFS action for one bot. Reads live WS data.

        Returns (act, item_idx, claimed_type) where claimed_type is the item type
        this bot is targeting (for coordination in _greedy_all), or None.

        target_types: set of type names this bot is allowed to target.
            If None, computed automatically from the active order.
        """
        bx, by = bot['position']
        inventory = list(bot.get('inventory', []))  # list of type-name strings
        inv_full = len(inventory) >= INV_CAP

        ms = self._map_state
        walkable = self._walkable
        # Use per-bot zone dropoff (nearest for multi-zone maps)
        drop_off = self._bot_dropoff(bot['id'], (bx, by))

        # 1. At ANY dropoff → deliver ONLY if we have items the active order still needs.
        at_dropoff = self._at_any_dropoff((bx, by))
        if at_dropoff:
            active_needs = set()
            for order in data.get('orders', []):
                if order.get('status') != 'active':
                    continue
                counts = {}
                for t in order.get('items_required', []):
                    counts[t] = counts.get(t, 0) + 1
                for t in order.get('items_delivered', []):
                    if t in counts:
                        counts[t] -= 1
                for t, c in counts.items():
                    if c > 0:
                        active_needs.add(t)
                break  # one active order
            if inventory and any(t in active_needs for t in inventory):
                return (ACT_DROPOFF, -1, None)
            # At dropoff with dead inventory (or empty) — move 1 step away to clear the tile.
            if not target_types:
                for _a in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                    _nx, _ny = bx + DX[_a], by + DY[_a]
                    if (_nx, _ny) in walkable and (_nx, _ny) not in occupied:
                        return (_a, -1, None)
                return (ACT_WAIT, -1, None)

        # 2. If target_types not pre-computed, compute from order minus own inventory
        if target_types is None:
            target_types = set()
            for order in data.get('orders', []):
                if order.get('status') != 'active':
                    continue
                counts = {}
                for t in order['items_required']:
                    counts[t] = counts.get(t, 0) + 1
                for t in order.get('items_delivered', []):
                    if t in counts:
                        counts[t] -= 1
                for t in inventory:
                    if t in counts:
                        counts[t] -= 1
                for t, c in counts.items():
                    if c > 0:
                        target_types.add(t)

        # 3. Inventory full → deliver or wait if dead inventory
        if inv_full:
            if at_dropoff:
                for _a in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                    _nx, _ny = bx + DX[_a], by + DY[_a]
                    if (_nx, _ny) in walkable and (_nx, _ny) not in occupied:
                        act = _a
                        break
                else:
                    act = ACT_WAIT
            else:
                # Route to nearest dropoff — dead or active inventory
                act = bfs_next_action((bx, by), drop_off, walkable, occupied, ms)
            return (act, -1, None)

        # 4. Pick up nearest target item.
        # If no active-order targets, fall back to preview-order prefetch.
        if allow_preview and not target_types and not inv_full:
            for order in data.get('orders', []):
                if order.get('status') != 'preview':
                    continue
                counts = {}
                for t in order['items_required']:
                    counts[t] = counts.get(t, 0) + 1
                for t in order.get('items_delivered', []):
                    if t in counts:
                        counts[t] -= 1
                for t in inventory:
                    if t in counts:
                        counts[t] -= 1
                target_types = {t for t, c in counts.items() if c > 0}
                break

        if target_types:
            best_idx, best_dist, best_adj = -1, 999999, None
            for item_idx, item in enumerate(ms.items):
                if item.get('type', '') not in target_types:
                    continue
                for adj in ms.item_adjacencies.get(item_idx, []):
                    d = self._dist((bx, by), adj)
                    if d < best_dist:
                        best_dist = d
                        best_idx = item_idx
                        best_adj = adj

            if best_adj is not None:
                claimed = ms.items[best_idx].get('type', '') if best_idx >= 0 else None
                if (bx, by) == best_adj:
                    return (ACT_PICKUP, best_idx, claimed)
                act = bfs_next_action((bx, by), best_adj, walkable, occupied, ms)
                return (act, -1, claimed)

        # 5. Have items but nothing more to pick up → deliver to nearest dropoff
        if inventory:
            act = bfs_next_action((bx, by), drop_off, walkable, occupied, ms)
            return (act, -1, None)

        # 6. Nothing to do → just wait
        return (ACT_WAIT, -1, None)

    def _smart_idle_action(self, bot, data, occupied, claimed_types=None):
        """Action for idle bots: target preview items with type coordination.

        Returns (act, item_idx, claimed_type).
        claimed_types: mutable set of types already claimed by other idle bots.
        """
        if claimed_types is None:
            claimed_types = set()
        inv = list(bot.get('inventory', []))

        # Find preview order types not yet claimed by other idle bots
        preview_targets = set()
        for order in data.get('orders', []):
            if order.get('status') != 'preview':
                continue
            counts = {}
            for t in order['items_required']:
                counts[t] = counts.get(t, 0) + 1
            for t in order.get('items_delivered', []):
                if t in counts:
                    counts[t] -= 1
            for t in inv:
                if t in counts:
                    counts[t] -= 1
            for t, c in counts.items():
                if c > 0 and t not in claimed_types:
                    preview_targets.add(t)
            break  # first preview order only

        if preview_targets:
            act, item_idx, claimed = self._greedy_action(
                bot, data, occupied, target_types=preview_targets, allow_preview=False)
            if claimed:
                claimed_types.add(claimed)
            return (act, item_idx, claimed)

        # No unclaimed preview targets — regular greedy
        return self._greedy_action(bot, data, occupied)

    # ── per-round action computation ──────────────────────────────────────────

    def _format_ws(self, act, item_idx, bot_id):
        """Format an internal (act, item_idx) pair as WS action dict."""
        ms = self._map_state
        a = {'bot': bot_id, 'action': ACTION_NAMES[act]}
        if act == ACT_PICKUP:
            if 0 <= item_idx < len(ms.items):
                a['item_id'] = ms.items[item_idx]['id']
            else:
                a['action'] = 'wait'
        return a

    def _get_actions(self, rnd, data):
        """Compute WS actions for this round.

        Returns (ws_actions, source_label).

        Multi-bot priority: plan (synced, lenient 1/3 miss) > per-round GPU > greedy
          Plan (MAPF or gpu_refine) is re-computed each time new orders arrive.
          Lenient sync allows up to 1/3 of bots to be out of position before fallback.

        Single-bot priority: plan (synced) > per-round GPU > plan (recovery) > greedy
          (single-bot exact DP plan is optimal within known orders; per-round is fallback)
        """
        live_bots = data.get('bots', [])
        occupied = {tuple(b['position']) for b in live_bots}

        # Signal per-round GPU worker for this round
        with self._pr_lock:
            self._pr_rnd = rnd
            self._pr_ws_data = data
        self._pr_event.set()

        # Per-round GPU result (background thread computes for this round)
        with self._pr_lock:
            pr_result = self._pr_actions.get(rnd)

        # Snapshot plan and offset under lock
        with self._lock:
            plan = PlanState(
                score=self._plan.score,
                actions=self._plan.actions,
                expected_pos=self._plan.expected_pos,
                goals=self._plan.goals,
                source=self._plan.source,
            )
            offset = self._round_offset

        # ── Multi-bot: per-bot plan adherence ─────────────────────────────
        if self._num_bots and self._num_bots > 1:
            if plan.actions:
                dp_rnd = rnd - offset
                if 0 <= dp_rnd < len(plan.actions):
                    # ── Per-bot sync classification ──
                    synced_bots = []   # (bid_idx, bot)
                    desynced_bots = []  # (bid_idx, bot)
                    for bid_idx, bot in enumerate(live_bots):
                        if (plan.expected_pos and dp_rnd < len(plan.expected_pos)
                                and bid_idx < len(plan.expected_pos[dp_rnd])):
                            if tuple(bot['position']) == tuple(plan.expected_pos[dp_rnd][bid_idx]):
                                synced_bots.append((bid_idx, bot))
                            else:
                                desynced_bots.append((bid_idx, bot))
                        else:
                            synced_bots.append((bid_idx, bot))

                    # ── Round offset detection (only when some bots desynced) ──
                    if desynced_bots and plan.expected_pos and dp_rnd > 0:
                        prev = dp_rnd - 1
                        num_lb = len(live_bots)
                        max_miss = max(1, num_lb // 3)
                        if 0 <= prev < len(plan.expected_pos):
                            prev_mm = sum(
                                1 for bi, b in enumerate(live_bots)
                                if bi < len(plan.expected_pos[prev])
                                and tuple(b['position']) != tuple(plan.expected_pos[prev][bi])
                            )
                            if prev_mm <= max_miss:
                                with self._lock:
                                    self._round_offset += 1
                                offset += 1
                                dp_rnd = rnd - offset
                                # Re-classify with adjusted dp_rnd
                                if 0 <= dp_rnd < len(plan.actions):
                                    synced_bots = []
                                    desynced_bots = []
                                    for bid_idx, bot in enumerate(live_bots):
                                        if (plan.expected_pos and dp_rnd < len(plan.expected_pos)
                                                and bid_idx < len(plan.expected_pos[dp_rnd])):
                                            if tuple(bot['position']) == tuple(plan.expected_pos[dp_rnd][bid_idx]):
                                                synced_bots.append((bid_idx, bot))
                                            else:
                                                desynced_bots.append((bid_idx, bot))
                                        else:
                                            synced_bots.append((bid_idx, bot))

                    n_synced = len(synced_bots)
                    n_desynced = len(desynced_bots)

                    if n_synced > 0 and 0 <= dp_rnd < len(plan.actions):
                        ws_actions = [None] * len(live_bots)
                        greedy_occ = set(tuple(b['position']) for b in live_bots)
                        idle_claimed = set()   # preview types claimed by idle bots
                        committed_next = set()  # cells synced bots will occupy next

                        # Phase 1: Synced bots follow plan
                        for bid_idx, bot in synced_bots:
                            bx, by = bot['position']
                            if bid_idx < len(plan.actions[dp_rnd]):
                                act, item_idx = plan.actions[dp_rnd][bid_idx]
                            else:
                                act, item_idx = ACT_WAIT, -1
                            # Override WAIT for idle bots (no future goals)
                            if act == ACT_WAIT and plan.goals is not None:
                                bot_goals = plan.goals.get(bid_idx, [])
                                if not any(g[0] >= dp_rnd for g in bot_goals):
                                    g_act, g_item, g_claimed = self._smart_idle_action(
                                        bot, data, greedy_occ, idle_claimed)
                                    if g_act != ACT_WAIT:
                                        act, item_idx = g_act, g_item
                            # Track where this bot will be
                            if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                                committed_next.add((bx + DX[act], by + DY[act]))
                            else:
                                committed_next.add((bx, by))
                            ws_actions[bid_idx] = self._format_ws(act, item_idx, bot['id'])

                        # Phase 2: Desynced bots get recovery actions
                        pr_bot_map = {}
                        if pr_result is not None:
                            for pa in pr_result[0]:
                                pr_bot_map[pa['bot']] = pa
                        recovery_occ = greedy_occ | committed_next

                        for bid_idx, bot in desynced_bots:
                            bx, by = bot['position']
                            act, item_idx = ACT_WAIT, -1

                            # Try 1: per-round GPU action for this bot
                            if bot['id'] in pr_bot_map:
                                ws_actions[bid_idx] = pr_bot_map[bot['id']]
                                _am = {'move_up': ACT_MOVE_UP, 'move_down': ACT_MOVE_DOWN,
                                       'move_left': ACT_MOVE_LEFT, 'move_right': ACT_MOVE_RIGHT}
                                a_int = _am.get(pr_bot_map[bot['id']].get('action', 'wait'))
                                if a_int is not None:
                                    committed_next.add((bx + DX[a_int], by + DY[a_int]))
                                else:
                                    committed_next.add((bx, by))
                                continue

                            # Try 2: BFS toward next plan goal
                            if plan.goals and bid_idx in plan.goals:
                                for g_rnd, g_pos, g_act, g_item in plan.goals[bid_idx]:
                                    if g_rnd >= dp_rnd:
                                        if (bx, by) == g_pos:
                                            act, item_idx = g_act, g_item
                                        else:
                                            act = bfs_next_action(
                                                (bx, by), g_pos, self._walkable,
                                                recovery_occ, self._map_state)
                                        break

                            # Try 3: greedy fallback
                            if act == ACT_WAIT:
                                g_act, g_item, _ = self._greedy_action(
                                    bot, data, recovery_occ)
                                if g_act != ACT_WAIT:
                                    act, item_idx = g_act, g_item

                            # Track destination
                            if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                                committed_next.add((bx + DX[act], by + DY[act]))
                            else:
                                committed_next.add((bx, by))
                            ws_actions[bid_idx] = self._format_ws(act, item_idx, bot['id'])

                        source = (plan.source if n_desynced == 0
                                  else f'{plan.source}_hybrid({n_synced}s/{n_desynced}d)')
                        return ws_actions, source

            # Fallback: per-round GPU or greedy
            if pr_result is not None:
                pr_actions, pr_score = pr_result
                return pr_actions, 'per_round_gpu'
            return self._greedy_all(live_bots, data, occupied), 'greedy'

        # ── Single-bot: plan (synced) > per-round GPU > plan (recovery) > greedy ──
        if not plan.actions:
            # No offline plan yet — use per-round GPU if available, else greedy
            if pr_result is not None:
                pr_actions, pr_score = pr_result
                return pr_actions, 'per_round_gpu'
            return self._greedy_all(live_bots, data, occupied), 'greedy'

        dp_rnd = rnd - offset

        if not (0 <= dp_rnd < len(plan.actions)):
            # Outside plan range: greedy
            return self._greedy_all(live_bots, data, occupied), 'greedy'

        # Check if all bots are at expected positions
        synced = True
        if plan.expected_pos and dp_rnd < len(plan.expected_pos):
            for bid_idx, bot in enumerate(live_bots):
                if bid_idx < len(plan.expected_pos[dp_rnd]):
                    ex, ey = plan.expected_pos[dp_rnd][bid_idx]
                    if tuple(bot['position']) != (ex, ey):
                        synced = False
                        break

        if synced:
            # Synced: execute plan actions directly.
            # Override WAIT when bot has no more future goals (plan exhausted for this bot).
            ws_actions = []
            for bid_idx, bot in enumerate(live_bots):
                if bid_idx < len(plan.actions[dp_rnd]):
                    act, item_idx = plan.actions[dp_rnd][bid_idx]
                else:
                    act, item_idx = ACT_WAIT, -1
                if act == ACT_WAIT and plan.goals is not None:
                    bot_goals = plan.goals.get(bid_idx, [])
                    if not any(g[0] >= dp_rnd for g in bot_goals):
                        g_act, g_item, _ = self._greedy_action(bot, data, occupied)
                        if g_act != ACT_WAIT:
                            act, item_idx = g_act, g_item
                ws_actions.append(self._format_ws(act, item_idx, bot['id']))
            return ws_actions, plan.source

        # Desynced: attempt round-offset detection, then BFS recovery
        if plan.expected_pos and dp_rnd > 0:
            prev = dp_rnd - 1
            if prev < len(plan.expected_pos):
                prev_match = all(
                    bid_idx >= len(plan.expected_pos[prev]) or
                    tuple(bot['position']) == tuple(plan.expected_pos[prev][bid_idx])
                    for bid_idx, bot in enumerate(live_bots)
                )
                if prev_match:
                    with self._lock:
                        self._round_offset += 1
                    dp_rnd = rnd - (offset + 1)
                    print(f"R{rnd}: missed round detected, offset→{offset + 1} dp_rnd={dp_rnd}",
                          file=sys.stderr)

        # BFS recovery: navigate each bot toward its next plan goal
        ws_actions = []
        for bid_idx, bot in enumerate(live_bots):
            bx, by = bot['position']
            act, item_idx = ACT_WAIT, -1

            if plan.goals and bid_idx in plan.goals:
                goals = plan.goals[bid_idx]
                # Find next goal at or after current dp_rnd
                next_goal = None
                for g_rnd, g_pos, g_act, g_item in goals:
                    if g_rnd >= dp_rnd:
                        next_goal = (g_pos, g_act, g_item)
                        break

                if next_goal:
                    g_pos, g_act, g_item = next_goal
                    # Safety: if goal is a PICKUP of wrong item type, fall back to greedy
                    # (prevents dead inventory from filler-order goals)
                    safe_goal = True
                    if g_act == ACT_PICKUP and 0 <= g_item < len(self._map_state.items):
                        g_itype = self._map_state.items[g_item].get('type', '')
                        active_needed = set()
                        for order in data.get('orders', []):
                            if order.get('status') == 'active':
                                delv = set(order.get('items_delivered', []))
                                for t in order['items_required']:
                                    if t not in delv:
                                        active_needed.add(t)
                                        delv.add(t)  # count once
                        if active_needed and g_itype not in active_needed:
                            safe_goal = False

                    if safe_goal:
                        if (bx, by) == g_pos:
                            act = g_act
                            item_idx = g_item
                        else:
                            act = bfs_next_action(
                                (bx, by), g_pos, self._walkable, occupied, self._map_state)
                    else:
                        act, item_idx, _ = self._greedy_action(bot, data, occupied)
                else:
                    # Past all goals: greedy
                    act, item_idx, _ = self._greedy_action(bot, data, occupied)
            else:
                act, item_idx, _ = self._greedy_action(bot, data, occupied)

            ws_actions.append(self._format_ws(act, item_idx, bot['id']))

        # If per-round GPU result is available, prefer it over BFS recovery
        if pr_result is not None:
            pr_actions, _ = pr_result
            return pr_actions, 'per_round_gpu'

        return ws_actions, f'{plan.source}_recovery'

    def _nightmare_v2(self, live_bots, data):
        """Nightmare solver: CascadeSolver (or V3 fallback).
        Falls back to greedy while solver initializes in background."""
        rnd = data.get('round', 0)
        if not hasattr(self, '_nm_solver_v3') or self._nm_solver_v3 is None:
            # Solver still initializing in background — use greedy fallback
            if rnd % 10 == 0:
                print(f"  [nm] R{rnd} GREEDY (solver not ready)", file=sys.stderr)
            occupied = {tuple(b['position']) for b in live_bots}
            return self._nightmare_greedy(live_bots, data, occupied)
        try:
            result = self._nm_solver_v3.ws_action(live_bots, data, self._map_state)
            score = data.get('score', 0)
            if rnd % 25 == 0 or rnd < 5:
                solver_name = type(self._nm_solver_v3).__name__
                print(f"  [nm] R{rnd} score={score} solver={solver_name}",
                      file=sys.stderr)
            return result
        except Exception as e:
            import traceback
            print(f"  [nm] R{rnd} ERROR: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            occupied = {tuple(b['position']) for b in live_bots}
            return self._nightmare_greedy(live_bots, data, occupied)

    def _load_nightmare_future_orders(self) -> list:
        """Try to load captured orders from PostgreSQL for nightmare chain planning."""
        try:
            from solution_store import load_capture
            cap = load_capture('nightmare')
            if cap and cap.get('orders'):
                from game_engine import Order
                ms = self._map_state
                orders = []
                for od in cap['orders']:
                    req_names = od.get('items_required', [])
                    req_ids = [ms.type_name_to_id.get(n, 0) for n in req_names]
                    orders.append(Order(len(orders), req_ids, 'preview'))
                print(f"  Loaded {len(orders)} captured orders for nightmare chain planning",
                      file=__import__('sys').stderr)
                return orders
        except Exception as e:
            print(f"  Could not load nightmare orders: {e}", file=__import__('sys').stderr)
        return []

    def _nightmare_greedy(self, live_bots, data, occupied):
        """Nightmare greedy: strict active-only picking, no preview, anti-stall.

        Key design principles:
        1. NEVER pick preview items (dead inventory is permanent and fatal)
        2. Only pick types bot doesn't already have (unless order needs >1)
        3. 1 picker per needed type (reduce aisle congestion)
        4. Dead-inv bots flee from dropoffs and corridors
        5. Anti-stall: escape after 3 rounds stuck
        """
        ms = self._map_state
        walkable = self._walkable
        drop_off_set = {tuple(dz) for dz in self._drop_off_zones}
        corridor_ys = {1, 9, 15}
        import random

        # ── Anti-stall tracking ──────────────────────────────────────────
        if not hasattr(self, '_nm_pos_hist'):
            self._nm_pos_hist = {}
            self._nm_stall = {}

        for bot in live_bots:
            bid = bot['id']
            bpos = tuple(bot['position'])
            hist = self._nm_pos_hist.get(bid, [])
            hist.append(bpos)
            if len(hist) > 8:
                hist = hist[-8:]
            self._nm_pos_hist[bid] = hist
            if len(hist) >= 2 and hist[-1] == hist[-2]:
                self._nm_stall[bid] = self._nm_stall.get(bid, 0) + 1
            else:
                self._nm_stall[bid] = 0

        # ── Active order analysis ────────────────────────────────────────
        order_need = {}  # type → count still needed
        for order in data.get('orders', []):
            if order.get('status') != 'active':
                continue
            for t in order['items_required']:
                order_need[t] = order_need.get(t, 0) + 1
            for t in order.get('items_delivered', []):
                if t in order_need:
                    order_need[t] -= 1
        order_need = {t: c for t, c in order_need.items() if c > 0}

        # Count ALL active-matching items across ALL bot inventories
        global_committed = {}  # type → total count in all inventories
        for bot in live_bots:
            for t in bot.get('inventory', []):
                if t in order_need:
                    global_committed[t] = global_committed.get(t, 0) + 1

        # still_needed: types where we need more items picked up
        still_needed = {}
        for t, need in order_need.items():
            shortfall = need - global_committed.get(t, 0)
            if shortfall > 0:
                still_needed[t] = shortfall

        # ── Helpers ──────────────────────────────────────────────────────
        ws_actions_by_id = {}
        tentative_occ = set(occupied)
        assigned_pickup = {}  # type → pickers assigned this round

        def _do_move(bot, act, idx=-1):
            bpos = tuple(bot['position'])
            bx, by = bpos
            tentative_occ.discard(bpos)
            dest = bpos
            if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                nx, ny = bx + DX[act], by + DY[act]
                if (nx, ny) in walkable:
                    dest = (nx, ny)
            tentative_occ.add(dest)
            ws_actions_by_id[bot['id']] = self._format_ws(act, idx, bot['id'])

        def _escape(bot):
            bpos = tuple(bot['position'])
            bx, by = bpos
            eff = tentative_occ - {bpos}
            dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
            random.shuffle(dirs)
            for a in dirs:
                nx, ny = bx + DX[a], by + DY[a]
                if (nx, ny) in walkable and (nx, ny) not in eff:
                    return a
            return ACT_WAIT

        def _flee_to_corridor(bot):
            """Move to nearest corridor row (y=1, 9, or 15)."""
            bpos = tuple(bot['position'])
            bx, by = bpos
            eff = tentative_occ - {bpos}
            best_target, best_d = None, 999999
            for cy in corridor_ys:
                # Try several x positions near the bot
                for dx in range(0, 15):
                    for cx in [bx + dx, bx - dx]:
                        if 0 <= cx < 30 and (cx, cy) in walkable:
                            d = self._dist(bpos, (cx, cy))
                            if d < best_d and d > 0:
                                best_d = d
                                best_target = (cx, cy)
            if best_target:
                return bfs_next_action(bpos, best_target, walkable, eff, ms)
            # Fallback: any direction
            for a in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                nx, ny = bx + DX[a], by + DY[a]
                if (nx, ny) in walkable and (nx, ny) not in eff:
                    return a
            return ACT_WAIT

        # ── Classify bots ────────────────────────────────────────────────
        deliverers = []   # has active-matching items → deliver immediately
        dead_bots = []    # has items, none match active → flee
        empty_bots = []   # no items → pick or wait

        for bot in live_bots:
            inv = bot.get('inventory', [])
            has_useful = any(t in order_need for t in inv)
            if has_useful:
                deliverers.append(bot)
            elif inv:
                dead_bots.append(bot)
            else:
                empty_bots.append(bot)

        # ════════════════════════════════════════════════════════════════
        # PASS 1: Dead-inv bots flee from dropoffs and busy areas
        # ════════════════════════════════════════════════════════════════
        for bot in dead_bots:
            bpos = tuple(bot['position'])
            stall = self._nm_stall.get(bot['id'], 0)

            if stall >= 3:
                _do_move(bot, _escape(bot))
                continue

            if bpos in drop_off_set:
                _do_move(bot, _flee_to_corridor(bot))
                continue

            # Near any dropoff (dist <= 4) → flee to corridor
            drop = self._nearest_dropoff(bpos)
            if self._dist(bpos, drop) <= 4:
                _do_move(bot, _flee_to_corridor(bot))
                continue

            # In bottom row (y=15 or y=16) → move up to clear delivery paths
            if bpos[1] >= 15:
                _do_move(bot, _flee_to_corridor(bot))
                continue

            # Otherwise wait in place
            _do_move(bot, ACT_WAIT)

        # ════════════════════════════════════════════════════════════════
        # PASS 2: Deliverers → go to nearest dropoff (no pickup en route)
        # ════════════════════════════════════════════════════════════════
        deliverers.sort(key=lambda b: self._dist(
            tuple(b['position']), self._nearest_dropoff(tuple(b['position']))))

        for bot in deliverers:
            bpos = tuple(bot['position'])
            drop_off = self._nearest_dropoff(bpos)
            eff = tentative_occ - {bpos}
            stall = self._nm_stall.get(bot['id'], 0)

            if stall >= 3:
                _do_move(bot, _escape(bot))
                continue

            if bpos in drop_off_set:
                _do_move(bot, ACT_DROPOFF)
                continue

            act = bfs_next_action(bpos, drop_off, walkable, eff, ms)
            _do_move(bot, act)

        # ════════════════════════════════════════════════════════════════
        # PASS 3: Empty bots → pick active items
        # ════════════════════════════════════════════════════════════════
        def _min_dist_to_needed(bot):
            bpos = tuple(bot['position'])
            best = 999999
            for item_idx_c, item in enumerate(ms.items):
                if item.get('type', '') in still_needed:
                    for adj in ms.item_adjacencies.get(item_idx_c, []):
                        d = self._dist(bpos, adj)
                        if d < best:
                            best = d
            return best
        empty_bots.sort(key=_min_dist_to_needed)

        for bot in empty_bots:
            bpos = tuple(bot['position'])
            bx, by = bpos
            eff = tentative_occ - {bpos}
            inv = bot.get('inventory', [])
            stall = self._nm_stall.get(bot['id'], 0)

            if stall >= 3:
                _do_move(bot, _escape(bot))
                continue

            if bpos in drop_off_set:
                _do_move(bot, _flee_to_corridor(bot))
                continue

            # Pick active-needed item
            inv_counts = {}
            for t in inv:
                inv_counts[t] = inv_counts.get(t, 0) + 1

            allowed = set()
            for t, shortfall in still_needed.items():
                n_assigned = assigned_pickup.get(t, 0)
                if n_assigned < shortfall:
                    already_has = inv_counts.get(t, 0)
                    if already_has < order_need.get(t, 0):
                        allowed.add(t)

            if allowed and len(inv) < INV_CAP:
                best_idx, best_dist, best_adj = -1, 999999, None
                for item_idx_c, item in enumerate(ms.items):
                    itype = item.get('type', '')
                    if itype not in allowed:
                        continue
                    for adj in ms.item_adjacencies.get(item_idx_c, []):
                        d = self._dist(bpos, adj)
                        if d < best_dist:
                            best_dist = d
                            best_idx = item_idx_c
                            best_adj = adj

                if best_adj is not None:
                    claimed = ms.items[best_idx].get('type', '')
                    if bpos == best_adj:
                        _do_move(bot, ACT_PICKUP, best_idx)
                    else:
                        act = bfs_next_action(bpos, best_adj, walkable, eff, ms)
                        _do_move(bot, act)
                    if claimed:
                        assigned_pickup[claimed] = assigned_pickup.get(claimed, 0) + 1
                    continue

            # Truly idle → wait
            _do_move(bot, ACT_WAIT)

        return [ws_actions_by_id[bot['id']] for bot in live_bots]

    def _mrta_action(self, live_bots, data):
        """MRTA solver for medium/hard/expert: task allocation + PIBT pathfinding."""
        if not hasattr(self, '_mrta_solver') or self._mrta_solver is None:
            from mrta_solver import MRTASolver
            if self._map_state and self._tables and self._difficulty:
                self._mrta_solver = MRTASolver(self._difficulty, self._map_state, self._tables)
            else:
                return [{'bot': b['id'], 'action': 'wait'} for b in live_bots]
        return self._mrta_solver.ws_action(live_bots, data, self._map_state)

    def _greedy_all(self, live_bots, data, occupied):
        """Coordinate bots with yield-first ordering and multi-dropoff support.

        Multi-dropoff (nightmare): uses zone-partitioned greedy with picker limits.
        Multi-bot (medium/hard/expert): uses MRTA solver.
        """
        # Multi-bot: use V3 solver for nightmare, MRTA for others
        if self._num_bots > 1:
            if self._difficulty == 'nightmare':
                return self._nightmare_v2(live_bots, data)
            return self._mrta_action(live_bots, data)

        ms = self._map_state
        walkable = self._walkable
        multi_drop = len(self._drop_off_zones) > 1
        drop_off_set = {tuple(dz) for dz in self._drop_off_zones}

        # ── Update per-bot position history and stall counts ─────────────
        for bot in live_bots:
            bid = bot['id']
            bpos = tuple(bot['position'])
            hist = self._bot_pos_history.get(bid, [])
            hist.append(bpos)
            if len(hist) > 6:
                hist = hist[-6:]
            self._bot_pos_history[bid] = hist
            if len(hist) >= 2 and hist[-1] == hist[-2]:
                self._bot_stall_count[bid] = self._bot_stall_count.get(bid, 0) + 1
            else:
                self._bot_stall_count[bid] = 0

        # Per-zone staging cells
        staging_by_zone = {}
        for zi, dz in self._zone_dropoff.items():
            dz_pos = tuple(dz)
            staging = dz_pos
            for _da in [ACT_MOVE_RIGHT, ACT_MOVE_LEFT, ACT_MOVE_UP, ACT_MOVE_DOWN]:
                sx, sy = dz_pos[0] + DX[_da], dz_pos[1] + DY[_da]
                if (sx, sy) in walkable:
                    if sx not in self._aisle_cols:
                        staging = (sx, sy)
                        break
                    elif staging == dz_pos:
                        staging = (sx, sy)
            staging_by_zone[zi] = staging

        # ── Active order analysis ────────────────────────────────────────────
        order_need = {}
        for order in data.get('orders', []):
            if order.get('status') != 'active':
                continue
            for t in order['items_required']:
                order_need[t] = order_need.get(t, 0) + 1
            for t in order.get('items_delivered', []):
                if t in order_need:
                    order_need[t] -= 1
        order_need = {t: c for t, c in order_need.items() if c > 0}

        committed = {}
        for bot in live_bots:
            for t in bot.get('inventory', []):
                if t in order_need:
                    committed[t] = committed.get(t, 0) + 1

        still_needed = {t: max(0, need - committed.get(t, 0))
                        for t, need in order_need.items()}
        still_needed = {t: c for t, c in still_needed.items() if c > 0}

        # ── Delivery bots' direct next steps (obstacle-free BFS) ─────────────
        delivery_wants = set()
        for bot in live_bots:
            inv = bot.get('inventory', [])
            if not any(t in order_need for t in inv):
                continue
            bpos = tuple(bot['position'])
            if bpos in drop_off_set:
                continue
            bot_drop = self._bot_dropoff(bot['id'], bpos)
            act = bfs_next_action(bpos, bot_drop, walkable, set(), ms)
            if act != ACT_WAIT:
                bx, by = bpos
                delivery_wants.add((bx + DX[act], by + DY[act]))

        def _sort_key(bot):
            bpos = tuple(bot['position'])
            bx, by = bpos
            inv = bot.get('inventory', [])
            has_useful = any(t in order_need for t in inv)
            is_yield = bpos in delivery_wants and not has_useful
            full_dead = len(inv) >= INV_CAP and not has_useful
            if is_yield:
                return (0, 0, bot['id'])
            elif has_useful:
                bot_drop = self._bot_dropoff(bot['id'], bpos)
                d = self._dist(bpos, bot_drop)
                return (1, d, bot['id'])
            elif full_dead:
                return (2, -by, bx, bot['id'])
            else:
                return (3, 0, bot['id'])

        assigned_pickup = {}
        ws_actions_by_id = {}
        tentative_occ = set(occupied)
        yield_stuck_at_drop = set()  # set of zone indices where yield bot is stuck

        # Per-zone dropoff reservation
        dropoff_reserved = {}
        for zi, dz in self._zone_dropoff.items():
            dz_pos = tuple(dz)
            dropoff_reserved[zi] = any(
                tuple(b['position']) == dz_pos for b in live_bots
                if any(t in order_need for t in b.get('inventory', []))
            )

        for bot in sorted(live_bots, key=_sort_key):
            bpos = tuple(bot['position'])
            bx, by = bpos
            inv = bot.get('inventory', [])
            has_useful = any(t in order_need for t in inv)
            is_yield = bpos in delivery_wants and not has_useful
            bot_drop = self._bot_dropoff(bot['id'], bpos)
            bot_drop_pos = tuple(bot_drop)
            bot_zi = self._zone_for_bot.get(bot['id'], 0)

            eff_occ = tentative_occ - {bpos}

            if is_yield:
                has_dead_inv = len(inv) > 0 and not has_useful
                if bpos in drop_off_set or has_dead_inv:
                    act = ACT_WAIT
                    for try_act in (ACT_MOVE_LEFT, ACT_MOVE_UP,
                                    ACT_MOVE_DOWN, ACT_MOVE_RIGHT):
                        tx, ty = bx + DX[try_act], by + DY[try_act]
                        if (tx, ty) in walkable and (tx, ty) not in eff_occ:
                            act = try_act
                            break
                    if act == ACT_WAIT and bpos in drop_off_set:
                        yield_stuck_at_drop.add(bot_zi)
                else:
                    act = bfs_next_action(bpos, bot_drop, walkable, eff_occ, ms)
                    if act == ACT_WAIT:
                        for try_act in (ACT_MOVE_LEFT, ACT_MOVE_UP,
                                        ACT_MOVE_DOWN, ACT_MOVE_RIGHT):
                            tx, ty = bx + DX[try_act], by + DY[try_act]
                            if (tx, ty) in walkable and (tx, ty) not in eff_occ:
                                act = try_act
                                break
                item_idx = -1

            elif has_useful:
                act, item_idx, _ = self._greedy_action(
                    bot, data, eff_occ, target_types=set(), allow_preview=False)
                # Yield-stuck-at-drop check: per-zone
                if bot_zi in yield_stuck_at_drop and act != ACT_WAIT:
                    ncx, ncy = bx + DX[act], by + DY[act]
                    if (ncx, ncy) == bot_drop_pos:
                        backed = False
                        for back_act in (ACT_MOVE_UP, ACT_MOVE_DOWN,
                                         ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                            rx, ry = bx + DX[back_act], by + DY[back_act]
                            if ((rx, ry) in walkable and (rx, ry) not in eff_occ
                                    and (rx, ry) not in drop_off_set):
                                act = back_act
                                item_idx = -1
                                backed = True
                                break
                        if not backed:
                            act = ACT_WAIT

                # Per-zone dropoff staging
                staging_pos = staging_by_zone.get(bot_zi, bot_drop_pos)
                if act != ACT_WAIT and bpos not in drop_off_set:
                    ncx, ncy = bx + DX[act], by + DY[act]
                    if (ncx, ncy) == bot_drop_pos and dropoff_reserved.get(bot_zi, False):
                        if staging_pos != bot_drop_pos and (bx, by) != staging_pos:
                            stage_act = bfs_next_action(bpos, staging_pos, walkable, eff_occ, ms)
                            if stage_act != ACT_WAIT:
                                act = stage_act
                                item_idx = -1
                            else:
                                act = ACT_WAIT
                if bpos not in drop_off_set and act != ACT_WAIT:
                    ncx, ncy = bx + DX[act], by + DY[act]
                    if (ncx, ncy) == bot_drop_pos and not dropoff_reserved.get(bot_zi, False):
                        dropoff_reserved[bot_zi] = True

            else:
                has_dead_inv_full = len(inv) >= INV_CAP and not has_useful
                if has_dead_inv_full:
                    act = bfs_next_action(bpos, bot_drop, walkable, eff_occ, ms)
                    if act == ACT_WAIT:
                        for try_act in (ACT_MOVE_UP, ACT_MOVE_DOWN,
                                        ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                            tx, ty = bx + DX[try_act], by + DY[try_act]
                            if (tx, ty) in walkable and (tx, ty) not in eff_occ:
                                act = try_act
                                break
                    item_idx = -1
                else:
                    allowed = {t for t, c in still_needed.items()
                               if assigned_pickup.get(t, 0) < c}
                    allow_prev = not allowed
                    act, item_idx, claimed = self._greedy_action(
                        bot, data, eff_occ, target_types=allowed, allow_preview=allow_prev)
                    if claimed:
                        assigned_pickup[claimed] = assigned_pickup.get(claimed, 0) + 1

            # ── Update tentative occupation ───────────────────────────────────
            tentative_occ.discard(bpos)
            dest = bpos
            if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
                nx2, ny2 = bx + DX[act], by + DY[act]
                if (nx2, ny2) in walkable:
                    dest = (nx2, ny2)
            tentative_occ.add(dest)

            ws_actions_by_id[bot['id']] = self._format_ws(act, item_idx, bot['id'])

        return [ws_actions_by_id[bot['id']] for bot in live_bots]

    # ── main async game loop ──────────────────────────────────────────────────

    async def run(self) -> int:
        """Connect to game server and play. Returns final score."""
        self._log_buffer = []  # in-memory log buffer (replaces file)

        final_score = 0
        last_source = 'none'
        expected_rnd = 0  # track expected round for desync detection

        print(f"Connecting to {self.ws_url}", file=sys.stderr)

        WS_CONNECT_TIMEOUT = 15
        WS_RECV_TIMEOUT = 10
        WS_SEND_TIMEOUT = 5

        try:
            ws = await asyncio.wait_for(
                websockets.connect(self.ws_url),
                timeout=WS_CONNECT_TIMEOUT
            )
            print(f"WebSocket connected", file=sys.stderr)
            last_recv_time = time.time()
            recv_count = 0
        except asyncio.TimeoutError:
            print(f"ERROR: WebSocket connection timed out after {WS_CONNECT_TIMEOUT}s", file=sys.stderr)
            return
        except Exception as e:
            print(f"ERROR: WebSocket connection failed: {e}", file=sys.stderr)
            return

        try:
            async with ws:
                game_over = False
                while not game_over:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                    except asyncio.TimeoutError:
                        elapsed_since = time.time() - last_recv_time
                        print(f"WARNING: No message for {elapsed_since:.1f}s (recv_count={recv_count})",
                              file=sys.stderr)
                        if elapsed_since > 30:
                            print(f"ERROR: WebSocket stalled for {elapsed_since:.1f}s, aborting",
                                  file=sys.stderr)
                            break
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"WebSocket closed: {e} (recv_count={recv_count})", file=sys.stderr)
                        break

                    recv_count += 1
                    last_recv_time = time.time()
                    data = json.loads(message)

                    self._log_buffer.append(data)

                    if data['type'] == 'game_over':
                        final_score = data.get('score', 0)
                        print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                        self._emit({"type": "game_over", "score": final_score})
                        break

                    if data['type'] != 'game_state':
                        continue

                    # ── Drain stale messages: if server sent newer states while ──
                    # ── we were busy, skip to the latest one.                   ──
                    # This prevents a 1-round action offset when GPU threads hold
                    # the GIL and delay our response past the server's 2s timeout.
                    drained = 0
                    while True:
                        try:
                            peek = await asyncio.wait_for(ws.recv(), timeout=0.002)
                            peek_data = json.loads(peek)
                            self._log_buffer.append(peek_data)
                            if peek_data.get('type') == 'game_over':
                                final_score = peek_data.get('score', 0)
                                print(f"GAME_OVER Score:{final_score}", file=sys.stderr)
                                self._emit({"type": "game_over", "score": final_score})
                                game_over = True
                                break
                            if peek_data.get('type') == 'game_state':
                                drained += 1
                                data = peek_data  # use the newer state
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            break
                    if game_over:
                        break
                    if drained > 0:
                        print(f"R{data['round']}: Drained {drained} stale round(s)",
                              file=sys.stderr)

                    rnd = data['round']
                    max_rounds = data.get('max_rounds', 300)
                    score = data.get('score', 0)

                    # Detect missed rounds (server advanced past us due to slow response)
                    if rnd > expected_rnd and expected_rnd > 0:
                        gap = rnd - expected_rnd
                        print(f"R{rnd}: ROUND GAP detected (expected R{expected_rnd}, "
                              f"missed {gap} rounds)", file=sys.stderr)
                    expected_rnd = rnd + 1

                    if rnd == 0:
                        self._init_round0(data)
                        # Emit init event with map structure for grid visualization
                        if self.json_stream:
                            ms = self._map_state
                            if ms:
                                walls, shelves = [], []
                                for _y in range(ms.height):
                                    for _x in range(ms.width):
                                        c = int(ms.grid[_y, _x])
                                        if c == CELL_WALL:
                                            walls.append([_x, _y])
                                        elif c == CELL_SHELF:
                                            shelves.append([_x, _y])
                                self._emit({
                                    "type": "init",
                                    "difficulty": self._difficulty,
                                    "num_bots": self._num_bots,
                                    "width": ms.width,
                                    "height": ms.height,
                                    "max_rounds": max_rounds,
                                    "walls": walls,
                                    "shelves": shelves,
                                    "drop_off": list(ms.drop_off),
                                    "spawn": list(ms.spawn),
                                    "items": [
                                        {"id": it["id"], "type": it["type"],
                                         "position": list(it["position"])}
                                        for it in ms.items
                                    ],
                                })
                    else:
                        self._check_new_orders(data)

                    t_recv = time.monotonic()
                    ws_actions, source = self._get_actions(rnd, data)
                    t_compute = time.monotonic() - t_recv

                    if t_compute > 0.5:
                        print(f"R{rnd}: SLOW response {t_compute:.2f}s (source={source})",
                              file=sys.stderr)

                    # Log: always on source change or every 10 rounds
                    plan_score = self._plan.score
                    if source != last_source:
                        print(f"R{rnd}/{max_rounds} Score:{score} plan={plan_score} "
                              f"[{last_source}→{source}]", file=sys.stderr)
                        last_source = source
                    elif rnd < 5 or rnd % 10 == 0 or rnd >= max_rounds - 5:
                        print(f"R{rnd}/{max_rounds} Score:{score} plan={plan_score} [{source}]",
                              file=sys.stderr)

                    # Emit round event for live dashboard
                    if self.json_stream and (rnd % 5 == 0 or rnd < 3 or rnd >= max_rounds - 2
                                             or source != last_source):
                        self._emit({
                            "type": "round",
                            "round": rnd,
                            "max_rounds": max_rounds,
                            "score": score,
                            "plan_score": plan_score,
                            "plan_source": source,
                            "bots": data.get('bots', []),
                            "orders": data.get('orders', []),
                        })

                    response = {'actions': ws_actions}
                    self._log_buffer.append(response)

                    try:
                        await asyncio.wait_for(ws.send(json.dumps(response)), timeout=WS_SEND_TIMEOUT)
                    except asyncio.TimeoutError:
                        print(f"WARNING: WebSocket send timed out at R{rnd}", file=sys.stderr)
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"WebSocket closed during send at R{rnd}: {e}", file=sys.stderr)
                        game_over = True

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        print(f"Game log: {len(self._log_buffer)} entries in memory", file=sys.stderr)

        # Import game log directly to PostgreSQL (controlled by --record flag)
        if self.do_record and self._log_buffer:
            try:
                sys.path.insert(0, os.path.normpath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), '..', 'replay')))
                from import_logs import parse_log_lines, save_to_db
                timestamp = int(time.time())
                record = parse_log_lines(self._log_buffer, pseudo_seed=timestamp)
                if record:
                    run_id = save_to_db(
                        os.environ.get("GROCERY_DB_URL",
                                       "postgres://grocery:grocery123@localhost:5433/grocery_bot"),
                        record, run_type='live')
                    print(f"  [db] Saved to PostgreSQL run_id={run_id}", file=sys.stderr)
                else:
                    print(f"  [db] No valid game data to import", file=sys.stderr)
            except Exception as _e:
                print(f"  [db] Direct DB import failed: {_e}", file=sys.stderr)

        # Post-game optimization: continue GPU solver for additional time
        if self.post_optimize_time > 0 and self._difficulty:
            print(f"\nPost-game optimization: GPU continues for {self.post_optimize_time}s...",
                  file=sys.stderr)
            print(f"(GPU threads still running — plan will keep improving)", file=sys.stderr)
            self._emit({"type": "post_optimize_start",
                        "duration": self.post_optimize_time,
                        "current_score": self._plan.score})
            t_post = time.time()
            last_score = self._plan.score
            while time.time() - t_post < self.post_optimize_time:
                elapsed = time.time() - t_post
                remaining = self.post_optimize_time - elapsed
                with self._lock:
                    cur_score = self._plan.score
                    cur_source = self._plan.source
                if cur_score > last_score:
                    print(f"  [{elapsed:.0f}s] Plan improved: {last_score}→{cur_score} "
                          f"({cur_source}), {remaining:.0f}s remaining", file=sys.stderr)
                    self._emit({"type": "post_optimize_progress",
                                "elapsed": int(elapsed), "remaining": int(remaining),
                                "score": cur_score, "source": cur_source})
                    last_score = cur_score
                else:
                    # Periodic heartbeat for dashboard
                    if int(elapsed) % 30 == 0:
                        self._emit({"type": "post_optimize_progress",
                                    "elapsed": int(elapsed), "remaining": int(remaining),
                                    "score": cur_score, "source": cur_source})
                time.sleep(5.0)
            print(f"Post-game optimization done. Final plan score: {self._plan.score}",
                  file=sys.stderr)
            self._emit({"type": "post_optimize_done", "final_score": self._plan.score})

        # Save best solution
        if self.do_save and self._difficulty:
            with self._lock:
                plan_score = self._plan.score
                plan_actions = self._plan.actions
                plan_source = self._plan.source
                cap = copy.deepcopy(self._capture)

            if plan_actions and plan_score > 0:
                meta = load_meta(self._difficulty)
                prev = meta.get('score', 0) if meta else 0
                saved = save_solution(self._difficulty, plan_score, plan_actions)
                if saved:
                    print(f"Saved solution: {self._difficulty} score={plan_score} "
                          f"source={plan_source} (was {prev})", file=sys.stderr)
                else:
                    print(f"Solution not saved (existing={prev} >= {plan_score})",
                          file=sys.stderr)

            if cap:
                merged, num_new, total = merge_capture(self._difficulty, cap)
                print(f"Saved capture: {total} orders ({num_new} new)", file=sys.stderr)

        reported_score = self._plan.score if self._plan.score > 0 else final_score
        self._emit({"type": "pipeline_done",
                    "final_score": reported_score,
                    "game_score": final_score,
                    "plan_score": self._plan.score,
                    "difficulty": self._difficulty,
                    "replay_ready": self.do_save and self._plan.score > 0,
                    "capture_ready": self.do_save and bool(self._capture)})
        return final_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Anytime online GPU stream solver for Grocery Bot')
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('--save', action='store_true',
                        help='Save best solution and capture after game')
    parser.add_argument('--max-states', type=int, default=None,
                        help='Override max states per GPU pass (all passes use this value)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip refinement iterations in GPU passes')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (no GPU)')
    parser.add_argument('--post-optimize-time', type=int, default=0, metavar='SECONDS',
                        help='Continue GPU optimization for N seconds after game ends. '
                             'Combines with --save to store the improved solution. '
                             'Use before replaying: python live_gpu_stream.py ... '
                             '--save --post-optimize-time 1800 (30min)')
    parser.add_argument('--json-stream', action='store_true',
                        help='Emit JSON events to stdout for SSE dashboard consumption. '
                             'Events: init, round, plan_upgrade, gpu_pass_done, '
                             'game_over, post_optimize_*, pipeline_done.')
    parser.add_argument('--preload-capture', action='store_true',
                        help='Pre-load existing capture from DB at round 0. '
                             'Injects all known orders so GPU starts optimizing with the '
                             'full order list immediately (prevents gen restarts per order).')
    parser.add_argument('--record', action='store_true',
                        help='Save game log directly to PostgreSQL after game ends.')
    parser.add_argument('--pipeline-mode', action='store_true',
                        help='Disable offline GPU passes (greedy + per-round GPU only). '
                             'Use when the pipeline handles offline optimization separately.')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'

    # Auto-set post-optimize time for hard/expert when saving (if not explicitly set)
    post_opt = args.post_optimize_time
    if post_opt == 0 and args.save:
        # Detect difficulty from WS URL if possible, otherwise default to 120s
        post_opt = 120

    solver = AnytimeGPUStream(
        ws_url=args.ws_url,
        save=args.save,
        max_states=args.max_states,
        no_refine=args.no_refine,
        device=device,
        post_optimize_time=post_opt,
        json_stream=args.json_stream,
        preload_capture=args.preload_capture,
        record=args.record,
        pipeline_mode=args.pipeline_mode,
    )
    asyncio.run(solver.run())


if __name__ == '__main__':
    main()
