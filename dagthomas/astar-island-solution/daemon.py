#!/usr/bin/env python3
"""Autonomous Astar Island daemon — runs everything while you're away.

Manages three concurrent activities:
  1. AUTOLOOP: Continuous parameter optimization (writes best_params.json)
  2. ROUND MONITOR: Detects new rounds, explores, submits with latest params
  3. CALIBRATION: Downloads completed round data, restarts autoloop with new data

Usage:
    python daemon.py                    # Run everything
    python daemon.py --no-autoloop      # Just monitor rounds (no optimization)
    python daemon.py --check-interval 60  # Check for rounds every 60s
"""
import argparse
import json
import math
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from config import MAP_H, MAP_W, NUM_CLASSES

# Flush all output immediately
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent / "data"
CAL_DIR = DATA_DIR / "calibration"
ROUNDS_DIR = DATA_DIR / "rounds"
PARAMS_FILE = Path(__file__).parent / "best_params.json"
LOG_FILE = DATA_DIR / "daemon.log"


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def download_round_analysis(client, round_id, round_number):
    """Download and save ground truth for a completed round."""
    cal_dir = CAL_DIR / f"round{round_number}"
    cal_dir.mkdir(parents=True, exist_ok=True)

    if (cal_dir / "analysis_seed_0.json").exists():
        return False  # Already have it

    try:
        detail = client.get_round_detail(round_id)
        with open(cal_dir / "round_detail.json", "w") as f:
            json.dump(detail, f)

        for si in range(detail["seeds_count"]):
            analysis = client.get_analysis(round_id, si)
            with open(cal_dir / f"analysis_seed_{si}.json", "w") as f:
                json.dump(analysis, f)

            score = analysis.get("score", 0)
            gt = np.array(analysis["ground_truth"])
            sett_pct = gt[:, :, 1].mean() * 100
            log(f"  R{round_number} seed {si}: score={score:.2f}, sett={sett_pct:.1f}%")

        log(f"  R{round_number} analysis saved to {cal_dir}")
        return True  # New data downloaded
    except Exception as e:
        log(f"  Failed to download R{round_number}: {e}", "ERROR")
        return False


def compute_expansion_radius(observations, detail):
    """Compute per-distance settlement observation rates.

    Returns dict {distance: (sett_count, total_count)} for use as
    obs_expansion_radius in gemini_predict.
    """
    from scipy.ndimage import distance_transform_cdt
    from utils import terrain_to_class

    dist_sett_counts = {}  # dist -> [sett_count, total_count]
    for obs in observations:
        sid = obs["seed_index"]
        if sid >= 5:
            continue
        vp, g = obs["viewport"], obs["grid"]
        state = detail["initial_states"][sid]
        terrain = np.array(state["grid"], dtype=int)
        is_sett = (terrain == 1) | (terrain == 2)
        if is_sett.any():
            dm = distance_transform_cdt(~is_sett, metric="taxicab")
        else:
            dm = np.full_like(terrain, 99, dtype=int)
        for row in range(len(g)):
            for col in range(len(g[0]) if g else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                    d = int(dm[my, mx])
                    if d > 15:
                        continue
                    oc = terrain_to_class(g[row][col])
                    if d not in dist_sett_counts:
                        dist_sett_counts[d] = [0, 0]
                    dist_sett_counts[d][1] += 1
                    if oc == 1:
                        dist_sett_counts[d][0] += 1

    return dist_sett_counts


def run_submission(client, round_id, detail):
    """Full pipeline: explore → predict → submit."""
    from explore import run_adaptive_exploration
    from predict_gemini import gemini_predict
    from predict import validate_prediction
    from utils import apply_floor, build_growth_front_map, build_obs_overlay, build_sett_survival

    # Load fresh calibration
    from calibration import CalibrationModel
    import predict
    cal = CalibrationModel.from_all_rounds()
    predict._calibration = cal
    log(f"  Calibration: {cal.get_stats()['rounds_loaded']} rounds, "
        f"{cal.get_stats()['total_cells']} cells")

    seeds_count = detail["seeds_count"]

    # Explore
    log("  Starting adaptive exploration...")
    exploration = run_adaptive_exploration(client, round_id, detail)

    global_mult = exploration.get("global_multipliers")
    fk_buckets = exploration.get("feature_key_buckets")
    multi_store = exploration.get("multi_sample_store")
    variance_regime = exploration.get("variance_regime")

    if variance_regime:
        log(f"  Variance regime: {variance_regime}")
    if global_mult is not None:
        mult = global_mult.get_multipliers()
        log(f"  Multipliers: sett={mult[1]:.3f}, port={mult[2]:.3f}, "
            f"forest={mult[4]:.3f}")

    # Compute expansion radius from observations
    obs_list = exploration.get("observations", [])
    exp_radius = compute_expansion_radius(obs_list, detail) if obs_list else None
    if exp_radius is not None:
        log(f"  Observed expansion radius: {exp_radius}")

    # Load current best params for logging
    try:
        bp = json.loads(PARAMS_FILE.read_text())
        log(f"  Using params: prior_w={bp.get('prior_w')}, T_high={bp.get('T_high')}, "
            f"score_avg={bp.get('score_avg', '?')}")
    except Exception:
        log("  Using default params (best_params.json not found)")

    # Estimate vigor from FK bucket observations for regime-conditional cal
    est_vigor = None
    if fk_buckets and hasattr(fk_buckets, 'counts'):
        sett_obs = sum(v[1] for v in fk_buckets.counts.values())
        total_obs = sum(fk_buckets.totals.values())
        if total_obs > 0:
            est_vigor = sett_obs / total_obs
            log(f"  Estimated vigor: {est_vigor:.4f}")

    # Run simulator inference from observations (if available)
    sim_predictions = {}
    sim_alpha = 0.20  # Default blend weight
    try:
        from sim_data import RoundData
        from sim_inference import fit_to_observations, detect_regime_from_obs, get_adaptive_alpha

        # Try GPU first, fall back to CPU
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                from sim_model_gpu import GPUSimulator
                use_gpu = True
                log("  Simulator: using GPU")
        except Exception:
            pass

        if obs_list:
            state0 = detail["initial_states"][0]
            terrain = np.array(state0["grid"], dtype=int)
            rd = RoundData("live", 0, terrain, state0["settlements"])

            # Detect regime and set adaptive alpha
            regime = detect_regime_from_obs(obs_list, terrain)
            sim_alpha = get_adaptive_alpha(regime)
            log(f"  Simulator regime={regime}, alpha={sim_alpha:.2f}")

            # Fit params: GPU uses 5000 sims/eval, CPU uses 500
            fit_sims = 5000 if use_gpu else 500
            sim_params, _ = fit_to_observations(
                rd, obs_list, n_sims=fit_sims, max_evals=200,
                verbose=False, use_gpu=use_gpu
            )
            log(f"  Simulator params fitted: base_surv={sim_params['base_survival']:.2f}, "
                f"exp_str={sim_params['expansion_str']:.2f}")

            # Generate predictions: GPU uses 10000 sims, CPU uses 2000
            pred_sims = 10000 if use_gpu else 2000
            for seed_idx in range(seeds_count):
                state_si = detail["initial_states"][seed_idx]
                terrain_si = np.array(state_si["grid"], dtype=int)
                rd_si = RoundData("live", seed_idx, terrain_si, state_si["settlements"])
                if use_gpu:
                    sim = GPUSimulator(rd_si)
                else:
                    from sim_model import Simulator
                    sim = Simulator(rd_si)
                sim_predictions[seed_idx] = sim.run(sim_params, n_sims=pred_sims, seed=42)
            log(f"  Simulator predictions: {len(sim_predictions)} seeds, {pred_sims} sims each")
    except Exception as e:
        log(f"  Simulator inference failed: {e}", "WARN")

    # Build per-seed local evidence from observations
    growth_front_maps = {}
    obs_overlays = {}
    sett_survivals = {}
    if obs_list:
        for seed_idx in range(seeds_count):
            terrain_si = np.array(detail["initial_states"][seed_idx]["grid"], dtype=int)
            seed_obs = [o for o in obs_list if o.get("seed_index") == seed_idx]
            if seed_obs:
                growth_front_maps[seed_idx] = build_growth_front_map(seed_obs, terrain_si)
            obs_overlays[seed_idx] = build_obs_overlay(obs_list, terrain_si, seed_idx)
            sett_survivals[seed_idx] = build_sett_survival(
                obs_list, detail["initial_states"][seed_idx]["settlements"], seed_idx
            )

    # Predict and submit
    for seed_idx in range(seeds_count):
        state = detail["initial_states"][seed_idx]
        prediction = gemini_predict(
            state, global_mult, fk_buckets,
            multi_store=multi_store,
            variance_regime=variance_regime,
            obs_expansion_radius=exp_radius,
            est_vigor=est_vigor,
            sim_pred=sim_predictions.get(seed_idx),
            sim_alpha=sim_alpha if seed_idx in sim_predictions else 0.0,
            growth_front_map=growth_front_maps.get(seed_idx),
            obs_overlay=obs_overlays.get(seed_idx),
            sett_survival=sett_survivals.get(seed_idx),
        )

        errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
        if errors:
            log(f"  Seed {seed_idx}: validation errors, applying floor", "WARN")
            prediction = apply_floor(prediction)

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        sett_avg = prediction[:, :, 1].mean()
        log(f"  Seed {seed_idx}: {resp.get('status', '?')} (sett={sett_avg:.4f})")

    log(f"  All {seeds_count} seeds submitted for R{detail['round_number']}")


def gpu_resubmit_round(client, round_id, detail, iteration: int = 0):
    """Iteratively improve predictions using GPU sim with different strategies.

    Called periodically while round is still active. Each iteration tries
    a different approach: varying alpha, more CMA-ES evals, different warm starts.
    """
    from predict_gemini import gemini_predict
    from predict import validate_prediction
    from utils import (apply_floor, build_growth_front_map, build_obs_overlay,
                       build_sett_survival, terrain_to_class, GlobalMultipliers,
                       FeatureKeyBuckets)

    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Load observations
    obs_dir = ROUNDS_DIR / round_id
    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    if not obs_files:
        return
    observations = [json.loads(f.read_text()) for f in obs_files]

    state0 = initial_states[0]
    terrain0 = np.array(state0["grid"], dtype=int)
    H, W = terrain0.shape

    # Build obs cells
    obs_cells = []
    for obs in observations:
        vp = obs["viewport"]
        grid = obs["grid"]
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < H and 0 <= mx < W:
                    cls = terrain_to_class(grid[row][col])
                    obs_cells.append((my, mx, cls))

    if len(obs_cells) < 100:
        return

    obs_y = np.array([c[0] for c in obs_cells])
    obs_x = np.array([c[1] for c in obs_cells])
    obs_cls = np.array([c[2] for c in obs_cells])

    try:
        from sim_data import RoundData
        from sim_model_gpu import GPUSimulator
        from sim_inference import (
            WARM_STARTS, PARAM_SPEC, PARAM_NAMES,
            params_to_vec, vec_to_params,
            detect_regime_from_obs, get_adaptive_alpha,
            _knn_warm_start, _compute_obs_features, _load_transfer_data,
        )
        import cma
    except ImportError as e:
        log(f"  GPU resubmit: missing dependency ({e})", "WARN")
        return

    regime = detect_regime_from_obs(observations, terrain0)

    # Each iteration varies strategy
    n_sims = 2000 + iteration * 500      # More sims each iteration
    max_evals = 200 + iteration * 100     # More CMA-ES evals
    seed_offset = iteration * 7           # Different random seed

    log(f"  GPU resubmit iter {iteration}: n_sims={n_sims}, max_evals={max_evals}, regime={regime}")

    rd0 = RoundData("live", 0, terrain0, state0["settlements"])
    sim_gpu = GPUSimulator(rd0, device="cuda")

    lo = np.array([PARAM_SPEC[k][1] for k in PARAM_NAMES])
    hi = np.array([PARAM_SPEC[k][2] for k in PARAM_NAMES])

    def objective(vec):
        params = vec_to_params(vec)
        pred = sim_gpu.run(params, n_sims=n_sims, seed=42 + seed_offset)
        pred_safe = np.maximum(pred, 1e-6)
        probs = pred_safe[obs_y, obs_x, obs_cls]
        return -float(np.mean(np.log(probs)))

    # Build starts
    td = _load_transfer_data()
    obs_features = _compute_obs_features(observations, terrain0)
    knn_params, _ = _knn_warm_start(obs_features, td, k=3)

    starts = [
        ("knn", params_to_vec(knn_params), 0.2),
        (regime, params_to_vec(WARM_STARTS[regime]), 0.5),
    ]
    for r in ["collapse", "moderate", "boom"]:
        if r != regime:
            starts.append((r, params_to_vec(WARM_STARTS[r]), 0.5))

    best_vec = None
    best_ll = float("inf")
    for name, x0, sig in starts:
        try:
            opts = {"maxfevals": max_evals // len(starts),
                    "bounds": [lo, hi], "verbose": -9,
                    "seed": 42 + seed_offset + hash(name) % 1000}
            es = cma.CMAEvolutionStrategy(x0, sig, opts)
            while not es.stop():
                solutions = es.ask()
                fitnesses = [objective(s) for s in solutions]
                es.tell(solutions, fitnesses)
            if es.result.fbest < best_ll:
                best_ll = es.result.fbest
                best_vec = es.result.xbest
        except Exception:
            continue

    if best_vec is None:
        log("  GPU resubmit: CMA-ES failed, skipping", "WARN")
        return

    sim_params = vec_to_params(best_vec)
    sim_alpha = get_adaptive_alpha(regime)

    # Generate sim predictions for all seeds
    sim_predictions = {}
    for seed_idx in range(seeds_count):
        state_si = initial_states[seed_idx]
        terrain_si = np.array(state_si["grid"], dtype=int)
        rd_si = RoundData("live", seed_idx, terrain_si, state_si["settlements"])
        sim_si = GPUSimulator(rd_si, device="cuda")
        sim_predictions[seed_idx] = sim_si.run(sim_params, n_sims=n_sims * 2,
                                                seed=42 + seed_offset)

    # Build per-seed evidence
    growth_front_maps = {}
    obs_overlays = {}
    sett_survivals = {}
    for seed_idx in range(seeds_count):
        terrain_si = np.array(initial_states[seed_idx]["grid"], dtype=int)
        seed_obs = [o for o in observations if o.get("seed_index") == seed_idx]
        if seed_obs:
            growth_front_maps[seed_idx] = build_growth_front_map(seed_obs, terrain_si)
        obs_overlays[seed_idx] = build_obs_overlay(observations, terrain_si, seed_idx)
        sett_survivals[seed_idx] = build_sett_survival(
            observations, initial_states[seed_idx]["settlements"], seed_idx
        )

    est_vigor = obs_features["sett_rate"] if obs_features["sett_rate"] > 0 else None
    gm = GlobalMultipliers()
    fk = FeatureKeyBuckets()

    # Submit
    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        prediction = gemini_predict(
            state, gm, fk,
            variance_regime=regime.upper() if regime == "boom" else None,
            est_vigor=est_vigor,
            sim_pred=sim_predictions.get(seed_idx),
            sim_alpha=sim_alpha,
            growth_front_map=growth_front_maps.get(seed_idx),
            obs_overlay=obs_overlays.get(seed_idx),
            sett_survival=sett_survivals.get(seed_idx),
        )

        errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
        if errors:
            prediction = apply_floor(prediction)

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        sett_avg = prediction[:, :, 1].mean()
        if seed_idx == 0:
            log(f"  GPU resubmit iter {iteration}: seed 0 = {resp.get('status', '?')} "
                f"(sett={sett_avg:.3f})")

    log(f"  GPU resubmit iter {iteration}: all {seeds_count} seeds re-submitted")


def fallback_submit(client, round_id, detail):
    """Submit predictions using whatever observations exist on disk."""
    import time
    from predict_gemini import gemini_predict
    from predict import validate_prediction
    from utils import (apply_floor, GlobalMultipliers, FeatureKeyBuckets, build_growth_front_map,
                       build_obs_overlay, build_sett_survival, terrain_to_class)
    from calibration import CalibrationModel, build_feature_keys
    import predict

    cal = CalibrationModel.from_all_rounds()
    predict._calibration = cal

    obs_dir = ROUNDS_DIR / round_id
    obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))
    log(f"  Fallback: found {len(obs_files)} observation files")

    gm = GlobalMultipliers()
    fk = FeatureKeyBuckets()
    initial_states = detail["initial_states"]

    for op in obs_files:
        obs = json.loads(op.read_text())
        sid = obs["seed_index"]
        vp, grid = obs["viewport"], obs["grid"]
        state = initial_states[sid]
        terrain = np.array(state["grid"], dtype=int)
        fkeys = build_feature_keys(terrain, state["settlements"])
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                my, mx = vp["y"] + row, vp["x"] + col
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                    oc = terrain_to_class(grid[row][col])
                    gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                    fk.add_observation(fkeys[my][mx], oc)

    # Estimate vigor for regime-conditional cal
    est_vigor_fb = None
    if fk and hasattr(fk, 'counts'):
        sett_obs = sum(v[1] for v in fk.counts.values())
        total_obs = sum(fk.totals.values())
        if total_obs > 0:
            est_vigor_fb = sett_obs / total_obs

    # Build per-seed local evidence from saved observations
    all_obs = [json.loads(op.read_text()) for op in obs_files]
    fb_growth_fronts = {}
    fb_obs_overlays = {}
    fb_sett_survivals = {}
    for si in range(detail["seeds_count"]):
        terrain_si = np.array(initial_states[si]["grid"], dtype=int)
        seed_obs = [o for o in all_obs if o.get("seed_index") == si]
        if seed_obs:
            fb_growth_fronts[si] = build_growth_front_map(seed_obs, terrain_si)
        fb_obs_overlays[si] = build_obs_overlay(all_obs, terrain_si, si)
        fb_sett_survivals[si] = build_sett_survival(
            all_obs, initial_states[si]["settlements"], si
        )

    for si in range(detail["seeds_count"]):
        state = initial_states[si]
        pred = gemini_predict(state, gm, fk, est_vigor=est_vigor_fb,
                              growth_front_map=fb_growth_fronts.get(si),
                              obs_overlay=fb_obs_overlays.get(si),
                              sett_survival=fb_sett_survivals.get(si))
        errors = validate_prediction(pred, detail["map_height"], detail["map_width"])
        if errors:
            pred = apply_floor(pred)
        time.sleep(1)
        resp = client.submit(round_id, si, pred.tolist())
        log(f"  Fallback seed {si}: {resp.get('status', '?')}")

    log("  Fallback submission complete")


def start_autoloop():
    """Start autoloop_fast.py as a subprocess."""
    log_path = DATA_DIR / "autoloop_fast_output.log"
    proc = subprocess.Popen(
        [sys.executable, "autoloop_fast.py", "--seeds", "5"],
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent,
    )
    log(f"Autoloop started (PID {proc.pid}), logging to {log_path}")
    return proc


def stop_autoloop(proc):
    """Gracefully stop autoloop and all child processes."""
    if proc is None:
        return
    if proc.poll() is None:
        log(f"Stopping autoloop (PID {proc.pid})...")
        try:
            # On Windows, kill process tree
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"],
                           capture_output=True, timeout=10)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        log("Autoloop stopped")
    else:
        # Already dead, but try to clean up any orphans
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"],
                           capture_output=True, timeout=5)
        except Exception:
            pass


def check_autoloop_health(proc):
    """Check if autoloop is still running, restart if crashed."""
    if proc is None:
        return None
    if proc.poll() is not None:
        log(f"Autoloop crashed (exit code {proc.returncode}), restarting...", "WARN")
        return start_autoloop()
    return proc


def get_autoloop_progress():
    """Read latest autoloop progress from log."""
    try:
        log_path = DATA_DIR / "autoloop_fast_log.jsonl"
        if not log_path.exists():
            return None
        # Read last line
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            pos = max(0, size - 2000)
            f.seek(pos)
            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
            last = json.loads(lines[-1])
            return last
    except Exception:
        return None


def sync_best_params():
    """Check if autoloop found better params, update best_params.json."""
    progress = get_autoloop_progress()
    if progress is None:
        return False

    new_avg = progress.get("scores_full", {}).get("avg", 0)
    if new_avg <= 0:
        new_avg = progress.get("scores_quick", {}).get("avg", 0)

    # Read current best
    try:
        current = json.loads(PARAMS_FILE.read_text())
        current_avg = current.get("score_avg", 0)
    except Exception:
        current_avg = 0

    if new_avg > current_avg + 0.01:  # Only update if meaningfully better
        new_params = progress.get("params", {})
        # Map autoloop param names to production param names
        param_map = {
            "fk_prior_weight": "prior_w",
            "fk_max_strength": "emp_max",
            "dist_exp_damp": "exp_damp",
            "mult_power": "base_power",
            "temp_high": "T_high",
            "smooth_alpha": "smooth_alpha",
            "floor_nonzero": "floor",
            "dist_sharpen_alpha": "dist_sharpen",
            "obs_overlay_alpha": "obs_pseudo",
            # Multiplier bounds (same name in both)
            "mult_sett_lo": "mult_sett_lo",
            "mult_sett_hi": "mult_sett_hi",
            "mult_port_lo": "mult_port_lo",
            "mult_port_hi": "mult_port_hi",
            "mult_forest_lo": "mult_forest_lo",
            "mult_forest_hi": "mult_forest_hi",
            "mult_empty_lo": "mult_empty_lo",
            "mult_empty_hi": "mult_empty_hi",
            # Per-class power
            "mult_power_sett": "mult_power_sett",
            "mult_power_port": "mult_power_port",
            # Calibration params
            "cal_fine_base": "cal_fine_base",
            "cal_fine_divisor": "cal_fine_divisor",
            "cal_fine_max": "cal_fine_max",
            "cal_coarse_base": "cal_coarse_base",
            "cal_coarse_divisor": "cal_coarse_divisor",
            "cal_coarse_max": "cal_coarse_max",
            "cal_base_base": "cal_base_base",
            "cal_base_divisor": "cal_base_divisor",
            "cal_base_max": "cal_base_max",
            "cal_global_weight": "cal_global_weight",
            # Additional features
            "growth_front_boost": "growth_front_boost",
            "barrier_strength": "barrier_strength",
            "temp_low": "T_low",
            "temp_ent_lo": "T_ent_lo",
            "temp_ent_hi": "T_ent_hi",
        }

        updated = dict(json.loads(PARAMS_FILE.read_text())) if PARAMS_FILE.exists() else {}
        for al_key, prod_key in param_map.items():
            if al_key in new_params:
                updated[prod_key] = new_params[al_key]

        # Also compute boom/nonboom from scores
        scores = progress.get("scores_full", progress.get("scores_quick", {}))
        boom_rounds = {"round6", "round7", "round11", "round14", "round17"}
        boom_scores = [scores[r] for r in boom_rounds if r in scores]
        nonboom_scores = [scores[r] for r in scores
                         if r.startswith("round") and r not in boom_rounds
                         and r not in ("avg", "boom_avg", "nonboom_avg")]

        updated["score_avg"] = new_avg
        updated["score_boom"] = float(np.mean(boom_scores)) if boom_scores else 0
        updated["score_nonboom"] = float(np.mean(nonboom_scores)) if nonboom_scores else 0
        updated["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated["source"] = "autoloop"
        updated["experiment_id"] = progress.get("id", "?")

        with open(PARAMS_FILE, "w") as f:
            json.dump(updated, f, indent=2)

        log(f"PARAMS UPDATED: avg {current_avg:.2f} -> {new_avg:.2f} "
            f"(experiment #{progress.get('id', '?')})")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Astar Island daemon")
    parser.add_argument("--no-autoloop", action="store_true",
                        help="Disable autoloop optimization")
    parser.add_argument("--check-interval", type=int, default=90,
                        help="Seconds between round checks (default: 90)")
    args = parser.parse_args()

    log("=" * 60)
    log("ASTAR ISLAND DAEMON STARTING")
    log(f"  Autoloop: {'DISABLED' if args.no_autoloop else 'ENABLED'}")
    log(f"  Check interval: {args.check_interval}s")
    log(f"  Params file: {PARAMS_FILE}")
    log("=" * 60)

    from client import AstarIslandClient
    client = AstarIslandClient()

    # Step 1: Download any missing round data
    log("Checking for new completed round data...")
    new_data = False
    try:
        rounds = client.get_rounds()
        for r in rounds:
            rn = r.get("round_number", 0)
            status = r.get("status", "")
            if status == "completed" and rn >= 1:
                if download_round_analysis(client, r["id"], rn):
                    new_data = True
    except Exception as e:
        log(f"Failed to check rounds: {e}", "ERROR")

    # Step 2: Start autoloop
    autoloop_proc = None
    if not args.no_autoloop:
        autoloop_proc = start_autoloop()

    # Step 3: Main monitoring loop
    last_submitted = None
    last_param_sync = 0
    last_autoloop_check = 0
    last_gpu_resubmit = 0  # timestamp of last GPU sim re-submission
    gpu_resubmit_count = 0  # how many times we've re-submitted this round
    cycle = 0

    def shutdown(sig, frame):
        log("Shutting down...")
        stop_autoloop(autoloop_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        try:
            cycle += 1
            now = time.time()

            # --- Check for active round ---
            try:
                active = client.get_active_round()
            except Exception as e:
                log(f"API error checking active round: {e}", "ERROR")
                active = None

            if active:
                round_id = active["id"]
                round_number = active["round_number"]

                if round_id != last_submitted:
                    budget = client.get_budget()
                    remaining = budget["queries_max"] - budget["queries_used"]

                    if remaining > 0:
                        log("=" * 60)
                        log(f"ROUND {round_number} DETECTED — {remaining} queries available")
                        log("=" * 60)

                        # Pause autoloop during submission
                        if autoloop_proc:
                            stop_autoloop(autoloop_proc)
                            autoloop_proc = None

                        try:
                            detail = client.get_round_detail(round_id)
                            log(f"  Map: {detail['map_width']}x{detail['map_height']}, "
                                f"{detail['seeds_count']} seeds")
                            run_submission(client, round_id, detail)
                            last_submitted = round_id
                        except Exception as e:
                            log(f"SUBMISSION FAILED: {e}", "ERROR")
                            traceback.print_exc()
                            # Try fallback: submit with whatever observations exist
                            log("Attempting fallback submission with partial observations...")
                            try:
                                fallback_submit(client, round_id, detail)
                                last_submitted = round_id
                            except Exception as e2:
                                log(f"Fallback also failed: {e2}", "ERROR")

                        # Restart autoloop after submission
                        if not args.no_autoloop:
                            autoloop_proc = start_autoloop()
                    else:
                        if round_id != last_submitted:
                            log(f"Round {round_number} active but 0 queries remaining "
                                f"(already submitted or budget exhausted)")
                            last_submitted = round_id
                            last_gpu_resubmit = 0
                            gpu_resubmit_count = 0

                # --- Iterative GPU sim re-submission on active round ---
                # Re-submit every 10 min with potentially better sim params
                if (round_id == last_submitted
                        and now - last_gpu_resubmit > 600
                        and gpu_resubmit_count < 10):
                    try:
                        rd = client.get_round_detail(round_id)
                        gpu_resubmit_round(client, round_id, rd, gpu_resubmit_count)
                        last_gpu_resubmit = now
                        gpu_resubmit_count += 1
                    except Exception as e:
                        log(f"GPU re-submit failed: {e}", "WARN")
                        last_gpu_resubmit = now  # Don't spam on errors

            # --- Check for newly completed rounds ---
            if cycle % 10 == 0:  # Every ~15 min
                try:
                    rounds = client.get_rounds()
                    new_data = False
                    for r in rounds:
                        rn = r.get("round_number", 0)
                        status = r.get("status", "")
                        if status == "completed" and rn >= 1:
                            if download_round_analysis(client, r["id"], rn):
                                new_data = True
                                log(f"New calibration data: Round {rn}")

                    if new_data and autoloop_proc:
                        log("New data available — restarting autoloop with fresh calibration")
                        stop_autoloop(autoloop_proc)
                        autoloop_proc = start_autoloop()
                except Exception as e:
                    log(f"Error checking completed rounds: {e}", "ERROR")

            # --- Sync best params from autoloop ---
            if now - last_param_sync > 120:  # Every 2 min
                if sync_best_params():
                    pass  # Already logged
                last_param_sync = now

            # --- Health check autoloop ---
            if not args.no_autoloop and now - last_autoloop_check > 300:  # Every 5 min
                autoloop_proc = check_autoloop_health(autoloop_proc)
                last_autoloop_check = now

            # --- Status report ---
            if cycle % 20 == 0:
                progress = get_autoloop_progress()
                if progress:
                    exp_count = progress.get("id", "?")
                    best = progress.get("scores_full", {}).get("avg",
                           progress.get("scores_quick", {}).get("avg", 0))
                    log(f"Status: autoloop at experiment #{exp_count}, best={best:.2f}")
                try:
                    bp = json.loads(PARAMS_FILE.read_text())
                    log(f"  Current best_params: avg={bp.get('score_avg', '?')}, "
                        f"boom={bp.get('score_boom', '?')}, source={bp.get('source', '?')}")
                except Exception:
                    pass

        except Exception as e:
            log(f"Unexpected error in main loop: {e}", "ERROR")
            traceback.print_exc()

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
