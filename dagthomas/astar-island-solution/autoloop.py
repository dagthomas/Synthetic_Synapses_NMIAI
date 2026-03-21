"""Autonomous experiment loop for Astar Island prediction pipeline.

Runs indefinitely, proposing parameter changes, backtesting them,
keeping improvements, and logging everything.

Usage:
    python autoloop.py              # Run until Ctrl+C
    python autoloop.py --quick      # 1 seed/round (faster exploration)
    python autoloop.py --summary    # Print summary of past experiments
"""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autoexperiment import BacktestHarness, compute_score
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

DATA_DIR = Path(__file__).parent / "data" / "calibration"
LOG_PATH = Path(__file__).parent / "data" / "autoloop_log.jsonl"


# ============================================================
# PARAMETER SPACE
# ============================================================

PARAM_SPACE = {
    # FK blending
    "fk_prior_weight": {"type": "float", "lo": 0.5, "hi": 12.0, "step": 0.25},
    "fk_max_strength": {"type": "float", "lo": 2.0, "hi": 25.0, "step": 0.5},
    "fk_min_count": {"type": "int", "lo": 2, "hi": 25, "step": 1},
    "fk_strength_fn": {"type": "cat", "choices": ["sqrt", "log", "linear"]},

    # Multiplier
    "mult_power": {"type": "float", "lo": 0.1, "hi": 1.0, "step": 0.02},
    "mult_power_sett": {"type": "float", "lo": 0.1, "hi": 1.5, "step": 0.02},  # per-class power for settlement
    "mult_power_port": {"type": "float", "lo": 0.1, "hi": 1.5, "step": 0.02},  # per-class power for port
    "mult_smooth": {"type": "float", "lo": 1.0, "hi": 20.0, "step": 0.5},
    "mult_sett_lo": {"type": "float", "lo": 0.02, "hi": 0.5, "step": 0.02},
    "mult_sett_hi": {"type": "float", "lo": 1.5, "hi": 5.0, "step": 0.1},
    "mult_port_lo": {"type": "float", "lo": 0.02, "hi": 0.5, "step": 0.02},
    "mult_port_hi": {"type": "float", "lo": 1.5, "hi": 5.0, "step": 0.1},
    "mult_forest_lo": {"type": "float", "lo": 0.2, "hi": 0.8, "step": 0.02},
    "mult_forest_hi": {"type": "float", "lo": 1.2, "hi": 2.5, "step": 0.1},
    "mult_empty_lo": {"type": "float", "lo": 0.5, "hi": 0.95, "step": 0.02},
    "mult_empty_hi": {"type": "float", "lo": 1.05, "hi": 1.5, "step": 0.02},

    # Floor
    "floor_nonzero": {"type": "float", "lo": 0.001, "hi": 0.015, "step": 0.0005},

    # Entropy-weighted temperature scaling
    "temp_low": {"type": "float", "lo": 0.5, "hi": 1.0, "step": 0.02},
    "temp_high": {"type": "float", "lo": 1.0, "hi": 1.5, "step": 0.02},
    "temp_ent_lo": {"type": "float", "lo": 0.1, "hi": 0.5, "step": 0.02},
    "temp_ent_hi": {"type": "float", "lo": 0.6, "hi": 1.5, "step": 0.02},

    # Selective spatial smoothing (settlement/ruin)
    "smooth_alpha": {"type": "float", "lo": 0.0, "hi": 0.5, "step": 0.02},

    # Proportional redistribution of structural zeros
    "prop_redist": {"type": "cat", "choices": [True, False]},

    # Distance-aware multiplier dampening
    "dist_aware_mult": {"type": "cat", "choices": [True, False]},
    "dist_exp_damp": {"type": "float", "lo": 0.1, "hi": 0.9, "step": 0.05},

    # Regime-adaptive prior weighting
    "regime_prior_scale": {"type": "float", "lo": 0.0, "hi": 3.0, "step": 0.25},

    # Cluster density multiplier (post-hoc spatial correction)
    "cluster_sett_boost": {"type": "float", "lo": 0.0, "hi": 1.0, "step": 0.05},

    # Inverted-U cluster density (quadratic penalty in post-hoc correction)
    "cluster_optimal": {"type": "float", "lo": 0.5, "hi": 5.0, "step": 0.25},
    "cluster_quad_pen": {"type": "float", "lo": -2.0, "hi": 0.0, "step": 0.1},

    # Growth front boost (young settlements expansion activity)
    "growth_front_boost": {"type": "float", "lo": 0.0, "hi": 1.0, "step": 0.05},

    # Direct observation overlay (Dirichlet-Multinomial pseudo-count)
    # Higher = trust model more. 0 = disabled. Sweet spot ~50.
    "obs_overlay_alpha": {"type": "float", "lo": 0.0, "hi": 100.0, "step": 2.0},

    # Distance-ring sharpening (correct per-distance sett/ruin rates from obs)
    "dist_sharpen_alpha": {"type": "float", "lo": 0.0, "hi": 1.0, "step": 0.05},

    # Terrain barrier correction (diffusion-based)
    "barrier_strength": {"type": "float", "lo": 0.0, "hi": 1.0, "step": 0.05},

    # CalibrationModel weights (the big untapped knob)
    "cal_fine_base": {"type": "float", "lo": 0.3, "hi": 3.0, "step": 0.1},
    "cal_fine_divisor": {"type": "float", "lo": 30.0, "hi": 500.0, "step": 10.0},
    "cal_fine_max": {"type": "float", "lo": 1.0, "hi": 10.0, "step": 0.25},
    "cal_coarse_base": {"type": "float", "lo": 0.2, "hi": 2.0, "step": 0.1},
    "cal_coarse_divisor": {"type": "float", "lo": 50.0, "hi": 500.0, "step": 10.0},
    "cal_coarse_max": {"type": "float", "lo": 1.0, "hi": 8.0, "step": 0.25},
    "cal_base_base": {"type": "float", "lo": 0.1, "hi": 2.0, "step": 0.05},
    "cal_base_divisor": {"type": "float", "lo": 200.0, "hi": 3000.0, "step": 50.0},
    "cal_base_max": {"type": "float", "lo": 0.5, "hi": 5.0, "step": 0.1},
    "cal_global_weight": {"type": "float", "lo": 0.05, "hi": 2.0, "step": 0.05},
    "cal_heuristic_blend": {"type": "float", "lo": 0.0, "hi": 0.5, "step": 0.02},
}

DEFAULT_PARAMS = {
    "fk_prior_weight": 1.5,
    "fk_max_strength": 20.0,
    "fk_min_count": 5,
    "fk_strength_fn": "sqrt",
    "mult_power": 0.3,
    "mult_power_sett": 0.3,
    "mult_power_port": 0.3,
    "mult_smooth": 5.0,
    "mult_sett_lo": 0.15,
    "mult_sett_hi": 2.0,
    "mult_port_lo": 0.15,
    "mult_port_hi": 2.0,
    "mult_forest_lo": 0.5,
    "mult_forest_hi": 1.8,
    "mult_empty_lo": 0.75,
    "mult_empty_hi": 1.25,
    "floor_nonzero": 0.008,
    "temp_low": 1.0,
    "temp_high": 1.15,
    "temp_ent_lo": 0.2,
    "temp_ent_hi": 1.0,
    "smooth_alpha": 0.15,
    "prop_redist": False,
    "dist_aware_mult": True,
    "dist_exp_damp": 0.4,
    "regime_prior_scale": 0.0,
    "cluster_sett_boost": 0.0,
    "cluster_optimal": 2.0,
    "cluster_quad_pen": 0.0,
    "growth_front_boost": 0.0,
    "obs_overlay_alpha": 0.0,
    "dist_sharpen_alpha": 0.0,
    "barrier_strength": 0.0,
    "cal_fine_base": 1.0,
    "cal_fine_divisor": 120.0,
    "cal_fine_max": 4.0,
    "cal_coarse_base": 0.75,
    "cal_coarse_divisor": 200.0,
    "cal_coarse_max": 3.0,
    "cal_base_base": 0.5,
    "cal_base_divisor": 1000.0,
    "cal_base_max": 1.5,
    "cal_global_weight": 0.4,
    "cal_heuristic_blend": 0.0,
}


# ============================================================
# EXPERIMENT LOG
# ============================================================

class ExperimentLog:
    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.entries = []
        self.best_score = 0.0
        self.best_params = dict(DEFAULT_PARAMS)
        self._load()

    def _load(self):
        if self.path.exists():
            for line in self.path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                        if entry.get("accepted") and entry.get("scores_full"):
                            avg = entry["scores_full"]["avg"]
                            if avg > self.best_score:
                                self.best_score = avg
                                self.best_params = dict(DEFAULT_PARAMS)
                                self.best_params.update(entry.get("params", {}))
                    except json.JSONDecodeError:
                        continue
        if not self.entries:
            self.best_score = 0.0
            self.best_params = dict(DEFAULT_PARAMS)

    def append(self, entry: dict):
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def count(self) -> int:
        return len(self.entries)

    def print_summary(self):
        if not self.entries:
            print("No experiments yet.")
            return
        accepted = [e for e in self.entries if e.get("accepted")]
        print(f"\n{'='*60}")
        print(f"Experiments: {len(self.entries)}, Accepted: {len(accepted)}")
        print(f"Best score: {self.best_score:.3f}")
        print(f"Best params diff from default:")
        for k, v in self.best_params.items():
            if v != DEFAULT_PARAMS.get(k):
                print(f"  {k}: {DEFAULT_PARAMS.get(k)} -> {v}")
        if accepted:
            print(f"\nRecent accepted:")
            for e in accepted[-5:]:
                sf = e.get("scores_full", {})
                print(f"  [{e['id']}] {e['name']}: avg={sf.get('avg', '?'):.2f} "
                      f"(R2={sf.get('round2','?'):.1f} R3={sf.get('round3','?'):.1f} R4={sf.get('round4','?'):.1f})")
        print(f"{'='*60}\n")


# ============================================================
# PARAMETER PROPOSER
# ============================================================

def perturb_params(base: dict, n_changes: int = None) -> tuple[str, dict]:
    """Randomly perturb 1-3 parameters from base."""
    params = dict(base)
    if n_changes is None:
        n_changes = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]

    keys = random.sample(list(PARAM_SPACE.keys()), min(n_changes, len(PARAM_SPACE)))
    changed = []

    for key in keys:
        spec = PARAM_SPACE[key]
        old_val = params.get(key, DEFAULT_PARAMS.get(key))
        if old_val is None:
            old_val = spec.get("lo", 0) if spec["type"] != "cat" else spec["choices"][0]

        if spec["type"] == "float":
            delta = random.gauss(0, spec["step"] * 2)
            new_val = round(max(spec["lo"], min(spec["hi"], old_val + delta)), 4)
            params[key] = new_val
        elif spec["type"] == "int":
            delta = random.randint(-2, 2)
            new_val = max(spec["lo"], min(spec["hi"], old_val + delta))
            params[key] = new_val
        elif spec["type"] == "cat":
            if random.random() < 0.3:
                params[key] = random.choice(spec["choices"])

        if params[key] != old_val:
            changed.append(f"{key}={params[key]}")

    name = ", ".join(changed) if changed else "no_change"
    return name, params


# ============================================================
# PREDICTION FUNCTION FACTORY
# ============================================================

def make_pred_fn(params: dict):
    """Create a prediction function with given parameters."""

    def pred_fn(state, global_mult, fk_buckets):
        grid = state["grid"]
        settlements = state["settlements"]
        terrain = np.array(grid, dtype=int)
        h, w = terrain.shape

        # Get calibration model
        cal = predict.get_calibration()

        # Build calibrated prior with overridden weights
        fkeys = build_feature_keys(terrain, settlements)
        prior_grid = np.zeros((h, w, NUM_CLASSES), dtype=float)

        for y in range(h):
            for x in range(w):
                code = int(terrain[y, x])
                if code == 10:
                    prior_grid[y, x] = [1, 0, 0, 0, 0, 0]
                    continue
                if code == 5:
                    prior_grid[y, x] = [0, 0, 0, 0, 0, 1]
                    continue

                fk = fkeys[y][x]
                # Custom CalibrationModel prior_for with tunable weights
                coarse_key = (fk[0], fk[1], fk[2], fk[4])
                vector = np.zeros(NUM_CLASSES, dtype=float)
                total_weight = 0.0

                fine_count = cal.fine_counts.get(fk, 0)
                if fine_count > 0:
                    fw = min(params["cal_fine_max"],
                             params["cal_fine_base"] + fine_count / params["cal_fine_divisor"])
                    fine_dist = cal.fine_sums[fk] / cal.fine_sums[fk].sum()
                    vector += fw * fine_dist
                    total_weight += fw

                coarse_count = cal.coarse_counts.get(coarse_key, 0)
                if coarse_count > 0:
                    cw = min(params["cal_coarse_max"],
                             params["cal_coarse_base"] + coarse_count / params["cal_coarse_divisor"])
                    coarse_dist = cal.coarse_sums[coarse_key] / cal.coarse_sums[coarse_key].sum()
                    vector += cw * coarse_dist
                    total_weight += cw

                base_count = cal.base_counts.get(fk[0], 0)
                if base_count > 0:
                    bw = min(params["cal_base_max"],
                             params["cal_base_base"] + base_count / params["cal_base_divisor"])
                    base_dist = cal.base_sums[fk[0]] / cal.base_sums[fk[0]].sum()
                    vector += bw * base_dist
                    total_weight += bw

                if total_weight == 0:
                    prior_grid[y, x] = cal.global_probs
                    continue

                gw = params["cal_global_weight"]
                vector += gw * cal.global_probs
                total_weight += gw
                blended = vector / total_weight

                # Optional heuristic blend (for exploration)
                hb = params.get("cal_heuristic_blend", 0.0)
                if hb > 0:
                    from predict import _get_r1_prior
                    feat = predict._precompute_cell_features(grid, settlements)
                    r1 = _get_r1_prior(code, feat["dist_sett"][y, x],
                                       feat["coastal"][y, x], feat["sett_r5"][y, x])
                    blended = (1 - hb) * blended + hb * r1

                # Floor and normalize
                blended = np.maximum(blended, params["floor_nonzero"])
                prior_grid[y, x] = blended / blended.sum()

        # Custom multiplier
        if global_mult.observed.sum() > 0:
            smooth_val = params["mult_smooth"]
            smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            ratio = (global_mult.observed + smooth) / np.maximum(
                global_mult.expected + smooth, 1e-6
            )
            ratio = np.power(ratio, params["mult_power"])
            ratio[0] = np.clip(ratio[0], params["mult_empty_lo"], params["mult_empty_hi"])
            ratio[5] = np.clip(ratio[5], 0.85, 1.15)
            for c in (1, 2, 3):
                ratio[c] = np.clip(ratio[c], params["mult_sett_lo"], params["mult_sett_hi"])
            ratio[4] = np.clip(ratio[4], params["mult_forest_lo"], params["mult_forest_hi"])
            mult = ratio
        else:
            mult = np.ones(NUM_CLASSES)

        # Coastal map
        coastal = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if terrain[y, x] == 10:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and terrain[ny, nx] == 10:
                        coastal[y, x] = True
                        break

        # FK bucket blending + multiplier + smart floor
        pw = params["fk_prior_weight"]
        ms = params["fk_max_strength"]
        mc = params["fk_min_count"]
        fl = params["floor_nonzero"]
        strength_fn = params.get("fk_strength_fn", "sqrt")

        for y in range(h):
            for x in range(w):
                code = int(terrain[y, x])
                if code == 10 or code == 5:
                    continue

                p = prior_grid[y, x].copy()

                # FK bucket blending
                fk = fkeys[y][x]
                empirical, count = fk_buckets.get_empirical(fk)
                if empirical is not None and count >= mc:
                    if strength_fn == "sqrt":
                        strength = min(ms, math.sqrt(count))
                    elif strength_fn == "log":
                        strength = min(ms, math.log1p(count) * 2)
                    else:  # linear
                        strength = min(ms, count * 0.1)
                    p = p * pw + empirical * strength
                    s = p.sum()
                    if s > 0:
                        p /= s

                # Multiplier
                p *= mult
                s = p.sum()
                if s > 0:
                    p /= s

                # Structural zeros
                p[5] = 0.0  # mountain never on non-mountain
                if not coastal[y, x]:
                    p[2] = 0.0  # port never on non-coastal

                # Floor remaining
                nonzero = p > 0
                if nonzero.any():
                    p[nonzero] = np.maximum(p[nonzero], params["floor_nonzero"])
                    p /= p.sum()

                prior_grid[y, x] = p

        # Lock static cells
        prior_grid[terrain == 5] = [0, 0, 0, 0, 0, 1]
        prior_grid[terrain == 10] = [1, 0, 0, 0, 0, 0]

        return prior_grid

    return pred_fn


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment loop")
    parser.add_argument("--quick", action="store_true", help="1 seed/round (faster)")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    args = parser.parse_args()

    log = ExperimentLog()

    if args.summary:
        log.print_summary()
        return

    seeds = 1 if args.quick else 5
    harness_quick = BacktestHarness(seeds_per_round=1)
    harness_full = BacktestHarness(seeds_per_round=seeds)

    # Initialize best from log or run baseline
    if log.best_score > 0:
        best_params = dict(log.best_params)
        best_score = log.best_score
        print(f"Resuming from experiment {log.count()}, best={best_score:.3f}")
    else:
        print("Running baseline evaluation...")
        pred_fn = make_pred_fn(DEFAULT_PARAMS)
        baseline = harness_full.evaluate(pred_fn)
        best_score = baseline["avg"]
        best_params = dict(DEFAULT_PARAMS)
        log.append({
            "id": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": "baseline",
            "params": {},
            "scores_quick": baseline,
            "scores_full": baseline,
            "accepted": True,
            "baseline_avg": 0.0,
            "elapsed": 0.0,
        })
        log.best_score = best_score
        log.best_params = best_params
        print(f"Baseline: {best_score:.3f}")

    iteration = log.count()
    accepted_count = 0
    no_improvement_streak = 0
    start_time = time.time()

    # Temperature for exploration
    def tolerance(it):
        return max(0.05, 0.5 * math.exp(-it / 300))

    try:
        while True:
            # Propose
            n_changes = None
            if no_improvement_streak > 200:
                n_changes = random.randint(2, 4)  # Wider search after stagnation
            name, params = perturb_params(best_params, n_changes)

            t0 = time.time()

            # Quick screen
            pred_fn = make_pred_fn(params)
            try:
                scores_quick = harness_quick.evaluate(pred_fn)
            except Exception as e:
                print(f"  [{iteration}] CRASHED: {e}")
                iteration += 1
                continue

            quick_avg = scores_quick["avg"]
            promo_threshold = best_score - 1.0  # Promote if within 1 point

            scores_full = None
            accepted = False

            if quick_avg >= promo_threshold:
                # Full evaluation
                try:
                    scores_full = harness_full.evaluate(pred_fn)
                except Exception as e:
                    print(f"  [{iteration}] FULL CRASHED: {e}")
                    iteration += 1
                    continue

                full_avg = scores_full["avg"]
                tol = tolerance(iteration)

                # Accept if better, or with probability if within tolerance
                if full_avg > best_score:
                    accepted = True
                elif full_avg > best_score - tol and random.random() < 0.3:
                    accepted = True  # Metropolis exploration

                if accepted and full_avg > best_score:
                    best_score = full_avg
                    best_params = dict(params)
                    log.best_score = best_score
                    log.best_params = best_params
                    no_improvement_streak = 0
                    accepted_count += 1

            elapsed = time.time() - t0

            # Log
            entry = {
                "id": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "params": {k: v for k, v in params.items() if v != DEFAULT_PARAMS.get(k)},
                "scores_quick": {k: round(v, 3) for k, v in scores_quick.items()},
                "scores_full": {k: round(v, 3) for k, v in scores_full.items()} if scores_full else None,
                "accepted": accepted,
                "baseline_avg": round(best_score, 3),
                "elapsed": round(elapsed, 2),
            }
            log.append(entry)

            # Print
            marker = ""
            if accepted and scores_full and scores_full["avg"] >= best_score:
                marker = f" ***NEW BEST: {best_score:.3f}***"
            elif scores_full:
                marker = f" (full={scores_full['avg']:.2f})"

            if scores_full or iteration % 20 == 0:
                rate = iteration / max(time.time() - start_time, 1) * 3600
                print(f"[{iteration:5d}] quick={quick_avg:.2f} best={best_score:.3f} "
                      f"streak={no_improvement_streak} rate={rate:.0f}/hr | {name}{marker}")

            iteration += 1
            no_improvement_streak += 1

            # Periodic summary
            if iteration % 500 == 0:
                log.print_summary()

    except KeyboardInterrupt:
        elapsed_total = time.time() - start_time
        print(f"\n\nStopped after {iteration - log.count() + len(log.entries)} experiments "
              f"in {elapsed_total/60:.1f} minutes")
        print(f"Accepted: {accepted_count}")
        log.print_summary()


if __name__ == "__main__":
    main()
