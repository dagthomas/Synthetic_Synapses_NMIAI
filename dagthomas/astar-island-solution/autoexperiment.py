"""Automated experiment loop for Astar Island prediction pipeline.

Pattern inspired by Karpathy's autoresearch:
  1. Fixed eval harness (backtest on R1-R4 ground truth, leave-one-out)
  2. Mutable prediction code (tweaks applied via parameter overrides)
  3. Score-based keep/revert

Usage:
    python autoexperiment.py              # Run full experiment suite
    python autoexperiment.py --quick      # Quick mode (1 seed per round)
"""
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from predict import _enforce_floor, get_static_prior, FLOOR_MIN
from utils import (
    FeatureKeyBuckets,
    GlobalMultipliers,
    terrain_to_class,
)
import predict

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
}
ROUND_NAMES = ["round2", "round3", "round4", "round5", "round6", "round7"]


def compute_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    if dynamic.any():
        weighted_kl = float(
            np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum()
        )
    else:
        weighted_kl = 0.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl)))


class BacktestHarness:
    """Fixed evaluation harness — loads all data once, scores quickly."""

    def __init__(self, seeds_per_round: int = 5):
        self.seeds_per_round = seeds_per_round
        self.round_data = {}  # round_name -> list of (detail, gt, obs_files)

        for rnd in ROUND_NAMES:
            detail = json.loads((DATA_DIR / rnd / "round_detail.json").read_text())
            seeds = []
            for si in range(min(seeds_per_round, 5)):
                analysis = json.loads(
                    (DATA_DIR / rnd / f"analysis_seed_{si}.json").read_text()
                )
                gt = np.array(analysis["ground_truth"])
                seeds.append({"state": detail["initial_states"][si], "gt": gt})

            rid = ROUND_IDS[rnd]
            obs_dir = OBS_DIR / rid
            obs_files = sorted(obs_dir.glob("obs_s*_q*.json"))

            self.round_data[rnd] = {
                "detail": detail,
                "seeds": seeds,
                "obs_files": obs_files,
            }

        print(f"Harness loaded: {len(ROUND_NAMES)} rounds, "
              f"{seeds_per_round} seeds/round, "
              f"obs files: {sum(len(d['obs_files']) for d in self.round_data.values())}")

    def build_obs_context(self, test_round: str, train_rounds: list[str]):
        """Build calibration + multipliers + FK buckets for a test round."""
        cal = CalibrationModel()
        for tr in train_rounds:
            cal.add_round(DATA_DIR / tr)
        predict._calibration_model = cal

        data = self.round_data[test_round]
        detail = data["detail"]
        obs_files = data["obs_files"]

        global_mult = GlobalMultipliers()
        fk_buckets = FeatureKeyBuckets()

        seed_priors = []
        seed_fkeys = []
        for si in range(len(data["seeds"])):
            state = data["seeds"][si]["state"]
            prior = get_static_prior(state["grid"], state["settlements"])
            seed_priors.append(prior)
            terrain_np = np.array(state["grid"], dtype=int)
            fkeys = build_feature_keys(terrain_np, state["settlements"])
            seed_fkeys.append(fkeys)

        for obs_path in obs_files:
            obs = json.loads(obs_path.read_text())
            sid = obs["seed_index"]
            if sid >= len(data["seeds"]):
                continue
            vp = obs["viewport"]
            grid = obs["grid"]
            prior = seed_priors[sid]
            fkeys = seed_fkeys[sid]
            for row in range(len(grid)):
                for col in range(len(grid[0]) if grid else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        obs_cls = terrain_to_class(grid[row][col])
                        global_mult.add_observation(obs_cls, prior[my, mx])
                        fk_buckets.add_observation(fkeys[my][mx], obs_cls)

        return global_mult, fk_buckets

    def evaluate(self, pred_fn) -> dict:
        """Run leave-one-out evaluation.

        pred_fn(state, global_mult, fk_buckets) -> np.ndarray (40,40,6)
        Returns dict with per-round and average scores.
        """
        results = {}
        for test_round in ROUND_NAMES:
            train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]
            global_mult, fk_buckets = self.build_obs_context(test_round, train_rounds)

            scores = []
            for si in range(len(self.round_data[test_round]["seeds"])):
                seed_data = self.round_data[test_round]["seeds"][si]
                pred = pred_fn(seed_data["state"], global_mult, fk_buckets)
                score = compute_score(seed_data["gt"], pred)
                scores.append(score)

            results[test_round] = float(np.mean(scores))

        results["avg"] = float(np.mean([results[r] for r in ROUND_NAMES]))
        return results


def apply_smart_floor(pred: np.ndarray, grid: list, coastal: np.ndarray = None,
                      floor_nonzero: float = 0.005) -> np.ndarray:
    """Apply class-specific floors."""
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    if coastal is None:
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

    for y in range(h):
        for x in range(w):
            code = int(terrain[y, x])
            if code == 10 or code == 5:
                continue
            p = pred[y, x]
            p[5] = 0.0  # mountain never on non-mountain
            if not coastal[y, x]:
                p[2] = 0.0  # port never on non-coastal
            nonzero = p > 0
            if nonzero.any():
                p[nonzero] = np.maximum(p[nonzero], floor_nonzero)
                p[:] = p / p.sum()

    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]
    return pred


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def make_baseline_fn(params: dict = None):
    """Create a prediction function with given parameters."""
    p = {
        "fk_prior_weight": 3.0,
        "fk_max_strength": 12.0,
        "fk_min_count": 5,
        "mult_power": 0.4,
        "mult_sett_lo": 0.15,
        "mult_sett_hi": 2.0,
        "mult_forest_lo": 0.5,
        "mult_forest_hi": 1.8,
        "mult_empty_lo": 0.75,
        "mult_empty_hi": 1.25,
        "floor_nonzero": 0.005,
        "zero_mountain": True,
        "zero_port_inland": True,
        "zero_ruin_far": False,
        "ruin_far_dist": 99,
    }
    if params:
        p.update(params)

    def pred_fn(state, global_mult, fk_buckets):
        grid = state["grid"]
        settlements = state["settlements"]

        # Custom multiplier with experiment params
        gm = global_mult
        if gm.observed.sum() > 0:
            smooth = 5.0 * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            ratio = (gm.observed + smooth) / np.maximum(gm.expected + smooth, 1e-6)
            ratio = np.power(ratio, p["mult_power"])
            ratio[0] = np.clip(ratio[0], p["mult_empty_lo"], p["mult_empty_hi"])
            ratio[5] = np.clip(ratio[5], 0.85, 1.15)
            for c in (1, 2, 3):
                ratio[c] = np.clip(ratio[c], p["mult_sett_lo"], p["mult_sett_hi"])
            ratio[4] = np.clip(ratio[4], p["mult_forest_lo"], p["mult_forest_hi"])
            mult = ratio
        else:
            mult = np.ones(NUM_CLASSES)

        # Get calibrated prior (uses predict._calibration_model)
        pred = get_static_prior(grid, settlements)

        terrain = np.array(grid, dtype=int)
        h, w = terrain.shape

        # Build feature keys for FK blending
        fkeys = build_feature_keys(terrain, settlements)

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

        # Settlement distance for ruin zeroing
        sett_pos = [(s["y"], s["x"]) for s in settlements]

        for y in range(h):
            for x in range(w):
                code = int(terrain[y, x])
                if code == 10 or code == 5:
                    continue

                # FK bucket blending
                fk = fkeys[y][x]
                empirical, count = fk_buckets.get_empirical(fk)
                if empirical is not None and count >= p["fk_min_count"]:
                    strength = min(p["fk_max_strength"], math.sqrt(count))
                    pred[y, x] = pred[y, x] * p["fk_prior_weight"] + empirical * strength
                    s = pred[y, x].sum()
                    if s > 0:
                        pred[y, x] /= s

                # Apply multipliers
                pred[y, x] *= mult
                s = pred[y, x].sum()
                if s > 0:
                    pred[y, x] /= s

                # Structural zeros
                if p["zero_mountain"]:
                    pred[y, x, 5] = 0.0
                if p["zero_port_inland"] and not coastal[y, x]:
                    pred[y, x, 2] = 0.0
                if p["zero_ruin_far"] and sett_pos:
                    min_dist = min(abs(y - sy) + abs(x - sx) for sy, sx in sett_pos)
                    if min_dist > p["ruin_far_dist"]:
                        pred[y, x, 3] = 0.0

                # Floor
                nonzero = pred[y, x] > 0
                if nonzero.any():
                    pred[y, x, nonzero] = np.maximum(
                        pred[y, x, nonzero], p["floor_nonzero"]
                    )
                    pred[y, x] /= pred[y, x].sum()

        pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
        pred[terrain == 10] = [1, 0, 0, 0, 0, 0]
        return pred

    return pred_fn


EXPERIMENTS = [
    # Baseline
    {"name": "baseline", "params": {}},

    # FK blend sweep
    {"name": "fk_prior=2_str=15", "params": {"fk_prior_weight": 2.0, "fk_max_strength": 15.0}},
    {"name": "fk_prior=4_str=10", "params": {"fk_prior_weight": 4.0, "fk_max_strength": 10.0}},
    {"name": "fk_prior=5_str=8", "params": {"fk_prior_weight": 5.0, "fk_max_strength": 8.0}},
    {"name": "fk_prior=1.5_str=18", "params": {"fk_prior_weight": 1.5, "fk_max_strength": 18.0}},
    {"name": "fk_prior=3_str=15", "params": {"fk_prior_weight": 3.0, "fk_max_strength": 15.0}},
    {"name": "fk_min_count=3", "params": {"fk_min_count": 3}},
    {"name": "fk_min_count=10", "params": {"fk_min_count": 10}},

    # Multiplier sweep
    {"name": "mult_power=0.3", "params": {"mult_power": 0.3}},
    {"name": "mult_power=0.5", "params": {"mult_power": 0.5}},
    {"name": "mult_power=0.35", "params": {"mult_power": 0.35}},
    {"name": "mult_power=0.45", "params": {"mult_power": 0.45}},
    {"name": "mult_sett_lo=0.10", "params": {"mult_sett_lo": 0.10}},
    {"name": "mult_sett_lo=0.20", "params": {"mult_sett_lo": 0.20}},
    {"name": "mult_sett_lo=0.05", "params": {"mult_sett_lo": 0.05}},
    {"name": "mult_sett_hi=2.5", "params": {"mult_sett_hi": 2.5}},
    {"name": "mult_sett_hi=3.0", "params": {"mult_sett_hi": 3.0}},
    {"name": "mult_forest_lo=0.3", "params": {"mult_forest_lo": 0.3}},
    {"name": "mult_forest_hi=2.0", "params": {"mult_forest_hi": 2.0}},
    {"name": "mult_empty=0.65-1.35", "params": {"mult_empty_lo": 0.65, "mult_empty_hi": 1.35}},
    {"name": "mult_empty=0.85-1.15", "params": {"mult_empty_lo": 0.85, "mult_empty_hi": 1.15}},

    # Floor sweep
    {"name": "floor=0.003", "params": {"floor_nonzero": 0.003}},
    {"name": "floor=0.008", "params": {"floor_nonzero": 0.008}},
    {"name": "floor=0.010", "params": {"floor_nonzero": 0.010}},
    {"name": "floor=0.002", "params": {"floor_nonzero": 0.002}},

    # Structural zeros
    {"name": "keep_mountain", "params": {"zero_mountain": False}},
    {"name": "keep_port_inland", "params": {"zero_port_inland": False}},

    # Combo: best from each category
    {"name": "combo_power03_floor003", "params": {"mult_power": 0.3, "floor_nonzero": 0.003}},
    {"name": "combo_power035_sett010", "params": {"mult_power": 0.35, "mult_sett_lo": 0.10}},
    {"name": "combo_fk2_15_power035", "params": {"fk_prior_weight": 2.0, "fk_max_strength": 15.0, "mult_power": 0.35}},
]


def main():
    parser = argparse.ArgumentParser(description="Auto-experiment loop")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 seed/round")
    args = parser.parse_args()

    seeds_per_round = 1 if args.quick else 5
    harness = BacktestHarness(seeds_per_round=seeds_per_round)

    results = []
    best_score = 0.0
    best_name = ""

    print(f"\nRunning {len(EXPERIMENTS)} experiments...")
    print(f"{'='*80}")

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        params = exp.get("params", {})

        t0 = time.time()
        pred_fn = make_baseline_fn(params)
        scores = harness.evaluate(pred_fn)
        elapsed = time.time() - t0

        r2, r3, r4 = scores["round2"], scores["round3"], scores["round4"]
        avg = scores["avg"]

        marker = ""
        if avg > best_score:
            best_score = avg
            best_name = name
            marker = " ***BEST***"

        print(f"[{i+1:2d}/{len(EXPERIMENTS)}] {name:35s} "
              f"R2={r2:.1f} R3={r3:.1f} R4={r4:.1f} AVG={avg:.1f} "
              f"({elapsed:.1f}s){marker}")

        results.append({
            "name": name,
            "params": params,
            "scores": scores,
            "elapsed": round(elapsed, 1),
        })

    print(f"\n{'='*80}")
    print(f"BEST: {best_name} with AVG={best_score:.1f}")

    # Sort by avg score
    results.sort(key=lambda r: r["scores"]["avg"], reverse=True)
    print(f"\nTop 5:")
    for r in results[:5]:
        s = r["scores"]
        print(f"  {r['name']:35s} R2={s['round2']:.1f} R3={s['round3']:.1f} "
              f"R4={s['round4']:.1f} AVG={s['avg']:.1f}")

    print(f"\nBottom 3:")
    for r in results[-3:]:
        s = r["scores"]
        print(f"  {r['name']:35s} R2={s['round2']:.1f} R3={s['round3']:.1f} "
              f"R4={s['round4']:.1f} AVG={s['avg']:.1f}")

    # Save results
    out_path = Path(__file__).parent / "data" / "autoexperiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
