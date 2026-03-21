"""Evaluate predict_gemini.py directly against ground truth (LOO).

Unlike autoloop_fast which uses a simplified prediction, this runs the actual
production prediction function to verify improvements transfer.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from predict_gemini import gemini_predict
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round3": "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    "round4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
    "round9": "2a341ace-0f57-4309-9b89-e59fe0f09179",
    "round10": "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    "round11": "324fde07-1670-4202-b199-7aa92ecb40ee",
    "round12": "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
}
ROUND_NAMES = ["round2", "round3", "round4", "round5", "round6", "round7", "round9", "round10", "round11", "round12"]
BOOM_ROUNDS = {"round6", "round7", "round11"}


def compute_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    if dynamic.any():
        wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
    else:
        wkl = 0.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


def evaluate_production(seeds_per_round: int = 5):
    import predict
    results = {}

    for test_round in ROUND_NAMES:
        train_rounds = [r for r in ROUND_NAMES + ["round1"] if r != test_round]

        # Build LOO calibration
        cal = CalibrationModel()
        for tr in train_rounds:
            cal.add_round(DATA_DIR / tr)
        predict._calibration = cal  # Inject into predict module

        detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
        rid = ROUND_IDS[test_round]
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))

        # Build global multipliers and FK buckets from observations
        gm = GlobalMultipliers()
        fk = FeatureKeyBuckets()

        # Load all observations
        seed_obs = {}  # {seed_index: [(viewport, grid), ...]}
        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= seeds_per_round:
                continue
            vp, grid = obs["viewport"], obs["grid"]
            state = detail["initial_states"][sid]
            terrain = np.array(state["grid"], dtype=int)
            fkeys = build_feature_keys(terrain, state["settlements"])
            for row in range(len(grid)):
                for col in range(len(grid[0]) if grid else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(grid[row][col])
                        gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        fk.add_observation(fkeys[my][mx], oc)

        scores = []
        for si in range(min(seeds_per_round, 5)):
            state = detail["initial_states"][si]
            gt = np.array(
                json.loads((DATA_DIR / test_round / f"analysis_seed_{si}.json").read_text())["ground_truth"]
            )
            pred = gemini_predict(state, gm, fk)
            scores.append(compute_score(gt, pred))

        results[test_round] = float(np.mean(scores))

    results["avg"] = float(np.mean([results[r] for r in ROUND_NAMES]))
    boom_scores = [results[r] for r in ROUND_NAMES if r in BOOM_ROUNDS]
    nonboom_scores = [results[r] for r in ROUND_NAMES if r not in BOOM_ROUNDS]
    results["boom_avg"] = float(np.mean(boom_scores))
    results["nonboom_avg"] = float(np.mean(nonboom_scores))
    return results


if __name__ == "__main__":
    t0 = time.time()
    r = evaluate_production()
    elapsed = time.time() - t0

    print(f"\n=== PRODUCTION EVALUATION (predict_gemini.py) [{elapsed:.1f}s] ===")
    for rn in ROUND_NAMES:
        tag = " [BOOM]" if rn in BOOM_ROUNDS else ""
        print(f"  {rn}: {r[rn]:.2f}{tag}")
    print(f"  ---")
    print(f"  Overall avg:  {r['avg']:.2f}")
    print(f"  Boom avg:     {r['boom_avg']:.2f}")
    print(f"  Non-boom avg: {r['nonboom_avg']:.2f}")
    print(f"  Gap:          {r['nonboom_avg'] - r['boom_avg']:.2f}")
