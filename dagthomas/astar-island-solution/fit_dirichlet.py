#!/usr/bin/env python3
"""Fit Dirichlet Full Matrix calibrator from all available rounds.

Run after each new round completes to update dirichlet_params.json.
Uses ALL rounds (no LOO) since production uses all available data.

Usage:
    python fit_dirichlet.py              # Fit from all rounds
    python fit_dirichlet.py --validate   # Fit + LOO validation
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from dirichlet_cal import DirichletCalibrator, PARAMS_FILE
from predict_gemini import gemini_predict
from utils import GlobalMultipliers, FeatureKeyBuckets, terrain_to_class
import predict

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent / "data" / "calibration"
OBS_DIR = Path(__file__).parent / "data" / "rounds"

# Map round names to UUIDs
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
BOOM_ROUNDS = {"round6", "round7", "round11"}


def get_dynamic_mask(grid):
    terrain = np.array(grid, dtype=int)
    static = (terrain == 10) | (terrain == 5)
    dynamic = ~static
    dynamic[0, :] = False
    dynamic[-1, :] = False
    dynamic[:, 0] = False
    dynamic[:, -1] = False
    return dynamic


def compute_score(gt, pred):
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


def discover_rounds():
    """Find all available rounds with both detail and analysis files."""
    available = {}
    for rn, rid in ROUND_IDS.items():
        detail_path = DATA_DIR / rn / "round_detail.json"
        analysis_path = DATA_DIR / rn / "analysis_seed_0.json"
        if detail_path.exists() and analysis_path.exists():
            available[rn] = rid
    # Also scan for rounds not in the hardcoded map
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("round"):
            rn = d.name
            if rn in available:
                continue
            detail_path = d / "round_detail.json"
            analysis_path = d / "analysis_seed_0.json"
            if detail_path.exists() and analysis_path.exists():
                # Try to find round ID from detail
                try:
                    detail = json.loads(detail_path.read_text())
                    rid = detail.get("id", "")
                    if rid:
                        available[rn] = rid
                except Exception:
                    pass
    return available


def load_round_predictions(round_names, round_ids):
    """Generate LOO-free predictions for all rounds (uses all data for calibration)."""
    # Build full calibration model
    cal = CalibrationModel()
    for rn in round_names:
        cal.add_round(DATA_DIR / rn)
    predict._calibration_model = cal

    all_preds = []
    all_gts = []

    for rn in round_names:
        rid = round_ids[rn]
        detail = json.loads((DATA_DIR / rn / "round_detail.json").read_text())

        # Build observations
        obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))
        gm = GlobalMultipliers()
        fk = FeatureKeyBuckets()
        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= 5:
                continue
            vp, g = obs["viewport"], obs["grid"]
            state = detail["initial_states"][sid]
            terrain = np.array(state["grid"], dtype=int)
            fkeys = build_feature_keys(terrain, state["settlements"])
            for row in range(len(g)):
                for col in range(len(g[0]) if g else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(g[row][col])
                        gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        fk.add_observation(fkeys[my][mx], oc)

        for si in range(5):
            state = detail["initial_states"][si]
            analysis_path = DATA_DIR / rn / f"analysis_seed_{si}.json"
            if not analysis_path.exists():
                continue
            gt = np.array(json.loads(analysis_path.read_text())["ground_truth"])
            pred = gemini_predict(state, gm, fk)
            dynamic = get_dynamic_mask(state["grid"])
            all_preds.append(pred[dynamic])
            all_gts.append(gt[dynamic])

    return np.concatenate(all_preds), np.concatenate(all_gts)


def fit_production(round_names, round_ids):
    """Fit Dirichlet calibrator using ALL available rounds."""
    print(f"Fitting Dirichlet calibrator on {len(round_names)} rounds...")
    t0 = time.time()

    preds, gts = load_round_predictions(round_names, round_ids)
    print(f"  {len(preds)} dynamic cells loaded in {time.time()-t0:.1f}s")

    cal = DirichletCalibrator()
    result = cal.fit(preds, gts)
    print(f"  Optimization: {result.nit} iterations, loss={result.fun:.6f}")
    print(f"  W diagonal: {np.round(np.diag(cal.W), 3)}")
    print(f"  b: {np.round(cal.b, 3)}")

    cal.save()
    print(f"  Saved to {PARAMS_FILE}")
    return cal


def validate_loo(round_names, round_ids):
    """LOO validation: for each test round, fit on others and evaluate."""
    print(f"\nLOO validation on {len(round_names)} rounds...")

    results = {}
    for test_rn in round_names:
        train_rounds = [r for r in round_names if r != test_rn]

        # Build LOO calibration model
        cal_model = CalibrationModel()
        for tr in train_rounds:
            cal_model.add_round(DATA_DIR / tr)
        predict._calibration_model = cal_model

        # Collect training pairs
        train_preds, train_gts = [], []
        for tr in train_rounds:
            rid = round_ids[tr]
            detail = json.loads((DATA_DIR / tr / "round_detail.json").read_text())
            obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))
            gm = GlobalMultipliers()
            fk = FeatureKeyBuckets()
            for op in obs_files:
                obs = json.loads(op.read_text())
                sid = obs["seed_index"]
                if sid >= 5:
                    continue
                vp, g = obs["viewport"], obs["grid"]
                state = detail["initial_states"][sid]
                terrain = np.array(state["grid"], dtype=int)
                fkeys = build_feature_keys(terrain, state["settlements"])
                for row in range(len(g)):
                    for col in range(len(g[0]) if g else 0):
                        my, mx = vp["y"] + row, vp["x"] + col
                        if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                            oc = terrain_to_class(g[row][col])
                            gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                            fk.add_observation(fkeys[my][mx], oc)

            for si in range(5):
                state = detail["initial_states"][si]
                analysis_path = DATA_DIR / tr / f"analysis_seed_{si}.json"
                if not analysis_path.exists():
                    continue
                gt = np.array(json.loads(analysis_path.read_text())["ground_truth"])
                pred = gemini_predict(state, gm, fk)
                dynamic = get_dynamic_mask(state["grid"])
                train_preds.append(pred[dynamic])
                train_gts.append(gt[dynamic])

        train_preds = np.concatenate(train_preds)
        train_gts = np.concatenate(train_gts)

        # Fit calibrator on training data
        dcal = DirichletCalibrator()
        dcal.fit(train_preds, train_gts)

        # Evaluate on test round
        test_rid = round_ids[test_rn]
        test_detail = json.loads((DATA_DIR / test_rn / "round_detail.json").read_text())
        obs_files = sorted((OBS_DIR / test_rid).glob("obs_s*_q*.json"))
        test_gm = GlobalMultipliers()
        test_fk = FeatureKeyBuckets()
        for op in obs_files:
            obs = json.loads(op.read_text())
            sid = obs["seed_index"]
            if sid >= 5:
                continue
            vp, g = obs["viewport"], obs["grid"]
            state = test_detail["initial_states"][sid]
            terrain = np.array(state["grid"], dtype=int)
            fkeys = build_feature_keys(terrain, state["settlements"])
            for row in range(len(g)):
                for col in range(len(g[0]) if g else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(g[row][col])
                        test_gm.add_observation(oc, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
                        test_fk.add_observation(fkeys[my][mx], oc)

        raw_scores, cal_scores = [], []
        for si in range(5):
            state = test_detail["initial_states"][si]
            analysis_path = DATA_DIR / test_rn / f"analysis_seed_{si}.json"
            if not analysis_path.exists():
                continue
            gt = np.array(json.loads(analysis_path.read_text())["ground_truth"])
            pred = gemini_predict(state, test_gm, test_fk)
            raw_scores.append(compute_score(gt, pred))

            pred_cal = dcal.transform(pred)
            grid = np.array(state["grid"])
            pred_cal[grid == 10] = [1, 0, 0, 0, 0, 0]
            pred_cal[grid == 5] = [0, 0, 0, 0, 0, 1]
            pred_cal[0, :] = [1, 0, 0, 0, 0, 0]
            pred_cal[-1, :] = [1, 0, 0, 0, 0, 0]
            pred_cal[:, 0] = [1, 0, 0, 0, 0, 0]
            pred_cal[:, -1] = [1, 0, 0, 0, 0, 0]
            cal_scores.append(compute_score(gt, pred_cal))

        results[test_rn] = (np.mean(raw_scores), np.mean(cal_scores))

    # Print results
    print(f"\n{'Round':<10} {'Raw':>8} {'Dirichlet':>10} {'Delta':>8}")
    print("-" * 40)
    raw_all, cal_all = [], []
    for rn in round_names:
        raw, cal = results[rn]
        d = cal - raw
        tag = " *" if rn in BOOM_ROUNDS else ""
        print(f"{rn:<10} {raw:8.2f} {cal:10.2f} {d:+8.2f}{tag}")
        raw_all.append(raw)
        cal_all.append(cal)

    raw_avg, cal_avg = np.mean(raw_all), np.mean(cal_all)
    boom_rns = [r for r in round_names if r in BOOM_ROUNDS]
    if boom_rns:
        raw_boom = np.mean([results[r][0] for r in boom_rns])
        cal_boom = np.mean([results[r][1] for r in boom_rns])
    else:
        raw_boom = cal_boom = 0
    print("-" * 40)
    print(f"{'AVG':<10} {raw_avg:8.2f} {cal_avg:10.2f} {cal_avg-raw_avg:+8.2f}")
    if boom_rns:
        print(f"{'BOOM':<10} {raw_boom:8.2f} {cal_boom:10.2f} {cal_boom-raw_boom:+8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Fit Dirichlet calibrator")
    parser.add_argument("--validate", action="store_true",
                        help="Run LOO validation after fitting")
    args = parser.parse_args()

    available = discover_rounds()
    round_names = sorted(available.keys(), key=lambda r: int(r.replace("round", "")))
    print(f"Available rounds: {', '.join(round_names)}")

    fit_production(round_names, available)

    if args.validate:
        validate_loo(round_names, available)


if __name__ == "__main__":
    main()
