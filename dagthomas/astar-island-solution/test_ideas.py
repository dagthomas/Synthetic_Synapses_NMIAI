"""Test untested ideas from IDEAS.md against baseline."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_cdt

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask, _build_feature_key_index,
    build_calibration_lookup, build_fk_empirical_lookup,
)
from utils import GlobalMultipliers, FeatureKeyBuckets, terrain_to_class
import predict
from predict_gemini import gemini_predict

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
ALL_ROUNDS = ["round1", "round2", "round3", "round4", "round5", "round6", "round7"]


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


def build_context(test_round):
    train_rounds = [r for r in ALL_ROUNDS if r != test_round]
    cal = CalibrationModel()
    for tr in train_rounds:
        cal.add_round(DATA_DIR / tr)
    predict._calibration_model = cal
    rid = ROUND_IDS[test_round]
    detail = json.loads((DATA_DIR / test_round / "round_detail.json").read_text())
    obs_files = sorted((OBS_DIR / rid).glob("obs_s*_q*.json"))
    gm = GlobalMultipliers()
    fk = FeatureKeyBuckets()
    for si in range(5):
        state = detail["initial_states"][si]
        prior = predict.get_static_prior(state["grid"], state["settlements"])
        fkeys = build_feature_keys(
            np.array(state["grid"], dtype=int), state["settlements"]
        )
        for op in obs_files:
            obs = json.loads(op.read_text())
            if obs["seed_index"] != si:
                continue
            vp, grid = obs["viewport"], obs["grid"]
            for row in range(len(grid)):
                for col in range(len(grid[0]) if grid else 0):
                    my, mx = vp["y"] + row, vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        oc = terrain_to_class(grid[row][col])
                        gm.add_observation(oc, prior[my, mx])
                        fk.add_observation(fkeys[my][mx], oc)
    return detail, gm, fk


def eval_fn(pred_fn, label):
    results = {}
    for tr in ROUND_IDS:
        detail, gm, fk = build_context(tr)
        scores = []
        for si in range(5):
            state = detail["initial_states"][si]
            pred = pred_fn(state, gm, fk)
            gt = np.array(
                json.loads(
                    (DATA_DIR / tr / f"analysis_seed_{si}.json").read_text()
                )["ground_truth"]
            )
            scores.append(compute_score(gt, pred))
        results[tr] = np.mean(scores)
    avg = np.mean(list(results.values()))
    r = results
    print(
        f"{label:50s} R2={r['round2']:.1f} R3={r['round3']:.1f} "
        f"R4={r['round4']:.1f} R5={r['round5']:.1f} R6={r['round6']:.1f} AVG={avg:.1f}"
    )
    return avg


# ============================================================
# HELPERS for building prediction variants
# ============================================================

def _base_pipeline(state, global_mult, fk_buckets,
                   prior_w_perclass=None, emp_max_scale=1.0):
    """Shared pipeline with parameterizable blending."""
    grid = np.array(state["grid"])
    settlements = state["settlements"]
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)
    cal = predict.get_calibration()
    cal_params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 100.0, "cal_fine_max": 5.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 2.0,
        "cal_base_base": 0.1, "cal_base_divisor": 100.0, "cal_base_max": 1.0,
        "cal_global_weight": 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)
    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    obs = global_mult.observed
    exp = np.maximum(global_mult.expected, 1e-6)
    ratio = obs / exp

    emp_max = np.clip(12.0 - 4.0 * ratio[1], 6.0, 12.0) * emp_max_scale
    power_sett = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
    power_exp = np.array([0.4, 0.50, 0.60, 0.50, 0.4, 0.4])

    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)

    if prior_w_perclass is None:
        prior_w_perclass = np.full(6, 5.0)

    for i in range(N):
        prior = priors[i]
        emp = empiricals[i]
        c = counts[i]
        dist_bucket = unique_keys[i][1]
        emp_w = min(math.sqrt(c), emp_max)
        if c >= 5:
            blend = np.zeros(6)
            for cls in range(6):
                pw = prior_w_perclass[cls]
                blend[cls] = (prior[cls] * pw + emp[cls] * emp_w) / (pw + emp_w)
        else:
            blend = prior.copy()
        if dist_bucket == 0:
            mults = np.power(ratio, power_sett)
        else:
            mults = np.power(ratio, power_exp)
        for c_idx in (1, 2, 3):
            mults[c_idx] = np.clip(mults[c_idx], 0.15, 2.5)
        blend = blend * mults
        blend = blend / blend.sum()
        lookup[i] = blend

    probs = lookup[idx_grid]

    # Temperature softening
    is_sett = (grid == 1) | (grid == 2)
    dist_map = distance_transform_cdt(~is_sett, metric="taxicab")
    radius = 2 + int(3.0 * min(ratio[1], 1.2))
    T_max = 1.0 + 0.10 * math.sqrt(min(ratio[1], 1.0))
    T_grid = np.ones((40, 40, 1), dtype=np.float32)
    T_grid[dist_map <= radius] = T_max
    probs = np.power(probs, 1.0 / T_grid)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Selective smoothing (no port)
    alpha = 0.75
    for k in [1, 3]:
        smoothed = uniform_filter(probs[:, :, k], size=3, mode="reflect")
        probs[:, :, k] = probs[:, :, k] * alpha + smoothed * (1 - alpha)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Structural zeros + floor
    coastal = _build_coastal_mask(grid)
    probs[grid != 5, 5] = 0.0
    probs[~coastal, 2] = 0.0
    probs = np.maximum(probs, 0.005)
    probs[grid != 5, 5] = 0.0
    probs[~coastal, 2] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)
    probs[grid == 10] = [1, 0, 0, 0, 0, 0]
    probs[grid == 5] = [0, 0, 0, 0, 0, 1]
    probs[0, :] = [1, 0, 0, 0, 0, 0]
    probs[-1, :] = [1, 0, 0, 0, 0, 0]
    probs[:, 0] = [1, 0, 0, 0, 0, 0]
    probs[:, -1] = [1, 0, 0, 0, 0, 0]
    return probs


# ============================================================
# IDEAS TO TEST
# ============================================================

def idea_I_perclass_fk(state, gm, fk):
    """Per-class FK weights: trust empirical MORE for settlement, LESS for forest."""
    return _base_pipeline(state, gm, fk,
                          prior_w_perclass=np.array([5.0, 3.0, 4.0, 4.0, 7.0, 5.0]))


def idea_I_v2(state, gm, fk):
    """Per-class FK: even more extreme — sett=2, forest=9."""
    return _base_pipeline(state, gm, fk,
                          prior_w_perclass=np.array([5.0, 2.0, 3.0, 3.0, 9.0, 5.0]))


def idea_H_direct_scale(state, gm, fk):
    """Direct sett% regime: scale empirical trust based on observed settlement deviation."""
    obs_sett_pct = gm.observed[1] / max(gm.observed.sum(), 1)
    confidence = 1.0 + 2.0 * abs(obs_sett_pct - 0.15)
    return _base_pipeline(state, gm, fk, emp_max_scale=confidence)


def idea_combined(state, gm, fk):
    """Combine I (perclass) + H (regime scaling)."""
    obs_sett_pct = gm.observed[1] / max(gm.observed.sum(), 1)
    confidence = 1.0 + 2.0 * abs(obs_sett_pct - 0.15)
    return _base_pipeline(state, gm, fk,
                          prior_w_perclass=np.array([5.0, 3.0, 4.0, 4.0, 7.0, 5.0]),
                          emp_max_scale=confidence)


# ============================================================

if __name__ == "__main__":
    print("IDEA TESTING (LOO on R2-R6)")
    print("=" * 110)

    baseline_avg = eval_fn(gemini_predict, "BASELINE (gemini_predict)")
    print()

    ideas = [
        (idea_I_perclass_fk, "IDEA I: Per-class FK (sett=3, forest=7)"),
        (idea_I_v2, "IDEA I v2: Per-class FK (sett=2, forest=9)"),
        (idea_H_direct_scale, "IDEA H: Direct sett% regime scaling"),
        (idea_combined, "COMBO I+H: Per-class FK + regime scaling"),
    ]

    results = []
    for fn, label in ideas:
        avg = eval_fn(fn, label)
        delta = avg - baseline_avg
        results.append((label, avg, delta))
        print(f"  {'>>> BETTER' if delta > 0.05 else '    neutral' if delta > -0.05 else '    WORSE'} ({delta:+.2f})")
        print()

    print("\nSUMMARY:")
    print(f"  Baseline: {baseline_avg:.2f}")
    for label, avg, delta in sorted(results, key=lambda x: -x[1]):
        marker = "***" if delta > 0.05 else ""
        print(f"  {label:50s} {avg:.2f} ({delta:+.2f}) {marker}")
