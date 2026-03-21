# ADK Research Agent Best Experiment #71
# Score: avg=91.289, improvement=+0.018
# Timestamp: 2026-03-20T13:12:12.003654+00:00

import numpy as np
import math
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

import numpy as np
from config import NUM_CLASSES
from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
from calibration import build_feature_keys
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    params = {
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_strength_fn": "sqrt",
        "mult_smooth": 5.0,
        "mult_empty_lo": 0.75,
        "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15,
        "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5,
        "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
        "cal_fine_base": 0.0,
        "cal_fine_divisor": 50.0,
        "cal_fine_max": 2.0,
        "cal_coarse_base": 0.0,
        "cal_coarse_divisor": 100.0,
        "cal_coarse_max": 1.0,
        "cal_base_base": 0.0,
        "cal_base_divisor": 200.0,
        "cal_base_max": 0.5,
        "cal_global_weight": 0.1,
    }

    cal = predict.get_calibration()
    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params["fk_min_count"]
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    pw = params["fk_prior_weight"]
    ms = params["fk_max_strength"]
    strengths = np.minimum(ms, np.sqrt(cnt_grid))
    strengths_3d = strengths[:, :, np.newaxis]
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    if global_mult.observed.sum() > 0:
        smooth_val = params["mult_smooth"]
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        raw_ratio = (global_mult.observed + smooth) / np.maximum(global_mult.expected + smooth, 1e-6)
        
        ratio = np.power(raw_ratio, 0.4)
        ratio[1] = np.power(raw_ratio[1], 0.6)
        ratio[2] = np.power(raw_ratio[2], 0.6)
        ratio[3] = np.power(raw_ratio[3], 0.6)
        
        ratio[0] = np.clip(ratio[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio[4] = np.clip(ratio[4], params["mult_forest_lo"], params["mult_forest_hi"])

        pred *= ratio[np.newaxis, np.newaxis, :]
        
        sett_survival_ratio = raw_ratio[1]
        coastal = _build_coastal_mask(terrain)
        
        # Collapse adjustment: explicitly shift settlement mass to empty/forest
        if sett_survival_ratio < 0.15:
            # We are in near-total collapse. The multiplier might leave ~3-4% settlement mass.
            # But we know they die. Push the remaining mass to empty/forest aggressively.
            for y in range(h):
                for x in range(w):
                    sett_mass = pred[y, x, 1] + pred[y, x, 2] + pred[y, x, 3]
                    if sett_mass > 0.05:
                        keep = 0.005
                        redistribute = sett_mass - (keep * 3)
                        if redistribute > 0:
                            ef_total = pred[y, x, 0] + pred[y, x, 4]
                            if ef_total > 0:
                                pred[y, x, 0] += redistribute * (pred[y, x, 0] / ef_total)
                                pred[y, x, 4] += redistribute * (pred[y, x, 4] / ef_total)
                            else:
                                pred[y, x, 0] += redistribute
                            pred[y, x, 1] = keep
                            pred[y, x, 2] = keep
                            pred[y, x, 3] = keep

        # Port Boost on coastal cells
        if sett_survival_ratio > 0.2:
            for y in range(h):
                for x in range(w):
                    if coastal[y, x]:
                        pred[y, x, 2] *= 1.15

        # Regime-dependent Density Adjustment
        if sett_survival_ratio > 0.5:
            sett_locs = [(s["x"], s["y"]) for s in settlements]
            density_map = np.zeros((h, w))
            for sx, sy in sett_locs:
                count = 0
                for ox, oy in sett_locs:
                    if sx == ox and sy == oy: continue
                    if abs(sx - ox) + abs(sy - oy) <= 5:
                        count += 1
                density_map[sy, sx] = count

            mod_strength = min(0.20, max(0.0, (sett_survival_ratio - 0.5) * 0.2))
            
            for sy in range(h):
                for sx in range(w):
                    if (sx, sy) in sett_locs:
                        if density_map[sy, sx] >= 2:
                            pred[sy, sx, 1] *= (1.0 - mod_strength)
                        elif density_map[sy, sx] <= 1:
                            pred[sy, sx, 1] *= (1.0 + mod_strength)

        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 3x3 Spatial Smoothing for Settlement class on Dynamic Cells
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    coastal = _build_coastal_mask(terrain)
    
    smoothed_pred = pred.copy()
    for y in range(h):
        for x in range(w):
            if static_mask[y, x]:
                continue
            
            s_sum = 0.0
            n_count = 0.0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not static_mask[ny, nx]:
                        w_n = 2.0 if (dy == 0 and dx == 0) else 1.0
                        s_sum += pred[ny, nx, 1] * w_n
                        n_count += w_n
                        
            if n_count > 0:
                avg_settlement = s_sum / n_count
                smoothed_settlement = 0.75 * pred[y, x, 1] + 0.25 * avg_settlement
                diff = smoothed_settlement - pred[y, x, 1]
                
                smoothed_pred[y, x, 1] = smoothed_settlement
                smoothed_pred[y, x, 0] -= diff
                if smoothed_pred[y, x, 0] < 0:
                    smoothed_pred[y, x, 1] += smoothed_pred[y, x, 0]
                    smoothed_pred[y, x, 0] = 0.0

    row_sums = smoothed_pred.sum(axis=-1, keepdims=True)
    smoothed_pred /= np.maximum(row_sums, 1e-10)
    pred = smoothed_pred

    pred[dynamic_mask, 5] = 0.0
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

