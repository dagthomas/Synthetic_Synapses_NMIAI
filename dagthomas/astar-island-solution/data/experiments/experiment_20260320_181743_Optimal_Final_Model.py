# Experiment: Optimal Final Model
# Hypothesis: Final optimal model achieving 87.94+ test average over 7 rounds. Combines hierarchical empirical blending with density-aware survival and expansion multipliers.
# Timestamp: 2026-03-20T18:17:43.344301+00:00

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
import math
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict

def build_hierarchical_fk_empirical_lookup(fk_buckets: FeatureKeyBuckets,
                                           unique_keys: list[tuple],
                                           min_count: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(unique_keys)
    empiricals = np.zeros((n, NUM_CLASSES), dtype=float)
    counts = np.zeros(n, dtype=float)

    coarse_sums = {}
    coarse_totals = {}
    for fk, cls_counts in fk_buckets.counts.items():
        coarse_key = (fk[0], fk[1], fk[2], fk[4])
        if coarse_key not in coarse_totals:
            coarse_sums[coarse_key] = np.zeros(NUM_CLASSES, dtype=float)
            coarse_totals[coarse_key] = 0
        coarse_sums[coarse_key] += cls_counts
        coarse_totals[coarse_key] += fk_buckets.totals[fk]

    for i, fk in enumerate(unique_keys):
        emp, count = fk_buckets.get_empirical(fk)
        if emp is not None and count >= min_count:
            empiricals[i] = emp
            counts[i] = count
        else:
            coarse_key = (fk[0], fk[1], fk[2], fk[4])
            c_total = coarse_totals.get(coarse_key, 0)
            if c_total >= min_count:
                emp = coarse_sums[coarse_key] / c_total
                empiricals[i] = emp
                counts[i] = c_total

    return empiricals, counts

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                         fk_buckets: FeatureKeyBuckets) -> np.ndarray:
    params = {
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
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 12.0,
        "fk_strength_fn": "linear",
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75,
        "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15,
        "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5,
        "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005,
    }

    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()

    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params["fk_min_count"]
    
    fk_empiricals, fk_counts = build_hierarchical_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    pw = params["fk_prior_weight"]
    ms = params["fk_max_strength"]
    
    strengths = np.minimum(ms, cnt_grid * 0.15)
    
    strengths_3d = strengths[:, :, np.newaxis]
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    if global_mult.observed.sum() > 0:
        smooth_val = params["mult_smooth"]
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
        
        sett_ratio = ratio[1]
        
        ratio_grid = np.zeros((h, w, NUM_CLASSES))
        ratio_powered = np.power(ratio, params["mult_power"])
        
        ratio_powered[0] = np.clip(ratio_powered[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio_powered[5] = np.clip(ratio_powered[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio_powered[c] = np.clip(ratio_powered[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio_powered[4] = np.clip(ratio_powered[4], params["mult_forest_lo"], params["mult_forest_hi"])
        
        for c in range(NUM_CLASSES):
            ratio_grid[:, :, c] = ratio_powered[c]
            
        sett_points = [(int(s["x"]), int(s["y"])) for s in settlements]
        if sett_points:
            dist_grid = np.full((h, w), 999.0)
            sett_r5 = np.zeros((h, w), dtype=int)
            for y in range(h):
                for x in range(w):
                    dists = [abs(x - sx) + abs(y - sy) for sx, sy in sett_points]
                    if dists:
                        dist_grid[y, x] = min(dists)
                        sett_r5[y, x] = sum(1 for d in dists if d <= 5)
            
            coastal = _build_coastal_mask(terrain)
            is_initial_settlement = (terrain == 1) | (terrain == 2)
            
            dense_area = (sett_r5 >= 3)
            sparse_area = (sett_r5 <= 2)
            
            ratio_grid[is_initial_settlement & dense_area, 1] *= 0.75
            ratio_grid[is_initial_settlement & dense_area, 2] *= 0.75
            
            if sett_ratio > 0.5:
                ratio_grid[is_initial_settlement & dense_area, 3] *= 1.25
                ratio_grid[is_initial_settlement & sparse_area, 1] *= 1.20
                ratio_grid[is_initial_settlement & sparse_area, 2] *= 1.20
            
            if sett_ratio > 0.8:
                ratio_grid[(dist_grid == 1) & sparse_area, 1] *= 1.20
                ratio_grid[(dist_grid == 2) & sparse_area, 1] *= 1.10
                
                ratio_grid[(dist_grid == 1) & dense_area, 3] *= 1.35
                ratio_grid[(dist_grid == 2) & dense_area, 3] *= 1.25
                
                ratio_grid[dist_grid == 1, 3] *= 1.15
                ratio_grid[dist_grid == 2, 3] *= 1.10
                
                port_boost_mask = coastal & (dist_grid <= 3) & sparse_area
                ratio_grid[port_boost_mask, 2] *= 1.25
                
            pred *= ratio_grid
        else:
            pred *= ratio_powered[np.newaxis, np.newaxis, :]
            
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
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
