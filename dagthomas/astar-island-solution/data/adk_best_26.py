# ADK Research Agent Best Experiment #26
# Score: avg=87.902, improvement=+0.476
# Timestamp: 2026-03-20T18:28:23.403826+00:00

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

def build_smooth_hierarchical_empirical_lookup(fk_buckets, unique_keys):
    n = len(unique_keys)
    empiricals = np.zeros((n, NUM_CLASSES), dtype=float)
    counts = np.zeros(n, dtype=float)

    coarse_counts = {}
    coarse_totals = {}
    base_counts = {}
    base_totals = {}
    
    for fk, count_arr in fk_buckets.counts.items():
        total = fk_buckets.totals[fk]
        if total == 0: continue
        
        coarse = (fk[0], fk[1], fk[2], fk[4])
        if coarse not in coarse_counts:
            coarse_counts[coarse] = np.zeros(NUM_CLASSES, dtype=float)
            coarse_totals[coarse] = 0
        coarse_counts[coarse] += count_arr
        coarse_totals[coarse] += total
        
        base = (fk[0], fk[1], fk[2])
        if base not in base_counts:
            base_counts[base] = np.zeros(NUM_CLASSES, dtype=float)
            base_totals[base] = 0
        base_counts[base] += count_arr
        base_totals[base] += total
        
    global_emp_counts = np.zeros(NUM_CLASSES, dtype=float)
    global_emp_total = 0
    for count_arr in fk_buckets.counts.values():
        global_emp_counts += count_arr
        global_emp_total += np.sum(count_arr)
        
    global_emp_dist = global_emp_counts / max(global_emp_total, 1e-6)

    for i, fk in enumerate(unique_keys):
        fine_emp, fine_count = fk_buckets.get_empirical(fk)
        coarse = (fk[0], fk[1], fk[2], fk[4])
        base = (fk[0], fk[1], fk[2])
        
        coarse_count = coarse_totals.get(coarse, 0)
        base_count = base_totals.get(base, 0)
        
        vector = np.zeros(NUM_CLASSES, dtype=float)
        total_weight = 0.0
        
        # Blending parameters specifically tuned for current-round data volumes
        # Current round has ~11,000 observations total (vs 40,000 per historical round)
        if fine_count > 0:
            fw = min(3.0, 0.5 + fine_count / 10.0)
            vector += fw * fine_emp
            total_weight += fw
            
        if coarse_count > 0:
            cw = min(1.5, 0.2 + coarse_count / 20.0)
            coarse_dist = coarse_counts[coarse] / coarse_count
            vector += cw * coarse_dist
            total_weight += cw
            
        if base_count > 0:
            bw = min(0.5, 0.1 + base_count / 50.0)
            base_dist = base_counts[base] / base_count
            vector += bw * base_dist
            total_weight += bw
            
        gw = 0.1
        vector += gw * global_emp_dist
        total_weight += gw
        
        empiricals[i] = vector / total_weight
        # Represent the count as the fine count, but boosted slightly if we relied on coarse
        counts[i] = max(fine_count, coarse_count * 0.2, base_count * 0.05)
            
    return empiricals, counts

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.5,
        "cal_fine_divisor": 50.0,
        "cal_fine_max": 3.0,
        "cal_coarse_base": 0.2,
        "cal_coarse_divisor": 100.0,
        "cal_coarse_max": 1.5,
        "cal_base_base": 0.1,
        "cal_base_divisor": 200.0,
        "cal_base_max": 0.5,
        "cal_global_weight": 0.1,
        
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_min_count": 0, # Unused now because smooth lookup handles zero counts gracefully
        
        "mult_smooth": 10.0,
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
    
    fk_empiricals, fk_counts = build_smooth_hierarchical_empirical_lookup(fk_buckets, unique_keys)

    # 1. Base Prior
    pred = cal_priors[idx_grid]

    # 2. Density-Aware Adjustment
    sett_positions = [(int(s["y"]), int(s["x"])) for s in settlements if s.get("alive", True)]
    sett_r5 = np.zeros((h, w), dtype=int)
    for sy, sx in sett_positions:
        y_min = max(0, sy - 5)
        y_max = min(h, sy + 6)
        for y in range(y_min, y_max):
            dx = 5 - abs(y - sy)
            x_min = max(0, sx - dx)
            x_max = min(w, sx + dx + 1)
            sett_r5[y, x_min:x_max] += 1

    is_settlement = (terrain == 1) | (terrain == 2)
    survival_mult = np.clip(1.3 - 0.15 * sett_r5, 0.55, 1.15)
    death_mult = 2.0 - survival_mult
    
    sm_grid = np.ones((h, w), dtype=float)
    sm_grid[is_settlement] = survival_mult[is_settlement]
    
    dm_grid = np.ones((h, w), dtype=float)
    dm_grid[is_settlement] = death_mult[is_settlement]
    
    pred[:, :, 1] *= sm_grid
    pred[:, :, 2] *= sm_grid
    pred[:, :, 0] *= dm_grid
    pred[:, :, 3] *= dm_grid
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 3. Blend with empirical data
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]

    pw = params.get("fk_prior_weight", 5.0)
    ms = params.get("fk_max_strength", 8.0)
    strengths = np.minimum(ms, np.sqrt(cnt_grid))

    strengths_3d = strengths[:, :, np.newaxis]
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    # Since we smoothly fallback to global empiricals, we can apply empiricals everywhere!
    pred = blended

    # 4. Global Multipliers with Prior-Weighted Smoothing
    if global_mult.observed.sum() > 0:
        smooth_total = params.get("mult_smooth", 10.0)
        smooth = smooth_total * cal.global_probs
        
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
            
        power_array = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
        ratio = np.power(ratio, power_array)
        
        ratio[0] = np.clip(ratio[0], 0.6, 1.4)
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], 0.02, 2.5)
        ratio[4] = np.clip(ratio[4], 0.5, 1.8)
        
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 5. Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # 6. Vectorized Iterative Floor
    floor = params.get("floor_nonzero", 0.005)
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    
    for _ in range(3):
        dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
        row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        dynamic_pred /= row_sums
        
    pred[dynamic_mask] = dynamic_pred

    # 7. Lock static cells
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred

