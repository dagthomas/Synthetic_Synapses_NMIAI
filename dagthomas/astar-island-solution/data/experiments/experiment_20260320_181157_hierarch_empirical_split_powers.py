# Experiment: hierarch_empirical_split_powers
# Hypothesis: 1. Hierarchical empirical fallback: Groups rare cell observations (like coastal) by coarse keys if exact matches are <5. This captures current-round regime data instead of falling back to multi-round priors. 2. Split multiplier powers: Applies a stronger regime adaptation (power=0.6) to volatile classes (settlement, port, ruin) while preserving stable classes (empty, forest) at power=0.4.
# Timestamp: 2026-03-20T18:11:57.467874+00:00

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

def build_hierarchical_empirical_lookup(fk_buckets, unique_keys, min_count=5):
    """
    Builds empirical distribution lookup table with hierarchical fallback.
    If exact feature key has < 5 observations, falls back to coarse key
    (ignoring forest neighbors), then to base key (ignoring has_port).
    This ensures we use current-round data for rare cells like coastal plains
    instead of falling back to multi-round priors that miss the regime shift.
    """
    n = len(unique_keys)
    empiricals = np.zeros((n, NUM_CLASSES), dtype=float)
    counts = np.zeros(n, dtype=float)

    coarse_counts = {}
    coarse_totals = {}
    base_counts = {}
    base_totals = {}
    
    # Pre-aggregate empirical data
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

    # Build lookup
    for i, fk in enumerate(unique_keys):
        emp, count = fk_buckets.get_empirical(fk)
        if emp is not None and count >= min_count:
            empiricals[i] = emp
            counts[i] = count
            continue
            
        coarse = (fk[0], fk[1], fk[2], fk[4])
        if coarse in coarse_totals and coarse_totals[coarse] >= min_count:
            empiricals[i] = coarse_counts[coarse] / coarse_totals[coarse]
            counts[i] = coarse_totals[coarse]
            continue
            
        base = (fk[0], fk[1], fk[2])
        if base in base_totals and base_totals[base] >= min_count:
            empiricals[i] = base_counts[base] / base_totals[base]
            counts[i] = base_totals[base]
            continue
            
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
        "fk_min_count": 5,
        
        "mult_smooth": 5.0,
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
    fk_min = params.get("fk_min_count", 5)
    
    # 1. Structural Improvement: Hierarchical empirical fallback
    fk_empiricals, fk_counts = build_hierarchical_empirical_lookup(fk_buckets, unique_keys, fk_min)

    pred = cal_priors[idx_grid]
    emp_grid = fk_empiricals[idx_grid]
    cnt_grid = fk_counts[idx_grid]
    has_fk = cnt_grid >= fk_min

    pw = params.get("fk_prior_weight", 5.0)
    ms = params.get("fk_max_strength", 8.0)
    strengths = np.minimum(ms, np.sqrt(cnt_grid))

    strengths_3d = strengths[:, :, np.newaxis]
    blended = pred * pw + emp_grid * strengths_3d
    blended_sum = np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    blended /= blended_sum
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    # 2. Structural Improvement: Per-class global multiplier powers
    if global_mult.observed.sum() > 0:
        smooth_val = params.get("mult_smooth", 5.0)
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
            
        # Apply stronger adaptation (0.6) to volatile classes: settlement, port, ruin
        # Keep empty and forest at base adaptation (0.4)
        power_array = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
        ratio = np.power(ratio, power_array)
        
        ratio[0] = np.clip(ratio[0], 0.75, 1.25)
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], 0.15, 2.5)
        ratio[4] = np.clip(ratio[4], 0.5, 1.8)
        
        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # 3. Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # 4. Vectorized floor
    floor = params.get("floor_nonzero", 0.005)
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # 5. Lock static cells
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred
