# ADK Research Agent Best Experiment #14
# Score: avg=91.102, improvement=+0.032
# Timestamp: 2026-03-20T10:25:30.945609+00:00

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
import predict
from calibration import build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers, fk_buckets: FeatureKeyBuckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.0, "cal_fine_divisor": 100.0, "cal_fine_max": 2.0,
        "cal_coarse_base": 0.0, "cal_coarse_divisor": 200.0, "cal_coarse_max": 1.0,
        "cal_base_base": 0.0, "cal_base_divisor": 500.0, "cal_base_max": 0.5,
        "cal_global_weight": 0.1,
        "fk_min_count": 5,
        "fk_prior_weight": 5.0,
        "fk_max_strength": 8.0,
        "fk_strength_fn": "sqrt",
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
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
    
    # Hierarchical Empirical Blending!
    fk_min = params.get("fk_min_count", 5)
    
    # 1. We need to aggregate fk_buckets into coarse buckets!
    # coarse_key = (terrain_code, dist_bucket, coastal_bool, has_port_flag)
    coarse_emp = {}
    
    for fk in unique_keys:
        emp, count = fk_buckets.get_empirical(fk)
        if emp is not None and count > 0:
            coarse_key = (fk[0], fk[1], fk[2], fk[4])
            if coarse_key not in coarse_emp:
                coarse_emp[coarse_key] = {"sums": np.zeros(NUM_CLASSES), "count": 0}
            coarse_emp[coarse_key]["sums"] += emp * count
            coarse_emp[coarse_key]["count"] += count

    # 2. Build empirical lookup table with hierarchical fallback
    n = len(unique_keys)
    fk_empiricals = np.zeros((n, NUM_CLASSES), dtype=float)
    fk_counts = np.zeros(n, dtype=float)
    
    for i, fk in enumerate(unique_keys):
        emp, count = fk_buckets.get_empirical(fk)
        
        # If we have enough fine data, use it
        if emp is not None and count >= fk_min:
            fk_empiricals[i] = emp
            fk_counts[i] = count
        else:
            # Fallback to coarse empirical data for this round
            coarse_key = (fk[0], fk[1], fk[2], fk[4])
            if coarse_key in coarse_emp and coarse_emp[coarse_key]["count"] >= fk_min:
                c_data = coarse_emp[coarse_key]
                fk_empiricals[i] = c_data["sums"] / c_data["count"]
                # We discount the count slightly since it's coarse data
                fk_counts[i] = c_data["count"] * 0.5

    # Vectorized prediction
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

    # Multipliers
    if global_mult.observed.sum() > 0:
        smooth_val = params.get("mult_smooth", 5.0)
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        raw_ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
        
        ratio = np.power(raw_ratio, params.get("mult_power", 0.4))
        ratio_sett = np.power(raw_ratio, 0.6)
        ratio[1:4] = ratio_sett[1:4]
        
        ratio[0] = np.clip(ratio[0], params.get("mult_empty_lo", 0.75), params.get("mult_empty_hi", 1.25))
        ratio[5] = np.clip(ratio[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio[c] = np.clip(ratio[c], params.get("mult_sett_lo", 0.15), params.get("mult_sett_hi", 2.5))
        ratio[4] = np.clip(ratio[4], params.get("mult_forest_lo", 0.5), params.get("mult_forest_hi", 1.8))

        pred *= ratio[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    coastal = _build_coastal_mask(terrain)
    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # Vectorized floor: set all zeros to 0, nonzeros to max(val, floor)
    floor = params.get("floor_nonzero", 0.005)
    dynamic_pred = pred[dynamic_mask]  # (N_dynamic, 6)
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    # Renormalize
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred
