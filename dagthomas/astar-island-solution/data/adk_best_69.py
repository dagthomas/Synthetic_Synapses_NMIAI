# ADK Research Agent Best Experiment #69
# Score: avg=91.276, improvement=+0.033
# Timestamp: 2026-03-20T13:04:24.946386+00:00

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
from fast_predict import (
    _build_feature_key_index, _build_coastal_mask,
    build_calibration_lookup, build_fk_empirical_lookup
)
import predict

def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    params = {
        "cal_fine_base": 0.0, "cal_fine_divisor": 10.0, "cal_fine_max": 20.0,
        "cal_coarse_base": 0.0, "cal_coarse_divisor": 20.0, "cal_coarse_max": 10.0,
        "cal_base_base": 0.0, "cal_base_divisor": 50.0, "cal_base_max": 5.0,
        "cal_global_weight": 1.0,
        "fk_min_count": 5,
        "fk_max_strength": 8.0,
        "fk_prior_weight": 5.0,
        "mult_smooth": 5.0,
        "mult_power": 0.4,
        "mult_empty_lo": 0.75, "mult_empty_hi": 1.25,
        "mult_sett_lo": 0.15, "mult_sett_hi": 2.5,
        "mult_forest_lo": 0.5, "mult_forest_hi": 1.8,
        "floor_nonzero": 0.005
    }

    grid = state["grid"]
    settlements = state["settlements"]
    terrain = np.array(grid, dtype=int)
    h, w = terrain.shape

    cal = predict.get_calibration()
    from calibration import build_feature_keys
    fkeys = build_feature_keys(terrain, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal_priors = build_calibration_lookup(cal, unique_keys, params)
    fk_min = params["fk_min_count"]
    fk_empiricals, fk_counts = build_fk_empirical_lookup(fk_buckets, unique_keys, fk_min)

    # Vectorized prediction
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

    # Multipliers & Collapse Detection
    collapse_detected = False
    base_ratio_sett = 1.0
    if global_mult.observed.sum() > 0:
        smooth_val = params["mult_smooth"]
        smooth = smooth_val * np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        ratio = (global_mult.observed + smooth) / np.maximum(
            global_mult.expected + smooth, 1e-6)
            
        base_ratio_sett = ratio[1]
        if base_ratio_sett < 0.5:
            collapse_detected = True
            
        power = np.array([0.4, 0.6, 0.6, 0.6, 0.4, 0.4])
        ratio_powered = np.power(ratio, power)
        
        ratio_powered[0] = np.clip(ratio_powered[0], params["mult_empty_lo"], params["mult_empty_hi"])
        ratio_powered[5] = np.clip(ratio_powered[5], 0.85, 1.15)
        for c in (1, 2, 3):
            ratio_powered[c] = np.clip(ratio_powered[c], params["mult_sett_lo"], params["mult_sett_hi"])
        ratio_powered[4] = np.clip(ratio_powered[4], params["mult_forest_lo"], params["mult_forest_hi"])
        
        pred *= ratio_powered[np.newaxis, np.newaxis, :]
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    # Settlement Density Logic
    if not collapse_detected:
        sett_pos = [(s["y"], s["x"]) for s in settlements]
        density_map = np.zeros((h, w), dtype=int)
        
        for (sy1, sx1) in sett_pos:
            for (sy2, sx2) in sett_pos:
                if sy1 == sy2 and sx1 == sx2:
                    continue
                dist = abs(sy1 - sy2) + abs(sx1 - sx2)
                if dist <= 5:
                    density_map[sy1, sx1] += 1
                    
        for (sy, sx) in sett_pos:
            if terrain[sy, sx] in (1, 2):
                neighbors = density_map[sy, sx]
                if neighbors == 0:
                    pred[sy, sx, 1] *= 1.15
                    pred[sy, sx, 2] *= 1.15
                elif neighbors == 1:
                    pred[sy, sx, 1] *= 1.05
                    pred[sy, sx, 2] *= 1.05
                else:
                    pred[sy, sx, 1] *= 0.85
                    pred[sy, sx, 2] *= 0.85
                pred[sy, sx] /= np.maximum(pred[sy, sx].sum(), 1e-10)

    # Port Boosting on Coastal Cells
    coastal = _build_coastal_mask(terrain)
    port_boost = 1.05 if collapse_detected else 1.25
    pred[coastal, 2] *= port_boost

    row_sums = pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    pred /= row_sums

    # RUIN RECLAMATION FIX
    if not collapse_detected and 0.1 < base_ratio_sett < 0.9:
        for (sy, sx) in [(s["y"], s["x"]) for s in settlements]:
            if terrain[sy, sx] in (1, 2):
                if pred[sy, sx, 1] < 0.5:
                    pred[sy, sx, 3] *= 1.50
                    pred[sy, sx] /= np.maximum(pred[sy, sx].sum(), 1e-10)

    is_ruin = terrain == 3
    if is_ruin.any():
        pred[is_ruin, 3] *= 1.50
        pred[is_ruin] /= np.maximum(pred[is_ruin].sum(axis=-1, keepdims=True), 1e-10)


    # Collapse Mass Redistribution
    if collapse_detected:
        static_mask = (terrain == 10) | (terrain == 5)
        dynamic_mask = ~static_mask
        dyn_pred = pred[dynamic_mask]
        
        volatile_mass = dyn_pred[:, 1] + dyn_pred[:, 2] + dyn_pred[:, 3]
        mask = volatile_mass > 0.05
        
        keep = params["floor_nonzero"]
        redistribute = volatile_mass[mask] - (keep * 3)
        
        pos_mask = redistribute > 0
        if pos_mask.any():
            final_mask = np.zeros(len(dyn_pred), dtype=bool)
            final_mask[mask] = pos_mask
            
            p_to_mod = dyn_pred[final_mask]
            r_amount = redistribute[pos_mask]
            
            p_to_mod[:, 1] = keep
            p_to_mod[:, 2] = keep
            p_to_mod[:, 3] = keep
            
            ef_total = p_to_mod[:, 0] + p_to_mod[:, 4]
            ef_mask = ef_total > 0
            
            p_to_mod[ef_mask, 0] += r_amount[ef_mask] * (p_to_mod[ef_mask, 0] / ef_total[ef_mask])
            p_to_mod[ef_mask, 4] += r_amount[ef_mask] * (p_to_mod[ef_mask, 4] / ef_total[ef_mask])
            p_to_mod[~ef_mask, 0] += r_amount[~ef_mask]
            
            dyn_pred[final_mask] = p_to_mod
            pred[dynamic_mask] = dyn_pred

    # EXPERIMENTAL NEW FEATURE: REGIME-AWARE EMPIRICAL BLEND SATURATION
    # When empirical data perfectly matches 0.0 (like mountain on dynamic cell),
    # the prior pulls it up to something nonzero.
    # In thriving regimes (> 1.0) we want regularization. 
    # But for empty cells specifically, they are incredibly stable.
    # We should aggressively saturate empirical data for empty/forest when count >= 15.
    high_conf = cnt_grid >= 15
    if high_conf.any():
        e_emp = emp_grid[high_conf, 0]
        f_emp = emp_grid[high_conf, 4]
        # Blend another 50% specifically toward empirical for these highly observed cells
        pred[high_conf, 0] = 0.5 * pred[high_conf, 0] + 0.5 * e_emp
        pred[high_conf, 4] = 0.5 * pred[high_conf, 4] + 0.5 * f_emp

        row_sums = pred.sum(axis=-1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        pred /= row_sums

    # Structural zeros
    static_mask = (terrain == 10) | (terrain == 5)
    dynamic_mask = ~static_mask
    pred[dynamic_mask, 5] = 0.0

    inland_dynamic = dynamic_mask & ~coastal
    pred[inland_dynamic, 2] = 0.0

    # ZERO FORESTS ON PLAINS AWAY FROM FORESTS IN THRIVING ROUNDS
    if not collapse_detected:
        n_keys = len(unique_keys)
        fn_1d = np.zeros(n_keys, dtype=float)
        for i, fk in enumerate(unique_keys):
            fn_1d[i] = fk[3]
        
        fn_grid = fn_1d[idx_grid]
        is_plains = terrain == 0
        no_forest_nbr = fn_grid == 0.0
        
        pred[is_plains & no_forest_nbr, 4] *= 0.1
        row_sums = pred.sum(axis=-1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        pred /= row_sums

    # Vectorized floor
    floor = params["floor_nonzero"]
    dynamic_pred = pred[dynamic_mask]
    nonzero_mask = dynamic_pred > 0
    dynamic_pred = np.where(nonzero_mask, np.maximum(dynamic_pred, floor), 0.0)
    
    row_sums = dynamic_pred.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    dynamic_pred /= row_sums
    pred[dynamic_mask] = dynamic_pred

    # Lock static
    pred[terrain == 5] = [0, 0, 0, 0, 0, 1]
    pred[terrain == 10] = [1, 0, 0, 0, 0, 0]

    return pred
