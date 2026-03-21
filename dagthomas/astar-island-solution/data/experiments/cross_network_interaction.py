import numpy as np
import math
from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import _build_coastal_mask, _build_feature_key_index, build_calibration_lookup, build_fk_empirical_lookup
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class
import predict
from scipy.ndimage import uniform_filter, distance_transform_cdt


def experimental_pred_fn(state, global_mult, fk_buckets):
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.0,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)

    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    obs = global_mult.observed
    exp = np.maximum(global_mult.expected, 1e-6)
    has_regime = obs.sum() > 1.0

    if has_regime:
        ratio = obs / exp
        sett_ratio = float(min(ratio[1], 2.5))
        collapse_sig = max(0.0, 1.0 - sett_ratio)
        thrive_sig = max(0.0, sett_ratio - 0.8)
        regime_extremity = abs(sett_ratio - 1.0)
    else:
        ratio = np.ones(NUM_CLASSES)
        sett_ratio = 1.0
        collapse_sig = 0.0
        thrive_sig = 0.2
        regime_extremity = 0.0

    base_emp_cap = float(np.clip(12.0 - 4.0 * sett_ratio, 6.0, 12.0))

    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)

    for i in range(N):
        terrain, dist_bucket, coastal, forest_nbrs, has_port = unique_keys[i]
        prior = priors[i]
        emp = empiricals[i]
        c = counts[i]

        f_dist = (4 - dist_bucket) / 4.0
        f_coastal = 1.0 if coastal else 0.0
        f_forest = forest_nbrs / 3.0
        f_sett = 1.0 if terrain in (1, 2, 3) else 0.0

        cross_dist_regime = f_dist * regime_extremity
        cross_coastal_collapse = f_coastal * collapse_sig
        cross_coastal_thrive = f_coastal * thrive_sig
        cross_sett_collapse = f_sett * collapse_sig
        cross_dist_thrive = f_dist * thrive_sig
        cross_forest = f_forest * (1.0 if terrain in (0, 11, 4) else 0.3)

        emp_cap = base_emp_cap + 2.0 * cross_dist_regime + 1.5 * cross_coastal_collapse
        emp_cap = max(5.0, min(emp_cap, 14.0))

        prior_w = 5.0
        emp_w = min(math.sqrt(c), emp_cap) if c >= 5 else 0.0

        if c >= 5:
            blend = (prior * prior_w + emp * emp_w) / (prior_w + emp_w)
        else:
            blend = prior.copy()

        if dist_bucket == 0:
            power = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
        else:
            power = np.array([0.4, 0.50, 0.60, 0.50, 0.4, 0.4])

        power[1] += 0.10 * cross_sett_collapse
        power[3] += 0.08 * cross_sett_collapse
        power[2] += 0.12 * cross_coastal_collapse + 0.08 * cross_coastal_thrive
        power[1] += 0.05 * cross_coastal_collapse + 0.06 * cross_dist_thrive

        if terrain in (0, 11):
            power[4] += 0.05 * cross_forest

        power = np.clip(power, 0.1, 0.95)

        mults = np.power(ratio, power)
        mults[1] = np.clip(mults[1], 0.15, 2.5)
        mults[2] = np.clip(mults[2], 0.15, 2.5)
        mults[3] = np.clip(mults[3], 0.15, 2.5)

        if cross_coastal_collapse > 0.2 and dist_bucket <= 2:
            delta = 0.01 * cross_coastal_collapse
            blend[0] += delta
            blend[1] = max(blend[1] - delta * 0.4, 0.001)
            blend[2] = max(blend[2] - delta * 0.6, 0.001)

        if cross_forest > 0.2 and dist_bucket <= 1 and thrive_sig > 0.2:
            blend[1] += 0.003 * cross_forest * thrive_sig

        blend = blend * mults
        s = blend.sum()
        if s > 0:
            blend /= s

        lookup[i] = blend

    probs = lookup[idx_grid]

    is_sett = (grid == 1) | (grid == 2)
    dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
    radius = 2 + int(3.0 * min(sett_ratio, 1.2))
    T_max = 1.0 + 0.10 * math.sqrt(min(sett_ratio, 1.0))
    coastal_mask = _build_coastal_mask(grid)
    T_vals = np.ones((MAP_H, MAP_W), dtype=np.float32)
    T_vals[dist_map <= radius] = T_max
    coastal_near = (dist_map <= radius) & coastal_mask
    T_vals[coastal_near] += 0.03 * regime_extremity
    T_grid = T_vals[:, :, np.newaxis]

    probs = np.power(probs, 1.0 / T_grid)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    alpha = 0.75
    for k in [1, 3]:
        smoothed = uniform_filter(probs[:, :, k], size=3, mode='reflect')
        probs[:, :, k] = probs[:, :, k] * alpha + smoothed * (1 - alpha)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0

    probs = np.maximum(probs, 0.005)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)

    probs[grid == 10] = [1, 0, 0, 0, 0, 0]
    probs[grid == 5] = [0, 0, 0, 0, 0, 1]
    probs[0, :] = [1, 0, 0, 0, 0, 0]
    probs[-1, :] = [1, 0, 0, 0, 0, 0]
    probs[:, 0] = [1, 0, 0, 0, 0, 0]
    probs[:, -1] = [1, 0, 0, 0, 0, 0]

    return probs
