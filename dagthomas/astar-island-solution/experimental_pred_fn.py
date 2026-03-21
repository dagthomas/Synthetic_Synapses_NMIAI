import numpy as np
import math
from calibration import build_feature_keys
from fast_predict import (
    _build_coastal_mask,
    _build_feature_key_index,
    build_calibration_lookup,
    build_fk_empirical_lookup,
)
from scipy.ndimage import uniform_filter, distance_transform_cdt
import predict


def experimental_pred_fn(state: dict, global_mult, fk_buckets) -> np.ndarray:
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0,
        'cal_fine_divisor': 100.0,
        'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5,
        'cal_coarse_divisor': 100.0,
        'cal_coarse_max': 2.2,
        'cal_base_base': 0.1,
        'cal_base_divisor': 100.0,
        'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)

    # 3. Empirical
    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    # Pre-calculate ratio for blending logic
    obs = global_mult.observed
    exp = global_mult.expected
    exp = np.maximum(exp, 1e-6)
    ratio = obs / exp

    # 4. Log-odds blending with observation-count-weighted alpha
    EPS = 1e-6
    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)

    power_sett = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
    power_exp  = np.array([0.4, 0.50, 0.60, 0.50, 0.4, 0.4])

    # Adaptive temperature for high-obs cells (scales up empirical influence in log-odds)
    # Higher obs_count -> higher effective temperature on empirical before blending
    for i in range(N):
        prior = priors[i].copy()
        emp = empiricals[i].copy()
        c = counts[i]
        dist_bucket = unique_keys[i][1]

        if c >= 5:
            # Clamp probabilities away from 0/1 for log-odds conversion
            prior_c = np.clip(prior, EPS, 1.0 - EPS)
            emp_c = np.clip(emp, EPS, 1.0 - EPS)

            # Normalize so they sum to 1 after clipping
            prior_c = prior_c / prior_c.sum()
            emp_c = emp_c / emp_c.sum()
            prior_c = np.clip(prior_c, EPS, 1.0 - EPS)
            emp_c = np.clip(emp_c, EPS, 1.0 - EPS)

            # Convert to log-odds (log(p / (1-p)))
            log_odds_prior = np.log(prior_c / (1.0 - prior_c))
            log_odds_emp = np.log(emp_c / (1.0 - emp_c))

            # Adaptive temperature: high obs_count scales up empirical signal
            # T_emp > 1 sharpens the empirical distribution in log-odds space
            T_emp = 1.0 + 0.3 * math.sqrt(min(c, 400.0)) / 20.0
            log_odds_emp_scaled = log_odds_emp * T_emp

            # Alpha increases with sqrt(obs_count): more data -> trust empirical more
            alpha = min(math.sqrt(c) / (math.sqrt(c) + 5.0), 0.75)

            # Blend in log-odds space
            log_odds_blend = alpha * log_odds_emp_scaled + (1.0 - alpha) * log_odds_prior

            # Convert back to probabilities via sigmoid
            blend = 1.0 / (1.0 + np.exp(-log_odds_blend))
            blend = np.maximum(blend, EPS)
            blend = blend / blend.sum()
        else:
            blend = prior.copy()

        # Apply multipliers based on distance bucket
        if dist_bucket == 0:
            mults = np.power(ratio, power_sett)
        else:
            mults = np.power(ratio, power_exp)

        mults[1] = np.clip(mults[1], 0.15, 2.8)
        mults[2] = np.clip(mults[2], 0.15, 2.5)
        mults[3] = np.clip(mults[3], 0.15, 2.5)

        blend = blend * mults
        blend = blend / blend.sum()

        lookup[i] = blend

    probs = lookup[idx_grid]

    # 5. Distance-aware Dynamic Softening
    is_sett = (grid == 1) | (grid == 2)
    dist_map = distance_transform_cdt(~is_sett, metric='taxicab')

    radius = 2 + int(3.0 * min(ratio[1], 1.2))

    T_max = 1.0 + 0.10 * math.sqrt(min(ratio[1], 1.0))
    T_grid = np.ones((40, 40, 1), dtype=np.float32)
    T_grid[dist_map <= radius] = T_max

    probs = np.power(probs, 1.0/T_grid)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Spatial smoothing (disabled for port class 2)
    alpha = 0.75
    for k in [1, 3]:
        smoothed = uniform_filter(probs[:, :, k], size=5, mode='reflect')
        probs[:, :, k] = probs[:, :, k] * alpha + smoothed * (1 - alpha)

    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 6. Structural zeros
    coastal_mask = _build_coastal_mask(grid)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0

    # 6b. Coastal settlement survival penalty
    probs[coastal_mask, 1] *= 0.6

    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 7. Floor
    probs = np.maximum(probs, 0.005)
    probs[grid != 5, 5] = 0.0
    probs[~coastal_mask, 2] = 0.0
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # 8. Lock static cells
    probs[grid == 10] = [1, 0, 0, 0, 0, 0]
    probs[grid == 5] = [0, 0, 0, 0, 0, 1]

    probs[0, :] = [1, 0, 0, 0, 0, 0]
    probs[-1, :] = [1, 0, 0, 0, 0, 0]
    probs[:, 0] = [1, 0, 0, 0, 0, 0]
    probs[:, -1] = [1, 0, 0, 0, 0, 0]

    return probs
