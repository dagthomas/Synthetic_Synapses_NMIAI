# Score: avg=86.70
# Direction: Looking at your error log, the `"Increase the port class prior multipli"` attempt crashed with a syn
# Error: None

def experimental_pred_fn(state, global_mult, fk_buckets):
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.3,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_max': 1.0,
        'cal_global_weight': 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, cal_params)

    # 3. Empirical
    empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    obs = global_mult.observed
    exp_arr = np.maximum(global_mult.expected, 1e-6)
    ratio = obs / exp_arr

    # 4. Blending with regime-adaptive empirical trust
    emp_max_weight = np.clip(12.0 - 4.0 * ratio[1], 6.0, 12.0)
    power_sett_coastal = np.array([0.4, 0.40, 0.75, 0.75, 0.4, 0.4])
    power_sett_inland  = np.array([0.4, 0.75, 0.75, 0.75, 0.4, 0.4])
    power_exp_coastal  = np.array([0.4, 0.40, 0.60, 0.50, 0.4, 0.4])
    power_exp_inland   = np.array([0.4, 0.50, 0.60, 0.50, 0.4, 0.4])

    regime_raw = float(np.clip(ratio[1], 0.3, 1.5))
    sett_prior_boost = 1.30 + 0.17 * (regime_raw - 0.3) / 1.2
    port_prior_boost_factor = 1.08  # +8% port class prior on coastal cells

    N = len(unique_keys)
    lookup = np.zeros((N, 6), dtype=np.float32)

    for i in range(N):
        prior = priors[i].copy()
        emp = empiricals[i]
        c = counts[i]
        dist_bucket = unique_keys[i][1]
        is_coastal = unique_keys[i][2]

        if dist_bucket <= 3:
            prior[1] *= sett_prior_boost
            prior = prior / prior.sum()

        # Port class prior boost on coastal cells
        if is_coastal:
            prior[2] *= port_prior_boost_factor
            prior = prior / prior.sum()

        prior_w = 5.0
        emp_w = min(math.sqrt(c), emp_max_weight)

        if c >= 5:
            blend = (prior * prior_w + emp * emp_w) / max(prior_w, emp_w)
        else:
            blend = prior

        if dist_bucket == 0:
            mults = np.power(ratio, power_sett_coastal if is_coastal else power_sett_inland)
        else:
            mults = np.power(ratio, power_exp_coastal if is_coastal else power_exp_inland)

        mults[1] = np.clip(mults[1], 0.08, 2.0)
        mults[2] = np.clip(mults[2], 0.08, 2.0)
        mults[3] = np.clip(mults[3], 0.08, 2.0)

        blend = blend * mults
        blend = blend / blend.sum()
        lookup[i] = blend

    probs = lookup[idx_grid]

    # 5. Sigmoid distance-from-settlement spatial multipliers
    is_sett = (grid == 1) | (grid == 2)
    land_mask = (grid != 10) & (grid != 5)

    if is_sett.any():
        dist_map = distance_transform_cdt(~is_sett, metric='taxicab')
    else:
        dist_map = np.full((MAP_H, MAP_W), 99, dtype=np.int32)

    regime = float(np.clip(ratio[1], 0.3, 1.5))
    coastal_mask = _build_coastal_mask(grid)
    dist_float = dist_map.astype(np.float32)

    # Sigmoid parameters
    beta = 0.8
    gamma = 3.5

    sigmoid_decay = 1.0 / (1.0 + np.exp(beta * (dist_float - gamma)))
    sigmoid_rise = 1.0 - sigmoid_decay

    # 5a. Settlement class
    alpha_sett = 0.35 + 0.10 * regime
    base_sett = 0.85 + 0.03 * regime
    sett_mult = base_sett + (1.0 + alpha_sett - base_sett) * sigmoid_decay
    coastal_penalty = 0.92 - 0.03 * regime
    sett_mult = np.where(coastal_mask & land_mask, sett_mult * coastal_penalty, sett_mult)
    sett_mult[~land_mask] = 1.0

    # 5b. Empty class
    empty_near = 0.82 + 0.02 * regime
    empty_mult = empty_near + (1.0 - empty_near) * sigmoid_rise
    empty_mult[~land_mask] = 1.0

    # 5c. Port class: coastal proximity Gaussian
    if coastal_mask.any():
        dist_to_coast = distance_transform_cdt(~coastal_mask)
    return probs