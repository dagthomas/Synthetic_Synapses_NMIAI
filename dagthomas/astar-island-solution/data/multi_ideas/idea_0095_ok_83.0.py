# Score: avg=82.99
# Direction: Increase the settlement class upper clamp from 2.5 to 2.8 to reduce entropy-weighted KL divergence c
# Error: None

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    try:
        p = _load_params()
    except NameError:
        p = {"prior_w": 2.0, "emp_max": 3.0, "base_power": 0.5}

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

    if fk_buckets is not None and hasattr(fk_buckets, 'get_empirical'):
        empiricals, counts = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)
    else:
        empiricals = np.zeros((len(unique_keys), NUM_CLASSES), dtype=float)
        counts = np.zeros(len(unique_keys), dtype=float)

    if global_mult is not None and hasattr(global_mult, 'observed') and global_mult.observed.sum() > 0:
        obs = global_mult.observed
        exp_arr = global_mult.expected
    else:
        obs = np.ones(NUM_CLASSES)
        exp_arr = np.ones(NUM_CLASSES)
    exp_arr = np.maximum(exp_arr, 1e-6)
    ratio = obs / exp_arr

    prior_w = p["prior_w"]
    emp_max = p["emp_max"]
    if variance_regime == 'EXTREME_BOOM':
        prior_w = max(prior_w - 0.5, 0.5)
        emp_max = emp_max * 1.2

    pred = priors[idx_grid]
    emp_grid = empiricals[idx_grid]
    cnt_grid = counts[idx_grid]
    has_fk = cnt_grid >= 5

    strengths = np.minimum(emp_max, np.sqrt(cnt_grid))
    blended = pred * prior_w + emp_grid * strengths[:, :, np.newaxis]
    blended /= np.maximum(blended.sum(axis=-1, keepdims=True), 1e-10)
    pred = np.where(has_fk[:, :, np.newaxis], blended, pred)

    bp = p["base_power"]
    mult = np.power(ratio, bp)
    mult[1] = np.power(ratio[1], bp)
    mult[2] = np.power(ratio[2], bp)
    mult[3] = np.power(ratio[3], bp)
    mult[0] = np.clip(mult[0], 0.75, 1.25)
    mult[1] = np.clip(mult[1], 0.15, 2.0)
    mult[2] = np.clip(mult[2], 0.15, 2.0)
    mult[3] = np.clip(mult[3], 0.15, 2.0)
    mult[4] = np.clip(mult[4], 0.5, 1.8)
    mult[5] = np.clip(mult[5], 0.85, 1.15)
    
    pred *= mult

    is_sett = (grid == 1) | (grid == 2)
    if is_sett.any():
        dist_map = distance_transform_cdt(~is_sett, metric='chessboard')
        sett_mult = np.clip(3.0 / (dist_map + 1.0), 0.1, 2.8)
        pred[:, :, 1] *= sett_mult
        pred[:, :, 2] *= sett_mult
        pred[:, :, 3] *= sett_mult
        
        if obs_expansion_radius is not None:
            out_of_bounds = dist_map > obs_expansion_radius
            pred[out_of_bounds, 1:4] *= 0.01

    coastal = _build_coastal_mask(grid)
    pred[~coastal, 2] = 0.0
    pred[grid != 3, 5] = 0.0

    pred = np.maximum(pred, 0.005)
    pred[~coastal, 2] = 0.0
    pred[grid != 3, 5] = 0.0

    probs = pred / np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    return probs