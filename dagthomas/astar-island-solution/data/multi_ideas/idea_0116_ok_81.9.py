# Score: avg=81.92
# Direction: Increase the `port` class global multiplier from 1.0 to 1.12 to compensate for the identified underp
# Error: None

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None,
                   est_vigor: float = None,
                   sim_pred: np.ndarray = None,
                   sim_alpha: float = 0.25,
                   growth_front_map: np.ndarray = None,
                   obs_overlay: tuple = None,
                   sett_survival: tuple = None) -> np.ndarray:
    p = {"prior_w": 1.5, "emp_max": 5.0}
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
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

    ratio[2] *= 1.12

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

    pred *= ratio

    if sim_pred is not None:
        pred = pred * (1.0 - sim_alpha) + sim_pred * sim_alpha

    is_mountain = (grid == 5)
    pred[~is_mountain, 5] = 0.0

    coastal_mask = _build_coastal_mask(grid)
    pred[~coastal_mask, 2] = 0.0

    if obs_expansion_radius is not None and len(settlements) > 0:
        y_coords, x_coords = np.indices((MAP_H, MAP_W))
        min_dist = np.full((MAP_H, MAP_W), 9999.0)
        for sy, sx in settlements:
            d = np.abs(y_coords - sy) + np.abs(x_coords - sx)
            min_dist = np.minimum(min_dist, d)
        pred[min_dist > obs_expansion_radius, 1:5] = 0.0

    if growth_front_map is not None:
        pred[:, :, 1:5] *= growth_front_map[:, :, np.newaxis]

    pred[pred > 0] = np.maximum(pred[pred > 0], 0.005)
    
    pred[~is_mountain, 5] = 0.0
    pred[~coastal_mask, 2] = 0.0

    row_sums = pred.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = pred / row_sums

    return probs