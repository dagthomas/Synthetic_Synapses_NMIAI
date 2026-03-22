# Score: avg=43.66
# Direction: Adjust the settlement class clamp range from [0.15, 2.5] to [0.15, 2.2]. This conservative reduction
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
    p = {"prior_w": 1.0, "emp_max": 2.0}
    grid = np.array(state['grid'])
    settlements = np.array(state['settlements'])

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
    
    pred[:, :, 1] = np.clip(pred[:, :, 1], 0.15, 2.2)

    if sim_pred is not None:
        pred = pred * (1.0 - sim_alpha) + sim_pred * sim_alpha

    for c in range(NUM_CLASSES):
        pred[:, :, c] = uniform_filter(pred[:, :, c], size=3)

    coastal_mask = _build_coastal_mask(grid)
    pred[:, :, 2] *= coastal_mask
    
    mountain_mask = (grid == 3)
    pred[:, :, 5] = np.where(mountain_mask, pred[:, :, 5], 0.0)

    pred = np.maximum(pred, 0.0)
    pred_mask = pred > 0
    pred[pred_mask] = np.maximum(pred[pred_mask], 0.005)
    
    if obs_expansion_radius is not None:
        init_sett = (settlements == 1)
        if init_sett.sum() > 0:
            dist = distance_transform_cdt(~init_sett)
            pred[dist > obs_expansion_radius, 1] = 0.0

    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    return pred
    return probs