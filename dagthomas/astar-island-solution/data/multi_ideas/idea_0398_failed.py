# FAILED: Compilation failed: '[' was never closed (<string>, line 106)
# Direction: Adjust the `settlement` class `clamp_range` from [0.15, 2.5] to [0.14, 2.6] to allow for slightly hi

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
    p = {"prior_w": 2.0, "emp_max": 3.0, "ratio_w": 0.5}
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
    raw_priors = priors.copy()

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

    ratio_w = p["ratio_w"]
    ratios = ratio * ratio_w + (1.0 - ratio_w)
    pred = pred * ratios

    clamp_ranges = {
        0: (0.15, 2.5),
        1: (0.14, 2.6),
        2: (0.15, 2.5),
        3: (0.15, 2.5),
        4: (0.15, 2.5),
        5: (0.5, 2.0)
    }
    raw_priors_grid = raw_priors[idx_grid]
    for c, (cmin, cmax) in clamp_ranges.items():
        if c < NUM_CLASSES:
            pred[:, :, c] = np.clip(pred[:, :, c], raw_priors_grid[:, :, c] * cmin, raw_priors_grid[:, :, c] * cmax)

    is_mountain = (grid == 3)
    is_coast = _build_coastal_mask(grid)

    pred = np.maximum(pred, 0.005)
    pred[~is_mountain, 5] = 0.0
    pred[~is_coast, 2] = 0.0
    pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    if sim_pred is not None:
        pred = (1.0 - sim_alpha) * pred + sim_alpha * sim_pred
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), 1e-10)

    if obs_overlay is not None:
        obs_mask, obs_classes = obs_overlay
        pred[obs_mask] = 0.0
        pred[obs_mask, obs_classes[obs_mask]] = 1.0

    if sett_survival is not None:
        surv_mask, surv_probs = sett_survival
        pred[surv_mask, 1] = surv_probs[surv_mask]
        rem_prob = 1.0 - pred[surv_mask, 1]
        rem_sum = pred[surv_mask].sum(axis=-1) - pred[surv_mask, 1]
        valid_rem = rem_sum > 1e-6
        for c in range(NUM_CLASSES):
            if c != 1:
                pred[surv_mask & valid_rem, c] = pred[surv_mask &
    return probs