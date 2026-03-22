# Score: avg=84.65
# Direction: Increase the `settlement` class `clamp_max` from 2.5 to 2.8 to allow for sharper probability peaks i
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
    p = {"prior_w": 1.5, "emp_max": 3.0}
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

    clamp_max = np.array([1.5, 2.0, 2.0, 2.8, 2.0, 1.5])
    ratio_clamped = np.clip(ratio, 0.5, clamp_max)
    pred *= ratio_clamped

    if sim_pred is not None:
        pred = pred * (1.0 - sim_alpha) + sim_pred * sim_alpha

    if obs_expansion_radius is not None and len(settlements) > 0:
        mask = np.ones((MAP_H, MAP_W), dtype=bool)
        for sr, sc in settlements:
            mask[sr, sc] = False
        dist = distance_transform_cdt(mask, metric='taxicab')
        far = dist > obs_expansion_radius
        pred[far, 3] *= 0.1

    coastal = _build_coastal_mask(grid)
    for r in range(MAP_H):
        for c in range(MAP_W):
            terrain = grid[r, c]
            terr_cls = terrain_to_class(terrain)
            if terr_cls == 5:
                pred[r, c, :] = 0.0
                pred[r, c, 5] = 1.0
            else:
                pred[r, c, 5] = 0.0
                if not coastal[r, c]:
                    pred[r, c, 2] = 0.0

    pred = np.maximum(pred, 0.0)
    non_zero = pred > 0
    pred[non_zero] = np.maximum(pred[non_zero], 0.005)
    row_sums = pred.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = pred / row_sums

    return probs