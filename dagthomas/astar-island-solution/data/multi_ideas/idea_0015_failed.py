# FAILED: Compilation failed: invalid syntax (<string>, line 23)
# Direction: Increase the settlement clamp upper bound from 2.5 to 2.8 to reduce the KL divergence error in high-

def experimental_pred_fn(state: dict, global_mult,
                       fk_buckets,
                       multi_store=None,
                       variance_regime: str = None,
                       obs_expansion_radius: int = None):
    grid = np.array(state['grid'])
    settlements = state['settlements']

    p = {"prior_w": 2.0, "emp_max": 3.0, "base_power": 0.5}

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

    if
    return probs