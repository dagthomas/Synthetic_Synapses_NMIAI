# FAILED: Compilation failed: unterminated string literal (detected at line 21) (<string>, line 21)
# Direction: Increase the settlement class upper clamp from 2.5 to 3.0 to allow the Bayesian prior more weight in

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    grid = np.array(state['grid'])
    settlements = state['settlements']

    p = {"prior_w": 1.0, "emp_max": 2.0, "base_power": 1.0}
    if hasattr(predict, '_load_params'):
        p = predict._load_params()

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.0,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_max': 1.0,
        'cal_global_
    return probs