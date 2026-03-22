# FAILED: Compilation failed: unterminated string literal (detected at line 26) (<string>, line 26)
# Direction: Increase the `settlement` class `coarse_max` calibration weight from 2.0 to 2.25. This provides a co

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
    p = {"prior_w": 2.0, "emp_max": 3.0, "smooth_w": 0.1}
    grid = np.array(state['grid'])
    settlements = state['settlements']

    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    if est_vigor is not None:
        cal = predict.get_regime_calibration(est_vigor)
    else:
        cal = predict.get_calibration()
        
    cal_coarse_max = np.array([2.0, 2.0, 2.0, 2.25, 2.0, 2.0])
    cal_params = {
        '
    return probs