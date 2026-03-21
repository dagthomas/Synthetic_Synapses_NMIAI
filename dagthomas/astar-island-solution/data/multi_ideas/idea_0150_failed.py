# FAILED: Compilation failed: unterminated string literal (detected at line 25) (<string>, line 25)
# Direction: Increase the global multiplier for the `port` class from 1.0 to 1.08 to address coastal underpredict

def experimental_pred_fn(state: dict, global_mult,
                   fk_buckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    """Production prediction with auto-loaded best params."""
    p = {"prior_w": 0.5, "emp_max": 2.0, "base_power": 0.5}
    
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    try:
        cal = predict.get_calibration()
    except Exception:
        cal = {}
        
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal_coarse_max': 2.0,
        'cal_base_base': 0.1, 'cal_base_divisor': 100.0, 'cal_base_
    return probs