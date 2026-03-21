# FAILED: Compilation failed: '{' was never closed (<string>, line 33)
# Direction: Change the FK blend normalization formula from (pw + ew) to max(pw, ew) to prevent low-count empiric

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                   fk_buckets: FeatureKeyBuckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
    """
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # Default parameters if _load_params is not available
    p = {
        "prior_w": 1.0,
        "emp_max": 5.0,
        "base_power": 0.5
    }

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    try:
        cal = predict.get_calibration()
    except Exception:
        cal = None
        
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
    return probs