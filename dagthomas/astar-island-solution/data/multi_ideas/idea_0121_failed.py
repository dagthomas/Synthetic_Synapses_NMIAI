# FAILED: Compilation failed: unterminated string literal (detected at line 28) (<string>, line 28)
# Direction: Increase the settlement class upper clamp from 2.5 to 3.2 to allow for higher-confidence predictions

def experimental_pred_fn(state: dict, global_mult,
                   fk_buckets,
                   multi_store=None,
                   variance_regime: str = None,
                   obs_expansion_radius: int = None) -> np.ndarray:
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
    """
    def _load_params():
        return {"prior_w": 2.0, "emp_max": 3.0, "base_power": 0.5}

    p = _load_params()
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # 2. Calibration
    cal = predict.get_calibration()
    cal_params = {
        'cal_fine_base': 1.0, 'cal_fine_divisor': 100.0, 'cal_fine_max': 5.0,
        'cal_coarse_base': 0.5, 'cal_coarse_divisor': 100.0, 'cal
    return probs