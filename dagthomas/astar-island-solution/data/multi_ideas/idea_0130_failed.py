# FAILED: Compilation failed: '(' was never closed (<string>, line 30)
# Direction: Increase the `settlement` class clamp range upper bound from 2.5 to 2.8 to better accommodate high-p

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
    """Production prediction with auto-loaded best params.

    Args:
        obs_expansion_radius: Maximum distance from initial settlements where
            settlements were observed during exploration. If provided, suppresses
            settlement predictions beyond this radius.
        est_vigor: Estimated settlement vigor from observations (settlement % on
            dynamic cells). If provided, uses regime-conditional calibration.
    """
    def _load_params():
        return {"prior_w": 1.0, "emp_max": 3.0}

    p = _load_params()
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    idx_grid, unique_keys = _build_feature_key_index(
    return probs