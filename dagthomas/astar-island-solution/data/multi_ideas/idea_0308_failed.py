# FAILED: Crashed on round2 seed 0: name 'probs' is not defined
# Direction: Adjust the `settlement` class `clamp_range` from `[0.15, 2.5]` to `[0.12, 2.7]` to allow for more nu

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
    try:
        p = _load_params()
    except NameError:
        p = getattr(predict, '_load_params', lambda: {"prior_w": 1.0, "emp_max": 3.0})()
        
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    fkeys = build_feature_keys(grid, settlements)
    return probs