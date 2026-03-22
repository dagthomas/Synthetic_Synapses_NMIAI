# FAILED: Compilation failed: unterminated string literal (detected at line 30) (<string>, line 30)
# Direction: Increase the `settlement` class `clamp_max` parameter from 2.5 to 2.85 to allow high-confidence feat

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
        try:
            p = predict._load_params()
        except AttributeError:
            p = {"prior_w": 2.0, "emp_max": 3.0, "clamp_min": 0.5, "clamp_max": 2.5}

    grid = np.array(state['grid'])
    settlements = state['
    return probs