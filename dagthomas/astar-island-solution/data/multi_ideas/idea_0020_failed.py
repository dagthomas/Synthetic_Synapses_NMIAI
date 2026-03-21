# FAILED: Crashed on round2 seed 0: name 'f' is not defined
# Direction: Increase the `settlement` class calibration `coarse_max` weight from 2.0 to 2.5. This allows the sys

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
    p = getattr(predict, '_load_params', lambda: {"prior_w": 2.0, "emp_max": 5.0})()
    if not isinstance(p, dict):
        p = {"prior_w": 2.0, "emp_max": 5.0}
        
    grid = np.array(state['grid'])
    settlements = state['settlements']

    # 1. Feature Keys
    f
    return probs