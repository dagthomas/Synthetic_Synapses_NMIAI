# FAILED: Crashed on round2 seed 0: experimental_pred_fn() takes 1 positional argument but 3 were given
# Direction: Change the FK blend normalization formula from `(pw + ew)` to `max(pw, ew)` to prevent prior-driven 

def experimental_pred_fn(grid):
    # Distance-aware multiplier
    is_sett = (grid == 1) | (grid == 2)
    if is_sett.any():
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(1 - is_sett)
        return dist_map
    return grid
    return probs