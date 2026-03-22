# FAILED: Crashed on round2 seed 0: experimental_pred_fn() missing 2 required positional arguments: 'ratio' and 'grid'
# Direction: Change the FK blend normalization from (pw+ew) to max(pw, ew) to prevent the empirical signal from b

def experimental_pred_fn(pred, blended, has_fk, ratio, grid):
    import numpy as np
    # Select between blended and pred based on has_fk
    out = np.where(has_fk[..., np.newaxis], blended, pred)
    # Apply ratio
    out = out * ratio
    # Normalize per cell (axis=-1) to 1.0
    out = out / out.sum(axis=-1, keepdims=True)
    return out
    return probs