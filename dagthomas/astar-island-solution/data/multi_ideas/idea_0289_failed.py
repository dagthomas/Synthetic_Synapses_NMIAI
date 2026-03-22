# FAILED: Crashed on round2 seed 0: experimental_pred_fn() missing 1 required positional argument: 'NUM_CLASSES'
# Direction: Change the settlement class FK (feature-key) blend normalization from `(pw + ew)` to `max(pw, ew)` t

def experimental_pred_fn(blended, pw, ew, NUM_CLASSES):
    blended_probs = np.zeros_like(blended)
    norm_all = pw + ew
    norm_sett = np.maximum(pw, ew)
    
    for c in range(NUM_CLASSES):
        if c == 1: # settlement class
            blended_probs[:, :, c] = blended[:, :, c] / np.maximum(norm_sett[:, :, 0], 1e-10)
        else:
            blended_probs[:, :, c] = blended[:, :, c] / np.maximum(norm_all[:, :, 0], 1e-10)
    return blended_probs
    return probs