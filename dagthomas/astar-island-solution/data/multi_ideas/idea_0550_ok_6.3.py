# Score: avg=6.32
# Direction: Increase the `settlement` class `clamp_max` upper bound from 2.5 to 2.85. This allows the Bayesian p
# Error: None

def experimental_pred_fn(state: dict, global_mult: GlobalMultipliers,
                       fk_buckets: FeatureKeyBuckets,
                       multi_store=None,
                       variance_regime: str = None,
                       obs_expansion_radius: int = None,
                       est_vigor: float = None,
                       sim_pred: np.ndarray = None,
                       sim_alpha: float = 0.25):
    grid = np.array(state['grid'])
    probs = np.zeros((40, 40, 6))

    def build_feature_keys(r, c, g):
        # Local neighborhood context as a feature key tuple
        res = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 40 and 0 <= nc < 40:
                    res.append(g[nr, nc])
                else:
                    res.append(-1)
        return tuple(res)

    for r in range(40):
        for c in range(40):
            # 1. Base probabilities
            if sim_pred is not None:
                p = sim_pred[r, c].copy()
            else:
                p = np.full(6, 1.0/6.0)

            # 2. Feature-based multipliers
            fk = build_feature_keys(r, c, grid)
            try:
                mults = fk_buckets.get(fk)
                if mults is not None:
                    p = p * mults
            except:
                pass

            # 3. Constraint: mountain (class 5) = 0 on non-mountain cells
            if grid[r, c] != 5:
                p[5] = 0.0

            # 4. Constraint: port (class 2) = 0 on non-coastal cells
            is_coastal = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 40 and 0 <= nc < 40:
                    if grid[nr, nc] == 0:
                        is_coastal = True
                        break
            if not is_coastal:
                p[2] = 0.0

            # 5. Apply floor >= 0.005 for nonzero classes
            p[p > 0] = np.maximum(p[p > 0], 0.005)

            # 6. Normalize to sum to 1.0
            total = np.sum(p)
            if total > 0:
                p = p / total
            else:
                # Absolute fallback
                p = np.zeros(6)
                p[0] = 1.0
            
            probs[r, c] = p

    return probs