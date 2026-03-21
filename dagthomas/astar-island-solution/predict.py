"""Prediction engine for Astar Island.

Tiered approach:
  Tier 1: Calibrated prior (CalibrationModel > GTM > R1 fallback)
  Tier 2: Per-cell Bayesian update with adaptive prior strength + floor
  Global multipliers adjust priors based on observed/expected ratios.
"""
import numpy as np

import math

from calibration import CalibrationModel, build_feature_keys
from config import MAP_H, MAP_W, NUM_CLASSES, PROB_FLOOR, TERRAIN_TO_CLASS
from utils import FeatureKeyBuckets, GlobalMultipliers, GlobalTransitionMatrix, ObservationAccumulator, apply_floor

from pathlib import Path

# Module-level calibration model (loaded once)
_calibration_model: CalibrationModel | None = None
_round_vigors: dict | None = None

DATA_DIR = Path(__file__).parent / "data" / "calibration"
REGIME_SIGMA = 0.04


def get_calibration() -> CalibrationModel:
    """Get or lazily load the calibration model."""
    global _calibration_model
    if _calibration_model is None:
        print("Loading calibration model...")
        _calibration_model = CalibrationModel.from_all_rounds()
    return _calibration_model


def _load_round_vigors() -> dict:
    """Load vigor for all calibration rounds (cached)."""
    global _round_vigors
    if _round_vigors is None:
        _round_vigors = {}
        if DATA_DIR.exists():
            for rd in sorted(DATA_DIR.iterdir()):
                if rd.is_dir() and (rd / "round_detail.json").exists():
                    _round_vigors[rd.name] = CalibrationModel.compute_round_vigor(rd)
    return _round_vigors


def get_regime_calibration(est_vigor: float) -> CalibrationModel:
    """Build a regime-conditional calibration model weighted by vigor similarity.

    Rounds with vigor close to est_vigor get higher weight.
    This produces a prior tuned for the current round's regime.
    """
    vigors = _load_round_vigors()
    cal = CalibrationModel()
    if not DATA_DIR.exists():
        return cal
    for rd in sorted(DATA_DIR.iterdir()):
        if rd.is_dir() and (rd / "round_detail.json").exists():
            v = vigors.get(rd.name, 0.1)
            w = math.exp(-((v - est_vigor) ** 2) / (2 * REGIME_SIGMA ** 2))
            w = max(w, 0.05)
            cal.add_round(rd, weight=w)
    return cal


# --- R1+R2 averaged fallback priors (used when GTM has no data) ---
PLAINS_BY_DIST = {
    0:  [0.360, 0.428, 0.004, 0.033, 0.175, 0.000],
    1:  [0.695, 0.234, 0.011, 0.018, 0.044, 0.000],
    2:  [0.713, 0.205, 0.016, 0.018, 0.049, 0.000],
    3:  [0.745, 0.183, 0.015, 0.017, 0.041, 0.000],
    4:  [0.823, 0.129, 0.012, 0.013, 0.025, 0.000],
    5:  [0.856, 0.106, 0.010, 0.011, 0.019, 0.000],
    6:  [0.914, 0.068, 0.006, 0.005, 0.009, 0.000],
    7:  [0.922, 0.061, 0.005, 0.004, 0.008, 0.000],
    8:  [0.960, 0.027, 0.004, 0.003, 0.005, 0.000],
    9:  [0.985, 0.010, 0.002, 0.001, 0.002, 0.000],
}
COASTAL_PLAINS_BY_DIST = {
    1:  [0.695, 0.080, 0.150, 0.018, 0.044, 0.000],
    2:  [0.713, 0.080, 0.130, 0.018, 0.049, 0.000],
    3:  [0.745, 0.070, 0.115, 0.017, 0.041, 0.000],
    4:  [0.823, 0.050, 0.080, 0.013, 0.025, 0.000],
    5:  [0.856, 0.040, 0.065, 0.011, 0.019, 0.000],
}
FOREST_BY_DIST = {
    1:  [0.098, 0.244, 0.015, 0.017, 0.629, 0.000],
    2:  [0.102, 0.207, 0.016, 0.016, 0.661, 0.000],
    3:  [0.092, 0.185, 0.015, 0.015, 0.696, 0.000],
    4:  [0.051, 0.132, 0.013, 0.010, 0.794, 0.000],
    5:  [0.043, 0.109, 0.010, 0.008, 0.829, 0.000],
    6:  [0.018, 0.067, 0.005, 0.005, 0.906, 0.000],
    7:  [0.016, 0.059, 0.003, 0.005, 0.917, 0.000],
    8:  [0.005, 0.018, 0.003, 0.002, 0.972, 0.000],
    9:  [0.005, 0.018, 0.001, 0.001, 0.975, 0.000],
}
SETTLEMENT_BY_CLUSTER = {
    1:  [0.364, 0.388, 0.048, 0.032, 0.168, 0.000],
    2:  [0.341, 0.446, 0.013, 0.031, 0.170, 0.000],
    3:  [0.376, 0.394, 0.018, 0.030, 0.183, 0.000],
    4:  [0.409, 0.359, 0.000, 0.031, 0.201, 0.000],
    5:  [0.431, 0.316, 0.000, 0.028, 0.225, 0.000],
}
SETTLEMENT_DEFAULT = [0.361, 0.428, 0.004, 0.033, 0.175, 0.000]
COASTAL_SETTLEMENT = [0.342, 0.125, 0.346, 0.022, 0.165, 0.000]
PORT_DEFAULT = [0.364, 0.120, 0.319, 0.021, 0.176, 0.000]

# Hard minimum: 0.01 per class (scoring requirement — 0.015 wastes mass on mountain/port/ruin)
FLOOR_MIN = 0.01

DEFAULT_PRIOR_STRENGTH = 0.8


def _lookup_with_interp(table: dict, dist: float, max_key: int = 9) -> np.ndarray:
    d_lo = int(dist)
    d_hi = d_lo + 1
    if d_lo >= max_key:
        return np.array(table[max_key])
    if d_lo < min(table.keys()):
        return np.array(table[min(table.keys())])
    if d_lo in table and d_hi in table:
        t = dist - d_lo
        return (1 - t) * np.array(table[d_lo]) + t * np.array(table[d_hi])
    elif d_lo in table:
        return np.array(table[d_lo])
    else:
        keys = sorted(table.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= d_lo <= keys[i + 1]:
                t = (dist - keys[i]) / (keys[i + 1] - keys[i])
                return (1 - t) * np.array(table[keys[i]]) + t * np.array(table[keys[i + 1]])
        return np.array(table[keys[-1]])


def _get_r1_prior(code: int, d: float, is_coastal: bool,
                  sett_r5: int) -> np.ndarray:
    if code == 4:
        return _lookup_with_interp(FOREST_BY_DIST, d)
    elif code == 1:
        if is_coastal:
            return np.array(COASTAL_SETTLEMENT)
        elif sett_r5 in SETTLEMENT_BY_CLUSTER:
            return np.array(SETTLEMENT_BY_CLUSTER[sett_r5])
        else:
            return np.array(SETTLEMENT_DEFAULT)
    elif code == 2:
        return np.array(PORT_DEFAULT)
    elif code == 3:
        return np.array(SETTLEMENT_DEFAULT)
    else:
        if is_coastal and d <= 5:
            return _lookup_with_interp(COASTAL_PLAINS_BY_DIST, d, max_key=5)
        return _lookup_with_interp(PLAINS_BY_DIST, d)


def _precompute_cell_features(grid: list[list[int]], settlements: list[dict]):
    """Precompute all per-cell features needed for prediction."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    raw = np.array(grid)
    ocean_mask = raw == 10

    # Distance to settlement (Manhattan, matching reference solution)
    sett_positions = [(s["y"], s["x"]) for s in settlements if s.get("alive", True)]
    dist_sett = np.full((h, w), 999.0)
    for y in range(h):
        for x in range(w):
            for sy, sx in sett_positions:
                d = abs(y - sy) + abs(x - sx)
                if d < dist_sett[y, x]:
                    dist_sett[y, x] = d

    # Coastal
    coastal = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if ocean_mask[y, x]:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and ocean_mask[ny, nx]:
                    coastal[y, x] = True
                    break

    # Forest neighbors count (0-3+, capped at 3)
    forest_neighbors = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            count = 0
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and raw[ny, nx] == 4:
                    count += 1
            forest_neighbors[y, x] = min(count, 3)

    # Cluster density
    sett_r5 = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            for sy, sx in sett_positions:
                if abs(y - sy) + abs(x - sx) <= 5:
                    sett_r5[y, x] += 1

    # Distance bucket (matching reference: 0, 1, 2, 3, 4+)
    dist_bucket = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            d = int(dist_sett[y, x])
            if d <= 0:
                dist_bucket[y, x] = 0
            elif d == 1:
                dist_bucket[y, x] = 1
            elif d == 2:
                dist_bucket[y, x] = 2
            elif d == 3:
                dist_bucket[y, x] = 3
            else:
                dist_bucket[y, x] = 4

    # Cell dynamism
    dynamism = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            code = raw[y, x]
            d = dist_sett[y, x]
            if code == 10 or code == 5:
                dynamism[y, x] = 0.0
            elif code in (1, 2, 3):
                dynamism[y, x] = 1.0
            elif d <= 2:
                dynamism[y, x] = 0.9
            elif d <= 5:
                dynamism[y, x] = 0.6
            elif d <= 8:
                dynamism[y, x] = 0.3
            else:
                dynamism[y, x] = 0.05

    return {
        "raw": raw, "dist_sett": dist_sett, "coastal": coastal,
        "forest_neighbors": forest_neighbors, "sett_r5": sett_r5,
        "dist_bucket": dist_bucket, "dynamism": dynamism,
    }


def _enforce_floor(p: np.ndarray, floor: float = FLOOR_MIN) -> np.ndarray:
    """Enforce minimum probability per class while maintaining sum=1.

    Sets any class below floor to floor, then redistributes the
    excess from classes above floor proportionally.
    """
    p = p.copy()
    # Normalize first if needed
    s = p.sum()
    if s > 0 and abs(s - 1.0) > 1e-8:
        p = p / s

    below = p < floor
    if not below.any():
        return p

    # Mass we need to add to bring below-floor up
    deficit = (floor - p[below]).sum()
    p[below] = floor

    # Take from above-floor classes proportionally
    above = ~below
    above_sum = p[above].sum()
    if above_sum > deficit:
        p[above] -= deficit * (p[above] / above_sum)
    else:
        # Edge case: everything is near floor, just set uniform
        p = np.full_like(p, 1.0 / len(p))

    # Ensure exact sum=1
    p = p / p.sum()

    # Final safety: if any class still below floor due to float precision
    if p.min() < floor - 1e-12:
        return _enforce_floor(p, floor)  # recurse once
    return p


def _apply_smart_floor(prediction: np.ndarray, raw: np.ndarray,
                       dynamism: np.ndarray, collapse_detected: bool = False,
                       coastal: np.ndarray = None) -> np.ndarray:
    """Apply class-specific floors based on what's actually possible.

    Key insight from GT analysis (N=200 simulations):
    - Mountain (class 5) is ALWAYS 0 on non-mountain dynamic cells
    - Port (class 2) is ALWAYS 0 on non-coastal cells
    Setting these to 0 saves 1-2% probability mass per cell for correct classes.
    Remaining nonzero classes get floor=0.005 (= 1/200, the minimum GT value).
    """
    h, w = raw.shape
    for y in range(h):
        for x in range(w):
            code = raw[y, x]
            if code == 10 or code == 5:
                continue  # handled separately

            p = prediction[y, x].copy()

            # Mountain never appears on non-mountain dynamic cells
            p[5] = 0.0

            # Port never appears on non-coastal cells
            if coastal is not None and not coastal[y, x]:
                p[2] = 0.0

            # Floor remaining nonzero classes at 0.005 (1/200 = min GT granularity)
            nonzero = p > 0
            if nonzero.any():
                p[nonzero] = np.maximum(p[nonzero], 0.005)
                p = p / p.sum()

            prediction[y, x] = p
    return prediction


def get_static_prior(grid: list[list[int]], settlements: list[dict],
                     adjustments: dict = None,
                     global_transitions: GlobalTransitionMatrix = None,
                     collapse_detected: bool = False,
                     global_multipliers: GlobalMultipliers = None,
                     feature_key_buckets: FeatureKeyBuckets = None) -> np.ndarray:
    """Build prior: CalibrationModel → feature-key empirical → multipliers → floor."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    prediction = np.zeros((h, w, NUM_CLASSES))

    feat = _precompute_cell_features(grid, settlements)
    raw = feat["raw"]
    dist_sett = feat["dist_sett"]
    coastal = feat["coastal"]
    sett_r5 = feat["sett_r5"]
    dynamism = feat["dynamism"]
    gtm = global_transitions

    # Load calibration model
    cal = get_calibration()
    use_calibration = cal.rounds_loaded > 0

    # Build feature keys for calibration lookup
    feature_keys = None
    if use_calibration:
        terrain_np = np.array(grid, dtype=int)
        feature_keys = build_feature_keys(terrain_np, settlements)

    # Get global multipliers if available
    mult = None
    if global_multipliers is not None:
        mult = global_multipliers.get_multipliers()

    for y in range(h):
        for x in range(w):
            code = grid[y][x]
            d = dist_sett[y, x]
            is_coastal = coastal[y, x]
            initial_cls = TERRAIN_TO_CLASS.get(code, 0)

            if code == 10 or code == 5:
                prediction[y, x] = [1, 0, 0, 0, 0, 0] if code == 10 else [0, 0, 0, 0, 0, 1]
                continue

            # Priority: CalibrationModel > GTM > R1 fallback
            if use_calibration:
                prior = cal.prior_for(feature_keys[y][x])
            else:
                # Try GTM
                global_prior = None
                if gtm is not None:
                    global_prior = gtm.get_transition_prob(initial_cls, d, coastal=is_coastal)

                if global_prior is not None:
                    bucket = gtm._bucket(d)
                    n_gtm = gtm.totals.get((initial_cls, bucket), 0)
                    gtm_weight = n_gtm / (n_gtm + 5)
                    r1_prior = _get_r1_prior(code, d, is_coastal, sett_r5[y, x])
                    prior = gtm_weight * global_prior + (1 - gtm_weight) * r1_prior
                else:
                    prior = _get_r1_prior(code, d, is_coastal, sett_r5[y, x])

            # Blend with feature-key empirical from current-round observations
            if feature_key_buckets is not None and feature_keys is not None:
                fk = feature_keys[y][x]
                empirical, count = feature_key_buckets.get_empirical(fk)
                if empirical is not None and count >= 5:
                    strength = min(8.0, math.sqrt(count))
                    prior = _enforce_floor(prior * 5.0 + empirical * strength)

            # Apply global multipliers (regime adaptation)
            if mult is not None:
                prior = prior * mult
                s = prior.sum()
                if s > 0:
                    prior = prior / s

            # Collapse adjustment: shift settlement mass to empty/forest
            if collapse_detected:
                sett_mass = prior[1] + prior[2] + prior[3]
                if sett_mass > 0.05:
                    keep = FLOOR_MIN
                    redistribute = sett_mass - keep * 3
                    if redistribute > 0:
                        ef_total = prior[0] + prior[4]
                        if ef_total > 0:
                            prior[0] += redistribute * (prior[0] / ef_total)
                            prior[4] += redistribute * (prior[4] / ef_total)
                        else:
                            prior[0] += redistribute
                        prior[1] = keep
                        prior[2] = keep
                        prior[3] = keep

            prediction[y, x] = prior

    # Smart per-cell, per-class floor
    prediction = _apply_smart_floor(prediction, raw, dynamism, collapse_detected, coastal=coastal)

    # Lock truly static cells (mountain/ocean never change)
    # Mountain: 100% mountain, 0% everything else
    prediction[raw == 5] = [0, 0, 0, 0, 0, 1]
    # Ocean: 100% empty, 0% everything else
    prediction[raw == 10] = [1, 0, 0, 0, 0, 0]

    return prediction


def get_observation_informed_prediction(
    grid: list[list[int]],
    settlements: list[dict],
    accumulator: ObservationAccumulator,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    adjustments: dict = None,
    global_transitions: GlobalTransitionMatrix = None,
    observed_settlement_stats: dict = None,
    collapse_detected: bool = False,
    global_multipliers: GlobalMultipliers = None,
) -> np.ndarray:
    """Per-cell Bayesian update with adaptive prior strength and floor.

    Dynamic cells (near settlements) get lower prior strength = trust observations more.
    Static cells skip Bayesian update entirely.
    """
    static = get_static_prior(grid, settlements, adjustments=adjustments,
                              global_transitions=global_transitions,
                              collapse_detected=collapse_detected,
                              global_multipliers=global_multipliers)
    n_obs = accumulator.get_observation_count()
    counts = accumulator.counts
    h, w = n_obs.shape

    feat = _precompute_cell_features(grid, settlements)
    raw = feat["raw"]
    dynamism = feat["dynamism"]
    dist_sett = feat["dist_sett"]

    prediction = np.zeros((h, w, NUM_CLASSES))

    for y in range(h):
        for x in range(w):
            code = raw[y, x]

            # Static cells: skip Bayesian update entirely
            if code == 10 or code == 5:
                prediction[y, x] = static[y, x]
                continue

            dyn = dynamism[y, x]
            n = n_obs[y, x]

            if n == 0:
                # Unobserved: use prior (+ spatial smoothing below)
                prediction[y, x] = static[y, x]
            else:
                # Per-cell prior strength: dynamic cells trust observations more
                # dyn=1.0 (settlement): strength = prior_strength * 0.3
                # dyn=0.05 (far): strength = prior_strength * 1.5
                cell_strength = prior_strength * (1.5 - 1.2 * dyn)
                # With collapse, trust observations even more
                if collapse_detected:
                    cell_strength *= 0.5
                alpha = cell_strength * static[y, x]
                posterior = alpha + counts[y, x]
                total = cell_strength + n
                prediction[y, x] = posterior / total

    # Settlement stats adjustment: if we observed a specific settlement's stats,
    # adjust its prediction based on population/food
    if observed_settlement_stats:
        all_dead = all(not s.get("alive", True) for s in observed_settlement_stats.values())
        for (sy, sx), stats in observed_settlement_stats.items():
            if 0 <= sy < h and 0 <= sx < w:
                pop = stats.get("population", 1.0)
                food = stats.get("food", 0.5)
                alive = stats.get("alive", True)

                if not alive or collapse_detected:
                    # Dead settlement: aggressively reduce settlement/port, boost empty/forest
                    prediction[sy, sx, 1] *= 0.1  # strongly reduce settlement
                    prediction[sy, sx, 2] *= 0.1  # strongly reduce port
                    prediction[sy, sx, 3] *= 0.5  # reduce ruin somewhat
                    prediction[sy, sx, 0] *= 2.0  # boost empty
                    prediction[sy, sx, 4] *= 2.0  # boost forest
                elif pop > 2.0 and food > 0.5:
                    # Thriving: boost settlement survival
                    prediction[sy, sx, 1] *= 1.3
                    prediction[sy, sx, 0] *= 0.8
                elif food < 0.2:
                    # Starving: boost collapse probability
                    prediction[sy, sx, 1] *= 0.7
                    prediction[sy, sx, 0] *= 1.2
                    prediction[sy, sx, 3] *= 1.3  # ruin

                # Renormalize this cell
                prediction[sy, sx] = prediction[sy, sx] / prediction[sy, sx].sum()

    # Spatial smoothing for unobserved cells only
    if (n_obs == 0).any():
        smoothed = prediction.copy()
        for y in range(h):
            for x in range(w):
                if n_obs[y, x] == 0 and dynamism[y, x] > 0.01:
                    neighbor_sum = np.zeros(NUM_CLASSES)
                    weight_sum = 0.0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and n_obs[ny, nx] > 0:
                                dist = max(abs(dy), abs(dx))
                                w_n = n_obs[ny, nx] / dist
                                neighbor_sum += prediction[ny, nx] * w_n
                                weight_sum += w_n
                    if weight_sum > 0:
                        neighbor_avg = neighbor_sum / weight_sum
                        # Trust neighbors more with collapse (less prior noise)
                        blend = 0.75 if collapse_detected else 0.85
                        smoothed[y, x] = blend * prediction[y, x] + (1 - blend) * neighbor_avg
        prediction = smoothed

    # Smart per-cell, per-class floor
    prediction = _apply_smart_floor(prediction, raw, dynamism, collapse_detected)

    return prediction


def predict_for_seed(
    grid: list[list[int]],
    settlements: list[dict],
    accumulator: ObservationAccumulator | None = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    adjustments: dict = None,
    global_transitions: GlobalTransitionMatrix = None,
    observed_settlement_stats: dict = None,
    collapse_detected: bool = False,
    global_multipliers: GlobalMultipliers = None,
    feature_key_buckets: FeatureKeyBuckets = None,
) -> np.ndarray:
    """Generate prediction for a single seed."""
    gtm = global_transitions
    gtm_str = ""
    if gtm:
        stats = gtm.get_stats()
        gtm_str = f", GTM={stats['total_observations']} obs"
    collapse_str = ", COLLAPSE" if collapse_detected else ""
    mult_str = ""
    if global_multipliers is not None:
        m = global_multipliers.get_multipliers()
        mult_str = f", mult=[s={m[1]:.2f},p={m[2]:.2f},f={m[4]:.2f}]"

    # NOTE: Tier 2 cell-level Bayesian update is disabled — it overfits to
    # single stochastic samples (1-2 obs/cell) and makes predictions worse.
    # Instead, use CalibrationModel + global multipliers (static prior).
    # TODO: Implement feature-key bucketed posterior (like reference solution)
    # which pools observations across cells with same features.
    fk_str = ""
    if feature_key_buckets is not None:
        fk_stats = feature_key_buckets.get_stats()
        fk_str = f", fk={fk_stats['keys_with_data']} keys"
    print(f"  Calibrated prior + multipliers{gtm_str}{collapse_str}{mult_str}{fk_str}")
    return get_static_prior(grid, settlements, adjustments=adjustments,
                            global_transitions=gtm,
                            collapse_detected=collapse_detected,
                            global_multipliers=global_multipliers,
                            feature_key_buckets=feature_key_buckets)


def validate_prediction(prediction: np.ndarray, height: int = MAP_H, width: int = MAP_W) -> list[str]:
    errors = []
    if prediction.shape != (height, width, NUM_CLASSES):
        errors.append(f"Shape mismatch: expected ({height},{width},{NUM_CLASSES}), got {prediction.shape}")
        return errors
    if np.any(prediction < 0):
        errors.append(f"{(prediction < 0).sum()} negative probabilities found")
    row_sums = prediction.sum(axis=-1)
    bad_sums = np.abs(row_sums - 1.0) > 0.01
    if bad_sums.any():
        errors.append(f"{bad_sums.sum()} cells don't sum to 1.0 (worst: {np.abs(row_sums - 1.0).max():.4f})")
    return errors
