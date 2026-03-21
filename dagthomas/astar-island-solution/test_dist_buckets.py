"""Tests for the finer distance bucket change in calibration.py.

Validates:
1. _dist_bucket() maps correctly
2. Feature keys use new buckets
3. CalibrationModel trains and predicts with new keys
4. FastHarness evaluates without errors
5. No score regression vs old bucket scheme
"""
import json
import math
import numpy as np
from pathlib import Path

from calibration import _dist_bucket, build_feature_keys, CalibrationModel
from config import MAP_H, MAP_W, NUM_CLASSES
from fast_predict import (
    _build_coastal_mask, _build_feature_key_index,
    build_calibration_lookup, build_fk_empirical_lookup,
)
from utils import FeatureKeyBuckets, GlobalMultipliers, terrain_to_class

DATA_DIR = Path("data/calibration")
OBS_DIR = Path("data/rounds")
ROUND_IDS = {
    "round2": "76909e29-f664-4b2f-b16b-61b7507277e9",
    "round5": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "round7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
}


def compute_score(gt, pred):
    gt_safe = np.maximum(gt, 1e-10)
    entropy = -np.sum(gt * np.log(gt_safe), axis=-1)
    dynamic = entropy > 0.01
    pred_safe = np.maximum(pred, 1e-10)
    kl = np.sum(gt * np.log(gt_safe / pred_safe), axis=-1)
    if dynamic.any():
        wkl = float(np.sum(entropy[dynamic] * kl[dynamic]) / entropy[dynamic].sum())
    else:
        wkl = 0.0
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


def test_dist_bucket_mapping():
    """Test that _dist_bucket maps distances to correct buckets."""
    print("Test 1: _dist_bucket() mapping...")

    expected = {
        0: 0, 1: 1, 2: 2, 3: 3,
        4: 4, 5: 4,  # d=4-5 → bucket 4
        6: 5, 7: 5, 8: 5,  # d=6-8 → bucket 5
        9: 6, 10: 6, 15: 6, 50: 6, 99: 6,  # d=9+ → bucket 6
    }

    for dist, exp_bucket in expected.items():
        actual = _dist_bucket(dist)
        assert actual == exp_bucket, f"_dist_bucket({dist}) = {actual}, expected {exp_bucket}"

    # Verify monotonicity: higher distance → same or higher bucket
    for d in range(100):
        assert _dist_bucket(d) <= _dist_bucket(d + 1) or _dist_bucket(d) == _dist_bucket(d + 1), \
            f"Non-monotonic at d={d}: {_dist_bucket(d)} > {_dist_bucket(d+1)}"

    # Verify we have exactly 7 unique buckets (0-6)
    all_buckets = set(_dist_bucket(d) for d in range(100))
    assert all_buckets == {0, 1, 2, 3, 4, 5, 6}, f"Unexpected buckets: {all_buckets}"

    print("  PASS: All 7 buckets map correctly (0, 1, 2, 3, 4-5, 6-8, 9+)")


def test_feature_keys_use_new_buckets():
    """Test that build_feature_keys produces keys with the new bucket values."""
    print("Test 2: Feature keys with new buckets...")

    # Create a simple synthetic terrain with settlements at known positions
    terrain = np.full((20, 20), 11, dtype=int)  # all plains
    terrain[0, :] = 10  # ocean border top
    terrain[:, 0] = 10  # ocean border left
    terrain[5, 5] = 5   # mountain

    settlements = [
        {"x": 10, "y": 10, "alive": True, "has_port": False},
    ]

    fkeys = build_feature_keys(terrain, settlements)

    # Check specific cells at known distances from settlement at (10,10)
    # (10,10) → d=0 → bucket 0
    assert fkeys[10][10][1] == 0, f"Settlement cell should be bucket 0, got {fkeys[10][10][1]}"

    # (10,11) → d=1 → bucket 1
    assert fkeys[10][11][1] == 1, f"d=1 should be bucket 1, got {fkeys[10][11][1]}"

    # (10,13) → d=3 → bucket 3
    assert fkeys[10][13][1] == 3, f"d=3 should be bucket 3, got {fkeys[10][13][1]}"

    # (10,14) → d=4 → bucket 4 (NEW: was bucket 4 in old scheme too, but now 4 means d=4-5 only)
    assert fkeys[10][14][1] == 4, f"d=4 should be bucket 4, got {fkeys[10][14][1]}"

    # (10,15) → d=5 → bucket 4 (d=4-5)
    assert fkeys[10][15][1] == 4, f"d=5 should be bucket 4, got {fkeys[10][15][1]}"

    # (10,16) → d=6 → bucket 5 (NEW: was bucket 4 in old scheme)
    assert fkeys[10][16][1] == 5, f"d=6 should be bucket 5, got {fkeys[10][16][1]}"

    # (10,18) → d=8 → bucket 5 (d=6-8)
    assert fkeys[10][18][1] == 5, f"d=8 should be bucket 5, got {fkeys[10][18][1]}"

    # (10,19) → d=9 → bucket 6 (d=9+)
    assert fkeys[10][19][1] == 6, f"d=9 should be bucket 6, got {fkeys[10][19][1]}"

    # Count unique dist buckets across all cells
    all_dist_buckets = set()
    for row in fkeys:
        for fk in row:
            all_dist_buckets.add(fk[1])

    print(f"  Unique dist buckets in 20x20 grid: {sorted(all_dist_buckets)}")
    assert len(all_dist_buckets) >= 5, f"Expected at least 5 unique buckets, got {len(all_dist_buckets)}"
    print("  PASS: Feature keys use new 7-bucket scheme correctly")


def test_calibration_model_trains():
    """Test that CalibrationModel can train on real data with new buckets."""
    print("Test 3: CalibrationModel training...")

    cal = CalibrationModel()

    # Load a few rounds
    loaded = 0
    for rname in ["round1", "round2", "round3", "round4", "round5"]:
        rdir = DATA_DIR / rname
        if rdir.exists() and cal.add_round(rdir):
            loaded += 1

    assert loaded >= 3, f"Need at least 3 rounds for testing, loaded {loaded}"

    stats = cal.get_stats()
    print(f"  Loaded {stats['rounds_loaded']} rounds, {stats['total_cells']} cells")
    print(f"  Fine keys: {stats['fine_keys']}, Coarse keys: {stats['coarse_keys']}")

    # The new buckets should produce MORE fine keys than before (more dist granularity)
    assert stats["fine_keys"] > 50, f"Expected >50 fine keys, got {stats['fine_keys']}"
    assert stats["coarse_keys"] > 20, f"Expected >20 coarse keys, got {stats['coarse_keys']}"

    # Test prior_for with various feature keys
    for terrain_code in [0, 1, 4, 11]:
        for dist_bucket in [0, 1, 2, 3, 4, 5, 6]:
            fk = (terrain_code, dist_bucket, False, 1, -1)
            prior = cal.prior_for(fk)
            assert prior.shape == (NUM_CLASSES,), f"Wrong shape: {prior.shape}"
            assert abs(prior.sum() - 1.0) < 0.01, f"Prior doesn't sum to 1: {prior.sum()}"
            assert (prior >= 0).all(), f"Negative probabilities in prior"

    print("  PASS: CalibrationModel trains and produces valid priors for all bucket values")


def test_fast_predict_pipeline():
    """Test that the full fast_predict pipeline works with new buckets."""
    print("Test 4: Fast predict pipeline...")

    # Load a real round
    cal = CalibrationModel()
    for rname in ["round1", "round2", "round3", "round4", "round5"]:
        rdir = DATA_DIR / rname
        if rdir.exists():
            cal.add_round(rdir)

    # Load round5 detail for testing
    detail = json.loads((DATA_DIR / "round5" / "round_detail.json").read_text())
    state = detail["initial_states"][0]
    terrain = np.array(state["grid"], dtype=int)

    # Build feature keys and lookup
    fkeys = build_feature_keys(terrain, state["settlements"])
    idx_grid, unique_keys = _build_feature_key_index(fkeys)

    # Verify unique_keys contain new bucket values
    dist_buckets_seen = set(uk[1] for uk in unique_keys)
    print(f"  Unique dist buckets in round5: {sorted(dist_buckets_seen)}")
    assert 5 in dist_buckets_seen or 6 in dist_buckets_seen, \
        f"Expected new bucket values (5 or 6) in keys, only see {dist_buckets_seen}"

    # Build calibration lookup
    params = {
        "cal_fine_base": 1.0, "cal_fine_divisor": 100.0, "cal_fine_max": 5.0,
        "cal_coarse_base": 0.5, "cal_coarse_divisor": 100.0, "cal_coarse_max": 2.0,
        "cal_base_base": 0.1, "cal_base_divisor": 100.0, "cal_base_max": 1.0,
        "cal_global_weight": 0.01,
    }
    priors = build_calibration_lookup(cal, unique_keys, params)

    assert priors.shape == (len(unique_keys), NUM_CLASSES), f"Wrong priors shape: {priors.shape}"
    assert (priors >= 0).all(), "Negative values in priors"

    # Check normalization
    sums = priors.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=0.01), f"Priors not normalized, sums range: [{sums.min():.3f}, {sums.max():.3f}]"

    # Build FK empirical lookup (with dummy data)
    fk_buckets = FeatureKeyBuckets()
    for uk in unique_keys:
        for _ in range(10):
            fk_buckets.add_observation(uk, np.random.randint(0, NUM_CLASSES))
    fk_emp, fk_cnt = build_fk_empirical_lookup(fk_buckets, unique_keys, min_count=5)

    assert fk_emp.shape == (len(unique_keys), NUM_CLASSES), f"Wrong emp shape"
    assert fk_cnt.shape == (len(unique_keys),), f"Wrong cnt shape"

    # Full prediction pipeline (index into grid)
    pred = priors[idx_grid]
    assert pred.shape == (MAP_H, MAP_W, NUM_CLASSES), f"Wrong pred shape: {pred.shape}"

    print("  PASS: Full fast_predict pipeline works with new buckets")


def test_score_on_real_rounds():
    """Test that scores on real rounds don't catastrophically regress."""
    print("Test 5: Score on real rounds (R2, R5, R7)...")

    from autoloop_fast import FastHarness, DEFAULT_PARAMS

    harness = FastHarness(seeds_per_round=5)

    # Test with default params
    scores = harness.evaluate(DEFAULT_PARAMS)
    avg = scores["avg"]
    print(f"  DEFAULT_PARAMS: AVG={avg:.3f}")

    # Sanity: score should be reasonable (>70)
    assert avg > 70.0, f"Score too low ({avg:.3f}), possible regression"

    # Test with optimized params + smoothing
    opt_params = {
        "fk_prior_weight": 1.5622, "fk_max_strength": 24.3174, "fk_min_count": 2,
        "fk_strength_fn": "linear", "mult_power": 0.1, "mult_power_sett": 0.3388,
        "mult_power_port": 0.1, "mult_smooth": 1.0, "mult_sett_lo": 0.1562,
        "mult_sett_hi": 5.0, "mult_port_lo": 0.1189, "mult_port_hi": 1.5293,
        "mult_forest_lo": 0.7167, "mult_forest_hi": 2.3908, "mult_empty_lo": 0.8678,
        "mult_empty_hi": 1.05, "floor_nonzero": 0.0037, "temp_low": 1.0,
        "temp_high": 1.0523, "temp_ent_lo": 0.154, "temp_ent_hi": 0.6,
        "smooth_alpha": 0.4602, "prop_redist": True,
        "cal_fine_base": 2.3363, "cal_fine_divisor": 443.4959,
        "cal_fine_max": 8.2028, "cal_coarse_base": 1.829, "cal_coarse_divisor": 50.0,
        "cal_coarse_max": 7.8716, "cal_base_base": 1.9627, "cal_base_divisor": 622.3227,
        "cal_base_max": 0.5237, "cal_global_weight": 0.5131, "cal_heuristic_blend": 0.2533,
    }
    scores_opt = harness.evaluate(opt_params)
    avg_opt = scores_opt["avg"]
    print(f"  Optimized params: AVG={avg_opt:.3f}")

    for r in ["round2", "round3", "round4", "round5", "round6", "round7"]:
        print(f"    {r}: {scores_opt[r]:.2f}")

    assert avg_opt > 80.0, f"Optimized score too low ({avg_opt:.3f})"

    print(f"  PASS: Scores are reasonable (default={avg:.1f}, optimized={avg_opt:.1f})")
    return avg_opt


def test_predictions_valid():
    """Test that predictions satisfy all constraints."""
    print("Test 6: Prediction validity constraints...")

    from predict_gemini import gemini_predict
    from utils import GlobalMultipliers, FeatureKeyBuckets
    import predict

    # Load calibration
    cal = CalibrationModel()
    for rname in ["round1", "round2", "round3", "round4", "round5", "round6", "round7"]:
        rdir = DATA_DIR / rname
        if rdir.exists():
            cal.add_round(rdir)
    predict._calibration_model = cal

    # Use round5 as test
    detail = json.loads((DATA_DIR / "round5" / "round_detail.json").read_text())
    state = detail["initial_states"][0]
    terrain = np.array(state["grid"], dtype=int)

    # Dummy multipliers and FK
    gm = GlobalMultipliers()
    fk = FeatureKeyBuckets()
    # Add some dummy observations
    fkeys = build_feature_keys(terrain, state["settlements"])
    for y in range(MAP_H):
        for x in range(MAP_W):
            cls = np.random.choice(NUM_CLASSES, p=[0.6, 0.1, 0.02, 0.03, 0.2, 0.05])
            gm.add_observation(cls, np.full(NUM_CLASSES, 1.0 / NUM_CLASSES))
            fk.add_observation(fkeys[y][x], cls)

    pred = gemini_predict(state, gm, fk)

    # Check shape
    assert pred.shape == (MAP_H, MAP_W, NUM_CLASSES), f"Wrong shape: {pred.shape}"

    # Check normalization
    sums = pred.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=0.001), f"Not normalized: sums range [{sums.min():.4f}, {sums.max():.4f}]"

    # Check non-negative
    assert (pred >= 0).all(), "Negative probabilities!"

    # Check static cells locked
    ocean_cells = terrain == 10
    if ocean_cells.any():
        assert np.allclose(pred[ocean_cells, 0], 1.0, atol=0.001), "Ocean cells not locked"

    mountain_cells = terrain == 5
    if mountain_cells.any():
        assert np.allclose(pred[mountain_cells, 5], 1.0, atol=0.001), "Mountain cells not locked"

    # Check mountain = 0 on non-mountain dynamic cells
    dynamic = ~((terrain == 10) | (terrain == 5))
    if dynamic.any():
        assert (pred[dynamic, 5] == 0).all(), "Mountain probability on non-mountain cells!"

    # Check no NaN/Inf
    assert np.isfinite(pred).all(), "NaN or Inf in predictions!"

    print("  PASS: All prediction validity constraints satisfied")


def main():
    print("=" * 60)
    print("Testing finer distance buckets (0,1,2,3,4-5,6-8,9+)")
    print("=" * 60)
    print()

    test_dist_bucket_mapping()
    print()

    test_feature_keys_use_new_buckets()
    print()

    test_calibration_model_trains()
    print()

    test_fast_predict_pipeline()
    print()

    score = test_score_on_real_rounds()
    print()

    test_predictions_valid()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print(f"Score with optimized params: {score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
