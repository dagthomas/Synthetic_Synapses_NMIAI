"""Strategic data collection for Astar Island.

Strategies:
  grid (default):
    1. Fixed 3x3 grid coverage (9 viewports/seed = 45 queries) for ~97% coverage
    2. 5 remaining queries on highest-uncertainty seeds
    3. Pool observations across all seeds into GlobalTransitionMatrix
    4. Run ParameterEstimator on settlement stats

  multi-sample:
    1. Compute per-cell expected entropy from CalibrationModel (free)
    2. Greedy select entropy-maximizing viewports (skip static waste)
    3. Query viewports — either multi-sample (stochastic API) or cross-seed
    4. Build MultiSampleStore for variance analysis
    5. Parameter estimation + FK bucketing

Usage:
    python explore.py                          # Grid strategy (default)
    python explore.py --strategy multi-sample  # Entropy-targeted
    python explore.py --dry-run                # Plan queries without calling API
    python explore.py --round-id UUID          # Explore specific round
"""
import argparse
import json
from pathlib import Path

import numpy as np

from client import AstarIslandClient
from config import MAP_H, MAP_W, NUM_CLASSES
from estimator import ParameterEstimator
from utils import (
    FeatureKeyBuckets,
    GlobalMultipliers,
    GlobalTransitionMatrix,
    MultiSampleStore,
    ObservationAccumulator,
    classify_cells,
    dynamism_heatmap,
    initial_grid_to_classes,
    terrain_to_class,
)
from calibration import build_feature_keys, CalibrationModel
from predict import get_static_prior
import predict

DATA_DIR = Path(__file__).parent / "data"

# Fixed 3x3 grid: 15x15 viewports at (0,0), (0,13), (0,25), (13,0), etc.
# Covers cells 0-14, 13-27, 25-39 per axis -- total coverage = 39/40 per axis
# Overlap at columns/rows 13-14 and 25-27 gives extra observations on those cells
GRID_POSITIONS = [(0, 0), (13, 0), (25, 0),
                  (0, 13), (13, 13), (25, 13),
                  (0, 25), (13, 25), (25, 25)]


def compute_expected_entropy_map(grid: np.ndarray, settlements: list, cal=None) -> np.ndarray:
    """Compute per-cell expected entropy from CalibrationModel (free, no queries).

    Returns (40, 40) float array. Ocean/mountain = 0 (static, no information value).
    High entropy = uncertain cell = more valuable to observe.
    """
    terrain = np.array(grid, dtype=int)
    if cal is None:
        cal = predict.get_calibration()

    fkeys = build_feature_keys(terrain, settlements)
    entropy_map = np.zeros((MAP_H, MAP_W), dtype=float)

    for y in range(MAP_H):
        for x in range(MAP_W):
            code = int(terrain[y, x])
            if code == 10 or code == 5:  # ocean or mountain — static
                continue
            prior = cal.prior_for(fkeys[y][x])
            # Shannon entropy
            safe_prior = np.maximum(prior, 1e-10)
            entropy_map[y, x] = -np.sum(prior * np.log(safe_prior))

    return entropy_map


def select_entropy_viewports(entropy_map: np.ndarray, n_viewports: int = 10,
                              vp_size: int = 15, min_overlap_frac: float = 0.5
                              ) -> list[dict]:
    """Greedy selection of viewports maximizing total expected entropy.

    Picks the highest-entropy viewport, masks covered area (blocks >50% overlap),
    repeats. Returns list of {x, y, w, h, entropy_score} dicts sorted by score.
    """
    h, w = entropy_map.shape
    selected = []
    # Track which cells are already covered (for overlap checking)
    covered = np.zeros((h, w), dtype=bool)

    for _ in range(n_viewports):
        best_score = -1.0
        best_pos = (0, 0)

        # Score all valid viewport positions
        for vy in range(h - vp_size + 1):
            for vx in range(w - vp_size + 1):
                # Check overlap with already selected viewports
                region = covered[vy:vy + vp_size, vx:vx + vp_size]
                overlap_frac = region.sum() / (vp_size * vp_size)
                if overlap_frac > min_overlap_frac:
                    continue

                # Score = sum of entropy in viewport (excluding already-covered cells)
                ent_region = entropy_map[vy:vy + vp_size, vx:vx + vp_size]
                uncovered = ~region
                score = float(ent_region[uncovered].sum())

                if score > best_score:
                    best_score = score
                    best_pos = (vx, vy)

        if best_score <= 0:
            break  # No more informative viewports

        vx, vy = best_pos
        selected.append({
            'x': vx, 'y': vy, 'w': vp_size, 'h': vp_size,
            'entropy_score': best_score,
        })
        covered[vy:vy + vp_size, vx:vx + vp_size] = True

    return selected


def run_multi_sample_exploration(client: AstarIslandClient, round_id: str,
                                  detail: dict, dry_run: bool = False) -> dict:
    """Multi-sample exploration: entropy-targeted viewports with variance analysis.

    Budget allocation (conservative 7+3 split):
      Phase 1: 7 entropy-targeted viewports x 5 seeds = 35 queries (66% coverage)
      Phase 2: 3 repeat viewports x 5 seeds = 15 queries (multi-sample on top 3)
    """
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Phase 0: Analyze initial states (free)
    print("=" * 60)
    print("Phase 0: Analyzing initial states (free)")
    print("=" * 60)
    analysis = analyze_initial_states(detail)

    # Phase 1: Compute per-seed entropy maps, average across seeds
    print("\n" + "=" * 60)
    print("Phase 1: Computing entropy maps (free)")
    print("=" * 60)

    cal = predict.get_calibration()
    seed_entropy_maps = []
    seed_feature_keys = []
    seed_priors = []

    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        terrain = np.array(state["grid"], dtype=int)
        emap = compute_expected_entropy_map(terrain, state["settlements"], cal)
        seed_entropy_maps.append(emap)

        prior = get_static_prior(state["grid"], state["settlements"])
        seed_priors.append(prior)

        fkeys = build_feature_keys(terrain, state["settlements"])
        seed_feature_keys.append(fkeys)

    # Average entropy across seeds (terrain is shared, so they should be similar)
    avg_entropy = np.mean(seed_entropy_maps, axis=0)
    total_entropy = float(avg_entropy.sum())
    dynamic_cells = (avg_entropy > 0).sum()
    print(f"  Total map entropy: {total_entropy:.1f}")
    print(f"  Dynamic cells: {dynamic_cells}/{MAP_H * MAP_W} ({100 * dynamic_cells / (MAP_H * MAP_W):.1f}%)")

    # Select entropy-targeted viewports with adaptive budget split
    # Strategy: maximize coverage first (permissive overlap), use remaining for repeats
    # Budget: 50 queries = n_unique * seeds_count + n_repeat * seeds_count
    max_budget = 50
    max_batches = max_budget // seeds_count  # 10 batches of 5 seeds

    # Try to get 8 unique viewports (40 queries), leaving 2 batches for repeats
    # Use permissive overlap (0.7) to maximize coverage
    viewports = select_entropy_viewports(avg_entropy, n_viewports=8, min_overlap_frac=0.7)

    # If we got fewer viewports, we have more budget for repeats
    n_unique_actual = len(viewports)
    n_repeat = min(max_batches - n_unique_actual, 3)  # cap at 3 repeats
    n_repeat = max(n_repeat, 0)

    coverage = compute_coverage(viewports)
    print(f"\n  Selected {n_unique_actual} entropy-targeted viewports (overlap_frac=0.7):")
    for i, vp in enumerate(viewports):
        print(f"    VP{i}: ({vp['x']},{vp['y']}) entropy={vp['entropy_score']:.1f}")
    print(f"  Coverage: {coverage:.1%} of map")

    # Top N viewports for repeat queries (multi-sampling)
    repeat_viewports = sorted(viewports, key=lambda v: v['entropy_score'], reverse=True)[:n_repeat]
    if repeat_viewports:
        print(f"\n  Repeat viewports (top {n_repeat} by entropy):")
        for vp in repeat_viewports:
            print(f"    ({vp['x']},{vp['y']}) entropy={vp['entropy_score']:.1f}")

    total_queries = n_unique_actual * seeds_count + len(repeat_viewports) * seeds_count
    remaining_budget = max_budget - total_queries
    print(f"\n  Total queries planned: {total_queries} "
          f"({n_unique_actual} unique + {len(repeat_viewports)} repeat) x {seeds_count} seeds"
          f"{f', {remaining_budget} unused' if remaining_budget > 0 else ''}")

    if dry_run:
        print(f"\n[DRY RUN] Would use {total_queries} queries")
        return {"analysis": analysis, "dry_run": True}

    # Check budget
    budget = client.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\nBudget: {remaining} queries remaining")
    if remaining == 0:
        print("No queries remaining!")
        return {"analysis": analysis, "accumulators": [], "observations": [],
                "estimates": {}, "global_transitions": None}

    # Initialize data structures
    accumulators = [ObservationAccumulator() for _ in range(seeds_count)]
    global_tm = GlobalTransitionMatrix()
    global_mult = GlobalMultipliers()
    fk_buckets = FeatureKeyBuckets()
    multi_store = MultiSampleStore()
    estimator = ParameterEstimator()
    all_observations = []

    # Set feature keys in multi-store
    for si in range(seeds_count):
        multi_store.set_feature_keys(si, seed_feature_keys[si])

    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    def _process_observation(seed_idx, vp, result, query_num, is_repeat=False):
        """Process a single observation — shared between unique and repeat phases."""
        accumulators[seed_idx].add_observation(result["grid"], result["viewport"])

        state = initial_states[seed_idx]
        global_tm.add_observation(
            state["grid"], result["grid"], result["viewport"], state["settlements"])

        obs_vp = result["viewport"]
        obs_grid = result["grid"]
        prior = seed_priors[seed_idx]
        fkeys = seed_feature_keys[seed_idx]
        for row in range(len(obs_grid)):
            for col in range(len(obs_grid[0]) if obs_grid else 0):
                my, mx = obs_vp["y"] + row, obs_vp["x"] + col
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                    obs_cls = terrain_to_class(obs_grid[row][col])
                    global_mult.add_observation(obs_cls, prior[my, mx])
                    fk_buckets.add_observation(fkeys[my][mx], obs_cls)

        # Add to multi-sample store
        multi_store.add_sample(seed_idx, vp["x"], vp["y"], result["grid"])

        estimator.add_observation(
            result, initial_grid=state["grid"],
            initial_settlements=state["settlements"])

        obs_entry = {
            "seed_index": seed_idx, "query_num": query_num,
            "viewport": result["viewport"], "grid": result["grid"],
            "settlements": result["settlements"],
            "is_repeat": is_repeat,
        }
        all_observations.append(obs_entry)

        # Incremental save
        fname = f"obs_s{seed_idx}_q{query_num}.json"
        with open(round_dir / fname, "w") as f:
            json.dump(obs_entry, f)

    # Phase 2: Unique viewport queries
    print("\n" + "=" * 60)
    print("Phase 2: Entropy-targeted coverage queries")
    print("=" * 60)

    query_num = 0
    for vp in viewports:
        for seed_idx in range(seeds_count):
            if client.queries_remaining <= 0:
                break
            print(f"  Q{query_num + 1}: seed={seed_idx}, vp=({vp['x']},{vp['y']})")
            result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])
            _process_observation(seed_idx, vp, result, query_num, is_repeat=False)
            query_num += 1
        if client.queries_remaining <= 0:
            break

    print(f"\nCoverage phase used {query_num} queries")
    for i, acc in enumerate(accumulators):
        n = acc.get_observation_count()
        obs_count = (n > 0).sum()
        print(f"  Seed {i}: {obs_count}/{MAP_H * MAP_W} cells ({100 * obs_count / (MAP_H * MAP_W):.1f}%)")

    # Phase 3: Repeat queries on highest-entropy viewports (for multi-sampling)
    if client.queries_remaining > 0 and repeat_viewports:
        print(f"\n" + "=" * 60)
        print(f"Phase 3: Multi-sample repeat queries ({client.queries_remaining} remaining)")
        print("=" * 60)

        for vp in repeat_viewports:
            for seed_idx in range(seeds_count):
                if client.queries_remaining <= 0:
                    break
                print(f"  Q{query_num + 1}: REPEAT seed={seed_idx}, vp=({vp['x']},{vp['y']})")
                result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])
                _process_observation(seed_idx, vp, result, query_num, is_repeat=True)
                query_num += 1
            if client.queries_remaining <= 0:
                break

    # Phase 4: Parameter estimation + variance analysis
    print("\n" + "=" * 60)
    print("Phase 4: Parameter estimation + variance analysis")
    print("=" * 60)

    estimator.print_summary()
    estimates = estimator.estimate()

    # Variance analysis from multi-samples
    ms_stats = multi_store.get_stats()
    print(f"\nMulti-sample store: {ms_stats['total_viewports']} viewports, "
          f"{ms_stats['multi_sample_viewports']} with multiple samples, "
          f"{ms_stats['total_grids']} total grids")

    if ms_stats['multi_sample_viewports'] > 0:
        var_overall = multi_store.get_overall_variance()
        print(f"  Avg settlement variance: {var_overall['avg_variance']:.6f}")
        print(f"  Max settlement variance: {var_overall['max_variance']:.6f}")
        print(f"  Avg settlement pct: {var_overall['avg_sett_pct']:.3f}")
        print(f"  Max settlement pct: {var_overall['max_sett_pct']:.3f}")

        # Detect regime
        regime = estimator.detect_regime_enhanced(multi_store)
        print(f"\n  Detected regime: {regime}")
    else:
        regime = estimator.detect_regime_enhanced(None)
        print(f"\n  Detected regime (no multi-sample): {regime}")

    # Save data
    save_exploration_data(round_id, detail, analysis, all_observations, accumulators,
                          estimates=estimates)

    tm_stats = global_tm.get_stats()
    print(f"\nGlobal transition matrix: {tm_stats['total_observations']} observations, "
          f"{tm_stats['buckets_with_data']} buckets with data")

    multipliers = global_mult.get_multipliers()
    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    mult_summary = global_mult.get_summary()
    print(f"\nGlobal multipliers ({mult_summary['total_cells_observed']} cells):")
    for name, m in zip(class_names, multipliers):
        print(f"  {name}: {m:.4f}")

    fk_stats = fk_buckets.get_stats()
    print(f"\nFeature-key buckets: {fk_stats['total_observations']} obs, "
          f"{fk_stats['keys_with_data']} keys, "
          f"avg {fk_stats['avg_per_key']:.0f}/key")

    print(f"\nExploration complete. Queries used: {client.queries_used}/{client._queries_max}")

    return {
        "analysis": analysis,
        "accumulators": accumulators,
        "observations": all_observations,
        "estimates": estimates,
        "global_transitions": global_tm,
        "global_multipliers": global_mult,
        "feature_key_buckets": fk_buckets,
        "multi_sample_store": multi_store,
        "variance_regime": regime,
    }


def run_adaptive_exploration(client: AstarIslandClient, round_id: str,
                              detail: dict, dry_run: bool = False) -> dict:
    """Adaptive exploration: detect regime early, then switch strategy.

    Phase 1 (15 queries): 3 grid viewports × 5 seeds → ~33% coverage, regime detection
    Decision: compute settlement multiplier from observations
      BOOM/EXTREME_BOOM → entropy-targeted + repeat queries for remaining budget
      MODERATE/COLLAPSE → continue grid coverage for maximum coverage

    This gets the best of both worlds: full coverage for easy rounds,
    targeted multi-sampling for boom rounds where variance matters most.
    """
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Phase 0: Analyze initial states (free)
    print("=" * 60)
    print("Phase 0: Analyzing initial states (free)")
    print("=" * 60)
    analysis = analyze_initial_states(detail)

    if dry_run:
        print("\n[DRY RUN] Adaptive strategy: would query 3 grid viewports first,")
        print("  then decide grid vs entropy-targeted based on settlement multiplier")
        return {"analysis": analysis, "dry_run": True}

    # Check budget
    budget = client.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\nBudget: {remaining} queries remaining")
    if remaining == 0:
        print("No queries remaining!")
        return {"analysis": analysis, "accumulators": [], "observations": [],
                "estimates": {}, "global_transitions": None}

    # Initialize shared data structures
    accumulators = [ObservationAccumulator() for _ in range(seeds_count)]
    global_tm = GlobalTransitionMatrix()
    global_mult = GlobalMultipliers()
    fk_buckets = FeatureKeyBuckets()
    multi_store = MultiSampleStore()
    estimator = ParameterEstimator()
    all_observations = []

    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build priors and feature keys for each seed
    cal = predict.get_calibration()
    seed_priors = []
    seed_feature_keys = []
    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        prior = get_static_prior(state["grid"], state["settlements"])
        seed_priors.append(prior)
        terrain_np = np.array(state["grid"], dtype=int)
        fkeys = build_feature_keys(terrain_np, state["settlements"])
        seed_feature_keys.append(fkeys)
        multi_store.set_feature_keys(seed_idx, fkeys)

    query_num = 0

    def _process_obs(seed_idx, vp_dict, result, is_repeat=False):
        nonlocal query_num
        accumulators[seed_idx].add_observation(result["grid"], result["viewport"])
        state = initial_states[seed_idx]
        global_tm.add_observation(
            state["grid"], result["grid"], result["viewport"], state["settlements"])
        obs_vp = result["viewport"]
        obs_grid = result["grid"]
        prior = seed_priors[seed_idx]
        fkeys = seed_feature_keys[seed_idx]
        for row in range(len(obs_grid)):
            for col in range(len(obs_grid[0]) if obs_grid else 0):
                my, mx = obs_vp["y"] + row, obs_vp["x"] + col
                if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                    obs_cls = terrain_to_class(obs_grid[row][col])
                    global_mult.add_observation(obs_cls, prior[my, mx])
                    fk_buckets.add_observation(fkeys[my][mx], obs_cls)
        multi_store.add_sample(seed_idx, vp_dict["x"], vp_dict["y"], result["grid"])
        estimator.add_observation(
            result, initial_grid=state["grid"],
            initial_settlements=state["settlements"])
        obs_entry = {
            "seed_index": seed_idx, "query_num": query_num,
            "viewport": result["viewport"], "grid": result["grid"],
            "settlements": result["settlements"], "is_repeat": is_repeat,
        }
        all_observations.append(obs_entry)
        fname = f"obs_s{seed_idx}_q{query_num}.json"
        with open(round_dir / fname, "w") as f:
            json.dump(obs_entry, f)
        query_num += 1

    # =========================================================
    # Phase 1: Entropy-targeted scout queries (5 viewports × 5 seeds = 25)
    # =========================================================
    # Compute entropy map from initial states (free, no queries)
    seed_entropy_maps = []
    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        terrain = np.array(state["grid"], dtype=int)
        emap = compute_expected_entropy_map(terrain, state["settlements"], cal)
        seed_entropy_maps.append(emap)
    avg_entropy = np.mean(seed_entropy_maps, axis=0)

    # Select top 5 entropy viewports for scout phase (25 queries)
    scout_vps = select_entropy_viewports(avg_entropy, n_viewports=5, min_overlap_frac=0.5)
    if len(scout_vps) < 3:
        # Fallback to diagonal if entropy map is too flat
        scout_positions = [(0, 0), (13, 13), (25, 25)]
        scout_vps = [{"x": x, "y": y, "w": 15, "h": 15, "entropy_score": 0}
                     for x, y in scout_positions]

    scout_coverage = compute_coverage(scout_vps)
    print("\n" + "=" * 60)
    print(f"Phase 1: Entropy-targeted scout ({len(scout_vps)} viewports × {seeds_count} seeds)")
    print("=" * 60)
    for i, vp in enumerate(scout_vps):
        print(f"  VP{i}: ({vp['x']},{vp['y']}) entropy={vp.get('entropy_score', 0):.1f}")
    print(f"  Coverage: {scout_coverage:.1%}")

    for vp in scout_vps:
        for seed_idx in range(seeds_count):
            if client.queries_remaining <= 0:
                break
            print(f"  Q{query_num + 1}: seed={seed_idx}, vp=({vp['x']},{vp['y']})")
            result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])
            _process_obs(seed_idx, vp, result)
        if client.queries_remaining <= 0:
            break

    print(f"\nScout phase: {query_num} queries used, {client.queries_remaining} remaining")

    # =========================================================
    # Decision point: detect regime from scout observations
    # =========================================================
    multipliers = global_mult.get_multipliers()
    sett_mult = multipliers[1]
    obs_total = global_mult.observed.sum()
    sett_pct = global_mult.observed[1] / max(obs_total, 1)

    print(f"\n" + "=" * 60)
    print(f"REGIME DETECTION (from {int(obs_total)} observed cells)")
    print(f"=" * 60)
    print(f"  Settlement multiplier: {sett_mult:.3f}")
    print(f"  Settlement observed %: {sett_pct:.1%}")

    # Classify regime
    if sett_mult > 1.3 or sett_pct > 0.15:
        detected_regime = "BOOM"
    elif sett_mult > 1.6 or sett_pct > 0.20:
        detected_regime = "EXTREME_BOOM"
    elif sett_mult < 0.5 or sett_pct < 0.02:
        detected_regime = "COLLAPSE"
    else:
        detected_regime = "MODERATE"

    # Override: check if extreme based on high variance in scout observations
    var_stats = multi_store.get_overall_variance()
    if var_stats["n_viewports"] > 0 and var_stats["avg_variance"] > 0.005:
        detected_regime = "EXTREME_BOOM"
        print(f"  Settlement variance: {var_stats['avg_variance']:.6f} → EXTREME_BOOM override")

    use_entropy_strategy = detected_regime in ("BOOM", "EXTREME_BOOM")
    print(f"  Detected regime: {detected_regime}")
    print(f"  Strategy for remaining budget: {'ENTROPY-TARGETED' if use_entropy_strategy else 'GRID COVERAGE'}")

    # =========================================================
    # Phase 2: Remaining queries — entropy-targeted coverage
    # =========================================================
    remaining_budget = client.queries_remaining

    print(f"\n" + "=" * 60)
    print(f"Phase 2: Entropy-targeted coverage ({remaining_budget} queries)")
    print("=" * 60)

    # Re-compute entropy map with scout regions down-weighted
    phase2_entropy = avg_entropy.copy()
    for vp in scout_vps:
        phase2_entropy[vp["y"]:vp["y"]+15, vp["x"]:vp["x"]+15] *= 0.3

    n_batches = remaining_budget // seeds_count
    if use_entropy_strategy:
        # BOOM: reserve 2 batches for repeat queries (multi-sampling)
        n_unique = max(n_batches - 2, 1)
        n_repeat = n_batches - n_unique
    else:
        # MODERATE/COLLAPSE: maximize coverage, 0 repeats
        n_unique = n_batches
        n_repeat = 0

    ent_vps = select_entropy_viewports(phase2_entropy, n_viewports=n_unique, min_overlap_frac=0.7)
    print(f"  Selected {len(ent_vps)} entropy viewports"
          f"{f' + {n_repeat} repeats' if n_repeat > 0 else ''}")
    for i, vp in enumerate(ent_vps):
        print(f"    VP{i}: ({vp['x']},{vp['y']}) entropy={vp['entropy_score']:.1f}")

    for vp in ent_vps:
        for seed_idx in range(seeds_count):
            if client.queries_remaining <= 0:
                break
            print(f"  Q{query_num + 1}: seed={seed_idx}, vp=({vp['x']},{vp['y']}) [entropy]")
            result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])
            _process_obs(seed_idx, vp, result)
        if client.queries_remaining <= 0:
            break

    # Repeat queries on highest-entropy viewports (boom only)
    if client.queries_remaining > 0 and n_repeat > 0:
        repeat_vps = sorted(ent_vps, key=lambda v: v["entropy_score"], reverse=True)[:n_repeat]
        for vp in repeat_vps:
            for seed_idx in range(seeds_count):
                if client.queries_remaining <= 0:
                    break
                print(f"  Q{query_num + 1}: REPEAT seed={seed_idx}, vp=({vp['x']},{vp['y']})")
                result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])
                _process_obs(seed_idx, vp, result, is_repeat=True)
            if client.queries_remaining <= 0:
                break

    # =========================================================
    # Phase 3: Final analysis
    # =========================================================
    print("\n" + "=" * 60)
    print("Phase 3: Parameter estimation + final analysis")
    print("=" * 60)

    estimator.print_summary()
    estimates = estimator.estimate()

    # Variance regime (may upgrade based on full data)
    variance_regime = estimator.detect_regime_enhanced(multi_store)
    print(f"\n  Final regime: {variance_regime} (scout detected: {detected_regime})")

    save_exploration_data(round_id, detail, analysis, all_observations, accumulators,
                          estimates=estimates)

    multipliers = global_mult.get_multipliers()
    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    mult_summary = global_mult.get_summary()
    print(f"\nGlobal multipliers ({mult_summary['total_cells_observed']} cells):")
    for name, m in zip(class_names, multipliers):
        print(f"  {name}: {m:.4f}")

    fk_stats = fk_buckets.get_stats()
    print(f"\nFeature-key buckets: {fk_stats['total_observations']} obs, "
          f"{fk_stats['keys_with_data']} keys, "
          f"avg {fk_stats['avg_per_key']:.0f}/key")

    for i, acc in enumerate(accumulators):
        n = acc.get_observation_count()
        obs_count = (n > 0).sum()
        print(f"  Seed {i}: {obs_count}/{MAP_H * MAP_W} cells "
              f"({100 * obs_count / (MAP_H * MAP_W):.1f}%)")

    print(f"\nExploration complete. Queries used: {client.queries_used}/{client._queries_max}")

    return {
        "analysis": analysis,
        "accumulators": accumulators,
        "observations": all_observations,
        "estimates": estimates,
        "global_transitions": global_tm,
        "global_multipliers": global_mult,
        "feature_key_buckets": fk_buckets,
        "multi_sample_store": multi_store,
        "variance_regime": variance_regime,
    }


def analyze_initial_states(detail: dict) -> dict:
    """Extract intelligence from initial states (free  no query cost)."""
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]
    analysis = {"seeds": [], "shared_terrain": None}

    for i, state in enumerate(initial_states):
        grid = state["grid"]
        settlements = state["settlements"]
        cell_info = classify_cells(grid)
        heatmap = dynamism_heatmap(grid, settlements)
        classes = initial_grid_to_classes(grid)

        unique, counts = np.unique(classes, return_counts=True)
        terrain_counts = dict(zip(unique.tolist(), counts.tolist()))

        seed_analysis = {
            "seed_index": i,
            "num_settlements": len([s for s in settlements if s.get("alive", True)]),
            "num_ports": len([s for s in settlements if s.get("has_port", False)]),
            "terrain_counts": terrain_counts,
            "static_cells": int(cell_info["static"].sum()),
            "dynamic_cells": int(cell_info["dynamic"].sum()),
            "heatmap": heatmap,
            "grid": grid,
            "settlements": settlements,
            "cell_info": cell_info,
        }
        analysis["seeds"].append(seed_analysis)

        print(f"Seed {i}: {seed_analysis['num_settlements']} settlements "
              f"({seed_analysis['num_ports']} ports), "
              f"{seed_analysis['static_cells']} static / "
              f"{seed_analysis['dynamic_cells']} dynamic cells")

    if seeds_count > 1:
        grids = [initial_grid_to_classes(s["grid"]) for s in initial_states]
        base_masks = []
        for g in grids:
            mask = np.zeros_like(g)
            mask[(g == 0) | (g == 4) | (g == 5)] = g[(g == 0) | (g == 4) | (g == 5)]
            base_masks.append(mask)
        shared = all(np.array_equal(base_masks[0], m) for m in base_masks[1:])
        analysis["shared_terrain"] = shared
        print(f"\nBase terrain shared across seeds: {shared}")

    return analysis


def compute_coverage(vp_list: list[dict], map_h: int = MAP_H, map_w: int = MAP_W) -> float:
    """Compute what fraction of the map is covered by a list of viewports."""
    covered = np.zeros((map_h, map_w), dtype=bool)
    for vp in vp_list:
        y, x, h, w = vp["y"], vp["x"], vp["h"], vp["w"]
        covered[y:y + h, x:x + w] = True
    return covered.sum() / (map_h * map_w)


def run_exploration(client: AstarIslandClient, round_id: str, detail: dict,
                    dry_run: bool = False) -> dict:
    """Run exploration with fixed 3x3 grid coverage + cross-seed pooling."""
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Phase 0: Analyze initial states (free)
    print("=" * 60)
    print("Phase 0: Analyzing initial states (free)")
    print("=" * 60)
    analysis = analyze_initial_states(detail)

    # Phase 1: Fixed grid plan
    print("\n" + "=" * 60)
    print("Phase 1: Fixed 3x3 grid coverage")
    print("=" * 60)
    viewports = [{"x": x, "y": y, "w": 15, "h": 15} for x, y in GRID_POSITIONS]
    coverage = compute_coverage(viewports)
    print(f"  9 viewports per seed, {coverage:.1%} map coverage")
    print(f"  Total: {len(viewports) * seeds_count} queries for coverage")
    print(f"  + {50 - len(viewports) * seeds_count} queries for adaptive")

    if dry_run:
        print(f"\n[DRY RUN] Would use {len(viewports) * seeds_count} coverage "
              f"+ {50 - len(viewports) * seeds_count} adaptive queries")
        return {"analysis": analysis, "dry_run": True}

    # Check budget
    budget = client.get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\nBudget: {remaining} queries remaining")
    if remaining == 0:
        print("No queries remaining!")
        return {"analysis": analysis, "accumulators": [], "observations": [],
                "estimates": {}, "global_transitions": None}

    # Phase 2: Coverage queries  9 viewports  5 seeds = 45 queries
    print("\n" + "=" * 60)
    print("Phase 2: Grid coverage queries")
    print("=" * 60)

    accumulators = [ObservationAccumulator() for _ in range(seeds_count)]
    global_tm = GlobalTransitionMatrix()
    global_mult = GlobalMultipliers()
    fk_buckets = FeatureKeyBuckets()
    estimator = ParameterEstimator()
    all_observations = []

    # Incremental save directory  write each observation immediately
    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build static priors and feature keys for each seed
    seed_priors = []
    seed_feature_keys = []
    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        prior = get_static_prior(state["grid"], state["settlements"])
        seed_priors.append(prior)
        terrain_np = np.array(state["grid"], dtype=int)
        fkeys = build_feature_keys(terrain_np, state["settlements"])
        seed_feature_keys.append(fkeys)

    coverage_budget = min(len(viewports) * seeds_count, remaining)
    query_num = 0

    # Round-robin: same viewport position across all seeds before moving to next
    for vp in viewports:
        for seed_idx in range(seeds_count):
            if query_num >= coverage_budget:
                break

            print(f"  Q{query_num + 1}: seed={seed_idx}, vp=({vp['x']},{vp['y']})")
            result = client.simulate(round_id, seed_idx, vp["x"], vp["y"], vp["w"], vp["h"])

            # Per-seed accumulator
            accumulators[seed_idx].add_observation(result["grid"], result["viewport"])

            # Cross-seed pooling
            state = initial_states[seed_idx]
            global_tm.add_observation(
                state["grid"], result["grid"], result["viewport"], state["settlements"])

            # Global multipliers + feature-key buckets
            obs_vp = result["viewport"]
            obs_grid = result["grid"]
            prior = seed_priors[seed_idx]
            fkeys = seed_feature_keys[seed_idx]
            for row in range(len(obs_grid)):
                for col in range(len(obs_grid[0]) if obs_grid else 0):
                    my, mx = obs_vp["y"] + row, obs_vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        obs_cls = terrain_to_class(obs_grid[row][col])
                        global_mult.add_observation(obs_cls, prior[my, mx])
                        fk_buckets.add_observation(fkeys[my][mx], obs_cls)

            # Parameter estimation
            estimator.add_observation(
                result, initial_grid=state["grid"],
                initial_settlements=state["settlements"])

            obs_entry = {
                "seed_index": seed_idx, "query_num": query_num,
                "viewport": result["viewport"], "grid": result["grid"],
                "settlements": result["settlements"],
            }
            all_observations.append(obs_entry)

            # Incremental save  NEVER lose observations
            fname = f"obs_s{seed_idx}_q{query_num}.json"
            with open(round_dir / fname, "w") as f:
                json.dump(obs_entry, f)

            query_num += 1

        if query_num >= coverage_budget:
            break

    print(f"\nCoverage phase used {query_num} queries")

    # Print coverage stats
    for i, acc in enumerate(accumulators):
        n = acc.get_observation_count()
        obs_count = (n > 0).sum()
        print(f"  Seed {i}: {obs_count}/{MAP_H * MAP_W} cells ({100 * obs_count / (MAP_H * MAP_W):.1f}%)")

    # Phase 3: Adaptive queries  target 2nd observations on most dynamic cells
    adaptive_budget = client.queries_remaining
    if adaptive_budget > 0:
        print(f"\n" + "=" * 60)
        print(f"Phase 3: Adaptive queries  2nd observations ({adaptive_budget} remaining)")
        print("=" * 60)

        # Build set of (seed, vx, vy) already queried in Phase 2
        phase2_queried = set()
        for vp in viewports:
            for seed_idx in range(seeds_count):
                phase2_queried.add((seed_idx, vp["x"], vp["y"]))

        # Improved adaptive: rank ALL seed-viewport combinations by dynamic value,
        # not just 1 per seed. This allows the most dynamic seeds to get multiple
        # 2nd observations while less dynamic seeds get fewer.
        all_candidates = []
        for seed_idx in range(seeds_count):
            heatmap = analysis["seeds"][seed_idx]["heatmap"]
            # Find top 3 non-overlapping viewports per seed
            used_positions = set()
            for _ in range(3):
                best_score, best_pos = -1, (0, 0)
                for vy in range(MAP_H - 15 + 1):
                    for vx in range(MAP_W - 15 + 1):
                        # Skip if too close to already-selected
                        too_close = any(abs(vy - py) < 8 and abs(vx - px) < 8
                                        for py, px in used_positions)
                        if too_close:
                            continue
                        s = heatmap[vy:vy + 15, vx:vx + 15].sum()
                        if s > best_score:
                            best_score = s
                            best_pos = (vx, vy)
                if best_score > 0:
                    used_positions.add(best_pos)
                    all_candidates.append((best_score, seed_idx, best_pos))

        # Sort by score, deduplicate (seed, viewport), skip Phase 2 repeats
        all_candidates.sort(reverse=True)
        seen = set()
        for score, seed_idx, (vx, vy) in all_candidates:
            key = (seed_idx, vx, vy)
            if key in seen or key in phase2_queried:
                continue
            seen.add(key)
            if client.queries_remaining <= 0:
                break

            print(f"  Adaptive: seed={seed_idx}, vp=({vx},{vy}), score={score:.2f}")
            result = client.simulate(round_id, seed_idx, vx, vy, 15, 15)
            accumulators[seed_idx].add_observation(result["grid"], result["viewport"])

            state = initial_states[seed_idx]
            global_tm.add_observation(
                state["grid"], result["grid"], result["viewport"], state["settlements"])

            obs_vp = result["viewport"]
            obs_grid = result["grid"]
            prior = seed_priors[seed_idx]
            fkeys = seed_feature_keys[seed_idx]
            for row in range(len(obs_grid)):
                for col in range(len(obs_grid[0]) if obs_grid else 0):
                    my, mx = obs_vp["y"] + row, obs_vp["x"] + col
                    if 0 <= my < MAP_H and 0 <= mx < MAP_W:
                        obs_cls = terrain_to_class(obs_grid[row][col])
                        global_mult.add_observation(obs_cls, prior[my, mx])
                        fk_buckets.add_observation(fkeys[my][mx], obs_cls)

            estimator.add_observation(
                result, initial_grid=state["grid"],
                initial_settlements=state["settlements"])

            obs_entry = {
                "seed_index": seed_idx,
                "query_num": len(all_observations),
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result["settlements"],
            }
            all_observations.append(obs_entry)

            # Incremental save  NEVER lose observations
            fname = f"obs_s{seed_idx}_q{obs_entry['query_num']}.json"
            with open(round_dir / fname, "w") as f:
                json.dump(obs_entry, f)

    # Phase 4: Parameter estimation
    print("\n" + "=" * 60)
    print("Phase 4: Parameter estimation")
    print("=" * 60)

    estimator.print_summary()
    estimates = estimator.estimate()

    # Save data FIRST (before any prints that might crash on unicode)
    save_exploration_data(round_id, detail, analysis, all_observations, accumulators,
                          estimates=estimates)

    tm_stats = global_tm.get_stats()
    print(f"\nGlobal transition matrix: {tm_stats['total_observations']} observations, "
          f"{tm_stats['buckets_with_data']} buckets with data")

    multipliers = global_mult.get_multipliers()
    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    mult_summary = global_mult.get_summary()
    print(f"\nGlobal multipliers ({mult_summary['total_cells_observed']} cells):")
    for name, m in zip(class_names, multipliers):
        print(f"  {name}: {m:.4f}")
    if multipliers[1] < 0.7:
        print("  >> Settlement multiplier low -- possible collapse scenario")

    fk_stats = fk_buckets.get_stats()
    print(f"\nFeature-key buckets: {fk_stats['total_observations']} obs, "
          f"{fk_stats['keys_with_data']} keys, "
          f"avg {fk_stats['avg_per_key']:.0f}/key")

    print(f"\nExploration complete. Queries used: {client.queries_used}/{client._queries_max}")

    return {
        "analysis": analysis,
        "accumulators": accumulators,
        "observations": all_observations,
        "estimates": estimates,
        "global_transitions": global_tm,
        "global_multipliers": global_mult,
        "feature_key_buckets": fk_buckets,
    }


def save_exploration_data(round_id: str, detail: dict, analysis: dict,
                          observations: list, accumulators: list,
                          estimates: dict = None):
    """Save all collected data to disk."""
    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    with open(round_dir / "initial_states.json", "w") as f:
        json.dump(detail, f)

    for obs in observations:
        fname = f"obs_s{obs['seed_index']}_q{obs['query_num']}.json"
        with open(round_dir / fname, "w") as f:
            json.dump(obs, f)

    summary = {
        "round_id": round_id,
        "num_observations": len(observations),
        "shared_terrain": analysis.get("shared_terrain"),
        "strategy": "fixed_3x3_grid",
        "seeds": [],
    }
    for i, seed_analysis in enumerate(analysis["seeds"]):
        acc = accumulators[i] if i < len(accumulators) else None
        seed_summary = {
            "seed_index": i,
            "num_settlements": seed_analysis["num_settlements"],
            "num_ports": seed_analysis["num_ports"],
            "terrain_counts": seed_analysis["terrain_counts"],
            "static_cells": seed_analysis["static_cells"],
            "dynamic_cells": seed_analysis["dynamic_cells"],
        }
        if acc is not None:
            n_obs = acc.get_observation_count()
            seed_summary["cells_observed"] = int((n_obs > 0).sum())
            seed_summary["max_observations"] = int(n_obs.max())
        summary["seeds"].append(seed_summary)

    with open(round_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if estimates:
        with open(round_dir / "estimates.json", "w") as f:
            json.dump(estimates, f, indent=2)

    print(f"\nData saved to {round_dir}")


def main():
    parser = argparse.ArgumentParser(description="Explore Astar Island")
    parser.add_argument("--dry-run", action="store_true", help="Plan queries without calling API")
    parser.add_argument("--round-id", type=str, help="Specific round UUID")
    parser.add_argument("--strategy", choices=["grid", "multi-sample", "adaptive"],
                        default="adaptive",
                        help="Exploration strategy: grid (3x3), multi-sample (entropy-targeted), "
                             "or adaptive (auto-detect regime, default)")
    args = parser.parse_args()

    client = AstarIslandClient()

    if args.round_id:
        round_id = args.round_id
    else:
        active = client.get_active_round()
        if not active:
            rounds = client.get_rounds()
            if not rounds:
                print("No rounds available")
                return
            round_id = rounds[-1]["id"]
            print(f"No active round. Using most recent: {rounds[-1].get('round_number')}")
        else:
            round_id = active["id"]
            print(f"Active round: {active['round_number']} ({round_id})")

    detail = client.get_round_detail(round_id)
    print(f"Map: {detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")
    print(f"Strategy: {args.strategy}\n")

    if args.strategy == "multi-sample":
        result = run_multi_sample_exploration(client, round_id, detail, dry_run=args.dry_run)
    elif args.strategy == "adaptive":
        result = run_adaptive_exploration(client, round_id, detail, dry_run=args.dry_run)
    else:
        result = run_exploration(client, round_id, detail, dry_run=args.dry_run)

    if not args.dry_run and "accumulators" in result:
        print("\n" + "=" * 60)
        print("Coverage Summary")
        print("=" * 60)
        for i, acc in enumerate(result["accumulators"]):
            n_obs = acc.get_observation_count()
            total = MAP_H * MAP_W
            observed = (n_obs > 0).sum()
            print(f"Seed {i}: {observed}/{total} cells observed "
                  f"({100 * observed / total:.1f}%), "
                  f"max obs/cell: {n_obs.max()}")


if __name__ == "__main__":
    main()
