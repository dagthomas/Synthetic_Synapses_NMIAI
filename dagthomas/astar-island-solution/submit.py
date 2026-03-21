"""Orchestrator: explore → predict → submit for a live round.

Usage:
    python submit.py                                # Full pipeline (grid strategy)
    python submit.py --strategy multi-sample        # Entropy-targeted exploration
    python submit.py --dry-run                      # Plan without API calls
    python submit.py --static-only                  # Submit static prior (no queries)
    python submit.py --uniform                      # Submit uniform baseline (1/6 each)
    python submit.py --round-id UUID                # Target specific round
"""
import argparse
import json
from pathlib import Path

import numpy as np

from client import AstarIslandClient
from config import MAP_H, MAP_W, NUM_CLASSES
from explore import run_exploration, run_multi_sample_exploration, run_adaptive_exploration, analyze_initial_states
from predict import (
    get_static_prior,
    predict_for_seed,
    validate_prediction,
)
from predict_gemini import gemini_predict
from utils import apply_floor, build_growth_front_map, build_obs_overlay, build_sett_survival

DATA_DIR = Path(__file__).parent / "data"


def submit_uniform(client: AstarIslandClient, round_id: str, seeds_count: int):
    """Submit uniform baseline (1/6 per class) for all seeds."""
    prediction = np.full((MAP_H, MAP_W, NUM_CLASSES), 1.0 / NUM_CLASSES)
    prediction = apply_floor(prediction)

    for seed_idx in range(seeds_count):
        errors = validate_prediction(prediction)
        if errors:
            print(f"  Seed {seed_idx} validation errors: {errors}")
            continue

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        print(f"  Seed {seed_idx}: {resp.get('status', 'unknown')}")


def submit_static_prior(client: AstarIslandClient, round_id: str, detail: dict):
    """Submit static prior predictions for all seeds."""
    initial_states = detail["initial_states"]

    for seed_idx, state in enumerate(initial_states):
        print(f"\nSeed {seed_idx}:")
        prediction = get_static_prior(state["grid"], state["settlements"])

        errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
        if errors:
            print(f"  Validation errors: {errors}")
            continue

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        print(f"  Submitted: {resp.get('status', 'unknown')}")


def submit_full_pipeline(client: AstarIslandClient, round_id: str, detail: dict,
                         dry_run: bool = False, strategy: str = "adaptive"):
    """Full pipeline: explore → predict → submit."""
    seeds_count = detail["seeds_count"]

    # Run exploration with selected strategy
    if strategy == "multi-sample":
        print(f"Using multi-sample exploration strategy (entropy-targeted)")
        exploration = run_multi_sample_exploration(client, round_id, detail, dry_run=dry_run)
    elif strategy == "adaptive":
        print(f"Using adaptive exploration strategy (auto-detect regime)")
        exploration = run_adaptive_exploration(client, round_id, detail, dry_run=dry_run)
    else:
        exploration = run_exploration(client, round_id, detail, dry_run=dry_run)

    if dry_run:
        print("\n[DRY RUN] Would generate predictions and submit for all seeds")
        return

    accumulators = exploration.get("accumulators", [None] * seeds_count)
    initial_states = detail["initial_states"]

    # Extract parameter estimates and global transitions
    estimates = exploration.get("estimates", {})
    prior_strength = estimates.get("prior_strength", 3.0)
    adjustments = estimates.get("adjustments", {})
    global_tm = exploration.get("global_transitions")
    global_mult = exploration.get("global_multipliers")
    fk_buckets = exploration.get("feature_key_buckets")
    multi_store = exploration.get("multi_sample_store")
    variance_regime = exploration.get("variance_regime")
    collapse_detected = estimates.get("collapse_detected", False)
    if collapse_detected:
        print("\n*** COLLAPSE DETECTED — adjusting predictions ***")
    if variance_regime:
        print(f"\nVariance regime: {variance_regime}")
    if global_mult is not None:
        mult = global_mult.get_multipliers()
        print(f"\nGlobal multipliers: sett={mult[1]:.3f}, port={mult[2]:.3f}, "
              f"ruin={mult[3]:.3f}, forest={mult[4]:.3f}")

    # Collect per-seed settlement stats from observations
    observations = exploration.get("observations", [])
    per_seed_sett_stats = {}  # seed_idx -> {(y,x): {pop, food, alive, ...}}
    for obs in observations:
        sid = obs["seed_index"]
        if sid not in per_seed_sett_stats:
            per_seed_sett_stats[sid] = {}
        for s in obs.get("settlements", []):
            key = (s["y"], s["x"])
            # Keep latest observation for each settlement position
            per_seed_sett_stats[sid][key] = s

    # Build per-seed local evidence from observations
    growth_front_maps = {}
    obs_overlays = {}
    sett_survivals = {}
    for seed_idx in range(seeds_count):
        terrain_si = np.array(initial_states[seed_idx]["grid"], dtype=int)
        seed_obs = [o for o in observations if o.get("seed_index") == seed_idx]
        if seed_obs:
            growth_front_maps[seed_idx] = build_growth_front_map(seed_obs, terrain_si)
        obs_overlays[seed_idx] = build_obs_overlay(observations, terrain_si, seed_idx)
        sett_survivals[seed_idx] = build_sett_survival(
            observations, initial_states[seed_idx]["settlements"], seed_idx
        )

    # Generate and submit predictions
    print("\n" + "=" * 60)
    print("Generating and submitting predictions")
    print("=" * 60)

    for seed_idx in range(seeds_count):
        state = initial_states[seed_idx]
        acc = accumulators[seed_idx] if seed_idx < len(accumulators) else None
        sett_stats = per_seed_sett_stats.get(seed_idx)

        print(f"\nSeed {seed_idx}:")
        if sett_stats:
            alive = sum(1 for s in sett_stats.values() if s.get("alive", True))
            print(f"  Settlement stats: {len(sett_stats)} observed ({alive} alive)")

        # Use Gemini-optimized prediction with variance regime
        if variance_regime:
            print(f"  Using gemini_predict with variance regime={variance_regime}")
        else:
            print(f"  Using gemini_predict (distance-aware multipliers + selective smoothing)")
        # Estimate vigor from FK bucket observations
        est_vigor = None
        if fk_buckets and hasattr(fk_buckets, 'counts'):
            sett_obs = sum(v[1] for v in fk_buckets.counts.values())
            total_obs = sum(fk_buckets.totals.values())
            if total_obs > 0:
                est_vigor = sett_obs / total_obs
                if seed_idx == 0:
                    print(f"  Estimated vigor: {est_vigor:.4f}")

        prediction = gemini_predict(
            state, global_mult, fk_buckets,
            multi_store=multi_store,
            variance_regime=variance_regime,
            est_vigor=est_vigor,
            growth_front_map=growth_front_maps.get(seed_idx),
            obs_overlay=obs_overlays.get(seed_idx),
            sett_survival=sett_survivals.get(seed_idx),
        )

        errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
        if errors:
            print(f"  Validation errors: {errors}")
            # Try to fix
            prediction = apply_floor(prediction)
            errors = validate_prediction(prediction, detail["map_height"], detail["map_width"])
            if errors:
                print(f"  Still invalid after fix: {errors}")
                continue

        resp = client.submit(round_id, seed_idx, prediction.tolist())
        print(f"  Submitted: {resp.get('status', 'unknown')}")

    # Save predictions
    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAll predictions submitted for round {round_id}")


def main():
    parser = argparse.ArgumentParser(description="Submit Astar Island predictions")
    parser.add_argument("--dry-run", action="store_true", help="Plan without API calls")
    parser.add_argument("--static-only", action="store_true", help="Submit static prior only")
    parser.add_argument("--uniform", action="store_true", help="Submit uniform baseline")
    parser.add_argument("--round-id", type=str, help="Specific round UUID")
    parser.add_argument("--strategy", choices=["grid", "multi-sample", "adaptive"],
                        default="adaptive",
                        help="Exploration strategy: grid (3x3), multi-sample (entropy), "
                             "or adaptive (auto-detect regime, default)")
    args = parser.parse_args()

    client = AstarIslandClient()

    if args.round_id:
        round_id = args.round_id
    else:
        active = client.get_active_round()
        if not active:
            print("No active round found.")
            rounds = client.get_rounds()
            if rounds:
                print(f"Most recent round: {rounds[-1].get('round_number')} ({rounds[-1]['id']})")
                print("Use --round-id to target a specific round")
            return
        round_id = active["id"]
        print(f"Active round: {active['round_number']} ({round_id})")

    detail = client.get_round_detail(round_id)
    print(f"Map: {detail['map_width']}x{detail['map_height']}, {detail['seeds_count']} seeds")

    if args.uniform:
        print("\nSubmitting uniform baseline...")
        submit_uniform(client, round_id, detail["seeds_count"])
    elif args.static_only:
        print("\nSubmitting static prior predictions...")
        submit_static_prior(client, round_id, detail)
    else:
        print(f"\nRunning full pipeline (strategy: {args.strategy})...")
        submit_full_pipeline(client, round_id, detail, dry_run=args.dry_run,
                             strategy=args.strategy)


if __name__ == "__main__":
    main()
