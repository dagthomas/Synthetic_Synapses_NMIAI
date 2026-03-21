"""Pre-compute simulator predictions for all rounds.

Run once to generate sim predictions, then autoloop can blend with sim_alpha
as a tunable parameter without the expensive fitting step.

Usage:
    python sim_precompute.py          # Fit all rounds, save predictions
    python sim_precompute.py --quick  # Fewer sims (faster)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim_data import load_round, ALL_ROUNDS
from sim_model import Simulator, compute_score
from sim_inference import fit_to_gt

CACHE_DIR = Path(__file__).parent / "data" / "sim_cache"


def precompute_all(n_sims_fit: int = 500, n_sims_pred: int = 2000,
                   max_evals: int = 400, seeds_per_round: int = 5):
    """Pre-compute and cache simulator predictions for all rounds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for rname in ALL_ROUNDS:
        print(f"\n{rname}:")

        # Fit params using seed 0
        rd0 = load_round(rname, 0)
        if rd0 is None or rd0.ground_truth is None:
            print("  Skipping (no GT)")
            continue

        t0 = time.perf_counter()
        params, fit_score = fit_to_gt(rd0, n_sims=n_sims_fit, max_evals=max_evals,
                                       verbose=False)
        t1 = time.perf_counter()
        print(f"  Fitted: score={fit_score:.2f} ({t1-t0:.0f}s)")

        # Generate predictions for all seeds
        seed_scores = []
        for si in range(seeds_per_round):
            rd = load_round(rname, si)
            if rd is None or rd.ground_truth is None:
                continue

            sim = Simulator(rd)
            pred = sim.run(params, n_sims=n_sims_pred, seed=42)
            score = compute_score(rd.ground_truth, pred)
            seed_scores.append(score)

            # Save prediction as numpy
            np.save(CACHE_DIR / f"{rname}_seed{si}_pred.npy", pred.astype(np.float32))

        # Save params and metadata
        meta = {
            "params": params,
            "fit_score": fit_score,
            "seed_scores": seed_scores,
            "avg_score": float(np.mean(seed_scores)) if seed_scores else 0.0,
            "n_sims_fit": n_sims_fit,
            "n_sims_pred": n_sims_pred,
        }
        (CACHE_DIR / f"{rname}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Seeds: {[f'{s:.1f}' for s in seed_scores]}")


def load_cached_prediction(round_name: str, seed_idx: int) -> np.ndarray | None:
    """Load pre-computed simulator prediction from cache."""
    path = CACHE_DIR / f"{round_name}_seed{seed_idx}_pred.npy"
    if path.exists():
        return np.load(path)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        precompute_all(n_sims_fit=300, n_sims_pred=1000, max_evals=200, seeds_per_round=1)
    else:
        precompute_all()
