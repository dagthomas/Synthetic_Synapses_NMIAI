"""Quick grid search of cluster_sett_boost values."""
import json
import time
from pathlib import Path

import numpy as np

from autoloop import DEFAULT_PARAMS
from autoloop_fast import FastHarness, ROUND_NAMES, BOOM_ROUNDS

def main():
    print("Loading harness...")
    harness = FastHarness(seeds_per_round=5)

    # Baseline: no cluster boost
    params = dict(DEFAULT_PARAMS)
    base_scores = harness.evaluate(params)
    print(f"Baseline: avg={base_scores['avg']:.3f} boom={base_scores['boom_avg']:.3f} "
          f"nonboom={base_scores['nonboom_avg']:.3f}")

    # Grid search cluster_sett_boost
    print(f"\n{'boost':>6} {'avg':>8} {'boom':>8} {'nonboom':>8} {'delta':>8}")
    print("-" * 42)
    for boost in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        params = dict(DEFAULT_PARAMS)
        params["cluster_sett_boost"] = boost
        scores = harness.evaluate(params)
        delta = scores["avg"] - base_scores["avg"]
        print(f"{boost:6.2f} {scores['avg']:8.3f} {scores['boom_avg']:8.3f} "
              f"{scores['nonboom_avg']:8.3f} {delta:+8.3f}")


if __name__ == "__main__":
    main()
