"""Clean A/B test: regime-conditional calibration.

Compares standard calibration vs vigor-weighted calibration across
multiple sigma values. Uses the FastHarness with regime_conditional flag.
"""
import time
from autoloop import DEFAULT_PARAMS
from autoloop_fast import FastHarness, ROUND_NAMES, BOOM_ROUNDS

import numpy as np


def main():
    params = dict(DEFAULT_PARAMS)

    # Baseline: standard calibration
    print("Loading BASELINE harness (standard calibration)...")
    t0 = time.time()
    h_base = FastHarness(seeds_per_round=5, regime_conditional=False)
    base = h_base.evaluate(params)
    print(f"  Baseline: avg={base['avg']:.3f} boom={base['boom_avg']:.3f} "
          f"nonboom={base['nonboom_avg']:.3f} ({time.time()-t0:.1f}s)")

    # Show estimated vigors
    print("\n  Estimated vigors from observations:")
    for rn in ROUND_NAMES:
        v = h_base.rounds[rn].get("est_vigor", "N/A")
        if isinstance(v, float):
            print(f"    {rn}: {v:.4f}")

    # Test regime-conditional at multiple sigma values
    print(f"\n{'sigma':>8} {'avg':>8} {'boom':>8} {'nonboom':>8} {'delta':>8}")
    print("-" * 44)

    for sigma in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.50]:
        t0 = time.time()
        h = FastHarness(seeds_per_round=5, regime_conditional=True, regime_sigma=sigma)
        scores = h.evaluate(params)
        elapsed = time.time() - t0
        delta = scores["avg"] - base["avg"]
        print(f"{sigma:8.2f} {scores['avg']:8.3f} {scores['boom_avg']:8.3f} "
              f"{scores['nonboom_avg']:8.3f} {delta:+8.3f}  ({elapsed:.1f}s)")

    # Per-round detail at best sigma
    best_sigma = 0.06
    print(f"\n--- Per-round detail at sigma={best_sigma} ---")
    h_best = FastHarness(seeds_per_round=5, regime_conditional=True, regime_sigma=best_sigma)
    scores = h_best.evaluate(params)
    print(f"{'Round':<10} {'Score':>8} {'Base':>8} {'Delta':>8} {'EstVigor':>10}")
    print("-" * 50)
    for r in ROUND_NAMES:
        tag = " *" if r in BOOM_ROUNDS else ""
        delta = scores[r] - base[r]
        ev = h_best.rounds[r].get("est_vigor", 0)
        print(f"{r:<10} {scores[r]:8.2f} {base[r]:8.2f} {delta:+8.2f} {ev:10.4f}{tag}")
    print("-" * 50)
    print(f"{'AVG':<10} {scores['avg']:8.3f} {base['avg']:8.3f} "
          f"{scores['avg'] - base['avg']:+8.3f}")
    print(f"{'BOOM':<10} {scores['boom_avg']:8.3f} {base['boom_avg']:8.3f} "
          f"{scores['boom_avg'] - base['boom_avg']:+8.3f}")
    print(f"{'NON-BOOM':<10} {scores['nonboom_avg']:8.3f} {base['nonboom_avg']:8.3f} "
          f"{scores['nonboom_avg'] - base['nonboom_avg']:+8.3f}")

    per_round = [scores[r] for r in ROUND_NAMES]
    base_per = [base[r] for r in ROUND_NAMES]
    print(f"\nStdDev: base={np.std(base_per):.2f} regime={np.std(per_round):.2f}")


if __name__ == "__main__":
    main()
