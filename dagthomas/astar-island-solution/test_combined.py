"""Test combined: regime-conditional calibration + barrier correction.

Tests each improvement independently and combined.
"""
import time
import numpy as np
from autoloop import DEFAULT_PARAMS
from autoloop_fast import FastHarness, ROUND_NAMES, BOOM_ROUNDS


def evaluate_and_print(label, harness, params):
    scores = harness.evaluate(params)
    per_round = [scores[r] for r in ROUND_NAMES]
    print(f"  {label}: avg={scores['avg']:.3f} boom={scores['boom_avg']:.3f} "
          f"nonboom={scores['nonboom_avg']:.3f} std={np.std(per_round):.2f}")
    return scores


def main():
    params = dict(DEFAULT_PARAMS)

    # 1. Baseline
    print("1. BASELINE (standard cal, no barrier)")
    h0 = FastHarness(seeds_per_round=5)
    base = evaluate_and_print("baseline", h0, params)

    # 2. Regime-conditional only (sigma=0.04)
    print("\n2. REGIME-CONDITIONAL ONLY (sigma=0.04)")
    h1 = FastHarness(seeds_per_round=5, regime_conditional=True, regime_sigma=0.04)
    rc = evaluate_and_print("regime", h1, params)

    # 3. Barrier correction only (grid search)
    print("\n3. BARRIER CORRECTION ONLY")
    for bs in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        p = dict(params)
        p["barrier_strength"] = bs
        scores = h0.evaluate(p)
        delta = scores["avg"] - base["avg"]
        print(f"  barrier={bs:.1f}: avg={scores['avg']:.3f} delta={delta:+.3f}")

    # 4. Combined: regime + best barrier
    print("\n4. COMBINED (regime sigma=0.04 + barrier grid)")
    for bs in [0.0, 0.1, 0.2, 0.3, 0.5]:
        p = dict(params)
        p["barrier_strength"] = bs
        scores = h1.evaluate(p)
        delta_from_base = scores["avg"] - base["avg"]
        delta_from_rc = scores["avg"] - rc["avg"]
        print(f"  barrier={bs:.1f}: avg={scores['avg']:.3f} "
              f"vs_base={delta_from_base:+.3f} vs_rc={delta_from_rc:+.3f}")

    # 5. Per-round detail for best combined
    print("\n5. PER-ROUND DETAIL (regime + barrier=0.2)")
    p = dict(params)
    p["barrier_strength"] = 0.2
    scores = h1.evaluate(p)
    print(f"{'Round':<10} {'Combined':>8} {'Base':>8} {'Delta':>8}")
    print("-" * 38)
    for r in ROUND_NAMES:
        tag = " *" if r in BOOM_ROUNDS else ""
        print(f"{r:<10} {scores[r]:8.2f} {base[r]:8.2f} {scores[r]-base[r]:+8.2f}{tag}")
    print("-" * 38)
    print(f"{'AVG':<10} {scores['avg']:8.3f} {base['avg']:8.3f} {scores['avg']-base['avg']:+8.3f}")
    print(f"{'BOOM':<10} {scores['boom_avg']:8.3f} {base['boom_avg']:8.3f} "
          f"{scores['boom_avg']-base['boom_avg']:+8.3f}")


if __name__ == "__main__":
    main()
