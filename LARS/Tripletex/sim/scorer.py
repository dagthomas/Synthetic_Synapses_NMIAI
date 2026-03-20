"""Scoring calculation for simulated submissions.

Replicates the competition scoring formula:
  final_score = correctness * tier_multiplier + efficiency_bonus (if correctness = 1.0)
"""

import math


def calculate_score(
    total_points: int,
    max_points: int,
    tier: int,
    api_calls: int,
    api_errors: int,
    baseline_calls: int,
) -> dict:
    """Calculate the final score for a simulated submission.

    Args:
        total_points: Points earned from field checks.
        max_points: Maximum possible points.
        tier: Task tier (1, 2, or 3).
        api_calls: Number of API calls the agent made.
        api_errors: Number of 4xx errors the agent encountered.
        baseline_calls: Minimum API calls for a perfect solution.

    Returns:
        {
            "correctness": float (0.0-1.0),
            "tier_multiplier": int,
            "base_score": float,
            "efficiency_bonus": float,
            "final_score": float,
            "max_possible": float,
        }
    """
    if max_points == 0:
        correctness = 0.0
    else:
        correctness = total_points / max_points

    tier_multiplier = tier
    base_score = correctness * tier_multiplier

    # Efficiency bonus only applies at perfect correctness
    efficiency_bonus = 0.0
    if correctness == 1.0 and baseline_calls > 0:
        # Call efficiency: ratio of baseline to actual calls
        call_ratio = baseline_calls / max(api_calls, 1)
        call_efficiency = min(call_ratio, 1.0)  # cap at 1.0

        # Error penalty: each error reduces bonus
        error_penalty = api_errors * 0.15

        # Efficiency bonus = up to tier_multiplier extra points
        raw_bonus = call_efficiency * tier_multiplier
        efficiency_bonus = max(0.0, raw_bonus - error_penalty)

    final_score = base_score + efficiency_bonus
    max_possible = tier_multiplier * 2  # perfect score + perfect efficiency

    return {
        "correctness": round(correctness, 4),
        "tier_multiplier": tier_multiplier,
        "base_score": round(base_score, 2),
        "efficiency_bonus": round(efficiency_bonus, 2),
        "final_score": round(final_score, 2),
        "max_possible": max_possible,
    }
