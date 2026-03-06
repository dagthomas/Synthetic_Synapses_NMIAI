"""Hardware profiles and compute budget estimates.

Each profile specifies max_states, chunk sizes, and time estimates for
different solver configurations. The actual runtime scales with memory
bandwidth (B200: ~8 TB/s vs 5090: ~1.8 TB/s) and VRAM (192GB vs 32GB).
"""
from dataclasses import dataclass


@dataclass
class HardwareProfile:
    name: str
    vram_gb: int
    # 3-bot joint DP limits
    joint3_max_states: int
    joint3_chunk_size: int       # max expanded states per chunk
    # 2-bot brute force limits
    brute2_max_states: int
    # Single-bot DP limits
    single_max_states: int
    # Time estimates (seconds per round at max states)
    est_3bot_per_round: float
    est_2bot_per_round: float
    est_1bot_per_round: float


RTX_5090 = HardwareProfile(
    name="RTX 5090",
    vram_gb=32,
    joint3_max_states=200_000,    # 200K × 343 = 68M expansion per round (~3 GB)
    joint3_chunk_size=50_000_000, # Process 50M states per chunk
    brute2_max_states=500_000,    # 500K × 49 = 24M expansion (~1.5 GB)
    single_max_states=1_000_000,  # 1M states for single bot
    est_3bot_per_round=0.5,       # ~150s for 300 rounds
    est_2bot_per_round=0.15,      # ~45s for 300 rounds
    est_1bot_per_round=0.05,      # ~15s for 300 rounds
)

B200 = HardwareProfile(
    name="NVIDIA B200",
    vram_gb=192,
    joint3_max_states=2_000_000,  # 2M × 343 = 686M expansion (~40 GB)
    joint3_chunk_size=500_000_000,
    brute2_max_states=5_000_000,  # 5M × 49 = 245M expansion (~15 GB)
    single_max_states=10_000_000,
    est_3bot_per_round=0.08,      # ~24s for 300 rounds
    est_2bot_per_round=0.025,
    est_1bot_per_round=0.008,
)


def estimate_time(profile: HardwareProfile, num_bots: int,
                  mode: str = 'sequential') -> dict:
    """Estimate training time for different solver configurations.

    Returns dict with time estimates in seconds for:
      - all_orderings: try all N! bot orderings
      - deep_refine: 50 refinement iterations
      - joint_3bot: 3-bot joint DP on all C(n,3) triples
      - total: sum of above
    """
    import math

    n_orderings = math.factorial(num_bots)
    n_triples = math.comb(num_bots, 3)
    n_pairs = math.comb(num_bots, 2)

    # All orderings: each ordering = num_bots × single-bot DP
    t_per_ordering = num_bots * 300 * profile.est_1bot_per_round
    t_orderings = n_orderings * t_per_ordering

    # Deep refine: 50 iterations × num_bots × single-bot DP
    t_refine = 50 * num_bots * 300 * profile.est_1bot_per_round

    # 3-bot joint refine on all triples
    t_joint3 = n_triples * 300 * profile.est_3bot_per_round

    # 2-bot brute force on all pairs
    t_brute2 = n_pairs * 300 * profile.est_2bot_per_round

    return {
        'all_orderings': t_orderings,
        'deep_refine_50': t_refine,
        'joint_3bot_all': t_joint3,
        'brute_2bot_all': t_brute2,
        'total': t_orderings + t_refine + t_joint3,
        'n_orderings': n_orderings,
        'n_triples': n_triples,
        'n_pairs': n_pairs,
    }


def print_estimates(num_bots: int):
    """Print time estimates for both hardware profiles."""
    for profile in [RTX_5090, B200]:
        est = estimate_time(profile, num_bots)
        print(f"\n{'='*50}")
        print(f"{profile.name} ({profile.vram_gb}GB VRAM)")
        print(f"  Bots: {num_bots}")
        print(f"  Max states (3-bot): {profile.joint3_max_states:,}")
        print(f"  Max states (2-bot): {profile.brute2_max_states:,}")
        print(f"  Max states (1-bot): {profile.single_max_states:,}")
        print(f"\n  All {est['n_orderings']} orderings: "
              f"{est['all_orderings']/60:.0f} min")
        print(f"  50-iter refinement: {est['deep_refine_50']/60:.0f} min")
        print(f"  3-bot joint ({est['n_triples']} triples): "
              f"{est['joint_3bot_all']/60:.0f} min")
        print(f"  2-bot brute ({est['n_pairs']} pairs): "
              f"{est['brute_2bot_all']/60:.0f} min")
        print(f"  Total deep treatment: {est['total']/3600:.1f} hours")


if __name__ == '__main__':
    for n in [3, 5, 7, 10]:
        print(f"\n{'#'*60}")
        print(f"  {n} BOTS")
        print(f"{'#'*60}")
        print_estimates(n)
