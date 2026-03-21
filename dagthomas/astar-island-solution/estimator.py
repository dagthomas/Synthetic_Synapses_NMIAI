"""Parameter estimation from simulation observations.

Analyzes settlement stats and terrain distributions from viewport observations
to infer the current round's hidden parameters and recommend adjustments to
the prediction engine.

Includes variance-based regime detection for distinguishing extreme boom (R7)
from moderate rounds (R5) using cross-sample settlement variance.
"""
import numpy as np

from config import CLASS_NAMES, NUM_CLASSES
from utils import terrain_to_class


def compute_stochastic_variance(multi_store, seed_feature_keys: dict = None) -> dict:
    """Compute per-feature-key settlement variance across multi-samples.

    Args:
        multi_store: MultiSampleStore with multiple grids per (seed, viewport)
        seed_feature_keys: Optional dict {seed_idx: fkeys} for FK-level analysis

    Returns dict: {
        'per_fk': {feature_key: {mean_sett_pct, variance_sett, max_sett_pct, n_samples}},
        'overall': {avg_variance, max_variance, avg_sett_pct, max_sett_pct, n_viewports},
    }
    """
    if multi_store is None:
        return {'per_fk': {}, 'overall': {
            'avg_variance': 0.0, 'max_variance': 0.0,
            'avg_sett_pct': 0.0, 'max_sett_pct': 0.0, 'n_viewports': 0,
        }}

    per_fk = multi_store.get_per_fk_variance()
    overall = multi_store.get_overall_variance()

    return {'per_fk': per_fk, 'overall': overall}


def detect_regime_from_variance(multi_store, observed_sett_pct: float) -> str:
    """Standalone regime detection from multi-sample variance.

    Args:
        multi_store: MultiSampleStore (or None)
        observed_sett_pct: fraction of observed cells that are settlement/port

    Returns: 'EXTREME_BOOM', 'BOOM', 'MODERATE', 'COLLAPSE', or 'UNKNOWN'
    """
    if multi_store is None:
        # Without variance data, fall back to simple threshold
        if observed_sett_pct >= 0.15:
            return "BOOM"
        elif observed_sett_pct < 0.02:
            return "COLLAPSE"
        else:
            return "MODERATE"

    var_stats = multi_store.get_overall_variance()
    avg_variance = var_stats.get('avg_variance', 0.0)
    max_sett_pct = max(observed_sett_pct, var_stats.get('max_sett_pct', 0.0))

    HIGH_VARIANCE_THRESHOLD = 0.005
    HIGH_SETT_THRESHOLD = 0.15

    if avg_variance > HIGH_VARIANCE_THRESHOLD:
        return "EXTREME_BOOM"
    elif max_sett_pct >= HIGH_SETT_THRESHOLD or observed_sett_pct >= HIGH_SETT_THRESHOLD:
        return "BOOM"
    elif observed_sett_pct < 0.02:
        return "COLLAPSE"
    else:
        return "MODERATE"


# R1 baseline stats (what we expect if parameters are similar to Round 1)
R1_BASELINE = {
    "avg_population": 2.5,
    "avg_food": 0.4,
    "avg_wealth": 0.15,
    "settlement_survival_rate": 0.41,  # 41% of initial settlements survive
    "expansion_rate": 0.19,  # 19% of near-plains become settlements
    "forest_loss_rate": 0.21,  # 21% of near-forest becomes settlement
    "ruin_rate": 0.031,  # 3.1% of initial settlements become ruins
    "port_formation_rate": 0.12,  # 12% of coastal near-plains become ports
}


class ParameterEstimator:
    """Estimates hidden round parameters from observed simulation data."""

    def __init__(self):
        self.settlement_stats = []  # All observed settlement dicts
        self.observed_terrains = []  # (initial_class, observed_class, dist_to_initial_sett)
        self.viewport_terrain_counts = np.zeros(NUM_CLASSES)
        self.viewport_cells_total = 0
        self.faction_sizes = {}  # owner_id -> count

    def add_observation(self, result: dict, initial_grid: list[list[int]] = None,
                        initial_settlements: list[dict] = None):
        """Process one simulate() result.

        Args:
            result: The simulate API response (grid, settlements, viewport)
            initial_grid: The initial terrain grid for this seed (for comparison)
            initial_settlements: Initial settlement list for this seed
        """
        # Settlement stats
        for s in result.get("settlements", []):
            self.settlement_stats.append({
                "population": s.get("population", 0),
                "food": s.get("food", 0),
                "wealth": s.get("wealth", 0),
                "defense": s.get("defense", 0),
                "has_port": s.get("has_port", False),
                "alive": s.get("alive", True),
                "owner_id": s.get("owner_id"),
            })

            # Track faction sizes
            oid = s.get("owner_id")
            if oid is not None and s.get("alive", True):
                self.faction_sizes[oid] = self.faction_sizes.get(oid, 0) + 1

        # Terrain counts from viewport
        viewport = result.get("viewport", {})
        grid = result.get("grid", [])
        for row in grid:
            for code in row:
                cls = terrain_to_class(code)
                self.viewport_terrain_counts[cls] += 1
                self.viewport_cells_total += 1

        # Compare observed terrain to initial if available
        if initial_grid is not None and viewport:
            vx, vy = viewport.get("x", 0), viewport.get("y", 0)
            # Build initial settlement distance (simplified — just check if near)
            initial_sett_pos = set()
            if initial_settlements:
                for s in initial_settlements:
                    if s.get("alive", True):
                        initial_sett_pos.add((s["y"], s["x"]))

            for row_idx, row in enumerate(grid):
                for col_idx, code in enumerate(row):
                    my, mx = vy + row_idx, vx + col_idx
                    if 0 <= my < len(initial_grid) and 0 <= mx < len(initial_grid[0]):
                        initial_code = initial_grid[my][mx]
                        initial_cls = terrain_to_class(initial_code)
                        observed_cls = terrain_to_class(code)

                        # Manhattan distance to nearest initial settlement
                        min_dist = 999
                        for sy, sx in initial_sett_pos:
                            d = abs(my - sy) + abs(mx - sx)
                            min_dist = min(min_dist, d)

                        self.observed_terrains.append(
                            (initial_cls, observed_cls, min_dist))

    def detect_collapse(self) -> bool:
        """Detect if all/most settlements have collapsed.

        Returns True if settlement survival rate is very low,
        indicating a total collapse scenario.
        """
        if not self.settlement_stats:
            return False
        alive = [s for s in self.settlement_stats if s["alive"]]
        total = len(self.settlement_stats)
        if total < 3:
            return False
        survival_rate = len(alive) / total
        # Collapse if <10% survival (R3 had 0%)
        return survival_rate < 0.10

    def estimate(self) -> dict:
        """Compute parameter estimates from accumulated observations.

        Returns dict with:
            - stats: raw aggregated statistics
            - regime: descriptive labels (harsh/mild winter, high/low expansion, etc.)
            - prior_strength: recommended Dirichlet prior strength
            - adjustments: multipliers to apply to R1 priors
            - collapse_detected: whether total settlement collapse was detected
        """
        result = {"stats": {}, "regime": {}, "prior_strength": 3.0, "adjustments": {},
                  "collapse_detected": self.detect_collapse()}

        # --- Settlement stats ---
        alive = [s for s in self.settlement_stats if s["alive"]]
        dead = [s for s in self.settlement_stats if not s["alive"]]

        if alive:
            avg_pop = np.mean([s["population"] for s in alive])
            avg_food = np.mean([s["food"] for s in alive])
            avg_wealth = np.mean([s["wealth"] for s in alive])
            avg_defense = np.mean([s["defense"] for s in alive])

            result["stats"]["avg_population"] = round(avg_pop, 3)
            result["stats"]["avg_food"] = round(avg_food, 3)
            result["stats"]["avg_wealth"] = round(avg_wealth, 3)
            result["stats"]["avg_defense"] = round(avg_defense, 3)
            result["stats"]["alive_count"] = len(alive)
            result["stats"]["dead_count"] = len(dead)

            # Winter harshness: low food = harsh winters
            food_ratio = avg_food / R1_BASELINE["avg_food"]
            if food_ratio < 0.6:
                result["regime"]["winter"] = "very_harsh"
            elif food_ratio < 0.85:
                result["regime"]["winter"] = "harsh"
            elif food_ratio > 1.3:
                result["regime"]["winter"] = "mild"
            else:
                result["regime"]["winter"] = "normal"

            # Population growth: high pop = good growth conditions
            pop_ratio = avg_pop / R1_BASELINE["avg_population"]
            result["stats"]["pop_ratio_vs_r1"] = round(pop_ratio, 3)

        # --- Port/trade activity ---
        port_count = len([s for s in alive if s.get("has_port")])
        if alive:
            port_fraction = port_count / len(alive)
            result["stats"]["port_fraction"] = round(port_fraction, 3)
            if port_fraction > 0.3:
                result["regime"]["trade"] = "high"
            elif port_fraction < 0.1:
                result["regime"]["trade"] = "low"
            else:
                result["regime"]["trade"] = "normal"

        # --- Faction analysis ---
        if self.faction_sizes:
            sizes = sorted(self.faction_sizes.values(), reverse=True)
            result["stats"]["num_factions"] = len(sizes)
            result["stats"]["largest_faction"] = sizes[0]
            result["stats"]["avg_faction_size"] = round(np.mean(sizes), 1)

            if len(sizes) >= 2 and sizes[0] > 3 * sizes[1]:
                result["regime"]["conflict"] = "dominant_faction"
            elif len(sizes) > 10:
                result["regime"]["conflict"] = "fragmented"
            else:
                result["regime"]["conflict"] = "balanced"

        # --- Terrain transition analysis ---
        if self.observed_terrains:
            # Expansion rate: initial plains/empty near settlements -> settlement
            near_plains = [(i, o) for i, o, d in self.observed_terrains
                           if i == 0 and d <= 5]  # class 0 = empty/plains/ocean
            if near_plains:
                became_sett = sum(1 for _, o in near_plains if o == 1)
                became_port = sum(1 for _, o in near_plains if o == 2)
                expansion = (became_sett + became_port) / len(near_plains)
                result["stats"]["observed_expansion_rate"] = round(expansion, 3)

                exp_ratio = expansion / max(R1_BASELINE["expansion_rate"], 0.01)
                if exp_ratio > 1.5:
                    result["regime"]["expansion"] = "high"
                elif exp_ratio < 0.5:
                    result["regime"]["expansion"] = "low"
                else:
                    result["regime"]["expansion"] = "normal"

            # Settlement survival: initial settlement -> still settlement
            initial_setts = [(i, o) for i, o, d in self.observed_terrains
                             if i in (1, 2)]  # settlements/ports
            if initial_setts:
                survived = sum(1 for _, o in initial_setts if o in (1, 2))
                survival_rate = survived / len(initial_setts)
                result["stats"]["observed_survival_rate"] = round(survival_rate, 3)

                surv_ratio = survival_rate / max(R1_BASELINE["settlement_survival_rate"], 0.01)
                result["stats"]["survival_ratio_vs_r1"] = round(surv_ratio, 3)

            # Forest loss
            near_forest = [(i, o) for i, o, d in self.observed_terrains
                           if i == 4 and d <= 5]
            if near_forest:
                became_sett = sum(1 for _, o in near_forest if o in (1, 2))
                forest_loss = became_sett / len(near_forest)
                result["stats"]["observed_forest_loss_rate"] = round(forest_loss, 3)

        # --- Compute prior strength recommendation ---
        # If observations diverge from R1, trust observations more (lower prior)
        divergence_signals = []

        if "observed_expansion_rate" in result["stats"]:
            exp_ratio = result["stats"]["observed_expansion_rate"] / max(R1_BASELINE["expansion_rate"], 0.01)
            divergence_signals.append(abs(exp_ratio - 1.0))

        if "observed_survival_rate" in result["stats"]:
            surv_ratio = result["stats"]["observed_survival_rate"] / max(R1_BASELINE["settlement_survival_rate"], 0.01)
            divergence_signals.append(abs(surv_ratio - 1.0))

        if "pop_ratio_vs_r1" in result["stats"]:
            divergence_signals.append(abs(result["stats"]["pop_ratio_vs_r1"] - 1.0))

        if divergence_signals:
            avg_divergence = np.mean(divergence_signals)
            # High divergence -> low prior strength (trust observations more)
            # 0 divergence -> prior_strength = 3.0 (trust R1 priors)
            # 1.0 divergence -> prior_strength = 0.5 (mostly trust observations)
            result["prior_strength"] = max(0.5, 3.0 - 2.5 * min(avg_divergence, 1.0))
            result["stats"]["avg_divergence_from_r1"] = round(avg_divergence, 3)

        # --- Compute prior adjustments ---
        # These are multipliers that scale the R1 priors toward observed reality
        if "observed_survival_rate" in result["stats"] and R1_BASELINE["settlement_survival_rate"] > 0:
            sr = result["stats"]["observed_survival_rate"]
            r1_sr = R1_BASELINE["settlement_survival_rate"]
            # If survival is higher than R1, boost settlement prob, reduce empty/ruin
            result["adjustments"]["settlement_survival"] = round(sr / r1_sr, 3)

        if "observed_expansion_rate" in result["stats"] and R1_BASELINE["expansion_rate"] > 0:
            er = result["stats"]["observed_expansion_rate"]
            r1_er = R1_BASELINE["expansion_rate"]
            result["adjustments"]["expansion"] = round(er / r1_er, 3)

        if "observed_forest_loss_rate" in result["stats"] and R1_BASELINE["forest_loss_rate"] > 0:
            fl = result["stats"]["observed_forest_loss_rate"]
            r1_fl = R1_BASELINE["forest_loss_rate"]
            result["adjustments"]["forest_loss"] = round(fl / r1_fl, 3)

        return result

    def detect_regime_enhanced(self, multi_store=None) -> str:
        """Enhanced regime detection using mean + variance from multi-samples.

        Returns one of: 'EXTREME_BOOM', 'BOOM', 'MODERATE', 'COLLAPSE', 'UNKNOWN'
        """
        est = self.estimate()

        # Basic collapse check
        if est.get("collapse_detected"):
            return "COLLAPSE"

        # Observed settlement percentage from terrain counts
        if self.viewport_cells_total > 0:
            observed_sett_pct = (
                self.viewport_terrain_counts[1] + self.viewport_terrain_counts[2]
            ) / self.viewport_cells_total
        else:
            return "UNKNOWN"

        # Variance-based detection from multi-samples
        avg_variance = 0.0
        max_sett_pct = observed_sett_pct
        if multi_store is not None:
            var_stats = multi_store.get_overall_variance()
            avg_variance = var_stats.get('avg_variance', 0.0)
            max_sett_pct = max(max_sett_pct, var_stats.get('max_sett_pct', 0.0))

        # Decision tree:
        # High variance + moderate mean → EXTREME_BOOM (R7-like)
        # High mean + low variance → BOOM (R6-like)
        # Low mean + low variance → MODERATE or COLLAPSE
        HIGH_VARIANCE_THRESHOLD = 0.005  # ~7% std dev in settlement rate
        HIGH_SETT_THRESHOLD = 0.15       # >15% settlement cells

        if avg_variance > HIGH_VARIANCE_THRESHOLD and observed_sett_pct < HIGH_SETT_THRESHOLD:
            return "EXTREME_BOOM"
        elif avg_variance > HIGH_VARIANCE_THRESHOLD and observed_sett_pct >= HIGH_SETT_THRESHOLD:
            return "EXTREME_BOOM"
        elif observed_sett_pct >= HIGH_SETT_THRESHOLD:
            return "BOOM"
        elif observed_sett_pct < 0.02:
            return "COLLAPSE"
        else:
            return "MODERATE"

    def print_summary(self):
        """Print a human-readable summary of estimated parameters."""
        est = self.estimate()

        print("\n--- Parameter Estimates ---")
        if est.get("collapse_detected"):
            print("  *** COLLAPSE DETECTED — all/most settlements dead ***")

        stats = est["stats"]
        if stats.get("alive_count"):
            print(f"  Settlements: {stats['alive_count']} alive, {stats.get('dead_count', 0)} dead")
            print(f"  Avg population: {stats.get('avg_population', '?')} "
                  f"(R1: {R1_BASELINE['avg_population']})")
            print(f"  Avg food: {stats.get('avg_food', '?')} "
                  f"(R1: {R1_BASELINE['avg_food']})")
            print(f"  Avg wealth: {stats.get('avg_wealth', '?')}")

        if stats.get("port_fraction") is not None:
            print(f"  Port fraction: {stats['port_fraction']:.1%}")

        if stats.get("num_factions"):
            print(f"  Factions: {stats['num_factions']}, largest: {stats['largest_faction']}")

        if stats.get("observed_survival_rate") is not None:
            print(f"  Settlement survival: {stats['observed_survival_rate']:.1%} "
                  f"(R1: {R1_BASELINE['settlement_survival_rate']:.1%})")

        if stats.get("observed_expansion_rate") is not None:
            print(f"  Expansion rate: {stats['observed_expansion_rate']:.1%} "
                  f"(R1: {R1_BASELINE['expansion_rate']:.1%})")

        regime = est["regime"]
        if regime:
            print(f"  Regime: {regime}")

        print(f"  Recommended prior strength: {est['prior_strength']:.2f} (R1 default: 3.0)")

        if est["adjustments"]:
            print(f"  Adjustments vs R1: {est['adjustments']}")
