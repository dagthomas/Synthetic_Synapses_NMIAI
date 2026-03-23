//! Hidden simulation parameters per round.
//!
//! Base 16 params from sim_model.py + 10 extended params from sca.rs
//! for richer faction dynamics, port mechanics, winter, and trade.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// The 26 hidden parameters controlling the simulation.
///
/// Original 16 (from sim_model.py):
///   base_survival, expansion_str, expansion_scale, decay_power, max_reach,
///   coastal_mod, food_coeff, cluster_pen, cluster_optimal, cluster_quad,
///   ruin_rate, port_factor, forest_resist, forest_clear, forest_reclaim, exp_death
///
/// Extended 10 (inspired by sca.rs):
///   neighbor_survival_bonus, faction_kill_prob, faction_range,
///   forest_survival_bonus, forest_repro_bonus,
///   ruin_to_settlement_prob, port_upgrade_prob, port_spawn_multiplier,
///   winter_severity, trade_bonus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimParams {
    // ── Original 16 parameters ──────────────────────────────────────

    /// Logit for overall settlement survival probability.
    pub base_survival: f64,
    /// Peak expansion probability (at distance=0).
    pub expansion_str: f64,
    /// Distance scale for Gaussian decay in expansion.
    pub expansion_scale: f64,
    /// Exponent on distance: exp(-(d/scale)^power); >1 = sharper cutoff.
    pub decay_power: f64,
    /// Hard cutoff: zero expansion beyond this distance.
    pub max_reach: f64,
    /// Coastal survival logit adjustment.
    pub coastal_mod: f64,
    /// Food adjacency survival boost coefficient.
    pub food_coeff: f64,
    /// Cluster density penalty on survival.
    pub cluster_pen: f64,
    /// Inverted-U peak density for clustering.
    pub cluster_optimal: f64,
    /// Quadratic penalty strength for clustering.
    pub cluster_quad: f64,
    /// Dead settlement → ruin probability (vs empty).
    pub ruin_rate: f64,
    /// Coastal expansion → port probability.
    pub port_factor: f64,
    /// Forest resistance to expansion (dampening).
    pub forest_resist: f64,
    /// Forest clearing rate near active settlements.
    pub forest_clear: f64,
    /// Empty → forest reclamation rate.
    pub forest_reclaim: f64,
    /// Probability expanded settlement immediately dies.
    pub exp_death: f64,

    // ── Extended parameters (factions, ports, winter, trade) ─────────

    /// Survival modifier per nearby same-faction settlement (d≤2).
    /// Positive = cooperative, negative = overcrowding penalty.
    pub neighbor_survival_bonus: f64,
    /// Probability of faction conflict killing a settlement when enemy is in range.
    pub faction_kill_prob: f64,
    /// Range for faction conflict (Manhattan distance).
    pub faction_range: f64,
    /// Survival bonus when settlement has forest within d≤2.
    pub forest_survival_bonus: f64,
    /// Spawn rate bonus for cells near forest.
    pub forest_repro_bonus: f64,
    /// Probability that a ruin becomes a settlement (if near existing settlement d≤3).
    pub ruin_to_settlement_prob: f64,
    /// Probability per year that a coastal settlement upgrades to port.
    pub port_upgrade_prob: f64,
    /// Spawn rate multiplier near ports vs regular settlements. >1 means ports attract growth.
    pub port_spawn_multiplier: f64,
    /// Winter severity (0-1). Higher = more starvation deaths each year.
    pub winter_severity: f64,
    /// Trade bonus for ports within range. Boosts food/survival of trading ports.
    pub trade_bonus: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            // Original 16
            base_survival: -0.5,
            expansion_str: 0.35,
            expansion_scale: 2.0,
            decay_power: 2.0,
            max_reach: 5.0,
            coastal_mod: -0.3,
            food_coeff: 0.5,
            cluster_pen: -0.3,
            cluster_optimal: 2.0,
            cluster_quad: -0.2,
            ruin_rate: 0.5,
            port_factor: 0.25,
            forest_resist: 0.3,
            forest_clear: 0.2,
            forest_reclaim: 0.05,
            exp_death: 0.3,
            // Extended 10
            neighbor_survival_bonus: 0.03,
            faction_kill_prob: 0.04,
            faction_range: 3.0,
            forest_survival_bonus: 0.03,
            forest_repro_bonus: 0.02,
            ruin_to_settlement_prob: 0.05,
            port_upgrade_prob: 0.08,
            port_spawn_multiplier: 1.3,
            winter_severity: 0.1,
            trade_bonus: 0.05,
        }
    }
}

/// Regime presets for creating interesting rounds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Regime {
    Collapse,
    Moderate,
    Boom,
    Random,
}

/// Parameter range specification for UI sliders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub name: &'static str,
    pub default: f64,
    pub min: f64,
    pub max: f64,
    pub description: &'static str,
}

/// All 26 parameter ranges for admin UI.
pub fn param_ranges() -> Vec<ParamRange> {
    vec![
        // Original 16
        ParamRange { name: "base_survival", default: -0.5, min: -6.0, max: 3.0, description: "Logit for settlement survival" },
        ParamRange { name: "expansion_str", default: 0.35, min: 0.005, max: 0.95, description: "Peak expansion probability" },
        ParamRange { name: "expansion_scale", default: 2.0, min: 0.5, max: 8.0, description: "Distance decay scale" },
        ParamRange { name: "decay_power", default: 2.0, min: 1.0, max: 4.0, description: "Gaussian-power exponent" },
        ParamRange { name: "max_reach", default: 5.0, min: 1.5, max: 15.0, description: "Hard expansion cutoff" },
        ParamRange { name: "coastal_mod", default: -0.3, min: -3.0, max: 1.0, description: "Coastal survival modifier" },
        ParamRange { name: "food_coeff", default: 0.5, min: 0.0, max: 3.0, description: "Food adjacency boost" },
        ParamRange { name: "cluster_pen", default: -0.3, min: -2.0, max: 0.5, description: "Cluster density penalty" },
        ParamRange { name: "cluster_optimal", default: 2.0, min: 0.5, max: 5.0, description: "Optimal neighbor count" },
        ParamRange { name: "cluster_quad", default: -0.2, min: -2.0, max: 0.0, description: "Quadratic penalty strength" },
        ParamRange { name: "ruin_rate", default: 0.5, min: 0.01, max: 0.99, description: "Dead → ruin probability" },
        ParamRange { name: "port_factor", default: 0.25, min: 0.01, max: 1.0, description: "Coastal expansion → port" },
        ParamRange { name: "forest_resist", default: 0.3, min: 0.0, max: 0.95, description: "Forest blocks expansion" },
        ParamRange { name: "forest_clear", default: 0.2, min: 0.0, max: 0.8, description: "Forest clearing near settlements" },
        ParamRange { name: "forest_reclaim", default: 0.05, min: 0.0, max: 0.5, description: "Empty → forest reclamation" },
        ParamRange { name: "exp_death", default: 0.3, min: 0.0, max: 0.9, description: "Expanded settlement death" },
        // Extended 10
        ParamRange { name: "neighbor_survival_bonus", default: 0.03, min: -0.1, max: 0.15, description: "Same-faction neighbor survival mod" },
        ParamRange { name: "faction_kill_prob", default: 0.04, min: 0.0, max: 0.3, description: "Faction conflict kill chance" },
        ParamRange { name: "faction_range", default: 3.0, min: 1.0, max: 6.0, description: "Faction conflict range" },
        ParamRange { name: "forest_survival_bonus", default: 0.03, min: 0.0, max: 0.15, description: "Forest adjacency survival boost" },
        ParamRange { name: "forest_repro_bonus", default: 0.02, min: 0.0, max: 0.1, description: "Forest adjacency spawn boost" },
        ParamRange { name: "ruin_to_settlement_prob", default: 0.05, min: 0.0, max: 0.2, description: "Ruin → settlement reclaim" },
        ParamRange { name: "port_upgrade_prob", default: 0.08, min: 0.0, max: 0.25, description: "Coastal settlement → port upgrade" },
        ParamRange { name: "port_spawn_multiplier", default: 1.3, min: 0.5, max: 3.0, description: "Port spawn rate multiplier" },
        ParamRange { name: "winter_severity", default: 0.1, min: 0.0, max: 0.5, description: "Annual winter death rate" },
        ParamRange { name: "trade_bonus", default: 0.05, min: 0.0, max: 0.2, description: "Port trade survival bonus" },
    ]
}

impl SimParams {
    /// Generate random parameters uniformly within valid ranges.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Self {
            base_survival: rng.gen_range(-6.0..=3.0),
            expansion_str: rng.gen_range(0.005..=0.95),
            expansion_scale: rng.gen_range(0.5..=8.0),
            decay_power: rng.gen_range(1.0..=4.0),
            max_reach: rng.gen_range(1.5..=15.0),
            coastal_mod: rng.gen_range(-3.0..=1.0),
            food_coeff: rng.gen_range(0.0..=3.0),
            cluster_pen: rng.gen_range(-2.0..=0.5),
            cluster_optimal: rng.gen_range(0.5..=5.0),
            cluster_quad: rng.gen_range(-2.0..=0.0),
            ruin_rate: rng.gen_range(0.01..=0.99),
            port_factor: rng.gen_range(0.01..=1.0),
            forest_resist: rng.gen_range(0.0..=0.95),
            forest_clear: rng.gen_range(0.0..=0.8),
            forest_reclaim: rng.gen_range(0.0..=0.5),
            exp_death: rng.gen_range(0.0..=0.9),
            neighbor_survival_bonus: rng.gen_range(-0.1..=0.15),
            faction_kill_prob: rng.gen_range(0.0..=0.3),
            faction_range: rng.gen_range(1.0..=6.0),
            forest_survival_bonus: rng.gen_range(0.0..=0.15),
            forest_repro_bonus: rng.gen_range(0.0..=0.1),
            ruin_to_settlement_prob: rng.gen_range(0.0..=0.2),
            port_upgrade_prob: rng.gen_range(0.0..=0.25),
            port_spawn_multiplier: rng.gen_range(0.5..=3.0),
            winter_severity: rng.gen_range(0.0..=0.5),
            trade_bonus: rng.gen_range(0.0..=0.2),
        }
    }

    /// Generate parameters biased toward a specific regime.
    pub fn from_regime<R: Rng>(regime: Regime, rng: &mut R) -> Self {
        match regime {
            Regime::Collapse => Self {
                base_survival: rng.gen_range(-5.0..=-2.0),
                expansion_str: rng.gen_range(0.05..=0.3),
                expansion_scale: rng.gen_range(0.5..=2.0),
                decay_power: rng.gen_range(2.0..=4.0),
                max_reach: rng.gen_range(1.5..=4.0),
                coastal_mod: rng.gen_range(-2.0..=-0.5),
                food_coeff: rng.gen_range(0.0..=1.0),
                cluster_pen: rng.gen_range(-1.5..=-0.3),
                cluster_optimal: rng.gen_range(0.5..=2.0),
                cluster_quad: rng.gen_range(-1.5..=-0.3),
                ruin_rate: rng.gen_range(0.5..=0.95),
                port_factor: rng.gen_range(0.01..=0.3),
                forest_resist: rng.gen_range(0.4..=0.9),
                forest_clear: rng.gen_range(0.3..=0.8),
                forest_reclaim: rng.gen_range(0.05..=0.3),
                exp_death: rng.gen_range(0.5..=0.9),
                neighbor_survival_bonus: rng.gen_range(-0.05..=0.02),
                faction_kill_prob: rng.gen_range(0.1..=0.25),
                faction_range: rng.gen_range(2.0..=5.0),
                forest_survival_bonus: rng.gen_range(0.0..=0.05),
                forest_repro_bonus: rng.gen_range(0.0..=0.03),
                ruin_to_settlement_prob: rng.gen_range(0.0..=0.03),
                port_upgrade_prob: rng.gen_range(0.0..=0.05),
                port_spawn_multiplier: rng.gen_range(0.5..=1.0),
                winter_severity: rng.gen_range(0.2..=0.5),
                trade_bonus: rng.gen_range(0.0..=0.03),
            },
            Regime::Moderate => Self {
                base_survival: rng.gen_range(-1.5..=0.5),
                expansion_str: rng.gen_range(0.2..=0.6),
                expansion_scale: rng.gen_range(1.5..=4.0),
                decay_power: rng.gen_range(1.5..=3.0),
                max_reach: rng.gen_range(3.0..=7.0),
                coastal_mod: rng.gen_range(-1.0..=0.3),
                food_coeff: rng.gen_range(0.2..=1.5),
                cluster_pen: rng.gen_range(-1.0..=0.0),
                cluster_optimal: rng.gen_range(1.0..=3.5),
                cluster_quad: rng.gen_range(-1.0..=-0.1),
                ruin_rate: rng.gen_range(0.2..=0.7),
                port_factor: rng.gen_range(0.1..=0.5),
                forest_resist: rng.gen_range(0.1..=0.5),
                forest_clear: rng.gen_range(0.1..=0.4),
                forest_reclaim: rng.gen_range(0.02..=0.15),
                exp_death: rng.gen_range(0.2..=0.5),
                neighbor_survival_bonus: rng.gen_range(0.0..=0.06),
                faction_kill_prob: rng.gen_range(0.02..=0.1),
                faction_range: rng.gen_range(2.0..=4.0),
                forest_survival_bonus: rng.gen_range(0.01..=0.06),
                forest_repro_bonus: rng.gen_range(0.01..=0.05),
                ruin_to_settlement_prob: rng.gen_range(0.02..=0.1),
                port_upgrade_prob: rng.gen_range(0.03..=0.12),
                port_spawn_multiplier: rng.gen_range(1.0..=1.8),
                winter_severity: rng.gen_range(0.05..=0.2),
                trade_bonus: rng.gen_range(0.02..=0.08),
            },
            Regime::Boom => Self {
                base_survival: rng.gen_range(0.0..=2.5),
                expansion_str: rng.gen_range(0.5..=0.95),
                expansion_scale: rng.gen_range(3.0..=8.0),
                decay_power: rng.gen_range(1.0..=2.0),
                max_reach: rng.gen_range(6.0..=15.0),
                coastal_mod: rng.gen_range(-0.5..=0.5),
                food_coeff: rng.gen_range(0.5..=2.5),
                cluster_pen: rng.gen_range(-0.5..=0.3),
                cluster_optimal: rng.gen_range(2.0..=5.0),
                cluster_quad: rng.gen_range(-0.5..=0.0),
                ruin_rate: rng.gen_range(0.1..=0.5),
                port_factor: rng.gen_range(0.3..=0.9),
                forest_resist: rng.gen_range(0.0..=0.3),
                forest_clear: rng.gen_range(0.1..=0.5),
                forest_reclaim: rng.gen_range(0.01..=0.1),
                exp_death: rng.gen_range(0.05..=0.35),
                neighbor_survival_bonus: rng.gen_range(0.03..=0.12),
                faction_kill_prob: rng.gen_range(0.0..=0.05),
                faction_range: rng.gen_range(1.0..=3.0),
                forest_survival_bonus: rng.gen_range(0.02..=0.1),
                forest_repro_bonus: rng.gen_range(0.02..=0.08),
                ruin_to_settlement_prob: rng.gen_range(0.05..=0.15),
                port_upgrade_prob: rng.gen_range(0.05..=0.2),
                port_spawn_multiplier: rng.gen_range(1.3..=2.5),
                winter_severity: rng.gen_range(0.0..=0.1),
                trade_bonus: rng.gen_range(0.05..=0.15),
            },
            Regime::Random => Self::random(rng),
        }
    }

    /// Clamp all parameters to their valid ranges.
    pub fn validate(&mut self) {
        self.base_survival = self.base_survival.clamp(-6.0, 3.0);
        self.expansion_str = self.expansion_str.clamp(0.005, 0.95);
        self.expansion_scale = self.expansion_scale.clamp(0.5, 8.0);
        self.decay_power = self.decay_power.clamp(1.0, 4.0);
        self.max_reach = self.max_reach.clamp(1.5, 15.0);
        self.coastal_mod = self.coastal_mod.clamp(-3.0, 1.0);
        self.food_coeff = self.food_coeff.clamp(0.0, 3.0);
        self.cluster_pen = self.cluster_pen.clamp(-2.0, 0.5);
        self.cluster_optimal = self.cluster_optimal.clamp(0.5, 5.0);
        self.cluster_quad = self.cluster_quad.clamp(-2.0, 0.0);
        self.ruin_rate = self.ruin_rate.clamp(0.01, 0.99);
        self.port_factor = self.port_factor.clamp(0.01, 1.0);
        self.forest_resist = self.forest_resist.clamp(0.0, 0.95);
        self.forest_clear = self.forest_clear.clamp(0.0, 0.8);
        self.forest_reclaim = self.forest_reclaim.clamp(0.0, 0.5);
        self.exp_death = self.exp_death.clamp(0.0, 0.9);
        self.neighbor_survival_bonus = self.neighbor_survival_bonus.clamp(-0.1, 0.15);
        self.faction_kill_prob = self.faction_kill_prob.clamp(0.0, 0.3);
        self.faction_range = self.faction_range.clamp(1.0, 6.0);
        self.forest_survival_bonus = self.forest_survival_bonus.clamp(0.0, 0.15);
        self.forest_repro_bonus = self.forest_repro_bonus.clamp(0.0, 0.1);
        self.ruin_to_settlement_prob = self.ruin_to_settlement_prob.clamp(0.0, 0.2);
        self.port_upgrade_prob = self.port_upgrade_prob.clamp(0.0, 0.25);
        self.port_spawn_multiplier = self.port_spawn_multiplier.clamp(0.5, 3.0);
        self.winter_severity = self.winter_severity.clamp(0.0, 0.5);
        self.trade_bonus = self.trade_bonus.clamp(0.0, 0.2);
    }
}
