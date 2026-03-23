//! Core Monte Carlo simulation engine.
//!
//! Combines the one-shot model (sim_model.py) with year-by-year dynamics (sca.rs):
//! - Settlement survival with spatial features (food, coastal, clustering)
//! - Faction-based conflict between different-owner settlements
//! - Expansion with Gaussian-power distance decay
//! - Port upgrade and trade mechanics
//! - Winter severity
//! - Forest clearing and reclamation
//! - Ruin reclamation

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use super::cell::*;
use super::grid::Grid;
use super::params::SimParams;
use super::terrain::{compute_settlement_features, is_coastal, SettlementFeatures, SettlementInfo};

const SIM_YEARS: usize = 50;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

/// Tracked settlement during year-by-year simulation.
#[derive(Clone)]
struct LiveSettlement {
    y: usize,
    x: usize,
    faction: u8,
    is_port: bool,
    food_r2: f64,
    is_coastal: bool,
}

/// Run a single Monte Carlo simulation, returning the final grid state.
///
/// This uses the extended year-by-year model with factions.
pub fn simulate_single(
    initial_grid: &Grid,
    settlements: &[SettlementInfo],
    params: &SimParams,
    seed: u64,
) -> Grid {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let h = initial_grid.height;
    let w = initial_grid.width;

    // Initialize terrain class grid
    let mut grid = initial_grid.clone();

    // Precompute static masks
    let coastal_mask: Vec<bool> = (0..h)
        .flat_map(|y| (0..w).map(move |x| is_coastal(initial_grid, y, x)))
        .collect();
    let buildable: Vec<bool> = initial_grid
        .cells
        .iter()
        .map(|&c| c != OCEAN && c != MOUNTAIN)
        .collect();
    let orig_forest: Vec<bool> = initial_grid.cells.iter().map(|&c| c == FOREST).collect();

    // Initialize live settlements with faction IDs
    let mut live_setts: Vec<LiveSettlement> = Vec::with_capacity(settlements.len() * 2);
    let features = compute_settlement_features(initial_grid, settlements);
    let mut next_faction: u8 = 1;
    for (i, s) in settlements.iter().enumerate() {
        live_setts.push(LiveSettlement {
            y: s.y,
            x: s.x,
            faction: next_faction,
            is_port: s.has_port,
            food_r2: features.food_r2[i],
            is_coastal: features.is_coastal[i],
        });
        next_faction = next_faction.wrapping_add(1);
        if next_faction == 0 {
            next_faction = 1;
        }
    }

    // Year-by-year simulation
    for _year in 0..SIM_YEARS {
        let mut dead_indices: Vec<usize> = Vec::new();
        let mut new_setts: Vec<LiveSettlement> = Vec::new();

        // ── 1. Settlement survival + faction conflict + winter ──
        let n_alive = live_setts.len();
        for si in 0..n_alive {
            let s = &live_setts[si];
            let idx = s.y * w + s.x;

            // Cluster count (same-faction neighbors within d≤3)
            let mut cluster_r3 = 0.0;
            let mut same_neighbors_d2 = 0i32;
            let mut has_enemy_in_range = false;
            for sj in 0..n_alive {
                if si == sj {
                    continue;
                }
                let other = &live_setts[sj];
                let d = (s.y as i32 - other.y as i32).abs()
                    + (s.x as i32 - other.x as i32).abs();
                if d <= 3 {
                    cluster_r3 += 1.0;
                }
                if d <= 2 && other.faction == s.faction {
                    same_neighbors_d2 += 1;
                }
                if other.faction != s.faction && (d as f64) <= params.faction_range {
                    has_enemy_in_range = true;
                }
            }

            // Faction conflict check
            if has_enemy_in_range
                && params.faction_kill_prob > 0.001
                && rand::Rng::gen::<f64>(&mut rng) < params.faction_kill_prob
            {
                // Killed by faction conflict → ruin
                grid.cells[idx] = RUIN;
                dead_indices.push(si);
                continue;
            }

            // Compute per-year survival from base logit
            let food_norm = s.food_r2 / 12.0;
            let coastal_val = if s.is_coastal { 1.0 } else { 0.0 };
            let cluster_norm = cluster_r3 / 3.0;

            let logits = params.base_survival
                + params.food_coeff * food_norm
                + params.coastal_mod * coastal_val
                + params.cluster_pen * cluster_norm
                + params.cluster_quad * (cluster_r3 - params.cluster_optimal).powi(2) / 9.0
                + params.neighbor_survival_bonus * same_neighbors_d2 as f64
                + if has_forest_nearby(&grid, w, h, s.y, s.x) {
                    params.forest_survival_bonus
                } else {
                    0.0
                };

            // Trade bonus for ports with nearby friendly ports
            let trade_mod = if s.is_port && params.trade_bonus > 0.001 {
                let has_trade_partner = live_setts.iter().enumerate().any(|(j, other)| {
                    j != si
                        && other.is_port
                        && other.faction == s.faction
                        && ((s.y as i32 - other.y as i32).abs()
                            + (s.x as i32 - other.x as i32).abs())
                            <= 8
                });
                if has_trade_partner {
                    params.trade_bonus
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // 50-year compound survival → per-year
            let compound_surv = sigmoid(logits + trade_mod);
            let per_year_surv = compound_surv.powf(1.0 / SIM_YEARS as f64);

            // Winter penalty
            let winter_death = params.winter_severity * rand::Rng::gen::<f64>(&mut rng);
            let final_surv = (per_year_surv - winter_death).clamp(0.01, 0.999);

            if rand::Rng::gen::<f64>(&mut rng) > final_surv {
                // Settlement dies
                if rand::Rng::gen::<f64>(&mut rng) < params.ruin_rate {
                    grid.cells[idx] = RUIN;
                } else {
                    grid.cells[idx] = EMPTY;
                }
                dead_indices.push(si);
                continue;
            }

            // Port upgrade for surviving coastal settlements
            if !s.is_port && coastal_mask[idx] && params.port_upgrade_prob > 0.001 {
                if rand::Rng::gen::<f64>(&mut rng) < params.port_upgrade_prob {
                    grid.cells[idx] = PORT;
                    // Mark port upgrade (applied after loop)
                }
            }
        }

        // Remove dead settlements (reverse order)
        dead_indices.sort_unstable();
        dead_indices.dedup();
        for &di in dead_indices.iter().rev() {
            live_setts.swap_remove(di);
        }

        // Update port status for upgraded settlements
        for s in live_setts.iter_mut() {
            if grid.get(s.y, s.x) == PORT {
                s.is_port = true;
            }
        }

        // ── 2. Compute nearest alive distance ──
        let alive_positions: Vec<(usize, usize)> =
            live_setts.iter().map(|s| (s.y, s.x)).collect();
        let nearest_dist = super::terrain::manhattan_distance_map(&alive_positions, h, w);

        // ── 3. Expansion ──
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                if !buildable[idx] {
                    continue;
                }
                let cell = grid.cells[idx];
                if cell == SETTLEMENT || cell == PORT {
                    continue;
                }

                let d = nearest_dist[idx];
                if d > params.max_reach || d == 0.0 {
                    continue;
                }

                // Gaussian-power expansion probability
                let normalized_d = d / params.expansion_scale;
                let mut exp_prob =
                    params.expansion_str * (-normalized_d.powf(params.decay_power)).exp();

                // Forest resistance
                if cell == FOREST {
                    exp_prob *= 1.0 - params.forest_resist;
                }

                // Port spawn multiplier
                if params.port_spawn_multiplier != 1.0 {
                    if let Some(nearest) = find_nearest_settlement(&live_setts, r, c) {
                        if nearest.is_port {
                            exp_prob *= params.port_spawn_multiplier;
                        }
                    }
                }

                // Forest repro bonus
                if params.forest_repro_bonus > 0.001 && has_forest_nearby(&grid, w, h, r, c) {
                    exp_prob += params.forest_repro_bonus;
                }

                // Per-year spawn rate (compound)
                let per_year_exp =
                    1.0 - (1.0 - exp_prob.clamp(0.0, 0.5)).powf(1.0 / SIM_YEARS as f64);

                if rand::Rng::gen::<f64>(&mut rng) < per_year_exp {
                    // Determine faction from nearest settlement
                    let faction = find_nearest_settlement(&live_setts, r, c)
                        .map(|s| s.faction)
                        .unwrap_or(0);

                    let is_port_new =
                        coastal_mask[idx] && rand::Rng::gen::<f64>(&mut rng) < params.port_factor;

                    if is_port_new {
                        grid.cells[idx] = PORT;
                    } else {
                        grid.cells[idx] = SETTLEMENT;
                    }

                    // Expansion death
                    if rand::Rng::gen::<f64>(&mut rng) < params.exp_death {
                        if rand::Rng::gen::<f64>(&mut rng) < params.ruin_rate {
                            grid.cells[idx] = RUIN;
                        } else {
                            grid.cells[idx] = EMPTY;
                        }
                    } else {
                        // Compute food for new settlement
                        let food = compute_food_r2(&grid, h, w, r, c);
                        new_setts.push(LiveSettlement {
                            y: r,
                            x: c,
                            faction,
                            is_port: is_port_new,
                            food_r2: food,
                            is_coastal: coastal_mask[idx],
                        });
                    }
                }

                // Ruin reclamation
                if cell == RUIN && d <= 3.0 && params.ruin_to_settlement_prob > 0.001 {
                    if rand::Rng::gen::<f64>(&mut rng) < params.ruin_to_settlement_prob {
                        let faction = find_nearest_settlement(&live_setts, r, c)
                            .map(|s| s.faction)
                            .unwrap_or(0);
                        grid.cells[idx] = SETTLEMENT;
                        let food = compute_food_r2(&grid, h, w, r, c);
                        new_setts.push(LiveSettlement {
                            y: r,
                            x: c,
                            faction,
                            is_port: false,
                            food_r2: food,
                            is_coastal: coastal_mask[idx],
                        });
                    }
                }
            }
        }

        // Add new settlements
        live_setts.extend(new_setts);

        // ── 4. Forest clearing near settlements ──
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                if grid.cells[idx] != FOREST {
                    continue;
                }
                let d = nearest_dist[idx];
                let clear_prob = params.forest_clear * (-(d / 3.0).powi(2)).exp();
                if rand::Rng::gen::<f64>(&mut rng) < clear_prob / SIM_YEARS as f64 {
                    grid.cells[idx] = EMPTY;
                }
            }
        }

        // ── 5. Forest reclamation on distant empty cells ──
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                if grid.cells[idx] != EMPTY {
                    continue;
                }
                if !buildable[idx] {
                    continue;
                }
                let d = nearest_dist[idx];
                let base = if orig_forest[idx] {
                    params.forest_reclaim * 2.0
                } else {
                    params.forest_reclaim * 0.5
                };
                let reclaim_prob = base * (d / 5.0).clamp(0.0, 1.0);
                if rand::Rng::gen::<f64>(&mut rng) < reclaim_prob / SIM_YEARS as f64 {
                    grid.cells[idx] = FOREST;
                }
            }
        }
    }

    grid
}

/// Run Monte Carlo simulations and return H×W×6 probability tensor.
pub fn simulate_monte_carlo(
    initial_grid: &Grid,
    settlements: &[SettlementInfo],
    params: &SimParams,
    n_runs: usize,
    base_seed: u64,
) -> Vec<[f64; NUM_CLASSES]> {
    let h = initial_grid.height;
    let w = initial_grid.width;
    let n = h * w;

    let mut counts = vec![[0u32; NUM_CLASSES]; n];

    for run in 0..n_runs {
        let seed = base_seed
            .wrapping_add(run as u64)
            .wrapping_mul(0x517cc1b727220a95);
        let final_grid = simulate_single(initial_grid, settlements, params, seed);

        for i in 0..n {
            let cls = to_prediction_class(final_grid.cells[i]);
            counts[i][cls] += 1;
        }
    }

    let inv = 1.0 / n_runs as f64;
    counts
        .iter()
        .map(|c| {
            let mut probs = [0.0; NUM_CLASSES];
            for k in 0..NUM_CLASSES {
                probs[k] = c[k] as f64 * inv;
            }
            probs
        })
        .collect()
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn has_forest_nearby(grid: &Grid, w: usize, h: usize, y: usize, x: usize) -> bool {
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            if dy.abs() + dx.abs() > 2 {
                continue;
            }
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                if grid.get(ny as usize, nx as usize) == FOREST {
                    return true;
                }
            }
        }
    }
    false
}

fn find_nearest_settlement(setts: &[LiveSettlement], y: usize, x: usize) -> Option<&LiveSettlement> {
    setts
        .iter()
        .min_by_key(|s| (s.y as i32 - y as i32).abs() + (s.x as i32 - x as i32).abs())
}

fn compute_food_r2(grid: &Grid, h: usize, w: usize, y: usize, x: usize) -> f64 {
    let mut food = 0;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            if dy.abs() + dx.abs() > 2 {
                continue;
            }
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                let cell = grid.get(ny as usize, nx as usize);
                if cell == FOREST || cell == PLAINS {
                    food += 1;
                }
            }
        }
    }
    food as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_single_returns_valid_grid() {
        let mut grid = Grid::new(10, 10);
        // Fill with plains
        for i in 0..100 {
            grid.cells[i] = PLAINS;
        }
        // Add ocean border (top row)
        for x in 0..10 {
            grid.set(0, x, OCEAN);
        }
        // Add settlement
        grid.set(5, 5, SETTLEMENT);

        let settlements = vec![SettlementInfo {
            x: 5,
            y: 5,
            has_port: false,
            alive: true,
        }];

        let params = SimParams::default();
        let result = simulate_single(&grid, &settlements, &params, 42);

        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
        // Ocean should stay ocean
        assert_eq!(result.get(0, 0), OCEAN);
    }

    #[test]
    fn test_monte_carlo_probs_sum_to_one() {
        let mut grid = Grid::new(10, 10);
        for i in 0..100 {
            grid.cells[i] = PLAINS;
        }
        grid.set(5, 5, SETTLEMENT);
        grid.set(3, 3, FOREST);

        let settlements = vec![SettlementInfo {
            x: 5,
            y: 5,
            has_port: false,
            alive: true,
        }];

        let params = SimParams::default();
        let probs = simulate_monte_carlo(&grid, &settlements, &params, 50, 42);

        for (i, p) in probs.iter().enumerate() {
            let sum: f64 = p.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Cell {i} probs don't sum to 1: {sum} ({p:?})"
            );
        }
    }
}
