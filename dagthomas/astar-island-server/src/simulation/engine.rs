//! Core Monte Carlo simulation engine.
//!
//! Fast single-step model ported from sim_model.py:
//! 1. Settlement survival (sigmoid of spatial features)
//! 2. Dead settlements → ruin or empty
//! 3. Expansion with Gaussian-power distance decay
//! 4. Expansion death
//! 5. Forest clearing/reclamation
//!
//! Extended params (faction, winter, trade, port upgrade) are folded into
//! the survival/expansion probabilities rather than simulated year-by-year.

use rand::Rng as RandRng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use super::cell::*;
use super::grid::Grid;
use super::params::SimParams;
use super::terrain::{compute_settlement_features, is_coastal, manhattan_distance_map, SettlementInfo};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

/// Run a single Monte Carlo simulation, returning the final grid state.
/// Uses the fast single-step model (no year-by-year loop).
pub fn simulate_single(
    initial_grid: &Grid,
    settlements: &[SettlementInfo],
    params: &SimParams,
    seed: u64,
) -> Grid {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let h = initial_grid.height;
    let w = initial_grid.width;
    let n_sett = settlements.len();

    // Precompute features
    let features = compute_settlement_features(initial_grid, settlements);
    let cell_coastal: Vec<bool> = (0..h)
        .flat_map(|y| (0..w).map(move |x| is_coastal(initial_grid, y, x)))
        .collect();
    let cell_buildable: Vec<bool> = initial_grid.cells.iter().map(|&c| c != OCEAN && c != MOUNTAIN).collect();
    let cell_is_settlement: Vec<bool> = initial_grid.cells.iter().map(|&c| c == SETTLEMENT || c == PORT).collect();
    let cell_is_forest: Vec<bool> = initial_grid.cells.iter().map(|&c| c == FOREST).collect();
    let orig_forest: Vec<bool> = cell_is_forest.clone();

    // Map initial terrain to class
    let terrain_class: Vec<u8> = initial_grid.cells.iter().map(|&c| to_prediction_class(c) as u8).collect();

    // Expand mask: buildable AND not already a settlement
    let expand_mask: Vec<bool> = cell_buildable.iter().zip(cell_is_settlement.iter())
        .map(|(&b, &s)| b && !s)
        .collect();

    // ── 1. Compute survival probabilities ──
    let mut surv_probs = vec![0.0f64; n_sett];
    for i in 0..n_sett {
        let food_norm = features.food_r2[i] / 12.0;
        let coastal = if features.is_coastal[i] { 1.0 } else { 0.0 };
        let cluster_norm = features.cluster_r3[i] / 3.0;

        let logits = params.base_survival
            + params.food_coeff * food_norm
            + params.coastal_mod * coastal
            + params.cluster_pen * cluster_norm
            + params.cluster_quad * (features.cluster_r3[i] - params.cluster_optimal).powi(2) / 9.0
            // Extended: neighbor bonus (approximate via cluster count)
            + params.neighbor_survival_bonus * features.cluster_r3[i].min(3.0)
            // Extended: forest survival bonus
            + if has_forest_nearby(initial_grid, w, h, settlements[i].y, settlements[i].x) {
                params.forest_survival_bonus
            } else { 0.0 }
            // Extended: trade bonus (ports near other ports)
            + if features.has_port[i] { params.trade_bonus * 2.0 } else { 0.0 }
            // Extended: winter penalty (reduces survival)
            - params.winter_severity * 0.5;

        surv_probs[i] = sigmoid(logits);
    }

    // ── 2. Sample settlement survival ──
    let mut alive = vec![false; n_sett];
    for i in 0..n_sett {
        alive[i] = rng.gen::<f64>() < surv_probs[i];

        // Extended: faction kill check (approximate — random kill chance)
        if alive[i] && params.faction_kill_prob > 0.001 {
            // Check if any other settlement is within faction range
            let has_neighbor = (0..n_sett).any(|j| {
                j != i && {
                    let d = (settlements[i].y as i32 - settlements[j].y as i32).abs()
                        + (settlements[i].x as i32 - settlements[j].x as i32).abs();
                    (d as f64) <= params.faction_range
                }
            });
            if has_neighbor && rng.gen::<f64>() < params.faction_kill_prob {
                alive[i] = false;
            }
        }
    }

    // Initialize output grid
    let mut grid_out = vec![0u8; h * w];
    grid_out.copy_from_slice(&terrain_class);

    // Place survived/dead settlements
    for i in 0..n_sett {
        let sy = settlements[i].y;
        let sx = settlements[i].x;
        let idx = sy * w + sx;
        if alive[i] {
            grid_out[idx] = if features.has_port[i] { 2 } else { 1 }; // Port or Settlement class

            // Extended: port upgrade for surviving non-port coastal settlements
            if !features.has_port[i] && cell_coastal[idx] && rng.gen::<f64>() < params.port_upgrade_prob {
                grid_out[idx] = 2; // Upgraded to port
            }
        } else {
            // Dead → ruin or empty
            grid_out[idx] = if rng.gen::<f64>() < params.ruin_rate { 3 } else { 0 };
        }
    }

    // ── 3. Compute nearest alive distance ──
    let alive_positions: Vec<(usize, usize)> = (0..n_sett)
        .filter(|&i| alive[i])
        .map(|i| (settlements[i].y, settlements[i].x))
        .collect();

    let nearest_dist = if alive_positions.is_empty() {
        vec![999.0; h * w]
    } else {
        manhattan_distance_map(&alive_positions, h, w)
    };

    // ── 4. Expansion ──
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if !expand_mask[idx] { continue; }
            if grid_out[idx] == 1 || grid_out[idx] == 2 { continue; } // already a settlement

            let d = nearest_dist[idx];
            if d > params.max_reach || d == 0.0 { continue; }

            // Gaussian-power expansion probability
            let normalized_d = d / params.expansion_scale;
            let mut exp_prob = params.expansion_str * (-normalized_d.powf(params.decay_power)).exp();

            // Forest resistance
            if cell_is_forest[idx] {
                exp_prob *= 1.0 - params.forest_resist;
            }

            // Extended: port spawn multiplier (boost near ports)
            if params.port_spawn_multiplier != 1.0 {
                // Check if nearest alive settlement is a port
                let nearest_is_port = alive_positions.iter().enumerate().any(|(_, &(py, px))| {
                    let dd = (r as i32 - py as i32).abs() + (c as i32 - px as i32).abs();
                    dd as f64 == d && initial_grid.get(py, px) == PORT
                });
                if nearest_is_port {
                    exp_prob *= params.port_spawn_multiplier;
                }
            }

            // Extended: forest reproduction bonus
            if params.forest_repro_bonus > 0.001 && has_forest_nearby(initial_grid, w, h, r, c) {
                exp_prob += params.forest_repro_bonus;
            }

            if rng.gen::<f64>() < exp_prob {
                // Expanded — determine type
                let is_port = cell_coastal[idx] && rng.gen::<f64>() < params.port_factor;
                grid_out[idx] = if is_port { 2 } else { 1 };

                // Expansion death
                if rng.gen::<f64>() < params.exp_death {
                    grid_out[idx] = if rng.gen::<f64>() < params.ruin_rate { 3 } else { 0 };
                }
            }

            // Extended: ruin reclamation
            if grid_out[idx] == 3 && d <= 3.0 && params.ruin_to_settlement_prob > 0.001 {
                if rng.gen::<f64>() < params.ruin_to_settlement_prob {
                    grid_out[idx] = 1;
                }
            }
        }
    }

    // ── 5. Forest clearing ──
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if grid_out[idx] != 4 { continue; }
            let d = nearest_dist[idx];
            let clear_prob = params.forest_clear * (-(d / 3.0).powi(2)).exp();
            if rng.gen::<f64>() < clear_prob {
                grid_out[idx] = 0;
            }
        }
    }

    // ── 6. Forest reclamation ──
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if grid_out[idx] != 0 { continue; }
            if !cell_buildable[idx] { continue; }
            let d = nearest_dist[idx];
            let base = if orig_forest[idx] {
                params.forest_reclaim * 2.0
            } else {
                params.forest_reclaim * 0.5
            };
            let reclaim_prob = base * (d / 5.0).clamp(0.0, 1.0);
            if rng.gen::<f64>() < reclaim_prob {
                grid_out[idx] = 4;
            }
        }
    }

    // Convert class indices back to terrain codes for the Grid
    let mut result = Grid::new(w, h);
    for i in 0..h * w {
        result.cells[i] = match grid_out[i] {
            0 => { // Empty class — restore original terrain code
                let orig = initial_grid.cells[i];
                if orig == OCEAN { OCEAN } else if orig == PLAINS { PLAINS } else { EMPTY }
            }
            1 => SETTLEMENT,
            2 => PORT,
            3 => RUIN,
            4 => FOREST,
            5 => MOUNTAIN,
            _ => EMPTY,
        };
    }
    result
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
        let seed = base_seed.wrapping_add(run as u64).wrapping_mul(0x517cc1b727220a95);
        let final_grid = simulate_single(initial_grid, settlements, params, seed);

        for i in 0..n {
            let cls = to_prediction_class(final_grid.cells[i]);
            counts[i][cls] += 1;
        }
    }

    let inv = 1.0 / n_runs as f64;
    counts.iter().map(|c| {
        let mut probs = [0.0; NUM_CLASSES];
        for k in 0..NUM_CLASSES {
            probs[k] = c[k] as f64 * inv;
        }
        probs
    }).collect()
}

fn has_forest_nearby(grid: &Grid, w: usize, h: usize, y: usize, x: usize) -> bool {
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            if dy.abs() + dx.abs() > 2 { continue; }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_single_returns_valid_grid() {
        let mut grid = Grid::new(10, 10);
        for i in 0..100 { grid.cells[i] = PLAINS; }
        for x in 0..10 { grid.set(0, x, OCEAN); }
        grid.set(5, 5, SETTLEMENT);

        let settlements = vec![SettlementInfo { x: 5, y: 5, has_port: false, alive: true }];
        let params = SimParams::default();
        let result = simulate_single(&grid, &settlements, &params, 42);

        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
        assert_eq!(result.get(0, 0), OCEAN);
    }

    #[test]
    fn test_monte_carlo_probs_sum_to_one() {
        let mut grid = Grid::new(10, 10);
        for i in 0..100 { grid.cells[i] = PLAINS; }
        grid.set(5, 5, SETTLEMENT);
        grid.set(3, 3, FOREST);

        let settlements = vec![SettlementInfo { x: 5, y: 5, has_port: false, alive: true }];
        let params = SimParams::default();
        let probs = simulate_monte_carlo(&grid, &settlements, &params, 50, 42);

        for (i, p) in probs.iter().enumerate() {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Cell {i} probs don't sum to 1: {sum}");
        }
    }
}
