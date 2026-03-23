//! Terrain analysis: distance maps, coastal detection, settlement features.

use super::cell::*;
use super::grid::Grid;

/// Check if cell (y, x) is adjacent to ocean (4-cardinal neighbors).
pub fn is_coastal(grid: &Grid, y: usize, x: usize) -> bool {
    let h = grid.height;
    let w = grid.width;
    const DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dy, dx) in DIRS {
        let ny = y as i32 + dy;
        let nx = x as i32 + dx;
        if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
            if grid.get(ny as usize, nx as usize) == OCEAN {
                return true;
            }
        }
    }
    false
}

/// Check if a terrain type is static (ocean or mountain — never changes).
pub fn is_static(api_code: u8) -> bool {
    api_code == OCEAN || api_code == MOUNTAIN
}

/// Extract settlement positions (y, x) from a grid.
pub fn find_settlements(grid: &Grid) -> Vec<(usize, usize)> {
    let mut settlements = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let cell = grid.get(y, x);
            if cell == SETTLEMENT || cell == PORT {
                settlements.push((y, x));
            }
        }
    }
    settlements
}

/// Compute Manhattan distance from each cell to nearest given position.
/// Returns flat Vec<f64> indexed by y*w+x.
pub fn manhattan_distance_map(positions: &[(usize, usize)], h: usize, w: usize) -> Vec<f64> {
    let mut dist_map = vec![f64::INFINITY; w * h];
    for &(py, px) in positions {
        for r in 0..h {
            for c in 0..w {
                let d = ((r as i32 - py as i32).abs() + (c as i32 - px as i32).abs()) as f64;
                let idx = r * w + c;
                if d < dist_map[idx] {
                    dist_map[idx] = d;
                }
            }
        }
    }
    dist_map
}

/// Per-settlement features needed for survival computation.
#[derive(Debug, Clone)]
pub struct SettlementFeatures {
    /// (x, y) positions.
    pub positions: Vec<(usize, usize)>,
    /// Is this settlement coastal (adjacent to ocean)?
    pub is_coastal: Vec<bool>,
    /// Has port status from initial state.
    pub has_port: Vec<bool>,
    /// Food-producing cells (forest or plains) within Manhattan distance 2.
    pub food_r2: Vec<f64>,
    /// Number of other settlements within Manhattan distance 3.
    pub cluster_r3: Vec<f64>,
    /// Manhattan distance from each settlement to every cell: [n_sett][h*w].
    pub dist_maps: Vec<Vec<f64>>,
}

/// Settlement info from initial state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SettlementInfo {
    pub x: usize,
    pub y: usize,
    pub has_port: bool,
    pub alive: bool,
}

/// Compute settlement features for the simulation engine.
pub fn compute_settlement_features(
    grid: &Grid,
    settlements: &[SettlementInfo],
) -> SettlementFeatures {
    let h = grid.height;
    let w = grid.width;
    let n = settlements.len();

    let positions: Vec<(usize, usize)> = settlements.iter().map(|s| (s.x, s.y)).collect();
    let is_coastal_vec: Vec<bool> = settlements.iter().map(|s| is_coastal(grid, s.y, s.x)).collect();
    let has_port_vec: Vec<bool> = settlements.iter().map(|s| s.has_port).collect();

    // Food within Manhattan distance 2
    let mut food_r2 = vec![0.0; n];
    for (i, s) in settlements.iter().enumerate() {
        let mut food = 0;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                if dy.abs() + dx.abs() > 2 {
                    continue;
                }
                let ny = s.y as i32 + dy;
                let nx = s.x as i32 + dx;
                if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                    let cell = grid.get(ny as usize, nx as usize);
                    if cell == FOREST || cell == PLAINS {
                        food += 1;
                    }
                }
            }
        }
        food_r2[i] = food as f64;
    }

    // Cluster count within Manhattan distance 3
    let mut cluster_r3 = vec![0.0; n];
    for i in 0..n {
        let mut count = 0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = (settlements[i].y as i32 - settlements[j].y as i32).abs()
                + (settlements[i].x as i32 - settlements[j].x as i32).abs();
            if d <= 3 {
                count += 1;
            }
        }
        cluster_r3[i] = count as f64;
    }

    // Per-settlement Manhattan distance maps
    let dist_maps: Vec<Vec<f64>> = settlements
        .iter()
        .map(|s| {
            let mut dm = vec![0.0; h * w];
            for r in 0..h {
                for c in 0..w {
                    dm[r * w + c] = ((r as i32 - s.y as i32).abs()
                        + (c as i32 - s.x as i32).abs())
                        as f64;
                }
            }
            dm
        })
        .collect();

    SettlementFeatures {
        positions,
        is_coastal: is_coastal_vec,
        has_port: has_port_vec,
        food_r2,
        cluster_r3,
        dist_maps,
    }
}
