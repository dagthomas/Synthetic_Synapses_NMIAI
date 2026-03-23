//! Procedural map generation for 40x40 Norse island worlds.
//!
//! Generates: ocean border → fjords → mountain chains → forest clusters → settlements.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use super::cell::*;
use super::grid::Grid;
use super::terrain::{is_coastal, SettlementInfo};

/// Generate a procedural map from a seed.
///
/// Returns (grid, settlements) where grid is a w×h terrain grid and
/// settlements lists the initial settlement positions.
pub fn generate_map(seed: u64, w: usize, h: usize) -> (Grid, Vec<SettlementInfo>) {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut grid = Grid::new(w, h);

    // 1. Fill with plains
    for i in 0..grid.cells.len() {
        grid.cells[i] = PLAINS;
    }

    // 2. Ocean border
    place_ocean_border(&mut grid, &mut rng);

    // 3. Fjords
    let n_fjords = rng.gen_range(2..=5);
    for _ in 0..n_fjords {
        place_fjord(&mut grid, &mut rng);
    }

    // 4. Mountain chains
    let n_chains = rng.gen_range(2..=4);
    for _ in 0..n_chains {
        place_mountain_chain(&mut grid, &mut rng);
    }

    // 5. Forest clusters
    let n_groves = rng.gen_range(8..=15);
    for _ in 0..n_groves {
        place_forest_cluster(&mut grid, &mut rng);
    }

    // 6. Settlements
    let n_settlements = rng.gen_range(15..=35);
    let settlements = place_settlements(&mut grid, &mut rng, n_settlements);

    (grid, settlements)
}

fn place_ocean_border(grid: &mut Grid, rng: &mut Xoshiro256StarStar) {
    let h = grid.height;
    let w = grid.width;

    // Always 1-cell border of ocean
    for x in 0..w {
        grid.set(0, x, OCEAN);
        grid.set(h - 1, x, OCEAN);
    }
    for y in 0..h {
        grid.set(y, 0, OCEAN);
        grid.set(y, w - 1, OCEAN);
    }

    // Sometimes 2-cell thick border on some edges
    if rng.gen_bool(0.6) {
        for x in 0..w {
            grid.set(1, x, OCEAN);
        }
    }
    if rng.gen_bool(0.6) {
        for x in 0..w {
            grid.set(h - 2, x, OCEAN);
        }
    }
    if rng.gen_bool(0.6) {
        for y in 0..h {
            grid.set(y, 1, OCEAN);
        }
    }
    if rng.gen_bool(0.6) {
        for y in 0..h {
            grid.set(y, w - 2, OCEAN);
        }
    }
}

fn place_fjord(grid: &mut Grid, rng: &mut Xoshiro256StarStar) {
    let h = grid.height as i32;
    let w = grid.width as i32;

    // Pick a random edge point
    let edge = rng.gen_range(0..4);
    let (mut y, mut x, mut dy, mut dx) = match edge {
        0 => (0i32, rng.gen_range(3..w - 3), 1i32, 0i32), // top
        1 => (h - 1, rng.gen_range(3..w - 3), -1, 0),      // bottom
        2 => (rng.gen_range(3..h - 3), 0, 0, 1),            // left
        _ => (rng.gen_range(3..h - 3), w - 1, 0, -1),       // right
    };

    let depth = rng.gen_range(5..=15);
    let fjord_width = rng.gen_range(1..=2);

    for _ in 0..depth {
        // Place ocean at current position
        for fw in 0..fjord_width {
            let fy = y + if dx != 0 { fw } else { 0 };
            let fx = x + if dy != 0 { fw } else { 0 };
            if fy >= 0 && fy < h && fx >= 0 && fx < w {
                grid.set(fy as usize, fx as usize, OCEAN);
            }
        }

        // Advance with slight randomness
        y += dy;
        x += dx;

        // Random drift perpendicular to main direction
        if rng.gen_bool(0.3) {
            if dy != 0 {
                x += if rng.gen_bool(0.5) { 1 } else { -1 };
            } else {
                y += if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }

        if y < 1 || y >= h - 1 || x < 1 || x >= w - 1 {
            break;
        }
    }
}

fn place_mountain_chain(grid: &mut Grid, rng: &mut Xoshiro256StarStar) {
    let h = grid.height;
    let w = grid.width;

    // Start from a random land position (avoiding borders)
    let mut y = rng.gen_range(4..h - 4) as i32;
    let mut x = rng.gen_range(4..w - 4) as i32;

    let chain_len = rng.gen_range(5..=15);

    // Pick a main direction with randomness
    let dir = rng.gen_range(0..8);
    let (base_dy, base_dx): (i32, i32) = match dir {
        0 => (-1, 0),
        1 => (-1, 1),
        2 => (0, 1),
        3 => (1, 1),
        4 => (1, 0),
        5 => (1, -1),
        6 => (0, -1),
        _ => (-1, -1),
    };

    for _ in 0..chain_len {
        if y >= 2 && y < (h - 2) as i32 && x >= 2 && x < (w - 2) as i32 {
            if grid.get(y as usize, x as usize) != OCEAN {
                grid.set(y as usize, x as usize, MOUNTAIN);
            }
        }

        // Follow main direction with random drift
        y += base_dy;
        x += base_dx;
        if rng.gen_bool(0.4) {
            y += rng.gen_range(-1..=1);
            x += rng.gen_range(-1..=1);
        }

        if y < 2 || y >= (h - 2) as i32 || x < 2 || x >= (w - 2) as i32 {
            break;
        }
    }
}

fn place_forest_cluster(grid: &mut Grid, rng: &mut Xoshiro256StarStar) {
    let h = grid.height;
    let w = grid.width;

    // Pick seed point on land
    let mut attempts = 0;
    let (sy, sx) = loop {
        let y = rng.gen_range(2..h - 2);
        let x = rng.gen_range(2..w - 2);
        if grid.get(y, x) == PLAINS {
            break (y, x);
        }
        attempts += 1;
        if attempts > 50 {
            return;
        }
    };

    grid.set(sy, sx, FOREST);
    let cluster_size = rng.gen_range(3..=10);

    let mut frontier = vec![(sy, sx)];
    let mut placed = 1;

    while placed < cluster_size && !frontier.is_empty() {
        let idx = rng.gen_range(0..frontier.len());
        let (fy, fx) = frontier[idx];

        // Try a random neighbor
        let dy = rng.gen_range(-1i32..=1);
        let dx = rng.gen_range(-1i32..=1);
        let ny = fy as i32 + dy;
        let nx = fx as i32 + dx;

        if ny >= 2 && ny < (h - 2) as i32 && nx >= 2 && nx < (w - 2) as i32 {
            let ny = ny as usize;
            let nx = nx as usize;
            if grid.get(ny, nx) == PLAINS {
                grid.set(ny, nx, FOREST);
                frontier.push((ny, nx));
                placed += 1;
            }
        }

        // Remove exhausted frontier cells occasionally
        if rng.gen_bool(0.2) && frontier.len() > 1 {
            frontier.swap_remove(idx);
        }
    }
}

fn place_settlements(
    grid: &mut Grid,
    rng: &mut Xoshiro256StarStar,
    target: usize,
) -> Vec<SettlementInfo> {
    let h = grid.height;
    let w = grid.width;
    let mut settlements = Vec::new();

    let mut attempts = 0;
    while settlements.len() < target && attempts < target * 20 {
        attempts += 1;

        let y = rng.gen_range(2..h - 2);
        let x = rng.gen_range(2..w - 2);

        let cell = grid.get(y, x);
        if cell != PLAINS && cell != FOREST {
            continue;
        }

        // Check minimum spacing (Manhattan distance ≥ 3)
        let too_close = settlements
            .iter()
            .any(|s: &SettlementInfo| (s.y as i32 - y as i32).abs() + (s.x as i32 - x as i32).abs() < 3);
        if too_close {
            continue;
        }

        // Coastal settlements may be ports
        let coastal = is_coastal(grid, y, x);
        let has_port = coastal && rng.gen_bool(0.4);

        if has_port {
            grid.set(y, x, PORT);
        } else {
            grid.set(y, x, SETTLEMENT);
        }

        settlements.push(SettlementInfo {
            x,
            y,
            has_port,
            alive: true,
        });
    }

    settlements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_map_dimensions() {
        let (grid, settlements) = generate_map(42, 40, 40);
        assert_eq!(grid.width, 40);
        assert_eq!(grid.height, 40);
        assert!(!settlements.is_empty(), "Should have at least some settlements");

        // Border should be ocean
        assert_eq!(grid.get(0, 0), OCEAN);
        assert_eq!(grid.get(0, 20), OCEAN);
        assert_eq!(grid.get(39, 39), OCEAN);
    }

    #[test]
    fn test_deterministic_generation() {
        let (g1, s1) = generate_map(123, 40, 40);
        let (g2, s2) = generate_map(123, 40, 40);
        assert_eq!(g1.cells, g2.cells, "Same seed should produce same map");
        assert_eq!(s1.len(), s2.len());
    }

    #[test]
    fn test_different_seeds_different_maps() {
        let (g1, _) = generate_map(1, 40, 40);
        let (g2, _) = generate_map(2, 40, 40);
        assert_ne!(g1.cells, g2.cells, "Different seeds should produce different maps");
    }
}
