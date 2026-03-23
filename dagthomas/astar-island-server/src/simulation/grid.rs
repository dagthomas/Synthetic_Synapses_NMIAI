use serde::{Deserialize, Serialize};

/// A 2D grid of cell types stored as flat Vec<u8>.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<u8>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![0; width * height],
        }
    }

    #[inline]
    pub fn get(&self, y: usize, x: usize) -> u8 {
        self.cells[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, y: usize, x: usize, val: u8) {
        self.cells[y * self.width + x] = val;
    }

    /// Create from a 2D array (API format: grid[y][x]).
    pub fn from_2d(grid: &[Vec<u8>]) -> Self {
        let height = grid.len();
        let width = if height > 0 { grid[0].len() } else { 0 };
        let mut cells = Vec::with_capacity(width * height);
        for row in grid {
            cells.extend_from_slice(row);
        }
        Self {
            width,
            height,
            cells,
        }
    }

    /// Extract a viewport sub-grid.
    pub fn extract_viewport(&self, vx: usize, vy: usize, vw: usize, vh: usize) -> Grid {
        let mut sub = Grid::new(vw, vh);
        for dy in 0..vh {
            for dx in 0..vw {
                let sy = vy + dy;
                let sx = vx + dx;
                if sy < self.height && sx < self.width {
                    sub.set(dy, dx, self.get(sy, sx));
                }
            }
        }
        sub
    }

    /// Convert to 2D vec for JSON serialization.
    pub fn to_2d(&self) -> Vec<Vec<u8>> {
        (0..self.height)
            .map(|y| (0..self.width).map(|x| self.get(y, x)).collect())
            .collect()
    }
}
