/// Cell type constants matching the API terrain codes.
pub const OCEAN: u8 = 10;
pub const PLAINS: u8 = 11;
pub const EMPTY: u8 = 0;
pub const SETTLEMENT: u8 = 1;
pub const PORT: u8 = 2;
pub const RUIN: u8 = 3;
pub const FOREST: u8 = 4;
pub const MOUNTAIN: u8 = 5;

pub const NUM_CLASSES: usize = 6;

/// Map API terrain code to prediction class (0-5).
#[inline]
pub fn to_prediction_class(cell: u8) -> usize {
    match cell {
        OCEAN | PLAINS | EMPTY => 0,
        SETTLEMENT => 1,
        PORT => 2,
        RUIN => 3,
        FOREST => 4,
        MOUNTAIN => 5,
        _ => 0,
    }
}

/// Human-readable name for a cell type.
pub fn cell_name(cell: u8) -> &'static str {
    match cell {
        OCEAN => "Void",
        PLAINS => "Regolith",
        EMPTY => "Empty",
        SETTLEMENT => "Crystal Node",
        PORT => "Refinery",
        RUIN => "Depleted Vein",
        FOREST => "Xenoflora",
        MOUNTAIN => "Obsidian Ridge",
        _ => "Unknown",
    }
}

/// CSS color for terrain visualization.
pub fn cell_color(cell: u8) -> &'static str {
    match cell {
        OCEAN => "#0a0a2e",
        PLAINS | EMPTY => "#2a1f3d",
        SETTLEMENT => "#00fff0",
        PORT => "#ff00ff",
        RUIN => "#4a4a5a",
        FOREST => "#39ff14",
        MOUNTAIN => "#1a1a2e",
        _ => "#ffffff",
    }
}

/// Class name for prediction classes.
pub fn class_name(cls: usize) -> &'static str {
    match cls {
        0 => "Empty",
        1 => "Crystal Node",
        2 => "Refinery",
        3 => "Depleted Vein",
        4 => "Xenoflora",
        5 => "Obsidian Ridge",
        _ => "Unknown",
    }
}
