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
        OCEAN => "Ocean",
        PLAINS => "Plains",
        EMPTY => "Empty",
        SETTLEMENT => "Settlement",
        PORT => "Port",
        RUIN => "Ruin",
        FOREST => "Forest",
        MOUNTAIN => "Mountain",
        _ => "Unknown",
    }
}

/// CSS color for terrain visualization.
pub fn cell_color(cell: u8) -> &'static str {
    match cell {
        OCEAN => "#1a5276",
        PLAINS | EMPTY => "#f9e79f",
        SETTLEMENT => "#e74c3c",
        PORT => "#8e44ad",
        RUIN => "#7f8c8d",
        FOREST => "#27ae60",
        MOUNTAIN => "#bdc3c7",
        _ => "#ffffff",
    }
}

/// Class name for prediction classes.
pub fn class_name(cls: usize) -> &'static str {
    match cls {
        0 => "Empty",
        1 => "Settlement",
        2 => "Port",
        3 => "Ruin",
        4 => "Forest",
        5 => "Mountain",
        _ => "Unknown",
    }
}
