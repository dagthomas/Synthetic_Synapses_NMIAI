/// Server configuration constants.

pub const MAP_W: usize = 40;
pub const MAP_H: usize = 40;
pub const NUM_CLASSES: usize = 6;
pub const MC_RUNS: usize = 50;
pub const QUERY_BUDGET: i32 = 50;
pub const SEEDS_PER_ROUND: usize = 5;
pub const VIEWPORT_MIN: usize = 5;
pub const VIEWPORT_MAX: usize = 15;

// Game scheduling
pub const DEFAULT_PREDICTION_WINDOW_MINUTES: i32 = 180;
pub const BREAK_BETWEEN_ROUNDS_MINUTES: i32 = 15;
pub const TOTAL_ROUNDS_PER_GAME: i32 = 50;

/// JWT secret — in production, load from env.
pub fn jwt_secret() -> String {
    std::env::var("JWT_SECRET").unwrap_or_else(|_| "astar-island-dev-secret-change-me".to_string())
}

/// Database URL.
pub fn database_url() -> String {
    std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://astar:astar@localhost:5432/astar_island".to_string())
}

/// Server listen address.
pub fn listen_addr() -> String {
    std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string())
}
