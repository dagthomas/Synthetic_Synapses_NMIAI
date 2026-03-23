CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rounds (
    id TEXT PRIMARY KEY,
    round_number INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    map_width INTEGER DEFAULT 40,
    map_height INTEGER DEFAULT 40,
    prediction_window_minutes INTEGER DEFAULT 165,
    started_at TEXT,
    closes_at TEXT,
    round_weight REAL DEFAULT 1.0,
    hidden_params TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS seeds (
    id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index INTEGER NOT NULL,
    map_seed INTEGER NOT NULL,
    initial_grid TEXT NOT NULL,
    settlements TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    UNIQUE(round_id, seed_index)
);

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL REFERENCES teams(id),
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index INTEGER NOT NULL,
    viewport_x INTEGER NOT NULL,
    viewport_y INTEGER NOT NULL,
    viewport_w INTEGER NOT NULL,
    viewport_h INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(id),
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index INTEGER NOT NULL,
    tensor TEXT NOT NULL,
    score REAL,
    submitted_at TEXT DEFAULT (datetime('now')),
    UNIQUE(team_id, round_id, seed_index)
);

CREATE INDEX IF NOT EXISTS idx_query_log_team_round ON query_log(team_id, round_id);
CREATE INDEX IF NOT EXISTS idx_predictions_team_round ON predictions(team_id, round_id);
CREATE INDEX IF NOT EXISTS idx_seeds_round ON seeds(round_id);
