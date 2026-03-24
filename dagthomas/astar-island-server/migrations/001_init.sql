CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    total_rounds BIGINT DEFAULT 50,
    round_duration_minutes BIGINT DEFAULT 180,
    break_duration_minutes BIGINT DEFAULT 15,
    status TEXT DEFAULT 'pending',
    current_round BIGINT DEFAULT 0,
    started_at TEXT,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

CREATE TABLE IF NOT EXISTS rounds (
    id TEXT PRIMARY KEY,
    game_id TEXT REFERENCES games(id),
    round_number BIGINT NOT NULL,
    status TEXT DEFAULT 'pending',
    map_width BIGINT DEFAULT 40,
    map_height BIGINT DEFAULT 40,
    prediction_window_minutes BIGINT DEFAULT 180,
    started_at TEXT,
    closes_at TEXT,
    round_weight DOUBLE PRECISION DEFAULT 1.0,
    hidden_params TEXT NOT NULL,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

CREATE TABLE IF NOT EXISTS seeds (
    id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index BIGINT NOT NULL,
    map_seed BIGINT NOT NULL,
    initial_grid TEXT NOT NULL,
    settlements TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    UNIQUE(round_id, seed_index)
);

CREATE TABLE IF NOT EXISTS query_log (
    id BIGSERIAL PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(id),
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index BIGINT NOT NULL,
    viewport_x BIGINT NOT NULL,
    viewport_y BIGINT NOT NULL,
    viewport_w BIGINT NOT NULL,
    viewport_h BIGINT NOT NULL,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(id),
    round_id TEXT NOT NULL REFERENCES rounds(id),
    seed_index BIGINT NOT NULL,
    tensor TEXT NOT NULL,
    score DOUBLE PRECISION,
    submitted_at TEXT DEFAULT (NOW()::TEXT),
    UNIQUE(team_id, round_id, seed_index)
);

CREATE INDEX IF NOT EXISTS idx_query_log_team_round ON query_log(team_id, round_id);
CREATE INDEX IF NOT EXISTS idx_predictions_team_round ON predictions(team_id, round_id);
CREATE INDEX IF NOT EXISTS idx_seeds_round ON seeds(round_id);
