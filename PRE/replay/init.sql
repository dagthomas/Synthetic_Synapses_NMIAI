CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    seed INTEGER NOT NULL,
    difficulty TEXT NOT NULL,
    grid_width INTEGER NOT NULL,
    grid_height INTEGER NOT NULL,
    bot_count INTEGER NOT NULL,
    item_types INTEGER NOT NULL,
    order_size_min INTEGER NOT NULL,
    order_size_max INTEGER NOT NULL,
    walls JSONB NOT NULL,
    shelves JSONB NOT NULL,
    items JSONB NOT NULL,
    drop_off JSONB NOT NULL,
    drop_off_zones JSONB,
    spawn JSONB NOT NULL,
    final_score INTEGER NOT NULL,
    items_delivered INTEGER NOT NULL,
    orders_completed INTEGER NOT NULL,
    run_type TEXT NOT NULL DEFAULT 'live',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rounds (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    bots JSONB NOT NULL,
    orders JSONB NOT NULL,
    actions JSONB NOT NULL DEFAULT '[]',
    score INTEGER NOT NULL,
    events JSONB NOT NULL DEFAULT '[]',
    UNIQUE(run_id, round_number)
);

CREATE INDEX IF NOT EXISTS idx_rounds_run_id ON rounds(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_difficulty ON runs(difficulty);
CREATE INDEX IF NOT EXISTS idx_runs_seed ON runs(seed);

CREATE TABLE IF NOT EXISTS tokens (
    id SERIAL PRIMARY KEY,
    ws_url TEXT NOT NULL UNIQUE,
    difficulty TEXT,
    map_seed INTEGER,
    token_raw TEXT,
    label TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tokens_difficulty ON tokens(difficulty);
CREATE INDEX IF NOT EXISTS idx_tokens_created_at ON tokens(created_at DESC);

CREATE TABLE IF NOT EXISTS ws_sessions (
    id SERIAL PRIMARY KEY,
    token_id INTEGER REFERENCES tokens(id) ON DELETE SET NULL,
    ws_url TEXT NOT NULL,
    difficulty TEXT,
    map_seed INTEGER,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    final_score INTEGER,
    rounds_received INTEGER DEFAULT 0,
    status TEXT DEFAULT 'connecting'
);

CREATE TABLE IF NOT EXISTS ws_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES ws_sessions(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    msg_type TEXT NOT NULL,
    round_num INTEGER,
    raw JSONB NOT NULL,
    received_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ws_messages_session ON ws_messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_ws_sessions_token ON ws_sessions(token_id);
