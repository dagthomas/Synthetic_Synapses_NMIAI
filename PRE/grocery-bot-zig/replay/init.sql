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
    spawn JSONB NOT NULL,
    final_score INTEGER NOT NULL,
    items_delivered INTEGER NOT NULL,
    orders_completed INTEGER NOT NULL,
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
