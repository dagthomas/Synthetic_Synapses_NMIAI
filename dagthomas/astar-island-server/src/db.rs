use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::str::FromStr;

use crate::config;

pub async fn setup_pool() -> SqlitePool {
    let url = config::database_url();
    let opts = SqliteConnectOptions::from_str(&url)
        .expect("Invalid DATABASE_URL")
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .connect_with(opts)
        .await
        .expect("Failed to connect to SQLite");

    // Run migrations
    let migration = include_str!("../migrations/001_init.sql");
    for statement in migration.split(';') {
        let stmt = statement.trim();
        if !stmt.is_empty() {
            sqlx::query(stmt)
                .execute(&pool)
                .await
                .unwrap_or_else(|e| panic!("Migration failed on: {stmt}\nError: {e}"));
        }
    }

    // Create admin user if not exists
    let admin_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM teams WHERE is_admin = TRUE)")
            .fetch_one(&pool)
            .await
            .unwrap_or(false);

    if !admin_exists {
        let id = uuid::Uuid::new_v4().to_string();
        let password_hash = crate::auth::jwt::hash_password("admin").expect("Failed to hash password");
        sqlx::query("INSERT INTO teams (id, name, password_hash, is_admin) VALUES (?, 'admin', ?, TRUE)")
            .bind(&id)
            .bind(&password_hash)
            .execute(&pool)
            .await
            .expect("Failed to create admin user");
        tracing::info!("Created admin user (password: admin)");
    }

    pool
}
