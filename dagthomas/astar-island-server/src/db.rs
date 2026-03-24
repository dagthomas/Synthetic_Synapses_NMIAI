use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::config;

pub async fn setup_pool() -> PgPool {
    let url = config::database_url();

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&url)
        .await
        .expect("Failed to connect to PostgreSQL");

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
    let admin_exists: Option<bool> =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM teams WHERE is_admin = TRUE)")
            .fetch_one(&pool)
            .await
            .ok();

    if !admin_exists.unwrap_or(false) {
        let id = uuid::Uuid::new_v4().to_string();
        let password_hash = crate::auth::jwt::hash_password("admin").expect("Failed to hash password");
        sqlx::query("INSERT INTO teams (id, name, password_hash, is_admin) VALUES ($1, 'admin', $2, TRUE)")
            .bind(&id)
            .bind(&password_hash)
            .execute(&pool)
            .await
            .expect("Failed to create admin user");
        tracing::info!("Created admin user (password: admin)");
    }

    pool
}
