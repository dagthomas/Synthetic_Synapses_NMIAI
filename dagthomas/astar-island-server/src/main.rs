mod admin;
mod api;
mod auth;
mod config;
mod db;
mod models;
mod simulation;

use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    tracing::info!("Starting Astar Island Server...");

    let pool = db::setup_pool().await;
    tracing::info!("Database ready");

    // Auth routes
    let auth_routes = Router::new()
        .route("/register", post(auth::routes::register))
        .route("/login", post(auth::routes::login))
        .route("/me", get(auth::routes::me));

    // Game API routes (compatible with official /astar-island/* endpoints)
    let game_routes = Router::new()
        .route("/rounds", get(api::rounds::list_rounds))
        .route("/rounds/{round_id}", get(api::rounds::get_round))
        .route("/simulate", post(api::simulate::simulate))
        .route("/submit", post(api::submit::submit))
        .route("/budget", get(api::budget::budget))
        .route("/leaderboard", get(api::leaderboard::leaderboard))
        .route("/my-rounds", get(api::my_rounds::my_rounds))
        .route("/my-predictions/{round_id}", get(api::my_rounds::my_predictions))
        .route("/analysis/{round_id}/{seed_index}", get(api::analysis::analysis));

    // Admin API routes (consumed by Svelte SPA)
    let admin_api_routes = Router::new()
        .route("/dashboard", get(admin::routes::dashboard))
        .route("/rounds", get(admin::routes::list_rounds_admin).post(admin::routes::create_round))
        .route("/rounds/{round_id}", get(admin::routes::round_detail))
        .route("/rounds/{round_id}/activate", post(admin::routes::activate_round))
        .route("/rounds/{round_id}/score", post(admin::routes::score_round))
        .route("/teams", get(admin::routes::list_teams))
        .route("/stats", get(admin::routes::stats));

    let app = Router::new()
        .nest("/auth", auth_routes)
        .nest("/astar-island", game_routes)
        .nest("/admin/api", admin_api_routes)
        // Serve Svelte SPA at /admin (if built)
        .nest_service(
            "/admin",
            ServeDir::new("frontend/build")
                .fallback(ServeFile::new("frontend/build/index.html")),
        )
        .layer(CorsLayer::permissive())
        .with_state(pool);

    let addr = config::listen_addr();
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
