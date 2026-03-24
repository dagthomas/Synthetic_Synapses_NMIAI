use axum::{extract::State, http::StatusCode, Json};
use sqlx::PgPool;

use crate::auth::middleware::AuthTeam;
use crate::config;
use crate::models::*;
use crate::simulation::{cell::*, engine, grid::Grid, params::SimParams, terrain::SettlementInfo};

/// POST /astar-island/simulate — Observe one simulation through a viewport.
pub async fn simulate(
    State(pool): State<PgPool>,
    AuthTeam(claims): AuthTeam,
    Json(req): Json<SimulateRequest>,
) -> Result<Json<SimulateResponse>, (StatusCode, String)> {
    // Validate round is active
    let round_row: Option<(String, String, i64, i64)> = sqlx::query_as(
        "SELECT id, status, map_width::BIGINT, map_height::BIGINT FROM rounds WHERE id = $1",
    )
    .bind(&req.round_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (round_id, status, map_width, map_height) =
        round_row.ok_or((StatusCode::NOT_FOUND, "Round not found".to_string()))?;

    if status != "active" {
        return Err((StatusCode::BAD_REQUEST, "Round not active".to_string()));
    }

    if req.seed_index < 0 || req.seed_index >= config::SEEDS_PER_ROUND as i64 {
        return Err((StatusCode::BAD_REQUEST, "Invalid seed_index".to_string()));
    }

    // Check budget
    let queries_used: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1 AND round_id = $2",
    )
    .bind(&claims.team_id)
    .bind(&round_id)
    .fetch_one(&pool)
    .await
    .unwrap_or(0);

    if queries_used >= config::QUERY_BUDGET as i64 {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Query budget exhausted".to_string()));
    }

    // Validate viewport
    let vx = req.viewport_x.unwrap_or(0);
    let vy = req.viewport_y.unwrap_or(0);
    let vw = req.viewport_w.unwrap_or(config::VIEWPORT_MAX).clamp(config::VIEWPORT_MIN, config::VIEWPORT_MAX);
    let vh = req.viewport_h.unwrap_or(config::VIEWPORT_MAX).clamp(config::VIEWPORT_MIN, config::VIEWPORT_MAX);

    // Clamp viewport to map bounds
    let vx = vx.min(map_width as usize - vw);
    let vy = vy.min(map_height as usize - vh);

    // Load seed data
    let seed_row: Option<(String, i64, String, String)> = sqlx::query_as(
        "SELECT id, map_seed::BIGINT, initial_grid, settlements FROM seeds WHERE round_id = $1 AND seed_index = $2",
    )
    .bind(&round_id)
    .bind(req.seed_index)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (_seed_id, _map_seed, grid_json, setts_json) =
        seed_row.ok_or((StatusCode::NOT_FOUND, "Seed not found".to_string()))?;

    let grid_2d: Vec<Vec<u8>> = serde_json::from_str(&grid_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Grid JSON: {e}")))?;
    let settlements: Vec<SettlementInfo> = serde_json::from_str(&setts_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Settlement JSON: {e}")))?;

    // Load hidden params
    let params_json: String = sqlx::query_scalar(
        "SELECT hidden_params FROM rounds WHERE id = $1",
    )
    .bind(&round_id)
    .fetch_one(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let params: SimParams = serde_json::from_str(&params_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Params JSON: {e}")))?;

    let grid = Grid::from_2d(&grid_2d);

    // Run ONE stochastic simulation (unique seed per query)
    let sim_seed = (queries_used as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(req.seed_index as u64 * 1000);
    let final_grid = engine::simulate_single(&grid, &settlements, &params, sim_seed);

    // Extract viewport
    let viewport_grid = final_grid.extract_viewport(vx, vy, vw, vh);

    // Find settlements within viewport (from final state)
    let mut sim_settlements = Vec::new();
    for r in 0..vh {
        for c in 0..vw {
            let cell = viewport_grid.get(r, c);
            if cell == SETTLEMENT || cell == PORT {
                let mut rng = rand::thread_rng();
                sim_settlements.push(SimSettlement {
                    x: vx + c,
                    y: vy + r,
                    population: rand::Rng::gen_range(&mut rng, 0.5..5.0),
                    food: rand::Rng::gen_range(&mut rng, 0.1..1.0),
                    wealth: rand::Rng::gen_range(&mut rng, 0.1..1.0),
                    defense: rand::Rng::gen_range(&mut rng, 0.1..1.0),
                    has_port: cell == PORT,
                    alive: true,
                    owner_id: 1,
                });
            }
        }
    }

    // Log query
    sqlx::query("INSERT INTO query_log (team_id, round_id, seed_index, viewport_x, viewport_y, viewport_w, viewport_h) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .bind(&claims.team_id)
        .bind(&round_id)
        .bind(req.seed_index)
        .bind(vx as i64)
        .bind(vy as i64)
        .bind(vw as i64)
        .bind(vh as i64)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    Ok(Json(SimulateResponse {
        grid: viewport_grid.to_2d(),
        settlements: sim_settlements,
        viewport: ViewportInfo { x: vx, y: vy, w: vw, h: vh },
        width: map_width as usize,
        height: map_height as usize,
        queries_used: queries_used + 1,
        queries_max: config::QUERY_BUDGET as i64,
    }))
}
