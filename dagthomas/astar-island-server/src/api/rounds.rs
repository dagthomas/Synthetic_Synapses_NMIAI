use axum::{extract::{Path, State}, http::StatusCode, Json};
use sqlx::SqlitePool;

use crate::models::*;
use crate::simulation::terrain::SettlementInfo;

/// GET /astar-island/rounds — List all rounds.
pub async fn list_rounds(
    State(pool): State<SqlitePool>,
) -> Result<Json<Vec<RoundListEntry>>, (StatusCode, String)> {
    let rows: Vec<(String, i64, String, i64, i64, i64, Option<String>, Option<String>, f64, String)> =
        sqlx::query_as(
            "SELECT id, round_number, status, map_width, map_height, prediction_window_minutes, started_at, closes_at, round_weight, created_at FROM rounds ORDER BY round_number DESC"
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let mut entries = Vec::new();
    for (id, round_number, status, map_width, map_height, pred_window, started_at, closes_at, round_weight, created_at) in rows {
        let seeds_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM seeds WHERE round_id = ?")
            .bind(&id)
            .fetch_one(&pool)
            .await
            .unwrap_or(0);

        entries.push(RoundListEntry {
            id,
            round_number,
            event_date: Some(created_at),
            status,
            map_width,
            map_height,
            prediction_window_minutes: pred_window,
            started_at,
            closes_at,
            round_weight,
            seeds_count: seeds_count as usize,
        });
    }

    Ok(Json(entries))
}

/// GET /astar-island/rounds/{round_id} — Round detail with initial states.
pub async fn get_round(
    State(pool): State<SqlitePool>,
    Path(round_id): Path<String>,
) -> Result<Json<RoundDetail>, (StatusCode, String)> {
    let row: Option<(String, i64, String, i64, i64)> = sqlx::query_as(
        "SELECT id, round_number, status, map_width, map_height FROM rounds WHERE id = ?"
    )
    .bind(&round_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (id, round_number, status, map_width, map_height) =
        row.ok_or((StatusCode::NOT_FOUND, "Round not found".to_string()))?;

    let seed_rows: Vec<(i64, String, String)> = sqlx::query_as(
        "SELECT seed_index, initial_grid, settlements FROM seeds WHERE round_id = ? ORDER BY seed_index"
    )
    .bind(&round_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let mut initial_states = Vec::new();
    for (_seed_index, grid_json, setts_json) in seed_rows {
        let grid: Vec<Vec<u8>> = serde_json::from_str(&grid_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON error: {e}")))?;
        let setts: Vec<SettlementInfo> = serde_json::from_str(&setts_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON error: {e}")))?;

        let sett_responses: Vec<SettlementResponse> = setts
            .iter()
            .map(|s| SettlementResponse {
                x: s.x,
                y: s.y,
                has_port: s.has_port,
                alive: s.alive,
            })
            .collect();

        initial_states.push(InitialState {
            grid,
            settlements: sett_responses,
        });
    }

    Ok(Json(RoundDetail {
        id,
        round_number,
        status,
        map_width,
        map_height,
        seeds_count: initial_states.len(),
        initial_states,
    }))
}
