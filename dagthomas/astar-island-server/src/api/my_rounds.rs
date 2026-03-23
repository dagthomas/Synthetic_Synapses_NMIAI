use axum::{extract::{Path, State}, http::StatusCode, Json};
use sqlx::SqlitePool;

use crate::auth::middleware::AuthTeam;
use crate::config;
use crate::models::*;

/// GET /astar-island/my-rounds — Rounds enriched with team's scores.
pub async fn my_rounds(
    State(pool): State<SqlitePool>,
    AuthTeam(claims): AuthTeam,
) -> Result<Json<Vec<MyRoundEntry>>, (StatusCode, String)> {
    let rounds: Vec<(String, i64, String, i64, i64, i64, f64, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT id, round_number, status, map_width, map_height, prediction_window_minutes, round_weight, started_at, closes_at FROM rounds ORDER BY round_number DESC"
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let mut entries = Vec::new();
    for (id, round_number, status, mw, mh, pw, rw, started_at, closes_at) in rounds {
        let queries_used: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM query_log WHERE team_id = ? AND round_id = ?",
        )
        .bind(&claims.team_id)
        .bind(&id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let seeds_submitted: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM predictions WHERE team_id = ? AND round_id = ?",
        )
        .bind(&claims.team_id)
        .bind(&id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        // Get per-seed scores
        let seed_scores_rows: Vec<(i64, Option<f64>)> = sqlx::query_as(
            "SELECT seed_index, score FROM predictions WHERE team_id = ? AND round_id = ? ORDER BY seed_index",
        )
        .bind(&claims.team_id)
        .bind(&id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        let seed_scores: Vec<Option<f64>> = (0..config::SEEDS_PER_ROUND as i64)
            .map(|si| {
                seed_scores_rows
                    .iter()
                    .find(|(idx, _)| *idx == si)
                    .and_then(|(_, s)| *s)
            })
            .collect();

        let round_score: Option<f64> = {
            let scored: Vec<f64> = seed_scores.iter().filter_map(|s| *s).collect();
            if scored.is_empty() {
                None
            } else {
                Some(scored.iter().sum::<f64>() / scored.len() as f64)
            }
        };

        entries.push(MyRoundEntry {
            id,
            round_number,
            status,
            map_width: mw,
            map_height: mh,
            seeds_count: config::SEEDS_PER_ROUND,
            round_weight: rw,
            started_at,
            closes_at,
            prediction_window_minutes: pw,
            round_score,
            seed_scores: Some(seed_scores),
            seeds_submitted,
            rank: None, // Could compute per-round rank if needed
            queries_used,
            queries_max: config::QUERY_BUDGET as i64,
        });
    }

    Ok(Json(entries))
}

/// GET /astar-island/my-predictions/{round_id} — Team's predictions with argmax/confidence.
pub async fn my_predictions(
    State(pool): State<SqlitePool>,
    AuthTeam(claims): AuthTeam,
    Path(round_id): Path<String>,
) -> Result<Json<Vec<MyPredictionEntry>>, (StatusCode, String)> {
    let rows: Vec<(i64, String, Option<f64>, Option<String>)> = sqlx::query_as(
        "SELECT seed_index, tensor, score, submitted_at FROM predictions WHERE team_id = ? AND round_id = ? ORDER BY seed_index",
    )
    .bind(&claims.team_id)
    .bind(&round_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let mut entries = Vec::new();
    for (seed_index, tensor_json, score, submitted_at) in rows {
        let tensor: Vec<Vec<Vec<f64>>> = serde_json::from_str(&tensor_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON error: {e}")))?;

        // Compute argmax and confidence grids
        let mut argmax_grid = Vec::new();
        let mut confidence_grid = Vec::new();

        for row in &tensor {
            let mut argmax_row = Vec::new();
            let mut conf_row = Vec::new();
            for probs in row {
                let (max_idx, max_val) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));
                argmax_row.push(max_idx);
                conf_row.push((*max_val * 1000.0).round() / 1000.0);
            }
            argmax_grid.push(argmax_row);
            confidence_grid.push(conf_row);
        }

        entries.push(MyPredictionEntry {
            seed_index,
            argmax_grid,
            confidence_grid,
            score,
            submitted_at,
        });
    }

    Ok(Json(entries))
}
