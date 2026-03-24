use axum::{extract::State, http::StatusCode, Json};
use sqlx::PgPool;

use crate::auth::middleware::AuthTeam;
use crate::config;
use crate::models::*;

/// POST /astar-island/submit — Submit prediction tensor for one seed.
pub async fn submit(
    State(pool): State<PgPool>,
    AuthTeam(claims): AuthTeam,
    Json(req): Json<SubmitRequest>,
) -> Result<Json<SubmitResponse>, (StatusCode, String)> {
    // Validate round is active
    let status: Option<String> = sqlx::query_scalar("SELECT status FROM rounds WHERE id = $1")
        .bind(&req.round_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let status = status.ok_or((StatusCode::NOT_FOUND, "Round not found".to_string()))?;
    if status != "active" {
        return Err((StatusCode::BAD_REQUEST, "Round not active".to_string()));
    }

    if req.seed_index < 0 || req.seed_index >= config::SEEDS_PER_ROUND as i64 {
        return Err((StatusCode::BAD_REQUEST, "Invalid seed_index".to_string()));
    }

    // Validate prediction shape: H×W×6
    if req.prediction.len() != config::MAP_H {
        return Err((StatusCode::BAD_REQUEST, format!(
            "Expected {} rows, got {}", config::MAP_H, req.prediction.len()
        )));
    }

    for (y, row) in req.prediction.iter().enumerate() {
        if row.len() != config::MAP_W {
            return Err((StatusCode::BAD_REQUEST, format!(
                "Row {y}: expected {} cols, got {}", config::MAP_W, row.len()
            )));
        }
        for (x, probs) in row.iter().enumerate() {
            if probs.len() != config::NUM_CLASSES {
                return Err((StatusCode::BAD_REQUEST, format!(
                    "Cell ({y},{x}): expected {} probs, got {}", config::NUM_CLASSES, probs.len()
                )));
            }
            let sum: f64 = probs.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err((StatusCode::BAD_REQUEST, format!(
                    "Cell ({y},{x}): probs sum to {sum:.4}, expected 1.0"
                )));
            }
            for &p in probs {
                if p < 0.0 {
                    return Err((StatusCode::BAD_REQUEST, format!(
                        "Cell ({y},{x}): negative probability"
                    )));
                }
            }
        }
    }

    let tensor_json = serde_json::to_string(&req.prediction)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON error: {e}")))?;

    let id = uuid::Uuid::new_v4().to_string();

    // Upsert: replace previous submission for same team+round+seed
    sqlx::query(
        "INSERT INTO predictions (id, team_id, round_id, seed_index, tensor) VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT(team_id, round_id, seed_index) DO UPDATE SET tensor = excluded.tensor, submitted_at = NOW()::TEXT, score = NULL"
    )
    .bind(&id)
    .bind(&claims.team_id)
    .bind(&req.round_id)
    .bind(req.seed_index)
    .bind(&tensor_json)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    Ok(Json(SubmitResponse {
        status: "accepted".to_string(),
        round_id: req.round_id,
        seed_index: req.seed_index,
    }))
}
