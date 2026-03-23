use axum::{extract::{Path, State}, http::StatusCode, Json};
use sqlx::SqlitePool;

use crate::auth::middleware::AuthTeam;
use crate::models::AnalysisResponse;

/// GET /astar-island/analysis/{round_id}/{seed_index} — Post-round ground truth comparison.
pub async fn analysis(
    State(pool): State<SqlitePool>,
    AuthTeam(claims): AuthTeam,
    Path((round_id, seed_index)): Path<(String, i64)>,
) -> Result<Json<AnalysisResponse>, (StatusCode, String)> {
    // Check round is completed or scoring
    let status: Option<String> = sqlx::query_scalar("SELECT status FROM rounds WHERE id = ?")
        .bind(&round_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let status = status.ok_or((StatusCode::NOT_FOUND, "Round not found".to_string()))?;
    if status != "completed" && status != "scoring" {
        return Err((StatusCode::BAD_REQUEST, "Round not yet completed".to_string()));
    }

    // Load ground truth
    let seed_row: Option<(String, String)> = sqlx::query_as(
        "SELECT initial_grid, ground_truth FROM seeds WHERE round_id = ? AND seed_index = ?",
    )
    .bind(&round_id)
    .bind(seed_index)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (grid_json, gt_json) =
        seed_row.ok_or((StatusCode::NOT_FOUND, "Seed not found".to_string()))?;

    let initial_grid: Vec<Vec<u8>> = serde_json::from_str(&grid_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Grid JSON: {e}")))?;

    // Ground truth is stored as flat [[f64; 6]; H*W], convert to [H][W][6]
    let gt_flat: Vec<[f64; 6]> = serde_json::from_str(&gt_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("GT JSON: {e}")))?;

    let h = initial_grid.len();
    let w = if h > 0 { initial_grid[0].len() } else { 0 };

    let ground_truth: Vec<Vec<Vec<f64>>> = (0..h)
        .map(|y| {
            (0..w)
                .map(|x| gt_flat[y * w + x].to_vec())
                .collect()
        })
        .collect();

    // Load team's prediction
    let pred_row: Option<(String, Option<f64>)> = sqlx::query_as(
        "SELECT tensor, score FROM predictions WHERE team_id = ? AND round_id = ? AND seed_index = ?",
    )
    .bind(&claims.team_id)
    .bind(&round_id)
    .bind(seed_index)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (prediction, score) = if let Some((tensor_json, score)) = pred_row {
        let prediction: Vec<Vec<Vec<f64>>> = serde_json::from_str(&tensor_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Pred JSON: {e}")))?;
        (prediction, score)
    } else {
        // No prediction submitted — return empty
        let empty: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 6]; w]; h];
        (empty, None)
    };

    Ok(Json(AnalysisResponse {
        prediction,
        ground_truth,
        score,
        width: w,
        height: h,
        initial_grid,
    }))
}
