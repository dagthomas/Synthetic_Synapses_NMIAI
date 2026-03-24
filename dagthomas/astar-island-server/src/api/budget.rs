use axum::{extract::State, http::StatusCode, Json};
use sqlx::PgPool;

use crate::auth::middleware::AuthTeam;
use crate::config;
use crate::models::BudgetResponse;

/// GET /astar-island/budget — Check remaining query budget.
pub async fn budget(
    State(pool): State<PgPool>,
    AuthTeam(claims): AuthTeam,
) -> Result<Json<BudgetResponse>, (StatusCode, String)> {
    // Find active round
    let round: Option<(String, String)> =
        sqlx::query_as("SELECT id, status FROM rounds WHERE status = 'active' LIMIT 1")
            .fetch_optional(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (round_id, _status) = round.ok_or((StatusCode::NOT_FOUND, "No active round".to_string()))?;

    let queries_used: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1 AND round_id = $2",
    )
    .bind(&claims.team_id)
    .bind(&round_id)
    .fetch_one(&pool)
    .await
    .unwrap_or(0);

    Ok(Json(BudgetResponse {
        round_id,
        queries_used,
        queries_max: config::QUERY_BUDGET as i64,
        active: true,
    }))
}
