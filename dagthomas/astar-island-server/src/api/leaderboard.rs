use axum::{extract::State, http::StatusCode, Json};
use sqlx::PgPool;

use crate::models::LeaderboardEntry;

/// GET /astar-island/leaderboard — Global leaderboard sorted by best weighted score.
pub async fn leaderboard(
    State(pool): State<PgPool>,
) -> Result<Json<Vec<LeaderboardEntry>>, (StatusCode, String)> {
    // For each team, compute: best (avg_seed_score * round_weight) across all completed rounds
    let rows: Vec<(String, String)> =
        sqlx::query_as("SELECT id, name FROM teams WHERE is_admin = FALSE")
            .fetch_all(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let mut entries: Vec<(String, String, f64, i64)> = Vec::new();

    for (team_id, team_name) in &rows {
        // Get all completed rounds this team participated in
        let round_scores: Vec<(String, f64)> = sqlx::query_as(
            "SELECT r.id, r.round_weight::DOUBLE PRECISION FROM rounds r
             WHERE r.status = 'completed'
             AND EXISTS (SELECT 1 FROM predictions p WHERE p.round_id = r.id AND p.team_id = $1)"
        )
        .bind(team_id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        let mut best_weighted = 0.0f64;
        let mut rounds_participated = 0i64;

        for (rid, round_weight) in &round_scores {
            rounds_participated += 1;

            // Average seed score for this team+round
            let avg_score: Option<f64> = sqlx::query_scalar(
                "SELECT AVG(score::DOUBLE PRECISION) FROM predictions WHERE team_id = $1 AND round_id = $2 AND score IS NOT NULL"
            )
            .bind(team_id)
            .bind(rid)
            .fetch_one(&pool)
            .await
            .unwrap_or(None);

            if let Some(avg) = avg_score {
                let weighted = avg * round_weight;
                if weighted > best_weighted {
                    best_weighted = weighted;
                }
            }
        }

        entries.push((team_id.clone(), team_name.clone(), best_weighted, rounds_participated));
    }

    // Sort by weighted score descending
    entries.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let result: Vec<LeaderboardEntry> = entries
        .into_iter()
        .enumerate()
        .map(|(i, (team_id, team_name, weighted_score, rounds_participated))| {
            LeaderboardEntry {
                team_id,
                team_name,
                weighted_score,
                rounds_participated,
                rank: (i + 1) as i64,
            }
        })
        .collect();

    Ok(Json(result))
}
