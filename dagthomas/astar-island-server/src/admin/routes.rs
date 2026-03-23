use axum::{extract::{Path, State}, http::StatusCode, Json};
use sqlx::SqlitePool;

use crate::auth::middleware::AdminAuth;
use crate::config;
use crate::models::*;
use crate::simulation::{
    cell::NUM_CLASSES, engine, grid::Grid, mapgen, params::{param_ranges, Regime, SimParams},
    scoring, terrain::SettlementInfo,
};

/// GET /admin/api/dashboard — Overview stats.
pub async fn dashboard(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<AdminDashboard>, (StatusCode, String)> {
    let active_round: Option<(String, i64, String, i64, i64, i64, Option<String>, Option<String>, f64, String)> =
        sqlx::query_as(
            "SELECT id, round_number, status, map_width, map_height, prediction_window_minutes, started_at, closes_at, round_weight, created_at FROM rounds WHERE status = 'active' LIMIT 1"
        )
        .fetch_optional(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let team_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM teams WHERE is_admin = FALSE")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_predictions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM predictions")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM rounds")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let active = active_round.map(|(id, rn, status, mw, mh, pw, sa, ca, rw, cd)| {
        RoundListEntry {
            id, round_number: rn, status, map_width: mw, map_height: mh,
            prediction_window_minutes: pw, started_at: sa, closes_at: ca,
            round_weight: rw, event_date: Some(cd), seeds_count: config::SEEDS_PER_ROUND,
        }
    });

    Ok(Json(AdminDashboard {
        active_round: active,
        team_count,
        total_predictions,
        total_rounds,
    }))
}

/// POST /admin/api/rounds — Create a new round.
#[axum::debug_handler]
pub async fn create_round(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
    Json(req): Json<CreateRoundRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Use a Send-safe seeded RNG (not thread_rng which is !Send across await points)
    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let (params, round_number, round_id, round_weight, params_json) = {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(time_seed);

        let params = if let Some(custom) = req.custom_params {
            custom
        } else {
            let regime = match req.regime.as_deref() {
                Some("collapse") => Regime::Collapse,
                Some("moderate") => Regime::Moderate,
                Some("boom") => Regime::Boom,
                _ => Regime::Random,
            };
            SimParams::from_regime(regime, &mut rng)
        };

        let params_json = serde_json::to_string(&params)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON: {e}")))?;

        // Determine round number
        let max_num: Option<i64> = sqlx::query_scalar("SELECT MAX(round_number) FROM rounds")
            .fetch_optional(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;
        let round_number = max_num.unwrap_or(0) + 1;

        let round_id = uuid::Uuid::new_v4().to_string();
        let round_weight = 1.05f64.powi(round_number as i32);

        (params, round_number, round_id, round_weight, params_json)
    };

    sqlx::query(
        "INSERT INTO rounds (id, round_number, status, hidden_params, round_weight) VALUES (?, ?, 'pending', ?, ?)"
    )
    .bind(&round_id)
    .bind(round_number)
    .bind(&params_json)
    .bind(round_weight)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    // Generate 5 seeds with maps and ground truth
    tracing::info!("Generating {n} seeds for round {round_number}...", n = config::SEEDS_PER_ROUND);
    for seed_idx in 0..config::SEEDS_PER_ROUND {
        let map_seed = time_seed.wrapping_add(seed_idx as u64).wrapping_mul(0x517cc1b727220a95);
        let (grid, settlements) = mapgen::generate_map(map_seed, config::MAP_W, config::MAP_H);

        // Compute ground truth (200 MC runs)
        let gt = engine::simulate_monte_carlo(
            &grid, &settlements, &params, config::MC_RUNS, map_seed,
        );

        let grid_json = serde_json::to_string(&grid.to_2d())
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON: {e}")))?;
        let setts_json = serde_json::to_string(&settlements)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON: {e}")))?;
        let gt_json = serde_json::to_string(&gt)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON: {e}")))?;

        let seed_id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO seeds (id, round_id, seed_index, map_seed, initial_grid, settlements, ground_truth) VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&seed_id)
        .bind(&round_id)
        .bind(seed_idx as i64)
        .bind(map_seed as i64)
        .bind(&grid_json)
        .bind(&setts_json)
        .bind(&gt_json)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

        tracing::info!("  Seed {seed_idx}: map_seed={map_seed}, {n} settlements", n = settlements.len());
    }

    Ok(Json(serde_json::json!({
        "id": round_id,
        "round_number": round_number,
        "status": "pending",
    })))
}

/// POST /admin/api/rounds/{id}/activate — Activate a round.
pub async fn activate_round(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let now = chrono::Utc::now();
    let closes_at = now + chrono::Duration::minutes(config::DEFAULT_PREDICTION_WINDOW_MINUTES as i64);

    sqlx::query("UPDATE rounds SET status = 'active', started_at = ?, closes_at = ? WHERE id = ?")
        .bind(now.to_rfc3339())
        .bind(closes_at.to_rfc3339())
        .bind(&round_id)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    Ok(Json(serde_json::json!({ "status": "active", "closes_at": closes_at.to_rfc3339() })))
}

/// POST /admin/api/rounds/{id}/score — Score all predictions for a round.
pub async fn score_round(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Set status to scoring
    sqlx::query("UPDATE rounds SET status = 'scoring' WHERE id = ?")
        .bind(&round_id)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    // Load ground truths for all seeds
    let seed_gts: Vec<(i64, String)> = sqlx::query_as(
        "SELECT seed_index, ground_truth FROM seeds WHERE round_id = ? ORDER BY seed_index",
    )
    .bind(&round_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut ground_truths: Vec<Vec<[f64; NUM_CLASSES]>> = Vec::new();
    for (_si, gt_json) in &seed_gts {
        let gt: Vec<[f64; NUM_CLASSES]> = serde_json::from_str(gt_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("GT JSON: {e}")))?;
        ground_truths.push(gt);
    }

    // Score all predictions
    let predictions: Vec<(String, String, i64, String)> = sqlx::query_as(
        "SELECT id, team_id, seed_index, tensor FROM predictions WHERE round_id = ?",
    )
    .bind(&round_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut scored = 0;
    for (pred_id, _team_id, seed_index, tensor_json) in &predictions {
        let pred_3d: Vec<Vec<Vec<f64>>> = serde_json::from_str(tensor_json)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Pred JSON: {e}")))?;

        // Convert [H][W][6] to flat [[f64; 6]; H*W]
        let mut pred_flat: Vec<[f64; NUM_CLASSES]> = Vec::new();
        for row in &pred_3d {
            for probs in row {
                let mut arr = [0.0; NUM_CLASSES];
                for (i, &v) in probs.iter().enumerate().take(NUM_CLASSES) {
                    arr[i] = v;
                }
                pred_flat.push(arr);
            }
        }

        if let Some(gt) = ground_truths.get(*seed_index as usize) {
            let score = scoring::compute_score(gt, &pred_flat);

            sqlx::query("UPDATE predictions SET score = ? WHERE id = ?")
                .bind(score)
                .bind(pred_id)
                .execute(&pool)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;
            scored += 1;
        }
    }

    // Set status to completed
    sqlx::query("UPDATE rounds SET status = 'completed' WHERE id = ?")
        .bind(&round_id)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    Ok(Json(serde_json::json!({ "status": "completed", "predictions_scored": scored })))
}

/// GET /admin/api/rounds/{id} — Round detail with hidden params and team scores.
pub async fn round_detail(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<AdminRoundDetail>, (StatusCode, String)> {
    let row: Option<(String, i64, String, String)> = sqlx::query_as(
        "SELECT id, round_number, status, hidden_params FROM rounds WHERE id = ?",
    )
    .bind(&round_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let (id, round_number, status, params_json) =
        row.ok_or((StatusCode::NOT_FOUND, "Round not found".to_string()))?;

    let hidden_params: SimParams = serde_json::from_str(&params_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Params JSON: {e}")))?;

    // Load seeds
    let seed_rows: Vec<(i64, i64, String, String)> = sqlx::query_as(
        "SELECT seed_index, map_seed, initial_grid, settlements FROM seeds WHERE round_id = ? ORDER BY seed_index",
    )
    .bind(&round_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let seeds: Vec<AdminSeedInfo> = seed_rows
        .into_iter()
        .map(|(si, ms, gj, sj)| {
            let grid: Vec<Vec<u8>> = serde_json::from_str(&gj).unwrap_or_default();
            let setts: Vec<SettlementInfo> = serde_json::from_str(&sj).unwrap_or_default();
            AdminSeedInfo {
                seed_index: si,
                map_seed: ms,
                initial_grid: grid,
                settlement_count: setts.len(),
            }
        })
        .collect();

    // Load team scores
    let teams: Vec<(String, String)> =
        sqlx::query_as("SELECT id, name FROM teams WHERE is_admin = FALSE")
            .fetch_all(&pool)
            .await
            .unwrap_or_default();

    let mut team_scores = Vec::new();
    for (team_id, team_name) in &teams {
        let scores: Vec<(i64, Option<f64>)> = sqlx::query_as(
            "SELECT seed_index, score FROM predictions WHERE team_id = ? AND round_id = ? ORDER BY seed_index",
        )
        .bind(team_id)
        .bind(&round_id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        if scores.is_empty() {
            continue;
        }

        let seed_scores: Vec<Option<f64>> = (0..config::SEEDS_PER_ROUND as i64)
            .map(|si| scores.iter().find(|(idx, _)| *idx == si).and_then(|(_, s)| *s))
            .collect();

        let scored: Vec<f64> = seed_scores.iter().filter_map(|s| *s).collect();
        let avg = if scored.is_empty() {
            None
        } else {
            Some(scored.iter().sum::<f64>() / scored.len() as f64)
        };

        let queries_used: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM query_log WHERE team_id = ? AND round_id = ?",
        )
        .bind(team_id)
        .bind(&round_id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        team_scores.push(TeamRoundScore {
            team_id: team_id.clone(),
            team_name: team_name.clone(),
            seed_scores,
            average_score: avg,
            queries_used,
        });
    }

    Ok(Json(AdminRoundDetail {
        id,
        round_number,
        status,
        hidden_params,
        seeds,
        team_scores,
    }))
}

/// GET /admin/api/teams — List all teams with stats.
pub async fn list_teams(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let teams: Vec<(String, String, bool, String)> =
        sqlx::query_as("SELECT id, name, is_admin, created_at FROM teams ORDER BY created_at")
            .fetch_all(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut result = Vec::new();
    for (id, name, is_admin, created_at) in &teams {
        let rounds_participated: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT round_id) FROM predictions WHERE team_id = ?",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let total_queries: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM query_log WHERE team_id = ?",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        result.push(serde_json::json!({
            "id": id,
            "name": name,
            "is_admin": is_admin,
            "created_at": created_at,
            "rounds_participated": rounds_participated,
            "total_queries": total_queries,
        }));
    }

    Ok(Json(result))
}

/// GET /admin/api/rounds — List all rounds (admin view, includes more detail).
pub async fn list_rounds_admin(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let rows: Vec<(String, i64, String, f64, Option<String>, Option<String>, String)> =
        sqlx::query_as(
            "SELECT id, round_number, status, round_weight, started_at, closes_at, created_at FROM rounds ORDER BY round_number DESC"
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut result = Vec::new();
    for (id, rn, status, rw, sa, ca, cd) in &rows {
        let team_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT team_id) FROM predictions WHERE round_id = ?",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let avg_score: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(score) FROM predictions WHERE round_id = ? AND score IS NOT NULL",
        )
        .bind(id)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        result.push(serde_json::json!({
            "id": id,
            "round_number": rn,
            "status": status,
            "round_weight": rw,
            "started_at": sa,
            "closes_at": ca,
            "created_at": cd,
            "teams_participated": team_count,
            "avg_score": avg_score,
        }));
    }

    Ok(Json(result))
}

/// GET /admin/api/stats — Global statistics.
pub async fn stats(
    State(pool): State<SqlitePool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let team_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM teams WHERE is_admin = FALSE")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM rounds")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let completed_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM rounds WHERE status = 'completed'")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_predictions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM predictions")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_queries: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM query_log")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let avg_score: Option<f64> = sqlx::query_scalar(
        "SELECT AVG(score) FROM predictions WHERE score IS NOT NULL"
    )
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);

    let param_ranges_data = param_ranges();

    Ok(Json(serde_json::json!({
        "team_count": team_count,
        "total_rounds": total_rounds,
        "completed_rounds": completed_rounds,
        "total_predictions": total_predictions,
        "total_queries": total_queries,
        "avg_score": avg_score,
        "param_ranges": param_ranges_data,
    })))
}
