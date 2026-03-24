use axum::{extract::{Path, State}, http::StatusCode, Json};
use sqlx::PgPool;

use crate::auth::middleware::AdminAuth;
use crate::config;
use crate::models::*;
use crate::simulation::{
    cell::NUM_CLASSES, engine, grid::Grid, mapgen, params::{param_ranges, Regime, SimParams},
    scoring, terrain::SettlementInfo,
};

/// GET /admin/api/dashboard — Overview stats.
pub async fn dashboard(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<AdminDashboard>, (StatusCode, String)> {
    let active_round: Option<(String, i64, String, i64, i64, i64, Option<String>, Option<String>, f64, String)> =
        sqlx::query_as(
            "SELECT id, round_number::BIGINT, status, map_width::BIGINT, map_height::BIGINT, prediction_window_minutes::BIGINT, started_at, closes_at, round_weight::DOUBLE PRECISION, created_at FROM rounds WHERE status = 'active' LIMIT 1"
        )
        .fetch_optional(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let team_count: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM teams WHERE is_admin = FALSE")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_predictions: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM predictions")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM rounds")
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
    State(pool): State<PgPool>,
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
        let max_num: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(round_number), 0)::BIGINT FROM rounds")
            .fetch_one(&pool)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;
        let round_number = max_num + 1;

        let round_id = uuid::Uuid::new_v4().to_string();
        let round_weight = 1.05f64.powi(round_number as i32);

        (params, round_number, round_id, round_weight, params_json)
    };

    sqlx::query(
        "INSERT INTO rounds (id, round_number, status, hidden_params, round_weight) VALUES ($1, $2, 'pending', $3, $4)"
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
            "INSERT INTO seeds (id, round_id, seed_index, map_seed, initial_grid, settlements, ground_truth) VALUES ($1, $2, $3, $4, $5, $6, $7)"
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
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let now = chrono::Utc::now();
    let closes_at = now + chrono::Duration::minutes(config::DEFAULT_PREDICTION_WINDOW_MINUTES as i64);

    sqlx::query("UPDATE rounds SET status = 'active', started_at = $1, closes_at = $2 WHERE id = $3")
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
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Set status to scoring
    sqlx::query("UPDATE rounds SET status = 'scoring' WHERE id = $1")
        .bind(&round_id)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    // Load ground truths for all seeds
    let seed_gts: Vec<(i64, String)> = sqlx::query_as(
        "SELECT seed_index::BIGINT, ground_truth FROM seeds WHERE round_id = $1 ORDER BY seed_index",
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
        "SELECT id, team_id, seed_index::BIGINT, tensor FROM predictions WHERE round_id = $1",
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

            sqlx::query("UPDATE predictions SET score = $1 WHERE id = $2")
                .bind(score)
                .bind(pred_id)
                .execute(&pool)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;
            scored += 1;
        }
    }

    // Set status to completed
    sqlx::query("UPDATE rounds SET status = 'completed' WHERE id = $1")
        .bind(&round_id)
        .execute(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    Ok(Json(serde_json::json!({ "status": "completed", "predictions_scored": scored })))
}

/// GET /admin/api/rounds/{id} — Round detail with hidden params and team scores.
pub async fn round_detail(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
    Path(round_id): Path<String>,
) -> Result<Json<AdminRoundDetail>, (StatusCode, String)> {
    let row: Option<(String, i64, String, String)> = sqlx::query_as(
        "SELECT id, round_number::BIGINT, status, hidden_params FROM rounds WHERE id = $1",
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
        "SELECT seed_index::BIGINT, map_seed::BIGINT, initial_grid, settlements FROM seeds WHERE round_id = $1 ORDER BY seed_index",
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
            "SELECT seed_index::BIGINT, score::DOUBLE PRECISION FROM predictions WHERE team_id = $1 AND round_id = $2 ORDER BY seed_index",
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
            "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1 AND round_id = $2",
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
    State(pool): State<PgPool>,
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
            "SELECT COUNT(DISTINCT round_id)::BIGINT FROM predictions WHERE team_id = $1",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let total_queries: i64 = sqlx::query_scalar(
            "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1",
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

/// GET /admin/api/rounds — List all rounds (admin view, includes first seed grid).
pub async fn list_rounds_admin(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<Vec<AdminRoundListEntry>>, (StatusCode, String)> {
    let rows: Vec<(String, i64, String, f64, Option<String>, Option<String>, String)> =
        sqlx::query_as(
            "SELECT id, round_number::BIGINT, status, round_weight::DOUBLE PRECISION, started_at, closes_at, created_at FROM rounds ORDER BY round_number DESC"
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut result = Vec::new();
    for (id, rn, status, rw, sa, ca, cd) in &rows {
        let team_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT team_id)::BIGINT FROM predictions WHERE round_id = $1",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let avg_score: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(score::DOUBLE PRECISION) FROM predictions WHERE round_id = $1 AND score IS NOT NULL",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap_or(None);

        // Fetch first seed grid for thumbnail
        let first_grid: Option<(String,)> = sqlx::query_as(
            "SELECT initial_grid FROM seeds WHERE round_id = $1 AND seed_index = 0 LIMIT 1",
        )
        .bind(id)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        let first_seed_grid: Option<Vec<Vec<u8>>> = first_grid
            .and_then(|(gj,)| serde_json::from_str(&gj).ok());

        result.push(AdminRoundListEntry {
            id: id.clone(),
            round_number: *rn,
            status: status.clone(),
            round_weight: *rw,
            started_at: sa.clone(),
            closes_at: ca.clone(),
            created_at: cd.clone(),
            teams_participated: team_count,
            avg_score,
            first_seed_grid,
        });
    }

    Ok(Json(result))
}

/// GET /admin/api/stats — Global statistics.
pub async fn stats(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let team_count: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM teams WHERE is_admin = FALSE")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM rounds")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let completed_rounds: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM rounds WHERE status = 'completed'")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_predictions: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM predictions")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let total_queries: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM query_log")
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

    let avg_score: Option<f64> = sqlx::query_scalar(
        "SELECT AVG(score::DOUBLE PRECISION) FROM predictions WHERE score IS NOT NULL"
    )
    .fetch_one(&pool)
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

/// GET /admin/api/stats/rounds — Per-round statistics with score distributions.
pub async fn stats_rounds(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<RoundStatsResponse>, (StatusCode, String)> {
    let rows: Vec<(String, i64, String, f64, String)> = sqlx::query_as(
        "SELECT id, round_number::BIGINT, status, round_weight::DOUBLE PRECISION, created_at FROM rounds ORDER BY round_number"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut rounds = Vec::new();
    for (rid, rn, status, rw, created_at) in &rows {
        let agg: Option<(i64, Option<f64>, Option<f64>, Option<f64>, Option<f64>)> = sqlx::query_as(
            "SELECT COUNT(*)::BIGINT, AVG(score::DOUBLE PRECISION), MIN(score::DOUBLE PRECISION), MAX(score::DOUBLE PRECISION), STDDEV(score::DOUBLE PRECISION) FROM predictions WHERE round_id = $1 AND score IS NOT NULL"
        )
        .bind(rid)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        let (pred_count, avg, min, max, stddev) = agg.unwrap_or((0, None, None, None, None));

        let teams_participated: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT team_id)::BIGINT FROM predictions WHERE round_id = $1"
        )
        .bind(rid)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        // Compute score buckets (10 bins: 0-10, 10-20, ..., 90-100)
        let scores: Vec<(f64,)> = sqlx::query_as(
            "SELECT score::DOUBLE PRECISION FROM predictions WHERE round_id = $1 AND score IS NOT NULL"
        )
        .bind(rid)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        let mut buckets = vec![0i64; 10];
        for (s,) in &scores {
            let idx = ((*s / 10.0).floor() as usize).min(9);
            buckets[idx] += 1;
        }

        rounds.push(RoundStatsEntry {
            round_number: *rn,
            round_id: rid.clone(),
            status: status.clone(),
            teams_participated,
            predictions_count: pred_count,
            avg_score: avg,
            min_score: min,
            max_score: max,
            stddev_score: stddev,
            score_buckets: buckets,
            round_weight: *rw,
            created_at: created_at.clone(),
        });
    }

    Ok(Json(RoundStatsResponse { rounds }))
}

/// GET /admin/api/stats/teams — Per-team performance metrics.
pub async fn stats_teams(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<TeamStatsResponse>, (StatusCode, String)> {
    let teams: Vec<(String, String)> = sqlx::query_as(
        "SELECT id, name FROM teams WHERE is_admin = FALSE ORDER BY name"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut result = Vec::new();
    for (tid, tname) in &teams {
        let agg: Option<(i64, i64, Option<f64>, Option<f64>, Option<f64>)> = sqlx::query_as(
            "SELECT COUNT(DISTINCT round_id)::BIGINT, COUNT(*)::BIGINT, AVG(score::DOUBLE PRECISION), MAX(score::DOUBLE PRECISION), MIN(score::DOUBLE PRECISION) FROM predictions WHERE team_id = $1 AND score IS NOT NULL"
        )
        .bind(tid)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        let (rounds_participated, total_preds, avg, best, worst) = agg.unwrap_or((0, 0, None, None, None));

        let total_queries: i64 = sqlx::query_scalar(
            "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1"
        )
        .bind(tid)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);

        let avg_qpr = if rounds_participated > 0 {
            total_queries as f64 / rounds_participated as f64
        } else {
            0.0
        };

        // Per-round scores
        let round_rows: Vec<(String, i64, Option<f64>, f64)> = sqlx::query_as(
            "SELECT p.round_id, r.round_number::BIGINT, AVG(p.score::DOUBLE PRECISION), r.round_weight::DOUBLE PRECISION FROM predictions p JOIN rounds r ON p.round_id = r.id WHERE p.team_id = $1 AND p.score IS NOT NULL GROUP BY p.round_id, r.round_number, r.round_weight ORDER BY r.round_number"
        )
        .bind(tid)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

        let mut cumulative = 0.0;
        let mut scores_by_round = Vec::new();
        for (rid, rnum, avg_s, rw) in &round_rows {
            let qs: i64 = sqlx::query_scalar(
                "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1 AND round_id = $2"
            )
            .bind(tid)
            .bind(rid)
            .fetch_one(&pool)
            .await
            .unwrap_or(0);

            if let Some(a) = avg_s {
                cumulative += a * rw;
            }

            scores_by_round.push(TeamRoundStatsEntry {
                round_number: *rnum,
                round_id: rid.clone(),
                avg_score: *avg_s,
                queries_used: qs,
                round_weight: *rw,
            });
        }

        result.push(TeamStatsEntry {
            team_id: tid.clone(),
            team_name: tname.clone(),
            rounds_participated,
            total_predictions: total_preds,
            total_queries,
            avg_score: avg,
            best_score: best,
            worst_score: worst,
            avg_queries_per_round: avg_qpr,
            scores_by_round,
            cumulative_weighted_score: cumulative,
        });
    }

    Ok(Json(TeamStatsResponse { teams: result }))
}

/// GET /admin/api/stats/predictions — Global prediction statistics.
pub async fn stats_predictions(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<PredictionStatsResponse>, (StatusCode, String)> {
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM predictions")
        .fetch_one(&pool).await.unwrap_or(0);
    let scored: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM predictions WHERE score IS NOT NULL")
        .fetch_one(&pool).await.unwrap_or(0);
    let unscored = total - scored;

    let global_agg: Option<(Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT AVG(score::DOUBLE PRECISION), STDDEV(score::DOUBLE PRECISION) FROM predictions WHERE score IS NOT NULL"
    )
    .fetch_optional(&pool).await.unwrap_or(None);
    let (global_avg, global_stddev) = global_agg.unwrap_or((None, None));

    // Score histogram
    let all_scores: Vec<(f64,)> = sqlx::query_as(
        "SELECT score::DOUBLE PRECISION FROM predictions WHERE score IS NOT NULL"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let mut histogram = vec![0i64; 10];
    for (s,) in &all_scores {
        let idx = ((*s / 10.0).floor() as usize).min(9);
        histogram[idx] += 1;
    }

    // Per-seed stats
    let seed_rows: Vec<(i64, f64, i64)> = sqlx::query_as(
        "SELECT seed_index::BIGINT, AVG(score::DOUBLE PRECISION)::DOUBLE PRECISION, COUNT(*)::BIGINT FROM predictions WHERE score IS NOT NULL GROUP BY seed_index ORDER BY seed_index"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let per_seed_stats: Vec<SeedStatsEntry> = seed_rows.into_iter()
        .map(|(si, avg, cnt)| SeedStatsEntry { seed_index: si, avg_score: avg, count: cnt })
        .collect();

    // Team x Round matrix
    let round_nums: Vec<(i64,)> = sqlx::query_as(
        "SELECT round_number::BIGINT FROM rounds ORDER BY round_number"
    )
    .fetch_all(&pool).await.unwrap_or_default();
    let round_numbers: Vec<i64> = round_nums.iter().map(|(n,)| *n).collect();

    let teams: Vec<(String, String)> = sqlx::query_as(
        "SELECT id, name FROM teams WHERE is_admin = FALSE ORDER BY name"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let mut matrix = Vec::new();
    for (tid, tname) in &teams {
        let scores: Vec<(i64, Option<f64>)> = sqlx::query_as(
            "SELECT r.round_number::BIGINT, AVG(p.score::DOUBLE PRECISION) FROM predictions p JOIN rounds r ON p.round_id = r.id WHERE p.team_id = $1 AND p.score IS NOT NULL GROUP BY r.round_number ORDER BY r.round_number"
        )
        .bind(tid)
        .fetch_all(&pool).await.unwrap_or_default();

        let round_scores: Vec<Option<f64>> = round_numbers.iter().map(|rn| {
            scores.iter().find(|(n, _)| n == rn).and_then(|(_, s)| *s)
        }).collect();

        matrix.push(TeamRoundMatrixEntry {
            team_id: tid.clone(),
            team_name: tname.clone(),
            round_scores,
        });
    }

    Ok(Json(PredictionStatsResponse {
        total, scored, unscored, global_avg, global_stddev,
        score_histogram: histogram,
        per_seed_stats,
        team_round_matrix: matrix,
        round_numbers,
    }))
}

/// GET /admin/api/stats/queries — Query usage analytics.
pub async fn stats_queries(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<QueryStatsResponse>, (StatusCode, String)> {
    let total_queries: i64 = sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM query_log")
        .fetch_one(&pool).await.unwrap_or(0);

    let team_count: i64 = sqlx::query_scalar("SELECT COUNT(DISTINCT team_id)::BIGINT FROM query_log")
        .fetch_one(&pool).await.unwrap_or(1).max(1);

    let round_count: i64 = sqlx::query_scalar("SELECT COUNT(DISTINCT round_id)::BIGINT FROM query_log")
        .fetch_one(&pool).await.unwrap_or(1).max(1);

    let avg_per_team = total_queries as f64 / team_count as f64;
    let avg_per_round = total_queries as f64 / round_count as f64;

    // Viewport heatmap (40x40)
    let vp_rows: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT viewport_x::BIGINT, viewport_y::BIGINT FROM query_log"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let mut heatmap = vec![vec![0i64; 40]; 40];
    for (vx, vy) in &vp_rows {
        let x = (*vx as usize).min(39);
        let y = (*vy as usize).min(39);
        heatmap[y][x] += 1;
    }

    // Per-team budget
    let team_budget_rows: Vec<(String, i64, i64)> = sqlx::query_as(
        "SELECT t.name, COUNT(*)::BIGINT, COUNT(DISTINCT q.round_id)::BIGINT FROM query_log q JOIN teams t ON q.team_id = t.id GROUP BY t.name ORDER BY COUNT(*) DESC"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let per_team_budget: Vec<TeamQueryBudget> = team_budget_rows.into_iter().map(|(name, total, rounds)| {
        TeamQueryBudget {
            team_name: name,
            total_queries: total,
            avg_per_round: if rounds > 0 { total as f64 / rounds as f64 } else { 0.0 },
        }
    }).collect();

    // Queries by round
    let qbr_rows: Vec<(i64, i64)> = sqlx::query_as(
        "SELECT r.round_number::BIGINT, COUNT(*)::BIGINT FROM query_log q JOIN rounds r ON q.round_id = r.id GROUP BY r.round_number ORDER BY r.round_number"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let queries_by_round: Vec<RoundQueryCount> = qbr_rows.into_iter()
        .map(|(rn, total)| RoundQueryCount { round_number: rn, total })
        .collect();

    Ok(Json(QueryStatsResponse {
        total_queries, avg_per_team, avg_per_round,
        viewport_heatmap: heatmap,
        per_team_budget,
        queries_by_round,
    }))
}

/// GET /admin/api/stats/params — Parameter correlation data.
pub async fn stats_params(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
) -> Result<Json<ParamStatsResponse>, (StatusCode, String)> {
    let rows: Vec<(i64, String)> = sqlx::query_as(
        "SELECT round_number::BIGINT, hidden_params FROM rounds ORDER BY round_number"
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let mut rounds = Vec::new();
    for (rn, params_json) in &rows {
        let params: serde_json::Value = serde_json::from_str(params_json).unwrap_or_default();

        let avg_score: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(score::DOUBLE PRECISION) FROM predictions p JOIN rounds r ON p.round_id = r.id WHERE r.round_number = $1 AND p.score IS NOT NULL"
        )
        .bind(rn)
        .fetch_one(&pool)
        .await
        .unwrap_or(None);

        rounds.push(ParamRoundEntry {
            round_number: *rn,
            avg_score,
            params,
        });
    }

    Ok(Json(ParamStatsResponse { rounds }))
}

/// GET /admin/api/teams/{team_id} — Individual team detail.
pub async fn team_detail(
    State(pool): State<PgPool>,
    AdminAuth(_): AdminAuth,
    Path(team_id): Path<String>,
) -> Result<Json<TeamDetailResponse>, (StatusCode, String)> {
    let team: Option<(String, String, bool, String)> = sqlx::query_as(
        "SELECT id, name, is_admin, created_at FROM teams WHERE id = $1"
    )
    .bind(&team_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB: {e}")))?;

    let (id, name, is_admin, created_at) = team
        .ok_or((StatusCode::NOT_FOUND, "Team not found".to_string()))?;

    let total_queries: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1"
    )
    .bind(&team_id)
    .fetch_one(&pool).await.unwrap_or(0);

    // Per-round detail
    let round_rows: Vec<(String, i64, String, f64)> = sqlx::query_as(
        "SELECT DISTINCT r.id, r.round_number::BIGINT, r.status, r.round_weight::DOUBLE PRECISION FROM rounds r JOIN predictions p ON r.id = p.round_id WHERE p.team_id = $1 ORDER BY r.round_number"
    )
    .bind(&team_id)
    .fetch_all(&pool).await.unwrap_or_default();

    let mut cumulative = 0.0;
    let mut rounds_detail = Vec::new();
    for (rid, rnum, rstatus, rw) in &round_rows {
        let scores: Vec<(i64, Option<f64>)> = sqlx::query_as(
            "SELECT seed_index::BIGINT, score::DOUBLE PRECISION FROM predictions WHERE team_id = $1 AND round_id = $2 ORDER BY seed_index"
        )
        .bind(&team_id)
        .bind(rid)
        .fetch_all(&pool).await.unwrap_or_default();

        let seed_scores: Vec<Option<f64>> = (0..config::SEEDS_PER_ROUND as i64)
            .map(|si| scores.iter().find(|(idx, _)| *idx == si).and_then(|(_, s)| *s))
            .collect();

        let scored: Vec<f64> = seed_scores.iter().filter_map(|s| *s).collect();
        let avg = if scored.is_empty() { None } else { Some(scored.iter().sum::<f64>() / scored.len() as f64) };

        let weighted = avg.unwrap_or(0.0) * rw;
        cumulative += weighted;

        let qs: i64 = sqlx::query_scalar(
            "SELECT COUNT(*)::BIGINT FROM query_log WHERE team_id = $1 AND round_id = $2"
        )
        .bind(&team_id)
        .bind(rid)
        .fetch_one(&pool).await.unwrap_or(0);

        rounds_detail.push(TeamDetailRound {
            round_id: rid.clone(),
            round_number: *rnum,
            status: rstatus.clone(),
            seed_scores,
            avg_score: avg,
            queries_used: qs,
            round_weight: *rw,
            weighted_contribution: weighted,
        });
    }

    // Compute rank (compare cumulative weighted scores of all teams)
    let all_teams: Vec<(String,)> = sqlx::query_as(
        "SELECT id FROM teams WHERE is_admin = FALSE"
    )
    .fetch_all(&pool).await.unwrap_or_default();

    let mut all_scores: Vec<(String, f64)> = Vec::new();
    for (tid,) in &all_teams {
        let ws: Option<f64> = sqlx::query_scalar(
            "SELECT SUM(sub.avg_score * sub.round_weight)::DOUBLE PRECISION FROM (SELECT AVG(p.score::DOUBLE PRECISION) as avg_score, r.round_weight::DOUBLE PRECISION as round_weight FROM predictions p JOIN rounds r ON p.round_id = r.id WHERE p.team_id = $1 AND p.score IS NOT NULL GROUP BY p.round_id, r.round_weight) sub"
        )
        .bind(tid)
        .fetch_one(&pool).await.unwrap_or(None);
        all_scores.push((tid.clone(), ws.unwrap_or(0.0)));
    }
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rank = all_scores.iter().position(|(t, _)| t == &team_id).map(|p| p as i64 + 1).unwrap_or(0);

    Ok(Json(TeamDetailResponse {
        id, name, created_at, is_admin, total_queries,
        rounds: rounds_detail,
        cumulative_weighted_score: cumulative,
        rank,
    }))
}
