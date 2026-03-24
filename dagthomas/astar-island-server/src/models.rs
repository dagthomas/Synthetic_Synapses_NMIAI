//! Shared request/response types for the API.

use serde::{Deserialize, Serialize};

// ── Round types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RoundListEntry {
    pub id: String,
    pub round_number: i64,
    pub event_date: Option<String>,
    pub status: String,
    pub map_width: i64,
    pub map_height: i64,
    pub prediction_window_minutes: i64,
    pub started_at: Option<String>,
    pub closes_at: Option<String>,
    pub round_weight: f64,
    pub seeds_count: usize,
}

#[derive(Serialize)]
pub struct RoundDetail {
    pub id: String,
    pub round_number: i64,
    pub status: String,
    pub map_width: i64,
    pub map_height: i64,
    pub seeds_count: usize,
    pub initial_states: Vec<InitialState>,
}

#[derive(Serialize)]
pub struct InitialState {
    pub grid: Vec<Vec<u8>>,
    pub settlements: Vec<SettlementResponse>,
}

#[derive(Serialize, Clone)]
pub struct SettlementResponse {
    pub x: usize,
    pub y: usize,
    pub has_port: bool,
    pub alive: bool,
}

// ── Simulate types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SimulateRequest {
    pub round_id: String,
    pub seed_index: i64,
    pub viewport_x: Option<usize>,
    pub viewport_y: Option<usize>,
    pub viewport_w: Option<usize>,
    pub viewport_h: Option<usize>,
}

#[derive(Serialize)]
pub struct SimulateResponse {
    pub grid: Vec<Vec<u8>>,
    pub settlements: Vec<SimSettlement>,
    pub viewport: ViewportInfo,
    pub width: usize,
    pub height: usize,
    pub queries_used: i64,
    pub queries_max: i64,
}

#[derive(Serialize)]
pub struct SimSettlement {
    pub x: usize,
    pub y: usize,
    pub population: f64,
    pub food: f64,
    pub wealth: f64,
    pub defense: f64,
    pub has_port: bool,
    pub alive: bool,
    pub owner_id: u8,
}

#[derive(Serialize)]
pub struct ViewportInfo {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

// ── Submit types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SubmitRequest {
    pub round_id: String,
    pub seed_index: i64,
    pub prediction: Vec<Vec<Vec<f64>>>,
}

#[derive(Serialize)]
pub struct SubmitResponse {
    pub status: String,
    pub round_id: String,
    pub seed_index: i64,
}

// ── Budget types ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct BudgetResponse {
    pub round_id: String,
    pub queries_used: i64,
    pub queries_max: i64,
    pub active: bool,
}

// ── Leaderboard types ───────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LeaderboardEntry {
    pub team_id: String,
    pub team_name: String,
    pub weighted_score: f64,
    pub rounds_participated: i64,
    pub rank: i64,
}

// ── Analysis types ──────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct AnalysisResponse {
    pub prediction: Vec<Vec<Vec<f64>>>,
    pub ground_truth: Vec<Vec<Vec<f64>>>,
    pub score: Option<f64>,
    pub width: usize,
    pub height: usize,
    pub initial_grid: Vec<Vec<u8>>,
}

// ── My rounds types ─────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MyRoundEntry {
    pub id: String,
    pub round_number: i64,
    pub status: String,
    pub map_width: i64,
    pub map_height: i64,
    pub seeds_count: usize,
    pub round_weight: f64,
    pub started_at: Option<String>,
    pub closes_at: Option<String>,
    pub prediction_window_minutes: i64,
    pub round_score: Option<f64>,
    pub seed_scores: Option<Vec<Option<f64>>>,
    pub seeds_submitted: i64,
    pub rank: Option<i64>,
    pub queries_used: i64,
    pub queries_max: i64,
}

#[derive(Serialize)]
pub struct MyPredictionEntry {
    pub seed_index: i64,
    pub argmax_grid: Vec<Vec<usize>>,
    pub confidence_grid: Vec<Vec<f64>>,
    pub score: Option<f64>,
    pub submitted_at: Option<String>,
}

// ── Admin types ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateRoundRequest {
    pub regime: Option<String>,
    pub custom_params: Option<crate::simulation::params::SimParams>,
}

#[derive(Serialize)]
pub struct AdminDashboard {
    pub active_round: Option<RoundListEntry>,
    pub team_count: i64,
    pub total_predictions: i64,
    pub total_rounds: i64,
}

#[derive(Serialize)]
pub struct AdminRoundDetail {
    pub id: String,
    pub round_number: i64,
    pub status: String,
    pub hidden_params: crate::simulation::params::SimParams,
    pub seeds: Vec<AdminSeedInfo>,
    pub team_scores: Vec<TeamRoundScore>,
}

#[derive(Serialize)]
pub struct AdminSeedInfo {
    pub seed_index: i64,
    pub map_seed: i64,
    pub initial_grid: Vec<Vec<u8>>,
    pub settlement_count: usize,
}

#[derive(Serialize)]
pub struct TeamRoundScore {
    pub team_id: String,
    pub team_name: String,
    pub seed_scores: Vec<Option<f64>>,
    pub average_score: Option<f64>,
    pub queries_used: i64,
}

// ── Admin rounds list with grid ────────────────────────────────────────

#[derive(Serialize)]
pub struct AdminRoundListEntry {
    pub id: String,
    pub round_number: i64,
    pub status: String,
    pub round_weight: f64,
    pub started_at: Option<String>,
    pub closes_at: Option<String>,
    pub created_at: String,
    pub teams_participated: i64,
    pub avg_score: Option<f64>,
    pub first_seed_grid: Option<Vec<Vec<u8>>>,
}

// ── Stats endpoint types ───────────────────────────────────────────────

#[derive(Serialize)]
pub struct RoundStatsEntry {
    pub round_number: i64,
    pub round_id: String,
    pub status: String,
    pub teams_participated: i64,
    pub predictions_count: i64,
    pub avg_score: Option<f64>,
    pub min_score: Option<f64>,
    pub max_score: Option<f64>,
    pub stddev_score: Option<f64>,
    pub score_buckets: Vec<i64>,
    pub round_weight: f64,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct RoundStatsResponse {
    pub rounds: Vec<RoundStatsEntry>,
}

#[derive(Serialize)]
pub struct TeamStatsEntry {
    pub team_id: String,
    pub team_name: String,
    pub rounds_participated: i64,
    pub total_predictions: i64,
    pub total_queries: i64,
    pub avg_score: Option<f64>,
    pub best_score: Option<f64>,
    pub worst_score: Option<f64>,
    pub avg_queries_per_round: f64,
    pub scores_by_round: Vec<TeamRoundStatsEntry>,
    pub cumulative_weighted_score: f64,
}

#[derive(Serialize)]
pub struct TeamRoundStatsEntry {
    pub round_number: i64,
    pub round_id: String,
    pub avg_score: Option<f64>,
    pub queries_used: i64,
    pub round_weight: f64,
}

#[derive(Serialize)]
pub struct TeamStatsResponse {
    pub teams: Vec<TeamStatsEntry>,
}

#[derive(Serialize)]
pub struct SeedStatsEntry {
    pub seed_index: i64,
    pub avg_score: f64,
    pub count: i64,
}

#[derive(Serialize)]
pub struct TeamRoundMatrixEntry {
    pub team_id: String,
    pub team_name: String,
    pub round_scores: Vec<Option<f64>>,
}

#[derive(Serialize)]
pub struct PredictionStatsResponse {
    pub total: i64,
    pub scored: i64,
    pub unscored: i64,
    pub global_avg: Option<f64>,
    pub global_stddev: Option<f64>,
    pub score_histogram: Vec<i64>,
    pub per_seed_stats: Vec<SeedStatsEntry>,
    pub team_round_matrix: Vec<TeamRoundMatrixEntry>,
    pub round_numbers: Vec<i64>,
}

#[derive(Serialize)]
pub struct TeamQueryBudget {
    pub team_name: String,
    pub total_queries: i64,
    pub avg_per_round: f64,
}

#[derive(Serialize)]
pub struct RoundQueryCount {
    pub round_number: i64,
    pub total: i64,
}

#[derive(Serialize)]
pub struct QueryStatsResponse {
    pub total_queries: i64,
    pub avg_per_team: f64,
    pub avg_per_round: f64,
    pub viewport_heatmap: Vec<Vec<i64>>,
    pub per_team_budget: Vec<TeamQueryBudget>,
    pub queries_by_round: Vec<RoundQueryCount>,
}

#[derive(Serialize)]
pub struct ParamRoundEntry {
    pub round_number: i64,
    pub avg_score: Option<f64>,
    pub params: serde_json::Value,
}

#[derive(Serialize)]
pub struct ParamStatsResponse {
    pub rounds: Vec<ParamRoundEntry>,
}

#[derive(Serialize)]
pub struct TeamDetailRound {
    pub round_id: String,
    pub round_number: i64,
    pub status: String,
    pub seed_scores: Vec<Option<f64>>,
    pub avg_score: Option<f64>,
    pub queries_used: i64,
    pub round_weight: f64,
    pub weighted_contribution: f64,
}

#[derive(Serialize)]
pub struct TeamDetailResponse {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub is_admin: bool,
    pub total_queries: i64,
    pub rounds: Vec<TeamDetailRound>,
    pub cumulative_weighted_score: f64,
    pub rank: i64,
}
