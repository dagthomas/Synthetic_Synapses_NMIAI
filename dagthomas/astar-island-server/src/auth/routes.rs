use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use super::jwt;
use super::middleware::AuthTeam;

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub name: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub name: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub token: String,
    pub team_id: String,
    pub team_name: String,
    pub is_admin: bool,
}

#[derive(Serialize)]
pub struct TeamInfo {
    pub team_id: String,
    pub team_name: String,
    pub is_admin: bool,
}

pub async fn register(
    State(pool): State<SqlitePool>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, (StatusCode, String)> {
    if req.name.trim().is_empty() || req.password.len() < 3 {
        return Err((StatusCode::BAD_REQUEST, "Name required, password min 3 chars".to_string()));
    }

    let id = uuid::Uuid::new_v4().to_string();
    let hash = jwt::hash_password(&req.password)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    sqlx::query("INSERT INTO teams (id, name, password_hash) VALUES (?, ?, ?)")
        .bind(&id)
        .bind(&req.name)
        .bind(&hash)
        .execute(&pool)
        .await
        .map_err(|e| {
            if e.to_string().contains("UNIQUE") {
                (StatusCode::CONFLICT, format!("Team name '{}' already taken", req.name))
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}"))
            }
        })?;

    let token = jwt::create_token(&id, &req.name, false)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(AuthResponse {
        token,
        team_id: id,
        team_name: req.name,
        is_admin: false,
    }))
}

pub async fn login(
    State(pool): State<SqlitePool>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, (StatusCode, String)> {
    let row: Option<(String, String, bool)> = sqlx::query_as(
        "SELECT id, password_hash, is_admin FROM teams WHERE name = ?",
    )
    .bind(&req.name)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    let (id, hash, is_admin) = row.ok_or((StatusCode::UNAUTHORIZED, "Invalid credentials".to_string()))?;

    let valid = jwt::verify_password(&req.password, &hash)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    if !valid {
        return Err((StatusCode::UNAUTHORIZED, "Invalid credentials".to_string()));
    }

    let token = jwt::create_token(&id, &req.name, is_admin)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(AuthResponse {
        token,
        team_id: id,
        team_name: req.name,
        is_admin,
    }))
}

pub async fn me(AuthTeam(claims): AuthTeam) -> Json<TeamInfo> {
    Json(TeamInfo {
        team_id: claims.team_id,
        team_name: claims.team_name,
        is_admin: claims.is_admin,
    })
}
