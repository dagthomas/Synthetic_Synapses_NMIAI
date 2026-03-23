use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};

use super::jwt::{verify_token, Claims};

/// Extractor for authenticated team. Checks Authorization header or access_token cookie.
pub struct AuthTeam(pub Claims);

impl<S: Send + Sync> FromRequestParts<S> for AuthTeam {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Try Authorization: Bearer <token>
        if let Some(auth_header) = parts.headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if let Some(token) = auth_str.strip_prefix("Bearer ") {
                    return verify_token(token)
                        .map(AuthTeam)
                        .map_err(|e| (StatusCode::UNAUTHORIZED, e));
                }
            }
        }

        // Try access_token cookie
        if let Some(cookie_header) = parts.headers.get("cookie") {
            if let Ok(cookies) = cookie_header.to_str() {
                for cookie in cookies.split(';') {
                    let cookie = cookie.trim();
                    if let Some(token) = cookie.strip_prefix("access_token=") {
                        return verify_token(token)
                            .map(AuthTeam)
                            .map_err(|e| (StatusCode::UNAUTHORIZED, e));
                    }
                }
            }
        }

        Err((
            StatusCode::UNAUTHORIZED,
            "Missing or invalid authentication".to_string(),
        ))
    }
}

/// Extractor that requires admin privileges.
pub struct AdminAuth(pub Claims);

impl<S: Send + Sync> FromRequestParts<S> for AdminAuth {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let AuthTeam(claims) = AuthTeam::from_request_parts(parts, state).await?;
        if !claims.is_admin {
            return Err((StatusCode::FORBIDDEN, "Admin access required".to_string()));
        }
        Ok(AdminAuth(claims))
    }
}
