//! Google OAuth 2.0 (PKCE) for Vertex AI — ported from layream-core.
//!
//! Flow (mobile, no client_secret):
//! 1. `generate_pkce()` → verifier persisted, challenge into auth URL
//! 2. user approves in browser/GeckoView → redirect carries `code`
//! 3. `exchange_code()` with the verifier → `Tokens`
//! 4. `get_valid_token()` refreshes via `refresh_token` when within the
//!    5-minute expiry margin
//!
//! The default client ID/redirect URI are app-specific Google OAuth client
//! values; override them in settings when shipping under a different
//! Google Cloud project.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::PomolluError;

const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const REVOKE_ENDPOINT: &str = "https://oauth2.googleapis.com/revoke";
const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/cloudplatformprojects.readonly";

const REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// Default Vertex AI OAuth client (Google Cloud project of the app author).
pub const DEFAULT_CLIENT_ID: &str =
    "317210024447-v4g6e0e1q5933vogajp0651vhkrgal06.apps.googleusercontent.com";
/// Reverse-client-ID custom scheme redirect, intercepted by the Android
/// deep-link handler / GeckoView OAuth dialog.
pub const DEFAULT_REDIRECT_URI: &str =
    "com.googleusercontent.apps.317210024447-v4g6e0e1q5933vogajp0651vhkrgal06:/oauth2callback";

/// Gemini Code Assist OAuth client (Google's public installed-app credentials
/// for the Code Assist / gemini-cli backend). `client_secret` here is not a
/// confidential secret in the OAuth sense — installed apps ship it in the
/// clear — but the `v1internal` endpoint requires this exact client.
pub const GCA_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
pub const GCA_CLIENT_SECRET: &str = "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";
pub const GCA_REDIRECT_URI: &str =
    "com.googleusercontent.apps.681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j:/oauth2callback";
pub const GCA_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub redirect_uri: String,
    /// OAuth scope string. Defaults to the Vertex scope when absent so that
    /// settings-derived credentials (which omit it) keep working.
    #[serde(default = "default_scope")]
    pub scope: String,
}

fn default_scope() -> String {
    SCOPE.to_string()
}

impl Default for OAuthCredentials {
    fn default() -> Self {
        Self {
            client_id: DEFAULT_CLIENT_ID.to_string(),
            client_secret: None,
            redirect_uri: DEFAULT_REDIRECT_URI.to_string(),
            scope: SCOPE.to_string(),
        }
    }
}

impl OAuthCredentials {
    /// Credentials for the Gemini Code Assist client_secret flow.
    pub fn gca() -> Self {
        Self {
            client_id: GCA_CLIENT_ID.to_string(),
            client_secret: Some(GCA_CLIENT_SECRET.to_string()),
            redirect_uri: GCA_REDIRECT_URI.to_string(),
            scope: GCA_SCOPE.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
}

pub fn generate_pkce() -> PkceChallenge {
    use rand::Rng;
    let verifier: String = rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(64)
        .map(char::from)
        .collect();
    let hash = Sha256::digest(verifier.as_bytes());
    let challenge = base64url_encode(&hash);
    PkceChallenge {
        verifier,
        challenge,
    }
}

fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: u64,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
}

impl Tokens {
    pub fn is_expired(&self) -> bool {
        now_secs() >= self.expires_at
    }

    pub fn needs_refresh(&self) -> bool {
        now_secs() + REFRESH_MARGIN.as_secs() >= self.expires_at
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn build_auth_url(creds: &OAuthCredentials, pkce: Option<&PkceChallenge>) -> String {
    let mut params = vec![
        ("client_id", creds.client_id.as_str()),
        ("redirect_uri", creds.redirect_uri.as_str()),
        ("response_type", "code"),
        ("scope", creds.scope.as_str()),
        ("access_type", "offline"),
        ("prompt", "select_account consent"),
    ];
    let challenge_str;
    if let Some(p) = pkce {
        challenge_str = p.challenge.clone();
        params.push(("code_challenge", &challenge_str));
        params.push(("code_challenge_method", "S256"));
    }
    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, uri_encode(v)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{AUTH_ENDPOINT}?{query}")
}

pub async fn exchange_code(
    client: &reqwest::Client,
    creds: &OAuthCredentials,
    code: &str,
    code_verifier: Option<&str>,
) -> Result<Tokens, PomolluError> {
    let mut params = vec![
        ("code".to_string(), code.to_string()),
        ("client_id".to_string(), creds.client_id.clone()),
        ("redirect_uri".to_string(), creds.redirect_uri.clone()),
        ("grant_type".to_string(), "authorization_code".to_string()),
    ];
    if let Some(secret) = &creds.client_secret {
        params.push(("client_secret".to_string(), secret.clone()));
    }
    if let Some(verifier) = code_verifier {
        params.push(("code_verifier".to_string(), verifier.to_string()));
    }

    let resp = client
        .post(TOKEN_ENDPOINT)
        .form(&params)
        .send()
        .await
        .map_err(|e| PomolluError::Http(format!("token exchange: {e:#}")))?;

    if !resp.status().is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(body read failed: {e})"));
        return Err(PomolluError::OAuth(body));
    }

    let token_resp: TokenResponse = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    Ok(to_tokens(token_resp))
}

pub async fn refresh_token(
    client: &reqwest::Client,
    creds: &OAuthCredentials,
    refresh: &str,
) -> Result<Tokens, PomolluError> {
    let mut params = vec![
        ("refresh_token".to_string(), refresh.to_string()),
        ("client_id".to_string(), creds.client_id.clone()),
        ("grant_type".to_string(), "refresh_token".to_string()),
    ];
    if let Some(secret) = &creds.client_secret {
        params.push(("client_secret".to_string(), secret.clone()));
    }

    let resp = client
        .post(TOKEN_ENDPOINT)
        .form(&params)
        .send()
        .await
        .map_err(|e| PomolluError::Http(format!("token refresh: {e:#}")))?;

    if !resp.status().is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(body read failed: {e})"));
        if body.contains("invalid_grant") {
            return Err(PomolluError::InvalidGrant);
        }
        return Err(PomolluError::OAuth(body));
    }

    let mut token_resp: TokenResponse = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    // Google may omit the refresh token on refresh; keep the existing one.
    if token_resp.refresh_token.is_none() {
        token_resp.refresh_token = Some(refresh.to_string());
    }

    Ok(to_tokens(token_resp))
}

/// Return a valid token, refreshing if within the expiry margin.
pub async fn get_valid_token(
    client: &reqwest::Client,
    creds: &OAuthCredentials,
    tokens: &Tokens,
) -> Result<Tokens, PomolluError> {
    if !tokens.needs_refresh() {
        return Ok(tokens.clone());
    }
    let refresh = tokens
        .refresh_token
        .as_deref()
        .ok_or(PomolluError::OAuth("no refresh token".into()))?;
    refresh_token(client, creds, refresh).await
}

pub async fn revoke_token(client: &reqwest::Client, token: &str) -> Result<(), PomolluError> {
    let resp = client
        .post(REVOKE_ENDPOINT)
        .form(&[("token", token)])
        .send()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(body read failed: {e})"));
        return Err(PomolluError::Http(format!("token revoke failed: {body}")));
    }
    Ok(())
}

fn to_tokens(resp: TokenResponse) -> Tokens {
    Tokens {
        access_token: resp.access_token,
        refresh_token: resp.refresh_token,
        expires_at: now_secs() + resp.expires_in,
    }
}

// ── GCP project listing ─────────────────────────────────────────────

const GCP_PROJECTS_ENDPOINT: &str = "https://cloudresourcemanager.googleapis.com/v1/projects";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpProject {
    #[serde(rename = "projectId")]
    pub project_id: String,
    pub name: String,
}

pub async fn list_gcp_projects(
    client: &reqwest::Client,
    access_token: &str,
) -> Result<Vec<GcpProject>, PomolluError> {
    let url = format!("{GCP_PROJECTS_ENDPOINT}?filter=lifecycleState:ACTIVE");
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {access_token}"))
        .send()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(body read failed: {e})"));
        return Err(PomolluError::ApiError { status, body });
    }

    let body: Value = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    let projects = body
        .get("projects")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(GcpProject {
                        project_id: p.get("projectId")?.as_str()?.to_string(),
                        name: p.get("name")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(projects)
}

pub fn uri_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                String::from(b as char)
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_url_format() {
        let creds = OAuthCredentials {
            client_id: "test-client".into(),
            client_secret: None,
            redirect_uri: "http://localhost:8080/callback".into(),
            scope: SCOPE.into(),
        };
        let url = build_auth_url(&creds, None);
        assert!(url.starts_with(AUTH_ENDPOINT));
        assert!(url.contains("client_id=test-client"));
        assert!(url.contains("access_type=offline"));
        assert!(url.contains("prompt=select_account%20consent"));
    }

    #[test]
    fn gca_credentials_use_client_secret_and_gca_scope() {
        let creds = OAuthCredentials::gca();
        assert!(creds.client_secret.is_some());
        assert_eq!(creds.scope, GCA_SCOPE);
        let url = build_auth_url(&creds, None);
        assert!(url.contains("userinfo.email"));
        assert!(url.contains(GCA_CLIENT_ID));
    }

    #[test]
    fn auth_url_with_pkce() {
        let creds = OAuthCredentials::default();
        let pkce = generate_pkce();
        let url = build_auth_url(&creds, Some(&pkce));
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert_eq!(pkce.verifier.len(), 64);
    }

    #[test]
    fn token_expiration() {
        let expired = Tokens {
            access_token: "t".into(),
            refresh_token: None,
            expires_at: 0,
        };
        assert!(expired.is_expired());
        assert!(expired.needs_refresh());

        let far_future = Tokens {
            access_token: "t".into(),
            refresh_token: None,
            expires_at: u64::MAX,
        };
        assert!(!far_future.is_expired());
        assert!(!far_future.needs_refresh());
    }

    #[test]
    fn pkce_challenge_is_base64url_of_sha256() {
        let pkce = generate_pkce();
        let expected = base64url_encode(&Sha256::digest(pkce.verifier.as_bytes()));
        assert_eq!(pkce.challenge, expected);
        assert!(!pkce.challenge.contains('='));
        assert!(!pkce.challenge.contains('+'));
        assert!(!pkce.challenge.contains('/'));
    }
}
