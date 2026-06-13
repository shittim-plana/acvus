//! Vertex AI OAuth commands — PKCE sign-in, token refresh, project listing.

use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;
use tauri::Manager;

use pomollu_core::oauth::{self, OAuthCredentials, Tokens};
use pomollu_core::persistence;

#[derive(Default)]
pub struct AuthState {
    pub vertex: Mutex<Option<Tokens>>,
    pub gca: Mutex<Option<Tokens>>,
    pub creds: Mutex<Option<OAuthCredentials>>,
}

impl AuthState {
    /// Restore tokens from the encrypted store at startup.
    pub fn load_persisted_tokens(&self, app: &tauri::AppHandle) {
        let Ok(data_dir) = crate::get_data_dir(app) else {
            return;
        };
        match persistence::load_tokens(&data_dir) {
            Ok(stored) => {
                if let Ok(mut v) = self.vertex.lock() {
                    *v = stored.vertex;
                }
                if let Ok(mut g) = self.gca.lock() {
                    *g = stored.gca;
                }
            }
            Err(e) => log::warn!("token store load failed: {e}"),
        }
    }

    fn credentials(&self, app: &tauri::AppHandle) -> OAuthCredentials {
        if let Ok(guard) = self.creds.lock() {
            if let Some(c) = guard.as_ref() {
                return c.clone();
            }
        }
        // Settings may override the OAuth client (custom GCP project).
        let from_settings = crate::get_data_dir(app)
            .ok()
            .and_then(|d| persistence::load_settings(&d).ok())
            .and_then(|s| {
                let client_id = s.get("oauthClientId")?.as_str()?.to_string();
                let redirect_uri = s
                    .get("oauthRedirectUri")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| oauth::DEFAULT_REDIRECT_URI.to_string());
                Some(OAuthCredentials {
                    client_id,
                    client_secret: None,
                    redirect_uri,
                })
            });
        let creds = from_settings.unwrap_or_default();
        if let Ok(mut guard) = self.creds.lock() {
            *guard = Some(creds.clone());
        }
        creds
    }

    fn persist(&self, app: &tauri::AppHandle) {
        let Ok(data_dir) = crate::get_data_dir(app) else {
            return;
        };
        let vertex = self.vertex.lock().ok().and_then(|g| g.clone());
        let gca = self.gca.lock().ok().and_then(|g| g.clone());
        let stored = persistence::StoredTokens { vertex, gca };
        if let Err(e) = persistence::save_tokens(&data_dir, &stored) {
            log::error!("token store save failed: {e}");
        }
    }
}

#[derive(Serialize)]
pub struct OAuthStatus {
    pub connected: bool,
    pub expired: bool,
}

/// Return a valid access token, refreshing (and re-persisting) if needed.
pub async fn valid_access_token(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
) -> Result<String, String> {
    let state = app.state::<AuthState>();
    let (tokens, creds) = {
        let guard = state.vertex.lock().map_err(|e| e.to_string())?;
        let tokens = guard.clone().ok_or("not connected to Vertex AI")?;
        (tokens, state.credentials(app))
    };

    let fresh = oauth::get_valid_token(client, &creds, &tokens)
        .await
        .map_err(String::from)?;

    if fresh.access_token != tokens.access_token {
        if let Ok(mut guard) = state.vertex.lock() {
            *guard = Some(fresh.clone());
        }
        state.persist(app);
    }
    Ok(fresh.access_token)
}

// ── Commands ────────────────────────────────────────────────────────

/// Begin the PKCE flow: persist the verifier, return the consent URL.
#[tauri::command]
pub fn vertex_oauth_start(
    app: tauri::AppHandle,
    state: tauri::State<'_, AuthState>,
) -> Result<String, String> {
    let creds = state.credentials(&app);
    let pkce = oauth::generate_pkce();
    let data_dir = crate::get_data_dir(&app)?;
    persistence::save_pkce_verifier(&data_dir, &pkce.verifier).map_err(String::from)?;
    Ok(oauth::build_auth_url(&creds, Some(&pkce)))
}

/// Exchange the redirect `code` (PKCE) for tokens and persist them.
#[tauri::command]
pub async fn vertex_oauth_callback(
    app: tauri::AppHandle,
    state: tauri::State<'_, AuthState>,
    code: String,
) -> Result<String, String> {
    let creds = state.credentials(&app);
    let data_dir = crate::get_data_dir(&app)?;
    let verifier = persistence::take_pkce_verifier(&data_dir);

    let client = reqwest::Client::new();
    let tokens = oauth::exchange_code(&client, &creds, &code, verifier.as_deref())
        .await
        .map_err(String::from)?;

    {
        let mut guard = state.vertex.lock().map_err(|e| e.to_string())?;
        *guard = Some(tokens);
    }
    state.persist(&app);
    Ok("connected".into())
}

#[tauri::command]
pub fn vertex_oauth_status(state: tauri::State<'_, AuthState>) -> Result<OAuthStatus, String> {
    let guard = state.vertex.lock().map_err(|e| e.to_string())?;
    Ok(match guard.as_ref() {
        Some(t) => OAuthStatus {
            connected: true,
            expired: t.is_expired() && t.refresh_token.is_none(),
        },
        None => OAuthStatus {
            connected: false,
            expired: false,
        },
    })
}

/// Revoke (best-effort) and clear stored tokens.
#[tauri::command]
pub async fn vertex_oauth_disconnect(
    app: tauri::AppHandle,
    state: tauri::State<'_, AuthState>,
) -> Result<String, String> {
    let token = {
        let mut guard = state.vertex.lock().map_err(|e| e.to_string())?;
        guard.take()
    };
    if let Some(t) = token {
        let client = reqwest::Client::new();
        if let Err(e) = oauth::revoke_token(&client, &t.access_token).await {
            log::warn!("token revoke failed (clearing local store anyway): {e}");
        }
    }
    let data_dir = crate::get_data_dir(&app)?;
    persistence::clear_tokens(&data_dir).map_err(String::from)?;
    Ok("disconnected".into())
}

#[tauri::command]
pub async fn vertex_list_projects(app: tauri::AppHandle) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let token = valid_access_token(&app, &client).await?;
    let projects = oauth::list_gcp_projects(&client, &token)
        .await
        .map_err(String::from)?;
    serde_json::to_value(projects).map_err(|e| e.to_string())
}

/// Check for an OAuth code handed off by the Android deep-link receiver.
#[tauri::command]
pub fn get_pending_oauth(app: tauri::AppHandle) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    Ok(match persistence::take_pending_oauth(&data_dir) {
        Some(code) => serde_json::json!({ "code": code }),
        None => serde_json::json!({}),
    })
}

// ── Gemini Code Assist (GCA) ────────────────────────────────────────
//
// GCA uses Google's installed-app client (client_secret flow, no PKCE) and a
// distinct scope. Tokens live in the same encrypted store under the `gca`
// field. The redirect is GCA's own reverse-client-ID scheme — register that
// scheme in AndroidManifest.xml alongside the Vertex one (see
// android-overlay/README.md).

/// Return a valid GCA access token, refreshing (and re-persisting) if needed.
pub async fn valid_gca_token(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
) -> Result<String, String> {
    let state = app.state::<AuthState>();
    let tokens = {
        let guard = state.gca.lock().map_err(|e| e.to_string())?;
        guard.clone().ok_or("not connected to Gemini Code Assist")?
    };
    let creds = OAuthCredentials::gca();

    let fresh = oauth::get_valid_token(client, &creds, &tokens)
        .await
        .map_err(String::from)?;

    if fresh.access_token != tokens.access_token {
        if let Ok(mut guard) = state.gca.lock() {
            *guard = Some(fresh.clone());
        }
        state.persist(app);
    }
    Ok(fresh.access_token)
}

/// Begin the GCA flow. Unlike Vertex this uses client_secret (no PKCE), so no
/// verifier is persisted — just return the consent URL.
#[tauri::command]
pub fn gca_oauth_start() -> Result<String, String> {
    let creds = OAuthCredentials::gca();
    Ok(oauth::build_auth_url(&creds, None))
}

/// Exchange the GCA redirect `code` for tokens and persist them.
#[tauri::command]
pub async fn gca_oauth_callback(
    app: tauri::AppHandle,
    state: tauri::State<'_, AuthState>,
    code: String,
) -> Result<String, String> {
    let creds = OAuthCredentials::gca();
    let client = reqwest::Client::new();
    let tokens = oauth::exchange_code(&client, &creds, &code, None)
        .await
        .map_err(String::from)?;

    {
        let mut guard = state.gca.lock().map_err(|e| e.to_string())?;
        *guard = Some(tokens);
    }
    state.persist(&app);
    Ok("connected".into())
}

#[tauri::command]
pub fn gca_oauth_status(state: tauri::State<'_, AuthState>) -> Result<OAuthStatus, String> {
    let guard = state.gca.lock().map_err(|e| e.to_string())?;
    Ok(match guard.as_ref() {
        Some(t) => OAuthStatus {
            connected: true,
            expired: t.is_expired() && t.refresh_token.is_none(),
        },
        None => OAuthStatus {
            connected: false,
            expired: false,
        },
    })
}

#[tauri::command]
pub async fn gca_oauth_disconnect(
    app: tauri::AppHandle,
    state: tauri::State<'_, AuthState>,
) -> Result<String, String> {
    let token = {
        let mut guard = state.gca.lock().map_err(|e| e.to_string())?;
        guard.take()
    };
    if let Some(t) = token {
        let client = reqwest::Client::new();
        if let Err(e) = oauth::revoke_token(&client, &t.access_token).await {
            log::warn!("GCA token revoke failed (clearing local store anyway): {e}");
        }
    }
    state.persist(&app);
    Ok("disconnected".into())
}

/// Resolve the GCA project ID (`loadCodeAssist`) and opt out of free-tier
/// data collection. Persists the project ID into settings for later chat calls.
#[tauri::command]
pub async fn gca_load_project(app: tauri::AppHandle) -> Result<String, String> {
    use pomollu_core::providers::gca;

    let client = reqwest::Client::new();
    let token = valid_gca_token(&app, &client).await?;

    if let Err(e) = gca::check_and_opt_out(&client, &token).await {
        log::warn!("GCA opt-out check failed: {e}");
    }

    let project_id = gca::load_code_assist(&client, &token)
        .await
        .map_err(String::from)?;

    // Persist into settings under `gcaProject` so chat_gca can read it.
    let data_dir = crate::get_data_dir(&app)?;
    if let Ok(mut settings) = persistence::load_settings(&data_dir) {
        if let Some(obj) = settings.as_object_mut() {
            obj.insert("gcaProject".into(), Value::String(project_id.clone()));
            let _ = persistence::save_settings(&data_dir, &settings);
        }
    }

    Ok(project_id)
}
