//! Gemini Code Assist (GCA) client — free-tier Gemini via the
//! `cloudcode-pa.googleapis.com/v1internal` surface, ported from layream.
//!
//! GCA wraps the same Gemini `GenerateRequest`/`StreamChunk` as Vertex AI
//! inside `{model, request, project?}` and authenticates with an OAuth2
//! access token (client_secret flow — distinct from Vertex's PKCE). The
//! `x-goog-api-client` header masquerades as the official IntelliJ plugin,
//! which the endpoint requires.
//!
//! Caveat: `v1internal` is an unofficial, undocumented API (the gemini-cli /
//! Code Assist backend). The client_id/secret below are Google's public
//! installed-app credentials — not a confidential secret in the OAuth sense —
//! but the endpoint shape can change without notice. The win is the free
//! quota; the cost is fragility.

use futures::StreamExt;
use serde_json::Value;

use crate::error::PomolluError;
use crate::providers::vertex::{GenerateRequest, StreamChunk};
use crate::retry::{self, CancelToken};

const GCA_BASE: &str = "https://cloudcode-pa.googleapis.com/v1internal";

pub const GCA_OAUTH_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
pub const GCA_OAUTH_CLIENT_SECRET: &str = "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";
pub const GCA_OAUTH_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile";

/// Known GCA-served models (source: risu-gca plugin). GCA has no model-list
/// endpoint, so this is a static catalog.
pub const GCA_MODELS: &[&str] = &[
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
];

pub fn build_stream_endpoint() -> String {
    format!("{GCA_BASE}:streamGenerateContent?alt=sse")
}

pub async fn stream_generate(
    client: &reqwest::Client,
    access_token: &str,
    model: &str,
    project: Option<&str>,
    request: &GenerateRequest,
    on_chunk: impl Fn(&str),
    cancel: Option<CancelToken>,
) -> Result<String, PomolluError> {
    let url = build_stream_endpoint();
    let auth = format!("Bearer {access_token}");
    let mut wrapped = serde_json::json!({ "model": model, "request": request });
    if let Some(p) = project {
        wrapped["project"] = Value::String(p.to_string());
    }

    let resp = retry::retry_request(&cancel, || {
        let req = client
            .post(&url)
            .header("Authorization", &auth)
            .header("x-goog-api-client", "google-cloud-intellij")
            .json(&wrapped);
        async { req.send().await.map_err(|e| PomolluError::Http(e.to_string())) }
    })
    .await?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(body read failed: {e})"));
        return Err(PomolluError::ApiError { status, body });
    }

    let mut full_text = String::new();
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        if retry::is_cancelled(&cancel) {
            return Err(PomolluError::Http("cancelled".to_string()));
        }
        let bytes = chunk.map_err(|e| PomolluError::Http(e.to_string()))?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            if let Some(json_str) = line.strip_prefix("data: ") {
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(json_str) {
                    if let Some(candidates) = &chunk.candidates {
                        for candidate in candidates {
                            if let Some(content) = &candidate.content {
                                for part in &content.parts {
                                    // Skip thought parts — reasoning traces are
                                    // not part of the answer text.
                                    if part.thought != Some(true) {
                                        if let Some(text) = &part.text {
                                            full_text.push_str(text);
                                            on_chunk(text);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(full_text)
}

/// Resolve the GCA project ID for this account (required by the chat call).
pub async fn load_code_assist(
    client: &reqwest::Client,
    access_token: &str,
) -> Result<String, PomolluError> {
    let url = format!("{GCA_BASE}/loadCodeAssist");
    let body = serde_json::json!({
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI"
        }
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {access_token}"))
        .json(&body)
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

    let resp_body: Value = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;
    let project_id = resp_body
        .get("projectId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(project_id)
}

/// Opt out of free-tier data collection if it is currently enabled.
/// Returns `true` if an opt-out write was performed.
pub async fn check_and_opt_out(
    client: &reqwest::Client,
    access_token: &str,
) -> Result<bool, PomolluError> {
    let url = format!("{GCA_BASE}/getCodeAssistGlobalUserSetting");
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

    if body.get("freeTierDataCollectionOptin") == Some(&Value::Bool(true)) {
        let set_url = format!("{GCA_BASE}/setCodeAssistGlobalUserSetting");
        let opt_resp = client
            .post(&set_url)
            .header("Authorization", format!("Bearer {access_token}"))
            .json(&serde_json::json!({ "freeTierDataCollectionOptin": false }))
            .send()
            .await
            .map_err(|e| PomolluError::Http(e.to_string()))?;
        if !opt_resp.status().is_success() {
            log::warn!("GCA opt-out failed: status {}", opt_resp.status().as_u16());
        }
        return Ok(true);
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gca_endpoint_format() {
        assert_eq!(
            build_stream_endpoint(),
            "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn model_catalog_has_expected_entries() {
        assert!(!GCA_MODELS.is_empty());
        assert!(GCA_MODELS.contains(&"gemini-2.5-flash"));
        assert!(GCA_MODELS.contains(&"gemini-3-pro-preview"));
    }
}
