//! Mistral AI chat client — streaming SSE + model listing.

use std::collections::HashMap;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::PomolluError;
use crate::retry::{self, CancelToken};

const API_BASE: &str = "https://api.mistral.ai/v1";

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub message: Option<ChatMessage>,
    pub delta: Option<DeltaMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeltaMessage {
    pub content: Option<String>,
}

pub async fn chat_stream(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatRequest,
    on_chunk: impl Fn(&str),
    cancel: Option<CancelToken>,
) -> Result<String, PomolluError> {
    let url = format!("{API_BASE}/chat/completions");
    let auth = format!("Bearer {api_key}");

    let mut stream_req = request.clone();
    stream_req.stream = Some(true);

    let resp = retry::retry_request(&cancel, || {
        let req = client
            .post(&url)
            .header("Authorization", &auth)
            .json(&stream_req);
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

            if line == "data: [DONE]" {
                break;
            }
            if let Some(json_str) = line.strip_prefix("data: ") {
                if let Ok(chunk) = serde_json::from_str::<ChatResponse>(json_str) {
                    for choice in &chunk.choices {
                        if let Some(delta) = &choice.delta {
                            if let Some(content) = &delta.content {
                                full_text.push_str(content);
                                on_chunk(content);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(full_text)
}

// ── Model listing ───────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub created: Option<u64>,
    #[serde(default)]
    pub capabilities: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelInfo>,
}

/// Only models that explicitly declare chat support (soundness over
/// completeness: unknown capability = not chat).
fn is_chat_model(model: &ModelInfo) -> bool {
    model
        .capabilities
        .as_ref()
        .and_then(|caps| caps.get("completion_chat"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Strip `-latest` and trailing `-YYMM`/`-YYMMDD` date suffixes.
fn model_base_name(id: &str) -> &str {
    let s = id.strip_suffix("-latest").unwrap_or(id);
    if let Some(pos) = s.rfind('-') {
        let suffix = &s[pos + 1..];
        let all_digits = !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit());
        if all_digits && (suffix.len() == 4 || suffix.len() == 6) {
            return &s[..pos];
        }
    }
    s
}

pub async fn list_models(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<Vec<ModelInfo>, PomolluError> {
    let url = format!("{API_BASE}/models");
    let auth = format!("Bearer {api_key}");

    let resp = retry::retry_request(&None, || {
        let req = client.get(&url).header("Authorization", &auth);
        async { req.send().await.map_err(|e| PomolluError::Http(e.to_string())) }
    })
    .await?;

    let list: ModelListResponse = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    let chat_models: Vec<ModelInfo> = list.data.into_iter().filter(is_chat_model).collect();

    // Per base name keep the `-latest` variant, else the newest by `created`.
    let mut best: HashMap<String, ModelInfo> = HashMap::new();
    for model in chat_models {
        let base = model_base_name(&model.id).to_owned();
        let dominated = best.get(&base).is_some_and(|existing| {
            let existing_is_latest = existing.id.ends_with("-latest");
            let new_is_latest = model.id.ends_with("-latest");
            if existing_is_latest && !new_is_latest {
                true
            } else if !existing_is_latest && new_is_latest {
                false
            } else {
                existing.created.unwrap_or(0) >= model.created.unwrap_or(0)
            }
        });
        if !dominated {
            best.insert(base, model);
        }
    }

    let mut models: Vec<ModelInfo> = best.into_values().collect();
    models.sort_by(|a, b| {
        let a_latest = a.id.ends_with("-latest");
        let b_latest = b.id.ends_with("-latest");
        match (a_latest, b_latest) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b
                .created
                .unwrap_or(0)
                .cmp(&a.created.unwrap_or(0))
                .then_with(|| a.id.cmp(&b.id)),
        }
    });

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_name_extraction() {
        assert_eq!(model_base_name("mistral-large-latest"), "mistral-large");
        assert_eq!(model_base_name("mistral-medium-2505"), "mistral-medium");
        assert_eq!(model_base_name("codestral-250115"), "codestral");
        assert_eq!(model_base_name("pixtral-large"), "pixtral-large");
        assert_eq!(model_base_name("model-123"), "model-123");
    }

    #[test]
    fn chat_capability_filter() {
        let with = ModelInfo {
            id: "m".into(),
            created: None,
            capabilities: Some(serde_json::json!({"completion_chat": true})),
        };
        let without = ModelInfo {
            id: "m".into(),
            created: None,
            capabilities: None,
        };
        assert!(is_chat_model(&with));
        assert!(!is_chat_model(&without));
    }

    #[test]
    fn stream_delta_parsing() {
        let json = r#"{"choices":[{"delta":{"content":"Hi"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("Hi")
        );
    }
}
