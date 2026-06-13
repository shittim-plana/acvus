//! Vertex AI Gemini client — SSE streaming + model listing.
//!
//! Auth is an OAuth2 access token (see `crate::oauth`), never an API key.

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::PomolluError;
use crate::retry::{self, CancelToken};

#[derive(Debug, Clone, Serialize)]
pub struct GenerateRequest {
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    pub max_output_tokens: u32,
    pub temperature: f64,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ThinkingConfig {
    Level {
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
    },
    Budget {
        #[serde(rename = "thinkingBudget")]
        thinking_budget: i32,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    pub candidates: Option<Vec<Candidate>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Candidate {
    pub content: Option<Content>,
}

fn host(region: &str) -> String {
    if region == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{region}-aiplatform.googleapis.com")
    }
}

pub fn build_stream_endpoint(project_id: &str, region: &str, model: &str) -> String {
    format!(
        "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent?alt=sse",
        host(region),
        project_id,
        region,
        model
    )
}

pub async fn stream_generate(
    client: &reqwest::Client,
    access_token: &str,
    project_id: &str,
    region: &str,
    model: &str,
    request: &GenerateRequest,
    on_chunk: impl Fn(&str),
    cancel: Option<CancelToken>,
) -> Result<String, PomolluError> {
    let url = build_stream_endpoint(project_id, region, model);
    let auth = format!("Bearer {access_token}");

    let resp = retry::retry_request(&cancel, || {
        let req = client.post(&url).header("Authorization", &auth).json(request);
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

pub async fn list_models(
    client: &reqwest::Client,
    access_token: &str,
    region: &str,
) -> Result<Vec<String>, PomolluError> {
    let url = format!("https://{}/v1/publishers/google/models", host(region));
    let auth = format!("Bearer {access_token}");

    let resp = retry::retry_request(&None, || {
        let req = client.get(&url).header("Authorization", &auth);
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

    let body: Value = resp
        .json()
        .await
        .map_err(|e| PomolluError::Http(e.to_string()))?;

    let mut models = Vec::new();
    if let Some(arr) = body.get("publisherModels").and_then(|m| m.as_array()) {
        for entry in arr {
            if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                let model_id = name.strip_prefix("publishers/google/models/").unwrap_or(name);
                models.push(model_id.to_string());
            }
        }
    }
    models.sort();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_endpoint_regional() {
        let url = build_stream_endpoint("my-project", "us-central1", "gemini-2.5-flash");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn stream_endpoint_global() {
        let url = build_stream_endpoint("proj", "global", "gemini-3.0-flash-preview");
        assert!(url.starts_with("https://aiplatform.googleapis.com/"));
        assert!(url.contains("locations/global"));
    }

    #[test]
    fn thinking_config_serialization() {
        let level = ThinkingConfig::Level {
            thinking_level: "high".into(),
        };
        assert_eq!(
            serde_json::to_string(&level).unwrap(),
            r#"{"thinkingLevel":"high"}"#
        );
        let budget = ThinkingConfig::Budget {
            thinking_budget: 2048,
        };
        assert_eq!(
            serde_json::to_string(&budget).unwrap(),
            r#"{"thinkingBudget":2048}"#
        );
    }
}
