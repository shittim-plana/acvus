//! Chat commands — per-stream buffer+poll streaming.
//!
//! acvus is a multi-session structure (workspaces are isolated), so streaming
//! is keyed by a caller-supplied `stream_id` rather than a single global slot.
//! Multiple chats — across workspaces, or Vertex + GCA + Mistral at once — run
//! concurrently without clobbering each other's buffers or cancel tokens.
//!
//! Streaming protocol (GeckoView-safe — does not rely on webview eval):
//! 1. a `chat_*` command opens a fresh buffer + cancel token under `stream_id`,
//!    starts the SSE request; each chunk is appended to that id's buffer and
//!    emitted as a `"chat-chunk"` event `{ streamId, chunk }`
//! 2. the frontend polls `poll_stream_chunks(stream_id)` (~100ms) and drains it
//! 3. on completion the cancel token is dropped; the final poll removes the
//!    now-empty buffer entry (no per-stream leak)

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Deserialize;
use serde_json::Value;
use tauri::Emitter;

use crate::commands_auth;
use pomollu_core::providers::{gca, mistral, vertex};
use pomollu_core::retry::{self, CancelToken};

/// Build a Gemini `GenerateRequest` shared by the Vertex and GCA chat paths.
fn build_gemini_request(
    messages: Vec<ChatApiMessage>,
    system_prompt: Option<String>,
    temperature: f64,
    max_tokens: u32,
    top_p: Option<f64>,
    top_k: Option<u32>,
    thinking_budget: Option<i32>,
    thinking_level: Option<String>,
) -> vertex::GenerateRequest {
    let contents: Vec<vertex::Content> = messages
        .into_iter()
        .map(|m| vertex::Content {
            role: if m.role == "assistant" { "model".into() } else { m.role },
            parts: vec![vertex::Part {
                text: Some(m.content),
                thought: None,
            }],
        })
        .collect();

    let thinking_config = match (thinking_level, thinking_budget) {
        (Some(level), _) => Some(vertex::ThinkingConfig::Level {
            thinking_level: level,
        }),
        (None, Some(budget)) => Some(vertex::ThinkingConfig::Budget {
            thinking_budget: budget,
        }),
        (None, None) => None,
    };

    vertex::GenerateRequest {
        contents,
        system_instruction: system_prompt.filter(|s| !s.is_empty()).map(|s| vertex::Content {
            role: "system".into(),
            parts: vec![vertex::Part {
                text: Some(s),
                thought: None,
            }],
        }),
        generation_config: vertex::GenerationConfig {
            max_output_tokens: max_tokens,
            temperature,
            thinking_config,
            top_p,
            top_k,
        },
    }
}

/// Per-stream chunk buffers, keyed by caller-supplied `stream_id`.
#[derive(Default)]
pub struct StreamBufferState {
    pub buffers: Mutex<HashMap<String, Vec<String>>>,
}

/// Per-stream cancel tokens. Presence of a key == stream is active.
#[derive(Default)]
pub struct StreamCancelState {
    pub tokens: Mutex<HashMap<String, CancelToken>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatApiMessage {
    pub role: String,
    pub content: String,
}

/// Open a fresh buffer + cancel token for `stream_id` (dropping any stale
/// chunks from a prior run under the same id).
fn begin_stream(
    buffer: &tauri::State<'_, StreamBufferState>,
    cancel: &tauri::State<'_, StreamCancelState>,
    stream_id: &str,
) -> Result<CancelToken, String> {
    buffer
        .buffers
        .lock()
        .map_err(|e| e.to_string())?
        .insert(stream_id.to_string(), Vec::new());
    let token = retry::new_cancel_token();
    cancel
        .tokens
        .lock()
        .map_err(|e| e.to_string())?
        .insert(stream_id.to_string(), token.clone());
    Ok(token)
}

fn make_on_chunk<'a>(
    app: &'a tauri::AppHandle,
    buffer: &'a tauri::State<'_, StreamBufferState>,
    stream_id: String,
) -> impl Fn(&str) + 'a {
    move |chunk: &str| {
        if let Ok(mut bufs) = buffer.buffers.lock() {
            bufs.entry(stream_id.clone()).or_default().push(chunk.to_string());
        }
        let _ = app.emit(
            "chat-chunk",
            serde_json::json!({ "streamId": stream_id, "chunk": chunk }),
        );
    }
}

/// Mark a stream finished by dropping its cancel token. The buffer stays until
/// the caller's final poll drains it (then poll removes the empty entry).
fn end_stream(cancel: &tauri::State<'_, StreamCancelState>, stream_id: &str) {
    if let Ok(mut tokens) = cancel.tokens.lock() {
        tokens.remove(stream_id);
    }
}

/// Drain a stream's buffered chunks (non-blocking). Removes the buffer entry
/// once the stream is no longer active and fully drained — no per-stream leak.
#[tauri::command]
pub fn poll_stream_chunks(
    buffer: tauri::State<'_, StreamBufferState>,
    cancel: tauri::State<'_, StreamCancelState>,
    stream_id: String,
) -> Result<Vec<String>, String> {
    let mut bufs = buffer.buffers.lock().map_err(|e| e.to_string())?;
    let drained = bufs.get_mut(&stream_id).map(std::mem::take).unwrap_or_default();

    let active = cancel
        .tokens
        .lock()
        .map_err(|e| e.to_string())?
        .contains_key(&stream_id);
    if !active {
        if let Some(v) = bufs.get(&stream_id) {
            if v.is_empty() {
                bufs.remove(&stream_id);
            }
        }
    }
    Ok(drained)
}

/// Cancel a specific in-flight chat stream.
#[tauri::command]
pub fn cancel_chat(
    cancel: tauri::State<'_, StreamCancelState>,
    stream_id: String,
) -> Result<(), String> {
    if let Some(token) = cancel.tokens.lock().map_err(|e| e.to_string())?.get(&stream_id) {
        token.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

// ── Mistral ─────────────────────────────────────────────────────────

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn chat_mistral(
    app: tauri::AppHandle,
    buffer: tauri::State<'_, StreamBufferState>,
    cancel: tauri::State<'_, StreamCancelState>,
    stream_id: String,
    messages: Vec<ChatApiMessage>,
    system_prompt: Option<String>,
    model: String,
    api_key: String,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    reasoning_effort: Option<String>,
) -> Result<String, String> {
    let token = begin_stream(&buffer, &cancel, &stream_id)?;

    let mut msgs: Vec<mistral::ChatMessage> = Vec::with_capacity(messages.len() + 1);
    if let Some(system) = system_prompt {
        if !system.is_empty() {
            msgs.push(mistral::ChatMessage {
                role: "system".into(),
                content: system,
            });
        }
    }
    msgs.extend(messages.into_iter().map(|m| mistral::ChatMessage {
        role: m.role,
        content: m.content,
    }));

    let request = mistral::ChatRequest {
        model,
        messages: msgs,
        temperature,
        top_p,
        max_tokens,
        stream: None,
        frequency_penalty: None,
        presence_penalty: None,
        reasoning_effort,
    };

    let client = reqwest::Client::new();
    let on_chunk = make_on_chunk(&app, &buffer, stream_id.clone());
    let result = mistral::chat_stream(&client, &api_key, &request, on_chunk, Some(token))
        .await
        .map_err(String::from);
    end_stream(&cancel, &stream_id);
    result
}

#[tauri::command]
pub async fn mistral_list_models(api_key: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let models = mistral::list_models(&client, &api_key)
        .await
        .map_err(String::from)?;
    serde_json::to_value(models).map_err(|e| e.to_string())
}

// ── Vertex AI ───────────────────────────────────────────────────────

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn chat_vertex(
    app: tauri::AppHandle,
    buffer: tauri::State<'_, StreamBufferState>,
    cancel: tauri::State<'_, StreamCancelState>,
    stream_id: String,
    messages: Vec<ChatApiMessage>,
    system_prompt: Option<String>,
    model: String,
    project_id: String,
    region: String,
    temperature: f64,
    max_tokens: u32,
    top_p: Option<f64>,
    top_k: Option<u32>,
    thinking_budget: Option<i32>,
    thinking_level: Option<String>,
) -> Result<String, String> {
    let token = begin_stream(&buffer, &cancel, &stream_id)?;

    let client = reqwest::Client::new();
    let access_token = match commands_auth::valid_access_token(&app, &client).await {
        Ok(t) => t,
        Err(e) => {
            end_stream(&cancel, &stream_id);
            return Err(e);
        }
    };

    let request = build_gemini_request(
        messages,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        top_k,
        thinking_budget,
        thinking_level,
    );

    let on_chunk = make_on_chunk(&app, &buffer, stream_id.clone());
    let result = vertex::stream_generate(
        &client,
        &access_token,
        &project_id,
        &region,
        &model,
        &request,
        on_chunk,
        Some(token),
    )
    .await
    .map_err(String::from);
    end_stream(&cancel, &stream_id);
    result
}

// ── Gemini Code Assist (GCA) ────────────────────────────────────────

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn chat_gca(
    app: tauri::AppHandle,
    buffer: tauri::State<'_, StreamBufferState>,
    cancel: tauri::State<'_, StreamCancelState>,
    stream_id: String,
    messages: Vec<ChatApiMessage>,
    system_prompt: Option<String>,
    model: String,
    project: Option<String>,
    temperature: f64,
    max_tokens: u32,
    top_p: Option<f64>,
    top_k: Option<u32>,
    thinking_budget: Option<i32>,
    thinking_level: Option<String>,
) -> Result<String, String> {
    let token = begin_stream(&buffer, &cancel, &stream_id)?;

    let client = reqwest::Client::new();
    let access_token = match commands_auth::valid_gca_token(&app, &client).await {
        Ok(t) => t,
        Err(e) => {
            end_stream(&cancel, &stream_id);
            return Err(e);
        }
    };

    let request = build_gemini_request(
        messages,
        system_prompt,
        temperature,
        max_tokens,
        top_p,
        top_k,
        thinking_budget,
        thinking_level,
    );

    let on_chunk = make_on_chunk(&app, &buffer, stream_id.clone());
    let result = gca::stream_generate(
        &client,
        &access_token,
        &model,
        project.as_deref(),
        &request,
        on_chunk,
        Some(token),
    )
    .await
    .map_err(String::from);
    end_stream(&cancel, &stream_id);
    result
}

/// GCA has no model-list endpoint — return the static catalog.
#[tauri::command]
pub fn gca_list_models() -> Vec<String> {
    gca::GCA_MODELS.iter().map(|m| m.to_string()).collect()
}

#[tauri::command]
pub async fn vertex_list_models(app: tauri::AppHandle, region: String) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let access_token = commands_auth::valid_access_token(&app, &client).await?;
    let models = vertex::list_models(&client, &access_token, &region)
        .await
        .map_err(String::from)?;
    serde_json::to_value(models).map_err(|e| e.to_string())
}
