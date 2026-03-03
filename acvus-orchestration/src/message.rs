use std::collections::HashMap;

/// A chat message with role, content, and optional tool call metadata.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
}

/// Model response: either text or tool calls.
#[derive(Debug, Clone)]
pub enum ModelResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

/// Tool specification passed to the model.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub params: HashMap<String, String>,
}

/// Token usage from a model response.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

/// Node output stored in storage.
#[derive(Debug, Clone)]
pub enum Output {
    Text(String),
    Json(serde_json::Value),
    Image(Vec<u8>),
}
