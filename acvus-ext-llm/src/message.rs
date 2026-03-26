//! Common message types shared across LLM providers.

/// Message content — text or binary blob.
#[derive(Debug, Clone)]
pub enum Content {
    Text(String),
    Blob { mime_type: String, data: String },
}

/// A chat message — explicit variants, no implicit fields.
#[derive(Debug, Clone)]
pub enum Message {
    Content { role: String, content: Content },
    ToolCalls(Vec<ToolCall>),
    ToolResult { call_id: String, content: String },
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A single content part from a model response.
#[derive(Debug, Clone)]
pub struct ContentItem {
    pub role: String,
    pub content: Content,
}

/// Model response: either content parts or tool calls.
#[derive(Debug, Clone)]
pub enum ModelResponse {
    Content(Vec<ContentItem>),
    ToolCalls(Vec<ToolCall>),
}

/// Token usage from a model response.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}
