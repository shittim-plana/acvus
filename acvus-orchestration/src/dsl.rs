use std::collections::HashMap;

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Llm,
    LlmCache {
        /// TTL string, e.g. "300s", "1h".
        ttl: String,
        /// Provider-specific cache config (e.g. display_name for Gemini).
        cache_config: HashMap<String, serde_json::Value>,
    },
}

/// Node specification — pure compilation input, no Serde.
#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub name: String,
    pub kind: NodeKind,
    pub provider: String,
    pub model: String,
    pub tools: Vec<ToolDecl>,
    pub messages: Vec<MessageSpec>,
    pub strategy: Strategy,
    pub generation: GenerationParams,
    pub cache_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Strategy {
    pub mode: StrategyMode,
    /// Template source for cache key (if-modified only).
    pub key_source: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub enum StrategyMode {
    #[default]
    Always,
    IfModified,
}

/// A message entry: either a template block or an iterator over a context key.
#[derive(Debug, Clone)]
pub enum MessageSpec {
    Block {
        role: String,
        source: String,
    },
    Iterator {
        key: String,
        source: Option<String>,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        slice: Option<Vec<i64>>,
        /// Bind each item to this context name (e.g. `bind = "msg"` → `@msg`).
        bind: Option<String>,
        /// Override the role for all messages from this iterator.
        role: Option<String>,
    },
}

/// Generation parameters for model calls.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
}

/// Tool declaration.
#[derive(Debug, Clone)]
pub struct ToolDecl {
    pub name: String,
    pub params: HashMap<String, String>,
}
