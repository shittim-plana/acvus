use std::collections::HashMap;

use serde::Deserialize;

/// Node kind — determines how the node is executed.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum NodeKind {
    Llm,
}

/// Node specification parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    pub kind: NodeKind,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    #[serde(default)]
    pub messages: Vec<MessageSpec>,
    pub strategy: Strategy,
    #[serde(default)]
    pub generation: GenerationParams,
    /// Output template file path — rendered after the model responds.
    pub output: Option<String>,
    /// Inline output template.
    pub inline_output: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Strategy {
    #[serde(default)]
    pub mode: StrategyMode,
    /// Template file for cache key (if-modified only).
    pub key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyMode {
    #[default]
    Always,
    IfModified,
}

/// A message entry: either a template block or an iterator over a storage key.
///
/// Iterator is tried first so that `{iterator, role, template}` matches Iterator, not Block.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageSpec {
    Iterator {
        iterator: String,
        template: Option<String>,
        inline_template: Option<String>,
        /// Python-style slice: `[start]` or `[start, end]`. Negative = from end.
        #[serde(default)]
        slice: Option<Vec<i64>>,
        /// Bind each item to this context name (e.g. `bind = "msg"` → `@msg`).
        /// Without bind, the legacy `@type`/`@text` injection is used.
        bind: Option<String>,
        /// Override the role for all messages from this iterator.
        role: Option<String>,
    },
    Block {
        role: String,
        template: Option<String>,
        inline_template: Option<String>,
    },
}

/// Generation parameters for model calls.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
}

/// Tool declaration.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDecl {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, String>,
}
