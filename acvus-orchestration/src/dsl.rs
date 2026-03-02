use std::collections::HashMap;

use serde::Deserialize;

/// Node specification parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    #[serde(default)]
    pub messages: Vec<MessageSpec>,
}

/// A message entry: either a template block or an iterator over a storage key.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageSpec {
    Block { role: String, template: String },
    Iterator { iterator: String, template: Option<String> },
}

/// Tool declaration.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDecl {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, String>,
}
