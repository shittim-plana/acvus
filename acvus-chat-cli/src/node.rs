use std::collections::HashMap;
use std::path::Path;

use acvus_orchestration::{
    GenerationParams, MessageSpec, NodeKind, NodeSpec, Strategy, StrategyMode, ToolDecl,
};
use serde::Deserialize;

/// A context reference like `"@node-name"`. Strips the `@` prefix on deserialization.
#[derive(Debug, Clone)]
struct ContextRef(String);

impl<'de> Deserialize<'de> for ContextRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.strip_prefix('@') {
            Some(name) => Ok(ContextRef(name.to_string())),
            None => Err(serde::de::Error::custom(format!(
                "context reference must start with '@', got: {s}"
            ))),
        }
    }
}

/// TOML-deserializable node definition.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum NodeKindDef {
    Llm,
    LlmCache {
        ttl: String,
        #[serde(default)]
        cache_config: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeDef {
    pub name: String,
    #[serde(flatten)]
    kind: NodeKindDef,
    provider: String,
    model: String,
    #[serde(default)]
    tools: Vec<ToolDeclDef>,
    #[serde(default)]
    messages: Vec<MessageDef>,
    #[serde(default)]
    strategy: StrategyDef,
    #[serde(default)]
    generation: GenerationParamsDef,
    cache_key: Option<ContextRef>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct StrategyDef {
    #[serde(default)]
    mode: StrategyModeDef,
    /// Template file for cache key (if-modified only).
    key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
enum StrategyModeDef {
    #[default]
    Always,
    IfModified,
}

/// Serde message entry — tried as Iterator first (untagged).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum MessageDef {
    Iterator {
        iterator: ContextRef,
        template: Option<String>,
        inline_template: Option<String>,
        #[serde(default)]
        slice: Option<Vec<i64>>,
        bind: Option<String>,
        role: Option<String>,
    },
    Block {
        role: String,
        template: Option<String>,
        inline_template: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Default)]
struct GenerationParamsDef {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolDeclDef {
    name: String,
    #[serde(default)]
    params: HashMap<String, String>,
}

/// Resolve a template: file path or inline source → source string.
fn resolve_template(
    base_dir: &Path,
    file: Option<&str>,
    inline: Option<&str>,
) -> Result<String, String> {
    match (file, inline) {
        (Some(path), _) => {
            let full = base_dir.join(path);
            std::fs::read_to_string(&full)
                .map_err(|e| format!("failed to load template '{}': {e}", full.display()))
        }
        (None, Some(src)) => Ok(src.to_string()),
        (None, None) => Err("no template or inline_template provided".into()),
    }
}

/// Convert a TOML `NodeDef` into a pure `NodeSpec`, reading template files from `base_dir`.
pub fn resolve_node(def: NodeDef, base_dir: &Path) -> Result<NodeSpec, String> {
    let kind = match def.kind {
        NodeKindDef::Llm => NodeKind::Llm,
        NodeKindDef::LlmCache { ttl, cache_config } => NodeKind::LlmCache { ttl, cache_config },
    };

    let mut messages = Vec::new();
    for (i, msg) in def.messages.into_iter().enumerate() {
        match msg {
            MessageDef::Block { role, template, inline_template } => {
                let source = resolve_template(base_dir, template.as_deref(), inline_template.as_deref())
                    .map_err(|e| format!("message {i}: {e}"))?;
                messages.push(MessageSpec::Block { role, source });
            }
            MessageDef::Iterator { iterator, template, inline_template, slice, bind, role } => {
                let source = if template.is_some() || inline_template.is_some() {
                    Some(
                        resolve_template(base_dir, template.as_deref(), inline_template.as_deref())
                            .map_err(|e| format!("message {i}: {e}"))?,
                    )
                } else {
                    None
                };
                messages.push(MessageSpec::Iterator {
                    key: iterator.0,
                    source,
                    slice,
                    bind,
                    role,
                });
            }
        }
    }

    let strategy = Strategy {
        mode: match def.strategy.mode {
            StrategyModeDef::Always => StrategyMode::Always,
            StrategyModeDef::IfModified => StrategyMode::IfModified,
        },
        key_source: match def.strategy.key {
            Some(path) => {
                let full = base_dir.join(&path);
                let src = std::fs::read_to_string(&full)
                    .map_err(|e| format!("failed to load strategy key '{}': {e}", full.display()))?;
                Some(src)
            }
            None => None,
        },
    };

    let generation = GenerationParams {
        temperature: def.generation.temperature,
        top_p: def.generation.top_p,
        top_k: def.generation.top_k,
        max_tokens: def.generation.max_tokens,
    };

    let tools = def.tools.into_iter().map(|t| ToolDecl {
        name: t.name,
        params: t.params,
    }).collect();

    let cache_key = def.cache_key.map(|r| r.0);

    Ok(NodeSpec {
        name: def.name,
        kind,
        provider: def.provider,
        model: def.model,
        tools,
        messages,
        strategy,
        generation,
        cache_key,
    })
}
