mod expr;
mod llm;
mod llm_cache;
mod plain;

pub use expr::{CompiledExpr, ExprSpec, compile_expr};
pub(crate) use llm::parse_type_name;
pub use llm::{
    CompiledLlm, CompiledToolBinding, CompiledToolParamInfo, GenerationParams, LlmSpec, MaxTokens,
    ThinkingConfig, ToolBinding, ToolParamInfo, compile_llm,
};
pub use llm_cache::{CompiledLlmCache, LlmCacheSpec, compile_llm_cache};
pub use plain::{CompiledPlain, PlainSpec, compile_plain};

use acvus_utils::{Astr, Interner};

use crate::compile::{CompiledMessage, CompiledScript};

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain(PlainSpec),
    Llm(LlmSpec),
    LlmCache(LlmCacheSpec),
    Expr(ExprSpec),
    Iterator(IteratorSpec),
}

/// Per-item transform for an iterator source.
#[derive(Debug, Clone)]
pub enum SourceTransform {
    /// Template interpolation — string with `{{ }}` blocks.
    /// Compiled via `compile_template`.
    Template(Astr),
    /// Script expression — arbitrary value transform.
    /// Compiled via `compile_script`.
    Script(Astr),
}

/// Per-item entry: condition (filter) + transform (map).
///
/// When processing each item, entries are evaluated in order.
/// The first entry whose condition matches has its transform applied.
/// Items with no matching entry are skipped.
#[derive(Debug, Clone)]
pub struct IteratorEntry {
    /// Condition script → Bool. None = always match.
    pub condition: Option<Astr>,
    /// Transform to apply when condition matches.
    pub transform: SourceTransform,
}

/// A single source in an iterator node.
///
/// Evaluates `expr` to get a value, then:
/// - If Iterator/List/Deque: applies skip(`start`) → take(`end`)
/// - If scalar: single item
///
/// For each item, `entries` are evaluated in order (first-match):
/// - No entries = pass-through (item yielded as-is)
/// - One entry (condition=None) = simple map
/// - Multiple entries = conditional map (first match wins, no match → skip)
///
/// Results are tagged with `name` as `{name: String, item: T}`.
#[derive(Debug, Clone)]
pub struct IteratorSource {
    pub name: String,
    pub expr: Astr,
    pub entries: Vec<IteratorEntry>,
    /// Skip N items from the start. Script → Int.
    pub start: Option<Astr>,
    /// Take up to N items. Script → Option<Int>. None = exhaust.
    pub end: Option<Astr>,
}

/// Spec for a composite iterator node.
///
/// Pulls items from multiple sources, tags each with the source name,
/// and yields `{name: String, item: T}` objects.
///
/// `unordered=false`: sequential — exhaust source A, then B, etc.
/// `unordered=true`: concurrent — yield from whichever source is ready first.
#[derive(Debug, Clone)]
pub struct IteratorSpec {
    pub sources: Vec<IteratorSource>,
    pub unordered: bool,
}

impl NodeKind {
    /// The raw output type produced by this node kind (before self.bind).
    pub fn raw_output_ty(&self, interner: &Interner) -> acvus_mir::ty::Ty {
        match self {
            NodeKind::Plain(spec) => spec.output_ty(),
            NodeKind::Llm(spec) => spec.output_ty(interner),
            NodeKind::LlmCache(spec) => spec.output_ty(),
            NodeKind::Expr(spec) => spec.output_ty.clone(),
            NodeKind::Iterator(_) => acvus_mir::ty::Ty::Infer,
        }
    }
}

/// Compiled per-item transform.
#[derive(Debug, Clone)]
pub enum CompiledSourceTransform {
    Template(CompiledScript),
    Script(CompiledScript),
}

/// Compiled entry: condition + transform.
#[derive(Debug, Clone)]
pub struct CompiledIteratorEntry {
    pub condition: Option<CompiledScript>,
    pub transform: CompiledSourceTransform,
}

/// Compiled iterator source.
#[derive(Debug, Clone)]
pub struct CompiledIteratorSource {
    pub name: String,
    pub expr: CompiledScript,
    pub entries: Vec<CompiledIteratorEntry>,
    pub start: Option<CompiledScript>,
    pub end: Option<CompiledScript>,
}

/// Compiled node kind — mirrors `NodeKind` but with compiled data.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain(CompiledPlain),
    Llm(CompiledLlm),
    LlmCache(CompiledLlmCache),
    Expr(CompiledExpr),
    Iterator {
        sources: Vec<CompiledIteratorSource>,
        unordered: bool,
    },
}

impl CompiledNodeKind {
    pub fn messages(&self) -> &[CompiledMessage] {
        match self {
            Self::Plain(_) => &[],
            Self::Llm(llm) => &llm.messages,
            Self::LlmCache(cache) => &cache.messages,
            Self::Expr(_) => &[],
            Self::Iterator { .. } => &[],
        }
    }
}
