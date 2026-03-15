pub(crate) mod display;
mod expr;
mod llm;
mod llm_cache;
mod plain;

pub use display::{CompiledDisplay, CompiledDisplayStatic};
pub use expr::{CompiledExpr, ExprSpec, compile_expr};
pub(crate) use llm::parse_type_name;
pub use llm::{
    CompiledLlm, CompiledToolBinding, CompiledToolParamInfo, GenerationParams, LlmSpec, MaxTokens,
    ThinkingConfig, ToolBinding, ToolParamInfo, compile_llm,
};
pub use llm_cache::{CompiledLlmCache, LlmCacheSpec, compile_llm_cache};
pub use plain::{CompiledPlain, PlainSpec, compile_plain};

use acvus_utils::{Astr, Interner};

use crate::compile::CompiledMessage;

/// Node kind — determines how the node is executed.
/// Config specific to each kind lives inside the variant.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Plain(PlainSpec),
    Llm(LlmSpec),
    LlmCache(LlmCacheSpec),
    Expr(ExprSpec),
    Display(DisplaySpec),
    DisplayStatic(DisplayStaticSpec),
    Iterator(IteratorSpec),
}

/// Spec for an iterable display node.
#[derive(Debug, Clone)]
pub struct DisplaySpec {
    pub iterator: String,
    pub template: String,
}

/// Spec for a static display node.
#[derive(Debug, Clone)]
pub struct DisplayStaticSpec {
    pub template: String,
}

/// Spec for a composite iterator node.
#[derive(Debug, Clone)]
pub struct IteratorSpec {
    pub sources: Vec<(String, Astr)>,
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
            NodeKind::Display(_) => acvus_mir::ty::Ty::String,
            NodeKind::DisplayStatic(_) => acvus_mir::ty::Ty::String,
            NodeKind::Iterator(_) => acvus_mir::ty::Ty::Infer,
        }
    }
}

/// Compiled node kind — mirrors `NodeKind` but with compiled data.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain(CompiledPlain),
    Llm(CompiledLlm),
    LlmCache(CompiledLlmCache),
    Expr(CompiledExpr),
    Display(CompiledDisplay),
    DisplayStatic(CompiledDisplayStatic),
    /// Composite iterator: pulls from multiple sources, yields items one by one.
    /// `unordered=false`: sequential (A then B).
    /// `unordered=true`: concurrent (FuturesUnordered, first-ready wins).
    Iterator {
        sources: Vec<(String, Astr)>,
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
            Self::Display(_) => &[],
            Self::DisplayStatic(_) => &[],
            Self::Iterator { .. } => &[],
        }
    }
}
