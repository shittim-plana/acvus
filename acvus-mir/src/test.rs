//! Test helpers for compiling source → MIR via the graph pipeline.
//!
//! These are convenience wrappers around extract → infer → lower.
//! Real callers should use the graph phases directly.

use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::error::MirError;
use crate::graph::*;
use crate::graph::{extract, infer, lower as graph_lower};
use crate::hints::HintTable;
use crate::ir::MirModule;
use crate::ty::Ty;

/// Build a single-unit CompilationGraph for testing.
/// Returns the graph and the `QualifiedRef` of the test unit.
fn make_graph(
    interner: &Interner,
    source: &str,
    kind: SourceKind,
    ctx: &[(&str, Ty)],
) -> (CompilationGraph, QualifiedRef) {
    let contexts = ctx
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(interner.intern(name)),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();
    let test_qref = QualifiedRef::root(interner.intern("test"));
    let mut functions = crate::builtins::standard_builtins(interner);
    functions.push(Function {
        qref: test_qref,
        kind: FnKind::Local(SourceCode {
            name: test_qref,
            source: interner.intern(source),
            kind,
        }),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    });
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };
    (graph, test_qref)
}

fn run_pipeline(
    interner: &Interner,
    graph: &CompilationGraph,
    target: QualifiedRef,
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let ext = extract::extract(interner, graph);
    let inf = infer::infer(interner, graph, &ext, &FxHashMap::default(), Freeze::default());
    let result = graph_lower::lower(interner, graph, &ext, &inf);
    if result.has_errors() {
        return Err(result.errors.into_iter().flat_map(|e| e.errors).collect());
    }
    result
        .modules
        .into_iter()
        .find(|(id, _)| *id == target)
        .map(|(_, pair)| pair)
        .ok_or_else(Vec::new)
}

/// Compile a template source string through the full graph pipeline.
pub fn compile_template(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let (graph, target) = make_graph(interner, source, SourceKind::Template, ctx);
    run_pipeline(interner, &graph, target)
}

/// Compile a script source string through the full graph pipeline.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    ctx: &[(&str, Ty)],
) -> Result<(MirModule, HintTable), Vec<MirError>> {
    let (graph, target) = make_graph(interner, source, SourceKind::Script, ctx);
    run_pipeline(interner, &graph, target)
}
