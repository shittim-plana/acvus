//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.
//!
//! Pass 1 (cross-module): SSA → Inline
//! Pass 2 (per-module):   SpawnSplit → CodeMotion → Reorder → SSA → RegColor → Validate
//!
//! Each body is promoted to CfgBody once, all passes run, then demoted once.
//! Validate runs on MirBody (after demote) to catch demotion bugs.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{self, CfgBody};
use crate::graph::inliner;
use crate::graph::QualifiedRef;
use crate::ir::MirModule;
use crate::optimize;
use crate::ty::Ty;
use crate::validate::{self, ValidationError};

/// Result of the optimization pipeline.
pub struct OptimizeResult {
    /// Optimized modules, keyed by function QualifiedRef.
    pub modules: FxHashMap<QualifiedRef, MirModule>,
    /// Validation errors per function (empty = valid).
    pub errors: Vec<(QualifiedRef, Vec<ValidationError>)>,
}

/// Run the full optimization pipeline.
///
/// `modules`: lowered MIR modules from Phase 3 (lower).
/// `fn_types`: QualifiedRef → Ty mapping for all functions.
/// `recursive_fns`: set of functions involved in recursion (SCC).
pub fn optimize(
    modules: FxHashMap<QualifiedRef, MirModule>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> OptimizeResult {
    // ── Pass 1: SSA (per-module) → Inline (cross-module) ────────────
    //
    // SSA runs on CfgBody, then demotes back for cross-module inlining.

    let mut ssa_modules = modules;
    for module in ssa_modules.values_mut() {
        run_pass1_body(&mut module.main, fn_types);
        for closure in module.closures.values_mut() {
            run_pass1_body(closure, fn_types);
        }
    }

    let inlined = inliner::inline(&ssa_modules, recursive_fns);

    // ── Pass 2: Optimize + Validate (per-module, direct calls) ──────
    //
    // promote once → all passes on CfgBody → demote once → validate on MirBody.

    let mut result_modules = FxHashMap::default();
    let mut all_errors = Vec::new();

    for (qref, mut module) in inlined.modules {
        run_pass2_body(&mut module.main, fn_types);
        for closure in module.closures.values_mut() {
            run_pass2_body(closure, fn_types);
        }

        // Validate on MirBody — after demote, catches demotion bugs.
        let errors = validate::validate(&module, fn_types);
        if !errors.is_empty() {
            all_errors.push((qref, errors));
        }

        result_modules.insert(qref, module);
    }

    OptimizeResult {
        modules: result_modules,
        errors: all_errors,
    }
}

/// Pass 1: SSA on a single body (promote → ssa → demote).
fn run_pass1_body(
    body: &mut crate::ir::MirBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
) {
    let mut cfg = cfg::promote(std::mem::take(body));
    optimize::ssa_pass::run(&mut cfg, fn_types);
    *body = cfg::demote(cfg);
}

/// Pass 2: Full optimization pipeline on a single body.
/// promote once → all passes → demote once.
fn run_pass2_body(
    body: &mut crate::ir::MirBody,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
) {
    let mut cfg = cfg::promote(std::mem::take(body));
    run_pass2(&mut cfg, fn_types);
    *body = cfg::demote(cfg);
}

/// Pass 2 pipeline on CfgBody: SpawnSplit → CodeMotion → Reorder → SSA → RegColor.
fn run_pass2(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>) {
    optimize::spawn_split::run(cfg, fn_types);
    optimize::code_motion::run(cfg, fn_types);
    optimize::reorder::run(cfg, fn_types);
    optimize::ssa_pass::run(cfg, fn_types);
    optimize::reg_color::color_body(cfg);
}
