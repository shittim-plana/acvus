//! Cross-block code motion: hoist pure instructions above branch points.
//!
//! After SpawnSplit, IO calls are Spawn (async start) + Eval (blocking wait).
//! This pass hoists Spawn and other pure instructions out of dominated blocks
//! into their dominator ancestors, maximizing the distance between Spawn and Eval.
//!
//! # Algorithm
//!
//! Each iteration:
//! 1. Build dominator tree + token liveness.
//! 2. For each hoistable instruction, walk UP the dominator chain to find the
//!    **highest ancestor** where all operands are available and no token conflict.
//!    This eliminates the need for multi-iteration fixpoint on deep merge chains.
//! 3. `def_block` is updated after each hoist decision, so later instructions
//!    in the same block see their dependencies' new locations — operand chains
//!    are resolved in a single pass.
//! 4. Repeat until no more instructions can be hoisted (fixpoint for cross-block
//!    chains, but typically converges in 1 iteration).
//!
//! # Hoistability (allowlist)
//!
//! Only provably pure instructions are hoisted. New/unknown instruction kinds
//! default to "not hoistable" (soundness by construction).
//!
//! Pure: arithmetic, value construction, field access, test predicates,
//! Spawn, LoadFunction, pure FunctionCall (no IO/tokens/context effects).
//!
//! NOT hoisted: Eval, context ops, variable ops, indirect calls.

use rustc_hash::FxHashMap;

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::token_liveness;
use crate::cfg::{BlockIdx, CfgBody};
use crate::graph::QualifiedRef;
use crate::ir::*;
use crate::ty::{Effect, EffectTarget, TokenId, Ty};

// ── Entry point ────────────────────────────────────────────────────

pub fn run(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>) {
    loop {
        if !hoist_pass(cfg, fn_types) {
            break;
        }
    }
}

// ── Single hoist pass ──────────────────────────────────────────────

/// One iteration: find hoistable instructions, move them to the highest
/// valid dominator. Returns true if anything moved.
fn hoist_pass(cfg: &mut CfgBody, fn_types: &FxHashMap<QualifiedRef, Ty>) -> bool {
    if cfg.blocks.len() < 2 {
        return false;
    }

    let domtree = DomTree::build(cfg);
    let tok_liveness = token_liveness::analyze(cfg, fn_types);
    let mut def_block = build_def_block(cfg);

    // ── Collect hoists ─────────────────────────────────────────────
    //
    // For each instruction, find the highest dominator ancestor where
    // all operands are available and no token conflicts exist.
    // def_block is updated after each decision so that operand chains
    // within a block are resolved in one pass.

    let mut hoists: Vec<(usize, usize, usize)> = Vec::new(); // (src_block, inst_idx, tgt_block)

    for (bi, block) in cfg.blocks.iter().enumerate() {
        if domtree.idom(BlockIdx(bi)).is_none() {
            continue;
        }

        for (i, inst) in block.insts.iter().enumerate() {
            let kind = &inst.kind;

            if !is_hoistable(kind, fn_types) || inst_info::defs(kind).is_empty() {
                continue;
            }

            let uses = inst_info::uses(kind);
            let tokens = token_ids_of(kind, fn_types, &cfg.val_types);

            if let Some(target) = find_highest_target(
                BlockIdx(bi), &uses, &tokens,
                &domtree, &def_block, &tok_liveness, cfg,
            ) {
                hoists.push((bi, i, target.0));
                for d in inst_info::defs(kind) {
                    def_block.insert(d, target);
                }
            }
        }
    }

    if hoists.is_empty() {
        return false;
    }

    // ── Apply hoists ───────────────────────────────────────────────

    let mut to_move: FxHashMap<usize, Vec<Inst>> = FxHashMap::default();
    for &(src_bi, inst_i, tgt_bi) in &hoists {
        to_move.entry(tgt_bi).or_default().push(cfg.blocks[src_bi].insts[inst_i].clone());
    }

    // Nop originals.
    for &(bi, i, _) in &hoists {
        cfg.blocks[bi].insts[i].kind = InstKind::Nop;
    }

    // Append to target blocks (before terminator, which is separate).
    for (tgt_bi, insts) in to_move {
        cfg.blocks[tgt_bi].insts.extend(insts);
    }

    true
}

// ── Def-block map ──────────────────────────────────────────────────

/// Build ValueId → BlockIdx mapping: where each value is defined.
fn build_def_block(cfg: &CfgBody) -> FxHashMap<ValueId, BlockIdx> {
    let mut def_block = FxHashMap::default();

    for (bi, block) in cfg.blocks.iter().enumerate() {
        let idx = BlockIdx(bi);
        for &p in &block.params {
            def_block.insert(p, idx);
        }
        for inst in &block.insts {
            for d in inst_info::defs(&inst.kind) {
                def_block.insert(d, idx);
            }
        }
        if let crate::cfg::Terminator::ListStep { dst, index_dst, .. } = &block.terminator {
            def_block.insert(*dst, idx);
            def_block.insert(*index_dst, idx);
        }
    }

    for &v in cfg.param_regs.iter().chain(cfg.capture_regs.iter()) {
        def_block.entry(v).or_insert(BlockIdx(0));
    }

    def_block
}

// ── Target finding ─────────────────────────────────────────────────

/// Walk up the dominator chain from `block_idx` to find the highest ancestor
/// where all operands are available (before the ancestor's terminator) and
/// no token conflicts exist.
fn find_highest_target(
    block_idx: BlockIdx,
    uses: &[ValueId],
    tokens: &smallvec::SmallVec<[TokenId; 2]>,
    domtree: &DomTree,
    def_block: &FxHashMap<ValueId, BlockIdx>,
    tok_liveness: &token_liveness::TokenLivenessResult,
    cfg: &CfgBody,
) -> Option<BlockIdx> {
    let mut best: Option<BlockIdx> = None;
    let mut candidate = domtree.idom(block_idx)?;

    loop {
        // All operands must be available at candidate's body (before terminator).
        let all_available = uses.iter().all(|u| {
            match def_block.get(u) {
                Some(&def_bi) if def_bi == candidate => {
                    !is_terminator_def(&cfg.blocks[candidate.0].terminator, *u)
                }
                Some(&def_bi) => domtree.dominates(def_bi, candidate),
                None => true,
            }
        });
        if !all_available {
            break;
        }

        // No token conflict at this level.
        if tokens.iter().any(|tid| tok_liveness.is_live_out(candidate, *tid)) {
            break;
        }

        best = Some(candidate);

        match domtree.idom(candidate) {
            Some(parent) => candidate = parent,
            None => break,
        }
    }

    best
}

// ── Hoistability (allowlist) ───────────────────────────────────────

/// Can this instruction be safely hoisted to a dominator block?
///
/// Allowlist: only provably pure instructions. Unknown kinds default to
/// not hoistable (soundness by construction).
fn is_hoistable(kind: &InstKind, fn_types: &FxHashMap<QualifiedRef, Ty>) -> bool {
    match kind {
        // Arithmetic / logic.
        InstKind::BinOp { .. }
        | InstKind::UnaryOp { .. }
        | InstKind::Cast { .. } => true,

        // Value construction.
        InstKind::Const { .. }
        | InstKind::MakeDeque { .. }
        | InstKind::MakeObject { .. }
        | InstKind::MakeTuple { .. }
        | InstKind::MakeRange { .. }
        | InstKind::MakeVariant { .. }
        | InstKind::MakeClosure { .. } => true,

        // Field / element access.
        InstKind::FieldGet { .. }
        | InstKind::ObjectGet { .. }
        | InstKind::ListIndex { .. }
        | InstKind::ListGet { .. }
        | InstKind::ListSlice { .. }
        | InstKind::TupleIndex { .. }
        | InstKind::UnwrapVariant { .. } => true,

        // Test predicates.
        InstKind::TestLiteral { .. }
        | InstKind::TestRange { .. }
        | InstKind::TestVariant { .. }
        | InstKind::TestListLen { .. }
        | InstKind::TestObjectKey { .. } => true,

        // Spawn (async start, no side-effect until Eval).
        // Only Direct: Indirect callee tokens can't be tracked.
        InstKind::Spawn { callee: Callee::Direct(..), .. } => true,

        // Function reference.
        InstKind::LoadFunction { .. } => true,

        // Pure direct FunctionCall.
        InstKind::FunctionCall { callee: Callee::Direct(qref), .. } => {
            is_pure_call(fn_types, qref)
        }

        // Everything else: NOT hoistable.
        _ => false,
    }
}

fn is_pure_call(fn_types: &FxHashMap<QualifiedRef, Ty>, qref: &QualifiedRef) -> bool {
    let Some(Ty::Fn { effect: Effect::Resolved(eff), .. }) = fn_types.get(qref) else {
        return false;
    };
    eff.is_pure()
}

// ── Terminator helpers ─────────────────────────────────────────────

fn is_terminator_def(term: &crate::cfg::Terminator, val: ValueId) -> bool {
    match term {
        crate::cfg::Terminator::ListStep { dst, index_dst, .. } => val == *dst || val == *index_dst,
        _ => false,
    }
}

fn token_ids_of(
    kind: &InstKind,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
    val_types: &FxHashMap<ValueId, Ty>,
) -> smallvec::SmallVec<[TokenId; 2]> {
    let effect_set = match kind {
        InstKind::FunctionCall { callee: Callee::Direct(qref), .. }
        | InstKind::Spawn { callee: Callee::Direct(qref), .. } => {
            fn_types.get(qref).and_then(|ty| match ty {
                Ty::Fn { effect: Effect::Resolved(eff), .. } => Some(eff),
                _ => None,
            })
        }
        _ => None,
    };

    let Some(eff) = effect_set else {
        return smallvec::SmallVec::new();
    };

    eff.reads
        .iter()
        .chain(eff.writes.iter())
        .filter_map(|t| match t {
            EffectTarget::Token(tid) => Some(*tid),
            _ => None,
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{self, CfgBody};
    use crate::ty::{Effect, EffectSet, EffectTarget, Param};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use std::collections::BTreeSet;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>, val_count: usize) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..val_count {
            factory.next();
        }
        cfg::promote(MirBody {
            insts: insts.into_iter().map(|kind| Inst { span: acvus_ast::Span::ZERO, kind }).collect(),
            val_types: FxHashMap::default(),
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    fn io_fn_type(i: &Interner, name: &str) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        (qref, Ty::Fn {
            params: vec![], ret: Box::new(Ty::Int), captures: vec![],
            effect: Effect::Resolved(EffectSet { io: true, ..Default::default() }),
        })
    }

    fn token_fn_type(i: &Interner, name: &str, tid: TokenId) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        let mut reads = BTreeSet::new();
        reads.insert(EffectTarget::Token(tid));
        (qref, Ty::Fn {
            params: vec![], ret: Box::new(Ty::Int), captures: vec![],
            effect: Effect::Resolved(EffectSet { reads, writes: BTreeSet::new(), io: true, self_modifying: false }),
        })
    }

    fn demoted(c: CfgBody) -> MirBody { cfg::demote(c) }

    fn kinds(body: &MirBody) -> Vec<&InstKind> {
        body.insts.iter().map(|i| &i.kind).collect()
    }

    #[test]
    fn spawn_hoisted_above_branch() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(0), then_label: Label(0), then_args: vec![], else_label: Label(1), else_args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::Spawn { dst: v(1), callee: Callee::Direct(qref), args: vec![v(0)], context_uses: vec![] },
            InstKind::Return(v(1)),
        ], 10);

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k.iter().position(|k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let jumpif_idx = k.iter().position(|k| matches!(k, InstKind::JumpIf { .. })).unwrap();
        assert!(spawn_idx < jumpif_idx, "spawn should be hoisted before branch");
    }

    #[test]
    fn block_param_dependency_prevents_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(0), then_label: Label(0), then_args: vec![v(0)], else_label: Label(1), else_args: vec![v(0)] },
            InstKind::BlockLabel { label: Label(0), params: vec![v(1)], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![v(1)] },
            InstKind::BlockLabel { label: Label(1), params: vec![v(2)], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![v(2)] },
            InstKind::BlockLabel { label: Label(2), params: vec![v(3)], merge_of: None },
            InstKind::Spawn { dst: v(4), callee: Callee::Direct(qref), args: vec![v(3)], context_uses: vec![] },
            InstKind::Return(v(4)),
        ], 10);

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k.iter().position(|k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let merge_idx = k.iter().position(|k| matches!(k, InstKind::BlockLabel { label: Label(2), .. })).unwrap();
        assert!(spawn_idx > merge_idx, "spawn should stay in merge block");
    }

    #[test]
    fn eval_not_hoisted() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        let mut cfg = make_cfg(vec![
            InstKind::Spawn { dst: v(0), callee: Callee::Direct(qref), args: vec![], context_uses: vec![] },
            InstKind::Const { dst: v(5), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(5), then_label: Label(0), then_args: vec![], else_label: Label(1), else_args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::Eval { dst: v(1), src: v(0), context_defs: vec![] },
            InstKind::Return(v(1)),
        ], 10);

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let eval_idx = k.iter().position(|k| matches!(k, InstKind::Eval { .. })).unwrap();
        let merge_idx = k.iter().position(|k| matches!(k, InstKind::BlockLabel { label: Label(2), .. })).unwrap();
        assert!(eval_idx > merge_idx, "eval must stay in merge block");
    }

    #[test]
    fn token_conflict_prevents_hoist() {
        let i = Interner::new();
        let tid = TokenId::alloc();
        let (qref1, ty1) = token_fn_type(&i, "io1", tid);
        let (qref2, ty2) = token_fn_type(&i, "io2", tid);
        let fn_types = FxHashMap::from_iter([(qref1, ty1), (qref2, ty2)]);

        let mut cfg = make_cfg(vec![
            InstKind::Spawn { dst: v(0), callee: Callee::Direct(qref1), args: vec![], context_uses: vec![] },
            InstKind::Const { dst: v(5), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(5), then_label: Label(0), then_args: vec![], else_label: Label(1), else_args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::Spawn { dst: v(1), callee: Callee::Direct(qref2), args: vec![], context_uses: vec![] },
            InstKind::Return(v(1)),
        ], 10);

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn2_idx = k.iter().rposition(|k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let merge_idx = k.iter().position(|k| matches!(k, InstKind::BlockLabel { label: Label(2), .. })).unwrap();
        assert!(spawn2_idx > merge_idx, "token-conflicting spawn should stay in merge block");
    }

    #[test]
    fn multi_level_hoist() {
        let i = Interner::new();
        let (qref, ty) = io_fn_type(&i, "io_fn");
        let fn_types = FxHashMap::from_iter([(qref, ty)]);

        // B0 → diamond → B3 → diamond → B6
        // Spawn in B6 should hoist directly to B0 in one pass.
        let mut cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(0), then_label: Label(0), then_args: vec![], else_label: Label(1), else_args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::JumpIf { cond: v(0), then_label: Label(3), then_args: vec![], else_label: Label(4), else_args: vec![] },
            InstKind::BlockLabel { label: Label(3), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(5), args: vec![] },
            InstKind::BlockLabel { label: Label(4), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(5), args: vec![] },
            InstKind::BlockLabel { label: Label(5), params: vec![], merge_of: None },
            InstKind::Spawn { dst: v(1), callee: Callee::Direct(qref), args: vec![v(0)], context_uses: vec![] },
            InstKind::Return(v(1)),
        ], 10);

        run(&mut cfg, &fn_types);
        let body = demoted(cfg);
        let k = kinds(&body);
        let spawn_idx = k.iter().position(|k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let first_jumpif = k.iter().position(|k| matches!(k, InstKind::JumpIf { .. })).unwrap();
        assert!(spawn_idx < first_jumpif, "spawn should be hoisted to B0 via highest-target");
    }

    #[test]
    fn pure_instruction_hoisted() {
        let mut cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf { cond: v(0), then_label: Label(0), then_args: vec![], else_label: Label(1), else_args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::BinOp { dst: v(1), op: acvus_ast::BinOp::Add, left: v(0), right: v(0) },
            InstKind::Return(v(1)),
        ], 10);

        run(&mut cfg, &FxHashMap::default());
        let body = demoted(cfg);
        let k = kinds(&body);
        let binop_idx = k.iter().position(|k| matches!(k, InstKind::BinOp { .. })).unwrap();
        let jumpif_idx = k.iter().position(|k| matches!(k, InstKind::JumpIf { .. })).unwrap();
        assert!(binop_idx < jumpif_idx, "pure BinOp should be hoisted before branch");
    }
}
