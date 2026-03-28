//! Reorder pass: dependency-preserving instruction reordering.
//!
//! After spawn_split, IO calls are expressed as Spawn + Eval pairs.
//! This pass reorders instructions so that:
//! - Spawns are scheduled as early as possible (latency hiding)
//! - Evals are scheduled as late as possible (just before first use)
//! - Independent instructions fill the gap between Spawn and Eval
//!
//! Dependency constraints:
//! - SSA use-def: instruction B uses value defined by A → A before B
//! - Token ordering: instructions sharing the same TokenId must preserve
//!   their original relative order
//!
//! Control flow boundaries (BlockLabel, Jump, JumpIf, Return) are pinned
//! and act as barriers — reordering happens only within each basic block.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::inst_info;
use crate::graph::QualifiedRef;
use crate::ir::*;
use crate::ty::{Effect, EffectTarget, Ty, TokenId};

/// Reorder instructions within each basic block for optimal Spawn/Eval scheduling.
///
/// `fn_types`: QualifiedRef → Ty for looking up callee effects (Token extraction).
pub fn run(body: &mut MirBody, fn_types: &FxHashMap<QualifiedRef, Ty>) {
    // Split instructions into basic blocks (separated by control flow).
    let blocks = split_into_blocks(&body.insts);
    let mut new_insts = Vec::with_capacity(body.insts.len());

    for block in blocks {
        let reordered = reorder_block(block, &body.val_types, fn_types);
        new_insts.extend(reordered);
    }

    body.insts = new_insts;
}

/// A contiguous run of instructions within a basic block.
/// Control flow instructions (BlockLabel at start, Jump/JumpIf/Return at end)
/// are pinned and not reordered.
struct Block {
    /// Pinned instructions at the start (BlockLabel, if any).
    header: Vec<Inst>,
    /// Reorderable instructions in the middle.
    body: Vec<Inst>,
    /// Pinned instructions at the end (Jump/JumpIf/Return, if any).
    footer: Vec<Inst>,
}

/// Split a flat instruction list into basic blocks.
fn split_into_blocks(insts: &[Inst]) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut footer = Vec::new();

    for inst in insts {
        if inst_info::is_control_flow(&inst.kind) {
            match &inst.kind {
                // BlockLabel starts a new block.
                InstKind::BlockLabel { .. } => {
                    // Flush previous block if there's anything.
                    if !header.is_empty() || !body.is_empty() || !footer.is_empty() {
                        blocks.push(Block {
                            header: std::mem::take(&mut header),
                            body: std::mem::take(&mut body),
                            footer: std::mem::take(&mut footer),
                        });
                    }
                    header.push(inst.clone());
                }
                // Jump/JumpIf/Return ends the current block.
                _ => {
                    footer.push(inst.clone());
                    blocks.push(Block {
                        header: std::mem::take(&mut header),
                        body: std::mem::take(&mut body),
                        footer: std::mem::take(&mut footer),
                    });
                }
            }
        } else {
            body.push(inst.clone());
        }
    }

    // Flush any remaining instructions (block without terminator).
    if !header.is_empty() || !body.is_empty() || !footer.is_empty() {
        blocks.push(Block { header, body, footer });
    }

    blocks
}

/// Priority for scheduling. Lower value = scheduled earlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    /// Spawn — schedule as early as possible.
    Spawn,
    /// Normal instruction.
    Normal(usize),
    /// Eval — schedule as late as possible.
    Eval,
}

fn classify(inst: &InstKind, original_index: usize) -> Priority {
    match inst {
        InstKind::Spawn { .. } => Priority::Spawn,
        InstKind::Eval { .. } => Priority::Eval,
        _ => Priority::Normal(original_index),
    }
}

/// Extract TokenIds from a FunctionCall/Spawn/Eval's effect.
fn token_deps(
    kind: &InstKind,
    val_types: &FxHashMap<ValueId, Ty>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
) -> SmallVec<[TokenId; 2]> {
    let effect_set = match kind {
        InstKind::FunctionCall {
            callee: Callee::Direct(qref),
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Direct(qref),
            ..
        } => fn_types.get(qref).and_then(|ty| match ty {
            Ty::Fn {
                effect: Effect::Resolved(eff),
                ..
            } => Some(eff),
            _ => None,
        }),
        InstKind::Eval { src, .. } => val_types.get(src).and_then(|ty| match ty {
            Ty::Handle(_, Effect::Resolved(eff)) => Some(eff),
            _ => None,
        }),
        _ => None,
    };

    let Some(eff) = effect_set else {
        return SmallVec::new();
    };

    eff.reads
        .iter()
        .chain(eff.writes.iter())
        .filter_map(|target| match target {
            EffectTarget::Token(tid) => Some(*tid),
            _ => None,
        })
        .collect()
}

/// Reorder instructions within a single basic block.
fn reorder_block(
    block: Block,
    val_types: &FxHashMap<ValueId, Ty>,
    fn_types: &FxHashMap<QualifiedRef, Ty>,
) -> Vec<Inst> {
    let n = block.body.len();
    if n <= 1 {
        // Nothing to reorder.
        let mut result = block.header;
        result.extend(block.body);
        result.extend(block.footer);
        return result;
    }

    // Build dependency edges: deps[i] = set of instruction indices that must come before i.
    let mut deps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];

    // 1. SSA use-def dependencies.
    // def_map: ValueId → instruction index within this block.
    let mut def_map: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (i, inst) in block.body.iter().enumerate() {
        for d in inst_info::defs(&inst.kind) {
            def_map.insert(d, i);
        }
    }
    for (i, inst) in block.body.iter().enumerate() {
        for u in inst_info::uses(&inst.kind) {
            if let Some(&def_idx) = def_map.get(&u) {
                if def_idx != i {
                    deps[i].push(def_idx);
                }
            }
            // If use is defined outside this block, no intra-block dependency.
        }
    }

    // 2. Token ordering: instructions sharing the same TokenId preserve original order.
    let mut last_token_user: FxHashMap<TokenId, usize> = FxHashMap::default();
    for (i, inst) in block.body.iter().enumerate() {
        let tokens = token_deps(&inst.kind, val_types, fn_types);
        for tid in &tokens {
            if let Some(&prev) = last_token_user.get(tid) {
                deps[i].push(prev);
            }
            last_token_user.insert(*tid, i);
        }
    }

    // 3. ContextStore ordering: stores to the same context preserve original order.
    // context_load→context_store is already handled by SSA use-def,
    // but store→store to the same context needs explicit ordering.
    let mut last_ctx_store: FxHashMap<QualifiedRef, usize> = FxHashMap::default();
    for (i, inst) in block.body.iter().enumerate() {
        if let InstKind::ContextStore { dst, .. } = &inst.kind {
            // dst is a context_project value; we need the context QualifiedRef.
            // Look up what dst was defined as.
            if let Some(&def_idx) = def_map.get(dst) {
                if let InstKind::ContextProject { ctx, .. } = &block.body[def_idx].kind {
                    if let Some(&prev) = last_ctx_store.get(ctx) {
                        if prev != i {
                            deps[i].push(prev);
                        }
                    }
                    last_ctx_store.insert(*ctx, i);
                }
            }
        }
    }

    // Priority-based topological sort.
    // Compute in-degree from deps.
    let mut in_degree = vec![0u32; n];
    let mut rdeps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        in_degree[i] = d.len() as u32;
        for &dep in d {
            rdeps[dep].push(i);
        }
    }

    // Priority queue: ready instructions sorted by priority.
    // Use a BinaryHeap with Reverse for min-heap behavior.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let priorities: Vec<Priority> = block
        .body
        .iter()
        .enumerate()
        .map(|(i, inst)| classify(&inst.kind, i))
        .collect();

    let mut ready: BinaryHeap<Reverse<(Priority, usize)>> = BinaryHeap::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            ready.push(Reverse((priorities[i], i)));
        }
    }

    let mut result = Vec::with_capacity(block.header.len() + n + block.footer.len());
    result.extend(block.header);

    let mut emitted = 0;
    while let Some(Reverse((_, idx))) = ready.pop() {
        result.push(block.body[idx].clone());
        emitted += 1;

        for &successor in &rdeps[idx] {
            in_degree[successor] -= 1;
            if in_degree[successor] == 0 {
                ready.push(Reverse((priorities[successor], successor)));
            }
        }
    }

    assert_eq!(
        emitted, n,
        "reorder: cycle detected in dependency graph ({emitted} emitted, {n} total)"
    );

    result.extend(block.footer);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{Effect, EffectSet, Param};
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_body(insts: Vec<InstKind>, val_count: usize) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            let vid = factory.next();
            val_types.insert(vid, Ty::Int);
        }
        MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types,
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    fn io_fn_type(i: &Interner, name: &str) -> (QualifiedRef, Ty) {
        let qref = QualifiedRef::root(i.intern(name));
        let ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::String),
            captures: vec![],
            effect: Effect::Resolved(EffectSet {
                io: true,
                ..Default::default()
            }),
        };
        (qref, ty)
    }

    /// Collect instruction kinds from a body (ignoring span).
    fn kinds(body: &MirBody) -> Vec<&InstKind> {
        body.insts.iter().map(|i| &i.kind).collect()
    }

    /// Find the index of the first instruction matching a predicate.
    fn find_idx(body: &MirBody, pred: impl Fn(&InstKind) -> bool) -> Option<usize> {
        body.insts.iter().position(|i| pred(&i.kind))
    }

    // ── Basic: Spawn moves before Eval ──────────────────────────────

    #[test]
    fn two_independent_spawns_before_evals() {
        // Before: spawn_a, eval_a, spawn_b, eval_b, add, return
        // After:  spawn_a, spawn_b, eval_a, eval_b, add, return
        //   (spawns first, evals later)
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch_a");
        let (fb, fb_ty) = io_fn_type(&i, "fetch_b");

        let mut fn_types = FxHashMap::default();
        fn_types.insert(fa, fa_ty);
        fn_types.insert(fb, fb_ty);

        // h0 = spawn fetch_a(); r0 = eval h0;
        // h1 = spawn fetch_b(); r1 = eval h1;
        // r2 = r0 + r1; return r2
        let mut body = make_body(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(fb),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(3),
                    src: v(2),
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(4),
                    op: acvus_ast::BinOp::Add,
                    left: v(1),
                    right: v(3),
                },
                InstKind::Return(v(4)),
            ],
            5,
        );
        // Set Handle types for eval.
        body.val_types.insert(
            v(0),
            Ty::Handle(
                Box::new(Ty::String),
                Effect::Resolved(EffectSet {
                    io: true,
                    ..Default::default()
                }),
            ),
        );
        body.val_types.insert(
            v(2),
            Ty::Handle(
                Box::new(Ty::String),
                Effect::Resolved(EffectSet {
                    io: true,
                    ..Default::default()
                }),
            ),
        );

        run(&mut body, &fn_types);

        // Both spawns should come before both evals.
        let spawn_a = find_idx(&body, |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(0))).unwrap();
        let spawn_b = find_idx(&body, |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(2))).unwrap();
        let eval_a = find_idx(&body, |k| matches!(k, InstKind::Eval { src, .. } if *src == v(0))).unwrap();
        let eval_b = find_idx(&body, |k| matches!(k, InstKind::Eval { src, .. } if *src == v(2))).unwrap();

        assert!(spawn_a < eval_a, "spawn_a must come before eval_a");
        assert!(spawn_b < eval_b, "spawn_b must come before eval_b");
        assert!(spawn_a < eval_b, "spawn_a should come before eval_b (parallelism)");
        assert!(spawn_b < eval_a, "spawn_b should come before eval_a (parallelism)");
    }

    // ── Dependency: Eval must wait for its Spawn ────────────────────

    #[test]
    fn eval_after_its_spawn() {
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch");
        let mut fn_types = FxHashMap::default();
        fn_types.insert(fa, fa_ty);

        let mut body = make_body(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Return(v(1)),
            ],
            2,
        );
        body.val_types.insert(
            v(0),
            Ty::Handle(Box::new(Ty::String), Effect::Resolved(EffectSet { io: true, ..Default::default() })),
        );

        run(&mut body, &fn_types);

        let spawn_idx = find_idx(&body, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&body, |k| matches!(k, InstKind::Eval { .. })).unwrap();
        assert!(spawn_idx < eval_idx);
    }

    // ── Independent work fills Spawn-Eval gap ───────────────────────

    #[test]
    fn independent_work_between_spawn_and_eval() {
        // spawn, const, const, eval, add, return
        // const instructions should land between spawn and eval.
        let i = Interner::new();
        let (fa, fa_ty) = io_fn_type(&i, "fetch");
        let mut fn_types = FxHashMap::default();
        fn_types.insert(fa, fa_ty);

        let mut body = make_body(
            vec![
                InstKind::Const {
                    dst: v(5),
                    value: acvus_ast::Literal::Int(10),
                },
                InstKind::Const {
                    dst: v(6),
                    value: acvus_ast::Literal::Int(20),
                },
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::BinOp {
                    dst: v(7),
                    op: acvus_ast::BinOp::Add,
                    left: v(5),
                    right: v(6),
                },
                InstKind::Return(v(1)),
            ],
            8,
        );
        body.val_types.insert(
            v(0),
            Ty::Handle(Box::new(Ty::String), Effect::Resolved(EffectSet { io: true, ..Default::default() })),
        );

        run(&mut body, &fn_types);

        let spawn_idx = find_idx(&body, |k| matches!(k, InstKind::Spawn { .. })).unwrap();
        let eval_idx = find_idx(&body, |k| matches!(k, InstKind::Eval { .. })).unwrap();

        // Spawn should be early, eval should be late.
        // BinOp(v5+v6) is independent of spawn/eval, can go between.
        assert!(spawn_idx < eval_idx);
    }

    // ── Token dependency preserves order ────────────────────────────

    #[test]
    fn token_dependency_preserves_order() {
        let i = Interner::new();
        let fa = QualifiedRef::root(i.intern("write_a"));
        let fb = QualifiedRef::root(i.intern("write_b"));
        let token = TokenId::alloc();

        let mut fn_types = FxHashMap::default();
        let token_effect = Effect::Resolved(EffectSet {
            writes: std::collections::BTreeSet::from([EffectTarget::Token(token)]),
            io: true,
            ..Default::default()
        });
        for &qref in &[fa, fb] {
            fn_types.insert(
                qref,
                Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Unit),
                    captures: vec![],
                    effect: token_effect.clone(),
                },
            );
        }

        let mut body = make_body(
            vec![
                InstKind::Spawn {
                    dst: v(0),
                    callee: Callee::Direct(fa),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(1),
                    src: v(0),
                    context_defs: vec![],
                },
                InstKind::Spawn {
                    dst: v(2),
                    callee: Callee::Direct(fb),
                    args: vec![],
                    context_uses: vec![],
                },
                InstKind::Eval {
                    dst: v(3),
                    src: v(2),
                    context_defs: vec![],
                },
                InstKind::Return(v(3)),
            ],
            4,
        );
        body.val_types.insert(v(0), Ty::Handle(Box::new(Ty::Unit), token_effect.clone()));
        body.val_types.insert(v(2), Ty::Handle(Box::new(Ty::Unit), token_effect.clone()));

        run(&mut body, &fn_types);

        // With shared token: spawn_a must come before spawn_b (original order preserved).
        let spawn_a = find_idx(&body, |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(0))).unwrap();
        let spawn_b = find_idx(&body, |k| matches!(k, InstKind::Spawn { dst, .. } if *dst == v(2))).unwrap();
        let eval_a = find_idx(&body, |k| matches!(k, InstKind::Eval { src, .. } if *src == v(0))).unwrap();
        let eval_b = find_idx(&body, |k| matches!(k, InstKind::Eval { src, .. } if *src == v(2))).unwrap();

        // Token forces sequential: a before b entirely.
        assert!(spawn_a < spawn_b, "token: spawn_a before spawn_b");
        assert!(eval_a < eval_b, "token: eval_a before eval_b");
    }

    // ── No-op: no spawns, no change ─────────────────────────────────

    #[test]
    fn no_spawns_preserves_order() {
        let mut body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Const {
                    dst: v(1),
                    value: acvus_ast::Literal::Int(2),
                },
                InstKind::BinOp {
                    dst: v(2),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(1),
                },
                InstKind::Return(v(2)),
            ],
            3,
        );

        let original: Vec<_> = body.insts.iter().map(|i| std::mem::discriminant(&i.kind)).collect();
        run(&mut body, &FxHashMap::default());
        let after: Vec<_> = body.insts.iter().map(|i| std::mem::discriminant(&i.kind)).collect();

        assert_eq!(original, after, "no spawns → order unchanged");
    }

    // ── Use-def chain prevents wrong reorder ────────────────────────

    #[test]
    fn use_def_prevents_reorder() {
        // r0 = const 1; r1 = r0 + r0; return r1
        // r0 must come before r1 (use-def).
        let mut body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::BinOp {
                    dst: v(1),
                    op: acvus_ast::BinOp::Add,
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return(v(1)),
            ],
            2,
        );

        run(&mut body, &FxHashMap::default());

        let const_idx = find_idx(&body, |k| matches!(k, InstKind::Const { .. })).unwrap();
        let binop_idx = find_idx(&body, |k| matches!(k, InstKind::BinOp { .. })).unwrap();
        assert!(const_idx < binop_idx);
    }
}
