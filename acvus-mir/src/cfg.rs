//! CFG — basic-block-based IR representation.
//!
//! The single CFG representation used by all analysis and optimization passes.
//! MirBody (flat Vec<Inst>) is promoted to CfgBody where each basic block owns
//! its instructions, eliminating index arithmetic entirely.
//!
//! Lifecycle: MirBody → promote → CfgBody → (passes) → demote → MirBody

use acvus_utils::LocalFactory;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::ir::{DebugInfo, Inst, InstKind, Label, MirBody, ValueId};
use crate::ty::Ty;

// ── BlockIdx ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockIdx(pub usize);

// ── Block ─────────────────────────────────────────────────────────

/// A basic block that owns its instructions.
#[derive(Debug, Clone)]
pub struct Block {
    /// Block label (None for entry block).
    pub label: Option<Label>,
    /// SSA block parameters (PHI equivalent).
    pub params: Vec<ValueId>,
    /// Body instructions (no control flow).
    pub insts: Vec<Inst>,
    /// Block terminator (control flow).
    pub terminator: Terminator,
    /// If this block is a merge point, which label it merges.
    pub merge_of: Option<Label>,
}

// ── Terminator ────────────────────────────────────────────────────

/// Block terminator — extracted from InstKind control flow variants.
#[derive(Debug, Clone)]
pub enum Terminator {
    Jump {
        label: Label,
        args: Vec<ValueId>,
    },
    JumpIf {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
    },
    ListStep {
        dst: ValueId,
        list: ValueId,
        index_src: ValueId,
        index_dst: ValueId,
        done: Label,
        done_args: Vec<ValueId>,
    },
    Return(ValueId),
    /// Implicit fallthrough to next block.
    Fallthrough,
}

// ── CfgBody ───────────────────────────────────────────────────────

/// Basic-block-based IR body for analysis and optimization passes.
#[derive(Debug, Clone)]
pub struct CfgBody {
    pub blocks: Vec<Block>,
    pub label_to_block: FxHashMap<Label, BlockIdx>,
    pub val_types: FxHashMap<ValueId, Ty>,
    pub param_regs: Vec<ValueId>,
    pub capture_regs: Vec<ValueId>,
    pub debug: DebugInfo,
    pub val_factory: LocalFactory<ValueId>,
}

impl CfgBody {
    /// Successors of a block.
    pub fn successors(&self, idx: BlockIdx) -> SmallVec<[BlockIdx; 2]> {
        let block = &self.blocks[idx.0];
        let mut succs = SmallVec::new();
        match &block.terminator {
            Terminator::Jump { label, .. } => {
                if let Some(&bi) = self.label_to_block.get(label) {
                    succs.push(bi);
                }
            }
            Terminator::JumpIf {
                then_label,
                else_label,
                ..
            } => {
                if let Some(&bi) = self.label_to_block.get(then_label) {
                    succs.push(bi);
                }
                if let Some(&bi) = self.label_to_block.get(else_label) {
                    succs.push(bi);
                }
            }
            Terminator::ListStep { done, .. } => {
                // Fallthrough (element available) + done branch (exhausted).
                let next = idx.0 + 1;
                if next < self.blocks.len() {
                    succs.push(BlockIdx(next));
                }
                if let Some(&bi) = self.label_to_block.get(done) {
                    succs.push(bi);
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < self.blocks.len() {
                    succs.push(BlockIdx(next));
                }
            }
            Terminator::Return(_) => {}
        }
        succs
    }

    /// Build a predecessors map from the CFG.
    pub fn predecessors(&self) -> FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>> {
        let mut preds: FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>> = FxHashMap::default();
        for i in 0..self.blocks.len() {
            let idx = BlockIdx(i);
            for succ in self.successors(idx) {
                preds.entry(succ).or_default().push(idx);
            }
        }
        preds
    }
}

// ── Promote: MirBody → CfgBody ───────────────────────────────────

pub fn promote(body: MirBody) -> CfgBody {
    let mut blocks: Vec<Block> = Vec::new();
    let mut current_label: Option<Label> = None;
    let mut current_params: Vec<ValueId> = Vec::new();
    let mut current_insts: Vec<Inst> = Vec::new();
    let mut current_merge: Option<Label> = None;

    let mut label_to_block: FxHashMap<Label, BlockIdx> = FxHashMap::default();

    for inst in body.insts {
        match &inst.kind {
            InstKind::BlockLabel {
                label,
                params,
                merge_of,
            } => {
                // Flush previous block.
                let terminator = extract_terminator(&mut current_insts);
                blocks.push(Block {
                    label: current_label,
                    params: current_params,
                    insts: std::mem::take(&mut current_insts),
                    terminator,
                    merge_of: current_merge,
                });

                // Start new block.
                label_to_block.insert(*label, BlockIdx(blocks.len()));
                current_label = Some(*label);
                current_params = params.clone();
                current_merge = *merge_of;
            }
            _ => {
                current_insts.push(inst);
            }
        }
    }

    // Flush last block.
    let terminator = extract_terminator(&mut current_insts);
    blocks.push(Block {
        label: current_label,
        params: current_params,
        insts: current_insts,
        terminator,
        merge_of: current_merge,
    });

    CfgBody {
        blocks,
        label_to_block,
        val_types: body.val_types,
        param_regs: body.param_regs,
        capture_regs: body.capture_regs,
        debug: body.debug,
        val_factory: body.val_factory,
    }
}

/// Extract the last control-flow instruction as a Terminator.
fn extract_terminator(insts: &mut Vec<Inst>) -> Terminator {
    if let Some(last) = insts.last() {
        match &last.kind {
            InstKind::Jump { label, args } => {
                let term = Terminator::Jump {
                    label: *label,
                    args: args.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let term = Terminator::JumpIf {
                    cond: *cond,
                    then_label: *then_label,
                    then_args: then_args.clone(),
                    else_label: *else_label,
                    else_args: else_args.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::Return(val) => {
                let term = Terminator::Return(*val);
                insts.pop();
                return term;
            }
            InstKind::ListStep {
                dst,
                list,
                index_src,
                index_dst,
                done,
                done_args,
            } => {
                let term = Terminator::ListStep {
                    dst: *dst,
                    list: *list,
                    index_src: *index_src,
                    index_dst: *index_dst,
                    done: *done,
                    done_args: done_args.clone(),
                };
                insts.pop();
                return term;
            }
            _ => {}
        }
    }
    Terminator::Fallthrough
}

// ── Demote: CfgBody → MirBody ────────────────────────────────────

pub fn demote(cfg: CfgBody) -> MirBody {
    let mut insts: Vec<Inst> = Vec::new();

    for block in cfg.blocks {
        // Emit BlockLabel for non-entry blocks.
        if let Some(label) = block.label {
            insts.push(Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::BlockLabel {
                    label,
                    params: block.params,
                    merge_of: block.merge_of,
                },
            });
        }

        // Emit body instructions.
        insts.extend(block.insts);

        // Emit terminator as instruction.
        match block.terminator {
            Terminator::Jump { label, args } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Jump { label, args },
                });
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::JumpIf {
                        cond,
                        then_label,
                        then_args,
                        else_label,
                        else_args,
                    },
                });
            }
            Terminator::Return(val) => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Return(val),
                });
            }
            Terminator::ListStep {
                dst,
                list,
                index_src,
                index_dst,
                done,
                done_args,
            } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::ListStep {
                        dst,
                        list,
                        index_src,
                        index_dst,
                        done,
                        done_args,
                    },
                });
            }
            Terminator::Fallthrough => {
                // No instruction — implicit fallthrough.
            }
        }
    }

    // Filter out Nops.
    insts.retain(|inst| !matches!(inst.kind, InstKind::Nop));

    MirBody {
        insts,
        val_types: cfg.val_types,
        param_regs: cfg.param_regs,
        capture_regs: cfg.capture_regs,
        debug: cfg.debug,
        val_factory: cfg.val_factory,
        label_count: 0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::LocalIdOps;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_body(insts: Vec<InstKind>) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            param_regs: Vec::new(),
            capture_regs: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    #[test]
    fn roundtrip_simple() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(42),
            },
            InstKind::Return(v(0)),
        ]);

        let original_len = body.insts.len();
        let cfg = promote(body);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].insts.len(), 1); // Const
        assert!(matches!(cfg.blocks[0].terminator, Terminator::Return(_)));

        let body = demote(cfg);
        assert_eq!(body.insts.len(), original_len);
    }

    #[test]
    fn roundtrip_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![v(1)],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            },
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![v(2)],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![v(3)],
                merge_of: None,
            },
            InstKind::Return(v(3)),
        ]);

        let cfg = promote(body);
        assert_eq!(cfg.blocks.len(), 4);
        assert!(matches!(
            cfg.blocks[0].terminator,
            Terminator::JumpIf { .. }
        ));
        assert!(matches!(cfg.blocks[1].terminator, Terminator::Jump { .. }));
        assert!(matches!(cfg.blocks[2].terminator, Terminator::Jump { .. }));
        assert!(matches!(cfg.blocks[3].terminator, Terminator::Return(_)));

        let body = demote(cfg);
        assert_eq!(body.insts.len(), 10);
    }

    #[test]
    fn label_to_block_mapping() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::Return(v(0)),
        ]);

        let cfg = promote(body);
        assert_eq!(cfg.label_to_block[&Label(0)], BlockIdx(1));
    }

    #[test]
    fn nop_filtered_on_demote() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Nop,
            InstKind::Return(v(0)),
        ]);

        let cfg = promote(body);
        let body = demote(cfg);
        assert!(!body.insts.iter().any(|i| matches!(i.kind, InstKind::Nop)));
    }

    #[test]
    fn successors_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            },
            InstKind::Return(v(0)),
        ]);

        let cfg = promote(body);
        // B0 → B1, B2
        let b0_succs = cfg.successors(BlockIdx(0));
        assert_eq!(b0_succs.len(), 2);
        // B1 → B3, B2 → B3
        let b1_succs = cfg.successors(BlockIdx(1));
        assert_eq!(b1_succs.len(), 1);
        assert_eq!(b1_succs[0], BlockIdx(3));
        // B3 has no successors (Return)
        let b3_succs = cfg.successors(BlockIdx(3));
        assert!(b3_succs.is_empty());
    }

    #[test]
    fn predecessors_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
                merge_of: None,
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
                merge_of: None,
            },
            InstKind::Return(v(0)),
        ]);

        let cfg = promote(body);
        let preds = cfg.predecessors();
        // B3 (merge) has two predecessors: B1, B2
        let b3_preds = preds.get(&BlockIdx(3)).unwrap();
        assert_eq!(b3_preds.len(), 2);
    }
}
