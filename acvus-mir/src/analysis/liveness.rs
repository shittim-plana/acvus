//! Liveness analysis — backward dataflow over the CFG.
//!
//! Computes which ValueIds are live at each program point.
//! A ValueId is live at a point if there exists a path from that point
//! to a use of the ValueId without an intervening redefinition.
//!
//! Result: `LivenessResult` provides `is_live(block, inst_offset, val)`.

use rustc_hash::FxHashSet;

use crate::analysis::cfg::{BlockIdx, Cfg};
use crate::analysis::dataflow::{backward_analysis, DataflowState, TransferFunction};
use crate::analysis::domain::SemiLattice;
use crate::analysis::inst_info;
use crate::ir::{Inst, MirBody, ValueId};

// ── Domain ──────────────────────────────────────────────────────────

/// Liveness domain: a ValueId is either Live or Dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    Dead,
    Live,
}

impl SemiLattice for Liveness {
    fn bottom() -> Self {
        Liveness::Dead
    }

    fn top() -> Self {
        Liveness::Live
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        match (*self, *other) {
            (Liveness::Live, _) => false,
            (Liveness::Dead, Liveness::Live) => {
                *self = Liveness::Live;
                true
            }
            (Liveness::Dead, Liveness::Dead) => false,
        }
    }
}

// ── Transfer function ───────────────────────────────────────────────

/// Backward transfer: kill defs, gen uses.
struct LivenessTransfer;

impl TransferFunction<Liveness> for LivenessTransfer {
    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<Liveness>) {
        // Kill: defs make the value dead (before this point).
        for d in inst_info::defs(&inst.kind) {
            state.values.remove(&d);
        }
        // Gen: uses make the value live (before this point).
        for u in inst_info::uses(&inst.kind) {
            state.set(u, Liveness::Live);
        }
    }
}

// ── Result ──────────────────────────────────────────────────────────

/// Per-block liveness: live-in set (values live at block entry).
pub struct LivenessResult {
    /// live_in[block_idx] = set of ValueIds live at the start of the block.
    pub live_in: Vec<FxHashSet<ValueId>>,
    /// live_out[block_idx] = set of ValueIds live at the end of the block.
    pub live_out: Vec<FxHashSet<ValueId>>,
}

impl LivenessResult {
    /// Is `val` live at the entry of block `block`?
    pub fn is_live_in(&self, block: BlockIdx, val: ValueId) -> bool {
        self.live_in
            .get(block.0)
            .map_or(false, |set| set.contains(&val))
    }

    /// Is `val` live at the exit of block `block`?
    pub fn is_live_out(&self, block: BlockIdx, val: ValueId) -> bool {
        self.live_out
            .get(block.0)
            .map_or(false, |set| set.contains(&val))
    }

    /// Compute the full liveness interval for each ValueId:
    /// (first_def_position, last_use_position) as flat instruction indices.
    ///
    /// This is what reg_color needs for linear scan allocation.
    pub fn intervals(&self, cfg: &Cfg, insts: &[Inst]) -> Vec<(ValueId, usize, usize)> {
        let mut def_pos: rustc_hash::FxHashMap<ValueId, usize> = rustc_hash::FxHashMap::default();
        let mut last_use: rustc_hash::FxHashMap<ValueId, usize> = rustc_hash::FxHashMap::default();

        for (bi, block) in cfg.blocks.iter().enumerate() {
            // Use the block's true start position (BlockLabel instruction index).
            // This is critical for blocks with empty inst_indices — without it,
            // block params and live-in/live-out values get position 0, causing
            // incorrect interval overlap and broken reg_color slot assignments.
            let block_start = block.start_pos;

            // Values live-in to this block: extend their interval to the block start.
            for &val in &self.live_in[bi] {
                // If val is live-in but not yet defined, its def is before this block.
                def_pos.entry(val).or_insert(0);
                // Extend last_use to at least the block start.
                let lu = last_use.entry(val).or_insert(block_start);
                *lu = (*lu).max(block_start);
            }

            // Block params are defs at the block start.
            // BlockLabel instructions are NOT in inst_indices (they're block headers),
            // so params must be registered separately to ensure reg_color can remap them.
            for &param in &block.params {
                def_pos.entry(param).or_insert(block_start);
            }

            // Values live-out of this block: extend their interval past the block end.
            // Terminators (Return, Jump, JumpIf) use values but are not in inst_indices,
            // so we extend to block_end + 1 to account for terminator uses.
            let block_end = block.inst_indices.last().copied().unwrap_or(block_start);
            let terminator_pos = block_end + 1;
            for &val in &self.live_out[bi] {
                let lu = last_use.entry(val).or_insert(terminator_pos);
                *lu = (*lu).max(terminator_pos);
            }

            // Defs and uses within the block.
            for &inst_idx in &block.inst_indices {
                for d in inst_info::defs(&insts[inst_idx].kind) {
                    def_pos.entry(d).or_insert(inst_idx);
                }
                for u in inst_info::uses(&insts[inst_idx].kind) {
                    let lu = last_use.entry(u).or_insert(inst_idx);
                    *lu = (*lu).max(inst_idx);
                }
            }
        }

        let mut intervals: Vec<(ValueId, usize, usize)> = Vec::new();
        for (&vid, &def) in &def_pos {
            let end = last_use.get(&vid).copied().unwrap_or(def);
            intervals.push((vid, def, end));
        }
        intervals.sort_by_key(|&(_, def, _)| def);
        intervals
    }
}

/// Run liveness analysis on a MirBody.
pub fn analyze(body: &MirBody) -> LivenessResult {
    let cfg = Cfg::build(&body.insts);

    if cfg.blocks.is_empty() {
        return LivenessResult {
            live_in: vec![],
            live_out: vec![],
        };
    }

    let result = backward_analysis(&cfg, &body.insts, &LivenessTransfer);

    let live_in = result
        .block_entry
        .iter()
        .map(|state| extract_live_set(state))
        .collect();
    let live_out = result
        .block_exit
        .iter()
        .map(|state| extract_live_set(state))
        .collect();

    LivenessResult { live_in, live_out }
}

/// Extract the set of live ValueIds from a DataflowState.
fn extract_live_set(state: &DataflowState<Liveness>) -> FxHashSet<ValueId> {
    state
        .values
        .iter()
        .filter_map(|(&vid, &liveness)| match liveness {
            Liveness::Live => Some(vid),
            Liveness::Dead => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::cfg::Cfg;
    use crate::ir::*;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_body(insts: Vec<InstKind>) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        // Allocate enough ValueIds.
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

    // ── Single block ────────────────────────────────────────────────

    #[test]
    fn simple_linear() {
        // r0 = const 1; r1 = const 2; r2 = r0 + r1; return r2
        let body = make_body(vec![
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
        ]);

        let result = analyze(&body);
        let cfg = Cfg::build(&body.insts);
        let intervals = result.intervals(&cfg, &body.insts);

        // r0: def at 0, last use at 2
        let r0 = intervals.iter().find(|(vid, _, _)| *vid == v(0)).unwrap();
        assert_eq!(r0.1, 0); // def
        assert_eq!(r0.2, 2); // last use

        // r1: def at 1, last use at 2
        let r1 = intervals.iter().find(|(vid, _, _)| *vid == v(1)).unwrap();
        assert_eq!(r1.1, 1);
        assert_eq!(r1.2, 2);

        // r2: def at 2, last use at 3 (Return)
        let r2 = intervals.iter().find(|(vid, _, _)| *vid == v(2)).unwrap();
        assert_eq!(r2.1, 2);
        assert_eq!(r2.2, 3);
    }

    #[test]
    fn dead_value_not_live() {
        // r0 = const 1; r1 = const 2; return r1
        // r0 is dead — never used.
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::Return(v(1)),
        ]);

        let result = analyze(&body);
        let cfg = Cfg::build(&body.insts);

        // r0 should not be live at any block exit.
        assert!(!result.is_live_out(BlockIdx(0), v(0)));
    }

    // ── Multi-block: branch ─────────────────────────────────────────

    #[test]
    fn branch_both_arms_use_value() {
        // r0 = const 1; r1 = const true
        // jump_if r1 then L0(r0) else L1(r0)
        // L0(r2): return r2
        // L1(r3): return r3
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(1),
                then_label: Label(0),
                then_args: vec![v(0)],
                else_label: Label(1),
                else_args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(2)],
                merge_of: None,
            },
            InstKind::Return(v(2)),
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(3)],
                merge_of: None,
            },
            InstKind::Return(v(3)),
        ]);

        let result = analyze(&body);

        // r0 should be live-out of block 0 (used in both branches via jump args).
        assert!(result.is_live_out(BlockIdx(0), v(0)));

        // r1 (cond) should be live-out of block 0 (used by JumpIf terminator).
        assert!(result.is_live_out(BlockIdx(0), v(1)));
    }

    #[test]
    fn value_live_across_blocks() {
        // Block 0: r0 = const 1; jump L0()
        // Block 1 (L0): r1 = r0 + r0; return r1
        // r0 must be live across the block boundary.
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
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
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Return(v(1)),
        ]);

        let result = analyze(&body);

        // r0 is defined in block 0, used in block 1.
        // Should be live-out of block 0.
        assert!(result.is_live_out(BlockIdx(0), v(0)));

        // Should be live-in of block 1.
        assert!(result.is_live_in(BlockIdx(1), v(0)));
    }

    // ── Loop ────────────────────────────────────────────────────────

    #[test]
    fn loop_keeps_value_live() {
        // Block 0: r0 = const 0; jump L0(r0)
        // Block 1 (L0, params=[r1]):
        //   r2 = r1 + 1
        //   r3 = r2 < 10
        //   jump_if r3 then L0(r2) else L1(r2)
        // Block 2 (L1, params=[r4]): return r4
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![v(0)],
            },
            // Loop header
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(1)],
                merge_of: None,
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(1),
            },
            InstKind::BinOp {
                dst: v(3),
                op: acvus_ast::BinOp::Lt,
                left: v(2),
                right: v(5), // pretend v(5) is 10
            },
            InstKind::JumpIf {
                cond: v(3),
                then_label: Label(0),
                then_args: vec![v(2)],
                else_label: Label(1),
                else_args: vec![v(2)],
            },
            // Exit
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(4)],
                merge_of: None,
            },
            InstKind::Return(v(4)),
        ]);

        let result = analyze(&body);

        // r1 (loop param) should be live-in to the loop header block.
        assert!(result.is_live_in(BlockIdx(1), v(1)));

        // r2 should be live-out of the loop body (used in jump args to both targets).
        assert!(result.is_live_out(BlockIdx(1), v(2)));
    }

    // ── Intervals ───────────────────────────────────────────────────

    #[test]
    fn intervals_non_overlapping_allows_reuse() {
        // r0 = const 1; r1 = r0 + r0; r2 = const 2; r3 = r2 + r2; r4 = r1 + r3; return r4
        // r0 is dead after inst 1. r2 can reuse r0's slot.
        let body = make_body(vec![
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
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(3),
                op: acvus_ast::BinOp::Add,
                left: v(2),
                right: v(2),
            },
            InstKind::BinOp {
                dst: v(4),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(3),
            },
            InstKind::Return(v(4)),
        ]);

        let result = analyze(&body);
        let cfg = Cfg::build(&body.insts);
        let intervals = result.intervals(&cfg, &body.insts);

        // r0: [0, 1], r2: [2, 3] — non-overlapping, can share a slot.
        let r0 = intervals.iter().find(|(vid, _, _)| *vid == v(0)).unwrap();
        let r2 = intervals.iter().find(|(vid, _, _)| *vid == v(2)).unwrap();
        assert!(r0.2 < r2.1, "r0 should end before r2 starts — slots reusable");
    }
}
