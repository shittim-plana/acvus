//! Dominator tree — Cooper, Harvey, Kennedy (2001) iterative algorithm.
//!
//! Block A **dominates** block B if every path from the entry to B passes
//! through A. The immediate dominator (idom) is the closest strict dominator.
//!
//! The entry block (index 0) dominates everything and has no dominator itself.
//!
//! # Unreachable blocks
//!
//! Blocks not reachable from the entry have `idom = UNREACHABLE`. All queries
//! (`idom`, `dominates`, `depth`) handle this explicitly — no panics.

use crate::cfg::{BlockIdx, CfgBody};
use smallvec::SmallVec;

/// Sentinel: idom not yet computed during fixpoint iteration.
const UNDEFINED: usize = usize::MAX;

/// Sentinel: block is unreachable from the entry.
/// After `build()` completes, any block still UNDEFINED is unreachable.
const UNREACHABLE: usize = usize::MAX;

/// Dominator tree computed from a CfgBody.
pub struct DomTree {
    /// `idom[b]` = immediate dominator of block `b`.
    ///
    /// - Entry block: `idom[0] == 0` (self-loop, no dominator).
    /// - Normal block: `idom[b]` = index of its immediate dominator.
    /// - Unreachable: `idom[b] == UNREACHABLE`.
    idom: Vec<usize>,
}

impl DomTree {
    /// Build the dominator tree from a CfgBody.
    pub fn build(cfg: &CfgBody) -> Self {
        let n = cfg.blocks.len();
        if n == 0 {
            return Self { idom: vec![] };
        }

        // ── Step 1: Reverse postorder ──────────────────────────────
        //
        // RPO gives a topological-ish ordering where dominators come
        // before dominated blocks. The fixpoint converges faster.

        let rpo = reverse_postorder(cfg, n);
        let mut rpo_order = vec![UNDEFINED; n];
        for (pos, &block) in rpo.iter().enumerate() {
            rpo_order[block] = pos;
        }

        // ── Step 2: Predecessors ───────────────────────────────────

        let preds = cfg.predecessors();

        // ── Step 3: Fixpoint iteration ─────────────────────────────
        //
        // Starting from "entry dominates itself, everything else undefined",
        // iterate until no idom changes. Each block's idom is the intersection
        // (nearest common dominator) of all its processed predecessors.

        let mut idom = vec![UNDEFINED; n];
        idom[rpo[0]] = rpo[0]; // Entry dominates itself.

        let mut changed = true;
        while changed {
            changed = false;

            // Process in RPO, skip entry.
            for &b in &rpo[1..] {
                let pred_list: SmallVec<[usize; 2]> = preds
                    .get(&BlockIdx(b))
                    .map(|ps| ps.iter().map(|p| p.0).collect())
                    .unwrap_or_default();

                // Find first predecessor with a computed idom.
                let Some(&first_processed) = pred_list.iter().find(|&&p| idom[p] != UNDEFINED) else {
                    continue; // All predecessors unreachable — skip.
                };

                // Intersect with remaining processed predecessors.
                let mut new_idom = first_processed;
                for &p in &pred_list {
                    if p != first_processed && idom[p] != UNDEFINED {
                        new_idom = intersect(&idom, &rpo_order, new_idom, p);
                    }
                }

                if idom[b] != new_idom {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        Self { idom }
    }

    /// Immediate dominator of `block`. None for entry or unreachable blocks.
    pub fn idom(&self, block: BlockIdx) -> Option<BlockIdx> {
        let b = block.0;
        if b >= self.idom.len() {
            return None;
        }
        let dom = self.idom[b];
        if dom == b || dom == UNREACHABLE {
            return None; // Entry (self-loop) or unreachable.
        }
        Some(BlockIdx(dom))
    }

    /// Does `a` dominate `b`? A block dominates itself.
    pub fn dominates(&self, a: BlockIdx, b: BlockIdx) -> bool {
        let mut cursor = b.0;
        loop {
            if cursor >= self.idom.len() {
                return false;
            }
            if cursor == a.0 {
                return true;
            }
            let parent = self.idom[cursor];
            if parent == UNREACHABLE {
                return false; // Hit unreachable block.
            }
            if parent == cursor {
                return false; // Reached entry without finding `a`.
            }
            cursor = parent;
        }
    }

    /// All blocks strictly dominated by `a` (excluding `a` itself).
    pub fn dominated_by(&self, a: BlockIdx) -> Vec<BlockIdx> {
        (0..self.idom.len())
            .filter(|&b| b != a.0 && self.dominates(a, BlockIdx(b)))
            .map(BlockIdx)
            .collect()
    }

    /// Depth in the dominator tree (entry = 0). Unreachable blocks return 0.
    pub fn depth(&self, block: BlockIdx) -> usize {
        let mut d = 0;
        let mut cursor = block.0;
        loop {
            if cursor >= self.idom.len() {
                break;
            }
            let parent = self.idom[cursor];
            if parent == cursor || parent == UNREACHABLE {
                break; // Entry or unreachable.
            }
            cursor = parent;
            d += 1;
        }
        d
    }
}

// ── Internal helpers ───────────────────────────────────────────────

/// Nearest common dominator of `a` and `b` (the "intersect" function from CHK01).
///
/// Walks both fingers up the dominator tree using RPO numbers until they meet.
/// Invariant: `idom[a]` and `idom[b]` are both != UNDEFINED.
fn intersect(idom: &[usize], rpo_order: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_order[a] > rpo_order[b] {
            a = idom[a];
        }
        while rpo_order[b] > rpo_order[a] {
            b = idom[b];
        }
    }
    a
}

/// Compute reverse postorder via iterative DFS from block 0.
fn reverse_postorder(cfg: &CfgBody, n: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);

    // Stack entries: (block_index, next_successor_to_visit).
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;

    while let Some(&mut (block, ref mut next_succ)) = stack.last_mut() {
        let succs = cfg.successors(BlockIdx(block));
        if *next_succ < succs.len() {
            let succ = succs[*next_succ].0;
            *next_succ += 1;
            if !visited[succ] {
                visited[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block);
            stack.pop();
        }
    }

    postorder.reverse();
    postorder
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        promote(MirBody {
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
        })
    }

    #[test]
    fn linear_chain() {
        let cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Int(0) },
            InstKind::Jump { label: Label(0), args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Const { dst: v(1), value: acvus_ast::Literal::Int(1) },
            InstKind::Jump { label: Label(1), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Return(v(1)),
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(0), BlockIdx(0)));
        assert!(dom.dominates(BlockIdx(0), BlockIdx(1)));
        assert!(dom.dominates(BlockIdx(0), BlockIdx(2)));
        assert!(dom.dominates(BlockIdx(1), BlockIdx(2)));
        assert!(!dom.dominates(BlockIdx(1), BlockIdx(0)));

        assert_eq!(dom.idom(BlockIdx(0)), None);
        assert_eq!(dom.idom(BlockIdx(1)), Some(BlockIdx(0)));
        assert_eq!(dom.idom(BlockIdx(2)), Some(BlockIdx(1)));
    }

    #[test]
    fn diamond() {
        let cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0), then_args: vec![],
                else_label: Label(1), else_args: vec![],
            },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Jump { label: Label(2), args: vec![] },
            InstKind::BlockLabel { label: Label(2), params: vec![], merge_of: None },
            InstKind::Const { dst: v(1), value: acvus_ast::Literal::Int(0) },
            InstKind::Return(v(1)),
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(0), BlockIdx(3)));
        assert!(!dom.dominates(BlockIdx(1), BlockIdx(3)));
        assert!(!dom.dominates(BlockIdx(2), BlockIdx(3)));
        assert_eq!(dom.idom(BlockIdx(3)), Some(BlockIdx(0)));
    }

    #[test]
    fn simple_loop() {
        let cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Int(0) },
            InstKind::Jump { label: Label(0), args: vec![v(0)] },
            InstKind::BlockLabel { label: Label(0), params: vec![v(1)], merge_of: None },
            InstKind::Const { dst: v(2), value: acvus_ast::Literal::Bool(true) },
            InstKind::JumpIf {
                cond: v(2),
                then_label: Label(0), then_args: vec![v(1)],
                else_label: Label(1), else_args: vec![],
            },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Return(v(1)),
        ]);
        let dom = DomTree::build(&cfg);

        assert!(dom.dominates(BlockIdx(1), BlockIdx(2)));
        assert_eq!(dom.idom(BlockIdx(1)), Some(BlockIdx(0)));
        assert_eq!(dom.idom(BlockIdx(2)), Some(BlockIdx(1)));
    }

    #[test]
    fn depth_linear() {
        let cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Int(0) },
            InstKind::Jump { label: Label(0), args: vec![] },
            InstKind::BlockLabel { label: Label(0), params: vec![], merge_of: None },
            InstKind::Const { dst: v(1), value: acvus_ast::Literal::Int(1) },
            InstKind::Jump { label: Label(1), args: vec![] },
            InstKind::BlockLabel { label: Label(1), params: vec![], merge_of: None },
            InstKind::Return(v(1)),
        ]);
        let dom = DomTree::build(&cfg);

        assert_eq!(dom.depth(BlockIdx(0)), 0);
        assert_eq!(dom.depth(BlockIdx(1)), 1);
        assert_eq!(dom.depth(BlockIdx(2)), 2);
    }

    #[test]
    fn single_block() {
        let cfg = make_cfg(vec![
            InstKind::Const { dst: v(0), value: acvus_ast::Literal::Int(0) },
            InstKind::Return(v(0)),
        ]);
        let dom = DomTree::build(&cfg);

        assert_eq!(dom.idom(BlockIdx(0)), None);
        assert!(dom.dominates(BlockIdx(0), BlockIdx(0)));
    }

    #[test]
    fn empty_cfg() {
        let cfg = make_cfg(vec![]);
        let dom = DomTree::build(&cfg);
        // promote creates one empty block even for empty input.
        assert_eq!(dom.idom.len(), 1);
    }
}
