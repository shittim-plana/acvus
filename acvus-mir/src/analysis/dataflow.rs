use std::collections::VecDeque;

use crate::ir::{Inst, ValueId};
use rustc_hash::FxHashMap;

use crate::analysis::cfg::{BlockIdx, Cfg, Terminator};
use crate::analysis::domain::SemiLattice;

/// Trait for domains that can produce definite boolean results.
/// Used by the dataflow engine to determine branch pruning.
pub trait BooleanDomain {
    fn as_definite_bool(&self) -> Option<bool>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataflowState<D: SemiLattice> {
    pub values: FxHashMap<ValueId, D>,
}

impl<D: SemiLattice> Default for DataflowState<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: SemiLattice> DataflowState<D> {
    pub fn new() -> Self {
        DataflowState {
            values: FxHashMap::default(),
        }
    }

    pub fn get(&self, val: ValueId) -> D {
        self.values.get(&val).cloned().unwrap_or_else(D::bottom)
    }

    pub fn set(&mut self, val: ValueId, domain: D) {
        self.values.insert(val, domain);
    }

    /// Join another state into this one. Returns true if anything changed.
    pub fn join_from(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (val, domain) in &other.values {
            let entry = self.values.entry(*val).or_insert_with(D::bottom);
            if entry.join_mut(domain) {
                changed = true;
            }
        }
        changed
    }
}

pub trait TransferFunction<D: SemiLattice> {
    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<D>);
}

pub struct DataflowResult<D: SemiLattice> {
    pub block_entry: Vec<DataflowState<D>>,
    pub block_exit: Vec<DataflowState<D>>,
    pub visited: Vec<bool>,
}

pub fn forward_analysis<D, T>(
    cfg: &Cfg,
    insts: &[Inst],
    transfer: &T,
    initial: DataflowState<D>,
) -> DataflowResult<D>
where
    D: SemiLattice + BooleanDomain,
    T: TransferFunction<D>,
{
    let n = cfg.blocks.len();
    let mut block_entry: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut block_exit: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut visited = vec![false; n];

    if n == 0 {
        return DataflowResult {
            block_entry,
            block_exit,
            visited,
        };
    }

    block_entry[0] = initial;
    let mut worklist = VecDeque::new();
    worklist.push_back(BlockIdx(0));

    while let Some(idx) = worklist.pop_front() {
        visited[idx.0] = true;
        let block = &cfg.blocks[idx.0];

        let mut state = block_entry[idx.0].clone();

        for &inst_idx in &block.inst_indices {
            transfer.transfer_inst(&insts[inst_idx], &mut state);
        }

        block_exit[idx.0] = state;

        match &block.terminator {
            Terminator::Jump { target, args } => {
                if let Some(&target_idx) = cfg.label_to_block.get(target) {
                    let changed = propagate_to_successor(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        args,
                        &mut block_entry[target_idx.0],
                    );
                    if changed {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let cond_val = block_exit[idx.0].get(*cond);
                let definite = cond_val.as_definite_bool();

                if definite != Some(false)
                    && let Some(&target_idx) = cfg.label_to_block.get(then_label)
                {
                    let changed = propagate_to_successor(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        then_args,
                        &mut block_entry[target_idx.0],
                    );
                    if changed {
                        worklist.push_back(target_idx);
                    }
                }

                if definite != Some(true)
                    && let Some(&target_idx) = cfg.label_to_block.get(else_label)
                {
                    let changed = propagate_to_successor(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        else_args,
                        &mut block_entry[target_idx.0],
                    );
                    if changed {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::ListStep { done, done_args } => {
                // Fallthrough (element available).
                let next = idx.0 + 1;
                if next < n {
                    let changed = block_entry[next].join_from(&block_exit[idx.0]);
                    if changed {
                        worklist.push_back(BlockIdx(next));
                    }
                }
                // Done branch (exhausted).
                if let Some(&target_idx) = cfg.label_to_block.get(done) {
                    let changed = propagate_to_successor(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        done_args,
                        &mut block_entry[target_idx.0],
                    );
                    if changed {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n {
                    let changed = block_entry[next].join_from(&block_exit[idx.0]);
                    if changed {
                        worklist.push_back(BlockIdx(next));
                    }
                }
            }
            Terminator::Return(_) => {}
        }
    }

    DataflowResult {
        block_entry,
        block_exit,
        visited,
    }
}

/// Backward dataflow analysis.
///
/// Mirror of `forward_analysis`: starts from exit blocks (Return), propagates
/// backward through predecessors. Transfer function is applied in reverse
/// instruction order within each block.
///
/// `block_exit` = state at the end of a block (before transfer, after successors).
/// `block_entry` = state at the start of a block (after backward transfer).
pub fn backward_analysis<D, T>(
    cfg: &Cfg,
    insts: &[Inst],
    transfer: &T,
) -> DataflowResult<D>
where
    D: SemiLattice,
    T: TransferFunction<D>,
{
    let n = cfg.blocks.len();
    let mut block_entry: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut block_exit: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut visited = vec![false; n];

    if n == 0 {
        return DataflowResult {
            block_entry,
            block_exit,
            visited,
        };
    }

    let preds = cfg.predecessors();

    // Seed worklist with all blocks (backward analysis starts from exits).
    let mut worklist = VecDeque::new();
    for i in (0..n).rev() {
        worklist.push_back(BlockIdx(i));
    }

    while let Some(idx) = worklist.pop_front() {
        visited[idx.0] = true;
        let block = &cfg.blocks[idx.0];

        // block_exit = join of all successor block_entry states.
        // For each successor, map the successor's block params back to
        // the terminator's args (reverse of forward propagation).
        let mut exit_state = DataflowState::new();
        match &block.terminator {
            Terminator::Jump { target, args } => {
                if let Some(&target_idx) = cfg.label_to_block.get(target) {
                    propagate_from_successor(
                        &block_entry[target_idx.0],
                        &cfg.blocks[target_idx.0].params,
                        args,
                        &mut exit_state,
                    );
                }
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                // cond is used by this terminator.
                exit_state.set(*cond, D::top());

                if let Some(&target_idx) = cfg.label_to_block.get(then_label) {
                    propagate_from_successor(
                        &block_entry[target_idx.0],
                        &cfg.blocks[target_idx.0].params,
                        then_args,
                        &mut exit_state,
                    );
                }
                if let Some(&target_idx) = cfg.label_to_block.get(else_label) {
                    propagate_from_successor(
                        &block_entry[target_idx.0],
                        &cfg.blocks[target_idx.0].params,
                        else_args,
                        &mut exit_state,
                    );
                }
            }
            Terminator::ListStep { done, done_args } => {
                // Fallthrough successor.
                let next = idx.0 + 1;
                if next < n {
                    exit_state.join_from(&block_entry[next]);
                }
                // Done branch successor.
                if let Some(&target_idx) = cfg.label_to_block.get(done) {
                    propagate_from_successor(
                        &block_entry[target_idx.0],
                        &cfg.blocks[target_idx.0].params,
                        done_args,
                        &mut exit_state,
                    );
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n {
                    exit_state.join_from(&block_entry[next]);
                }
            }
            Terminator::Return(val) => {
                // Return's value is live at the exit of this block.
                exit_state.set(*val, D::top());
            }
        }

        block_exit[idx.0] = exit_state;

        // Apply transfer in reverse instruction order.
        let mut state = block_exit[idx.0].clone();
        for &inst_idx in block.inst_indices.iter().rev() {
            transfer.transfer_inst(&insts[inst_idx], &mut state);
        }

        // Check if block_entry changed.
        let old = &block_entry[idx.0];
        if state != *old {
            block_entry[idx.0] = state;
            // Enqueue predecessors.
            if let Some(pred_list) = preds.get(&idx) {
                for &pred in pred_list {
                    worklist.push_back(pred);
                }
            }
        }
    }

    DataflowResult {
        block_entry,
        block_exit,
        visited,
    }
}

/// Backward propagation: map successor's live params back to this block's terminator args.
fn propagate_from_successor<D: SemiLattice>(
    succ_entry: &DataflowState<D>,
    succ_params: &[ValueId],
    term_args: &[ValueId],
    exit_state: &mut DataflowState<D>,
) {
    // If a successor's param is live at its entry, the corresponding
    // terminator arg is live at this block's exit.
    for (param, arg) in succ_params.iter().zip(term_args.iter()) {
        let param_val = succ_entry.get(*param);
        if param_val != D::bottom() {
            let entry = exit_state.values.entry(*arg).or_insert_with(D::bottom);
            entry.join_mut(&param_val);
        }
    }

    // Also propagate any values live at successor entry that aren't params
    // (they flow through unchanged).
    let param_set: rustc_hash::FxHashSet<ValueId> = succ_params.iter().copied().collect();
    for (val, domain) in &succ_entry.values {
        if !param_set.contains(val) {
            let entry = exit_state.values.entry(*val).or_insert_with(D::bottom);
            entry.join_mut(domain);
        }
    }
}

fn propagate_to_successor<D: SemiLattice>(
    source_exit: &DataflowState<D>,
    target_params: &[ValueId],
    args: &[ValueId],
    target_entry: &mut DataflowState<D>,
) -> bool {
    let mut changed = false;

    for (param, arg) in target_params.iter().zip(args.iter()) {
        let arg_val = source_exit.get(*arg);
        let entry = target_entry.values.entry(*param).or_insert_with(D::bottom);
        if entry.join_mut(&arg_val) {
            changed = true;
        }
    }

    if target_entry.join_from(source_exit) {
        changed = true;
    }

    changed
}
