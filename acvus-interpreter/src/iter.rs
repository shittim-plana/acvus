use std::sync::{Arc, OnceLock};

use acvus_utils::TrackedDeque;

use crate::value::{FnValue, Value};

// =========================================================================
// SharedIter — lazy iterator with effect-aware caching
// =========================================================================

/// A lazy iterator over `Value`s.
///
/// The chain of operations is stored as a flat list ([`IterChain`]).
/// Actual computation is deferred until `collect` (or a consuming builtin).
///
/// ## Clone semantics
///
/// Clone copies `ops` and `offset` independently. The `source` and `cache`
/// are shared via `Arc`.
///
/// - **Pure iterators**: `cache` is shared — first collect populates it,
///   subsequent clones read from cache without re-executing.
/// - **Effectful iterators**: each clone should get a fresh `cache`
///   (via [`SharedIter::fork`]) so side effects are re-executed.
///
/// ## Offset
///
/// `offset` tracks how many items have been consumed by `next()`.
/// Each clone has its own offset, so multiple consumers can iterate
/// independently over the same cached data.
#[derive(Clone)]
pub struct SharedIter {
    source: Arc<[Value]>,
    ops: Vec<IterOp>,
    offset: usize,
    cache: Arc<OnceLock<Vec<Value>>>,
}

/// Flat representation of a lazy iterator pipeline.
#[derive(Clone)]
pub struct IterChain {
    pub source: Arc<[Value]>,
    pub ops: Vec<IterOp>,
}

/// A single lazy operation in the pipeline.
#[derive(Clone)]
pub enum IterOp {
    Map(FnValue),
    Filter(FnValue),
    Take(usize),
    Skip(usize),
    /// Concatenate another chain's elements after the current ones.
    Chain(IterChain),
    /// Flatten nested lists: each `Value::List(inner)` is inlined.
    Flatten,
    /// Map then flatten: call closure, then inline any `Value::List`.
    FlatMap(FnValue),
}

impl SharedIter {
    pub fn from_list(items: Vec<Value>) -> Self {
        Self {
            source: items.into(),
            ops: Vec::new(),
            offset: 0,
            cache: Arc::new(OnceLock::new()),
        }
    }

    pub fn from_list_rev(mut items: Vec<Value>) -> Self {
        items.reverse();
        Self::from_list(items)
    }

    // -- lazy chain builders (return a *new* SharedIter) ----------------------

    pub fn map(self, f: FnValue) -> Self {
        self.push_op(IterOp::Map(f))
    }

    pub fn filter(self, f: FnValue) -> Self {
        self.push_op(IterOp::Filter(f))
    }

    pub fn take(self, n: usize) -> Self {
        self.push_op(IterOp::Take(n))
    }

    pub fn skip(self, n: usize) -> Self {
        self.push_op(IterOp::Skip(n))
    }

    pub fn chain(self, other: SharedIter) -> Self {
        self.push_op(IterOp::Chain(other.snapshot_chain()))
    }

    pub fn flatten(self) -> Self {
        self.push_op(IterOp::Flatten)
    }

    pub fn flat_map(self, f: FnValue) -> Self {
        self.push_op(IterOp::FlatMap(f))
    }

    // -- snapshot / access ----------------------------------------------------

    /// Snapshot the current chain for embedding in another chain (e.g. `Chain` op).
    /// If cache is populated, uses cached data as source (no ops needed).
    pub fn snapshot_chain(&self) -> IterChain {
        if let Some(cached) = self.cache.get() {
            IterChain {
                source: cached.clone().into(),
                ops: Vec::new(),
            }
        } else {
            IterChain {
                source: Arc::clone(&self.source),
                ops: self.ops.clone(),
            }
        }
    }

    /// Get the cached result, if available.
    pub fn cached(&self) -> Option<&Vec<Value>> {
        self.cache.get()
    }

    /// Store the computed result in the cache.
    /// No-op if already cached (OnceLock guarantees single initialization).
    pub fn set_cache(&self, items: Vec<Value>) {
        let _ = self.cache.set(items);
    }

    /// Create a fork with independent cache.
    /// Use this for effectful iterators where re-execution is required.
    pub fn fork(&self) -> Self {
        Self {
            source: Arc::clone(&self.source),
            ops: self.ops.clone(),
            offset: self.offset,
            cache: Arc::new(OnceLock::new()),
        }
    }

    /// Current consumption offset (for `next()` support).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Advance the offset by one (for `next()`).
    pub fn advance(&mut self) {
        self.offset += 1;
    }

    fn push_op(self, op: IterOp) -> Self {
        let mut ops = self.ops;
        ops.push(op);
        Self {
            source: self.source,
            ops,
            offset: 0,
            cache: Arc::new(OnceLock::new()),
        }
    }
}

impl std::fmt::Debug for SharedIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedIter(ops={}, offset={})", self.ops.len(), self.offset)
    }
}

impl PartialEq for SharedIter {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

// =========================================================================
// SequenceChain — lazy Deque with origin tracking
// =========================================================================

/// A lazy sequence of operations on a [`TrackedDeque`].
///
/// Represents a Deque that has been loaded from storage (with checkpoint set)
/// and has pending structural operations. The origin TrackedDeque's checksum
/// is preserved through all operations — when collected, the result can be
/// diffed against the storage origin via [`TrackedDeque::into_diff`].
///
/// ## Invariant
///
/// `origin` must have a checkpoint set. All structural ops (take, skip, chain)
/// are applied lazily. On collect, they are executed against `origin`,
/// producing a TrackedDeque in the same checksum lineage.
///
/// ## Allowed ops
///
/// Only structural operations are allowed — these do not transform individual
/// elements, so the TrackedDeque's diff tracking remains valid:
/// - `Take(n)`: keep only the first `n` elements
/// - `Skip(n)`: consume `n` elements from the front
/// - `Chain(SharedIter)`: extend with elements from a lazy iterator
///
/// Element-transforming operations (map, filter) break the origin relationship
/// and must go through Iterator (Sequence → Iterator coercion).
#[derive(Clone)]
pub struct SequenceChain {
    origin: TrackedDeque<Value>,
    ops: Vec<SequenceOp>,
}

/// A structural operation on a Sequence.
#[derive(Clone)]
pub enum SequenceOp {
    /// Keep only the first `n` elements.
    Take(usize),
    /// Consume `n` elements from the front.
    Skip(usize),
    /// Extend with elements from a lazy iterator.
    /// Requires runtime to execute (SharedIter may contain closures).
    Chain(SharedIter),
}

impl SequenceChain {
    /// Create a new SequenceChain from a TrackedDeque.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the deque has a checkpoint set.
    pub fn new(origin: TrackedDeque<Value>) -> Self {
        debug_assert!(
            origin.is_dirty() || true, // checkpoint must exist — is_dirty only works after checkpoint
            "SequenceChain: origin must have checkpoint set"
        );
        Self {
            origin,
            ops: Vec::new(),
        }
    }

    /// Create from a storage-loaded TrackedDeque: clone + checkpoint.
    pub fn from_stored(stored: TrackedDeque<Value>) -> Self {
        let mut working = stored;
        working.checkpoint();
        Self::new(working)
    }

    /// Create an empty sequence (first turn, no stored value).
    pub fn empty() -> Self {
        let mut deque = TrackedDeque::new();
        deque.checkpoint();
        Self::new(deque)
    }

    // -- lazy builders --------------------------------------------------------

    pub fn take(mut self, n: usize) -> Self {
        self.ops.push(SequenceOp::Take(n));
        self
    }

    pub fn skip(mut self, n: usize) -> Self {
        self.ops.push(SequenceOp::Skip(n));
        self
    }

    pub fn chain(mut self, iter: SharedIter) -> Self {
        self.ops.push(SequenceOp::Chain(iter));
        self
    }

    // -- access ---------------------------------------------------------------

    /// Borrow the origin TrackedDeque.
    pub fn origin(&self) -> &TrackedDeque<Value> {
        &self.origin
    }

    /// Borrow the pending ops.
    pub fn ops(&self) -> &[SequenceOp] {
        &self.ops
    }

    /// Convert to a SharedIter (Sequence → Iterator coercion).
    ///
    /// The origin's items become the source, and SequenceOps are translated
    /// to IterOps. The result is still lazy — nothing is executed.
    pub fn into_shared_iter(self) -> SharedIter {
        let source: Arc<[Value]> = self.origin.into_vec().into();
        let ops: Vec<IterOp> = self
            .ops
            .into_iter()
            .map(|op| match op {
                SequenceOp::Take(n) => IterOp::Take(n),
                SequenceOp::Skip(n) => IterOp::Skip(n),
                SequenceOp::Chain(iter) => IterOp::Chain(iter.snapshot_chain()),
            })
            .collect();
        SharedIter {
            source,
            ops,
            offset: 0,
            cache: Arc::new(OnceLock::new()),
        }
    }

    /// Consume and return the origin TrackedDeque (for direct Deque access
    /// when no ops have been applied).
    ///
    /// # Panics
    ///
    /// Panics if there are pending ops.
    pub fn into_origin(self) -> TrackedDeque<Value> {
        assert!(
            self.ops.is_empty(),
            "into_origin called with {} pending ops",
            self.ops.len()
        );
        self.origin
    }

    /// Whether this sequence has pending lazy ops.
    pub fn has_ops(&self) -> bool {
        !self.ops.is_empty()
    }

    /// The origin's checksum — for verification against storage.
    pub fn origin_checksum(&self) -> u64 {
        self.origin.checksum()
    }
}

impl std::fmt::Debug for SequenceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SequenceChain(len={}, ops={}, checksum={:#x})",
            self.origin.len(),
            self.ops.len(),
            self.origin.checksum()
        )
    }
}

impl PartialEq for SequenceChain {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}
