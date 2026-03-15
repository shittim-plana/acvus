use std::sync::Arc;

use acvus_interpreter::{Interpreter, LazyValue, RuntimeError, TypedValue, Value};
use acvus_mir::ir::{MirBody, MirModule};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;

/// Composite iterator node: pulls from multiple sources, yields items one by one.
///
/// Each source is a `(name_tag, source_node_name)` pair.
/// - If the source value is an Iterator: `exec_next` per item.
/// - If the source value is a scalar: yield once.
///
/// `unordered=false`: sequential — exhaust source A, then B, etc.
/// `unordered=true`: concurrent — yield from whichever source is ready first.
pub struct IteratorNode {
    sources: Vec<(String, Astr)>,
    unordered: bool,
    interner: Interner,
}

impl IteratorNode {
    pub fn new(sources: Vec<(String, Astr)>, unordered: bool, interner: &Interner) -> Self {
        Self {
            sources,
            unordered,
            interner: interner.clone(),
        }
    }
}

impl Node for IteratorNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<TypedValue>>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let sources = self.sources.clone();
        let unordered = self.unordered;
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            if unordered {
                // TODO: FuturesUnordered concurrent execution
                // For now, fall back to sequential
                sequential_iterate(&interner, &sources, &local, &handle).await?;
            } else {
                sequential_iterate(&interner, &sources, &local, &handle).await?;
            }
            Ok(())
        })
    }
}

/// Sequential iteration: exhaust each source in order.
async fn sequential_iterate(
    interner: &Interner,
    sources: &[(String, Astr)],
    _local: &FxHashMap<Astr, Arc<TypedValue>>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<(), RuntimeError> {
    for (_name_tag, source_name) in sources {
        // Request the source value from context
        let source_value = handle.request_context(*source_name).await;
        let value = Arc::unwrap_or_clone(source_value).into_value();
        let inner = Arc::unwrap_or_clone(value);

        // Check if it's an iterator or a scalar
        match inner {
            Value::Lazy(LazyValue::Iterator(ih)) => {
                // Iterator: pull one by one via exec_next
                let empty_module = MirModule { main: MirBody::default(), closures: Default::default() };
                let mut interp = Interpreter::new(interner, empty_module);
                let mut current = ih;
                loop {
                    let result;
                    (interp, result) = Interpreter::exec_next(interp, current, handle).await?;
                    match result {
                        Some((item, rest)) => {
                            current = rest;
                            handle.yield_val(TypedValue::new(
                                Arc::new(item),
                                acvus_mir::ty::Ty::Infer,
                            )).await;
                        }
                        None => break,
                    }
                }
            }
            other => {
                // Scalar: yield once
                handle.yield_val(TypedValue::new(
                    Arc::new(other),
                    acvus_mir::ty::Ty::Infer,
                )).await;
            }
        }
    }
    Ok(())
}
