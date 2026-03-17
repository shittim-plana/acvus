use std::sync::Arc;

use acvus_interpreter::{Interpreter, PureValue, RuntimeError, TypedValue, Value};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::Effect;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;
use super::helpers::{eval_script_in_coroutine, render_block_in_coroutine};
use crate::kind::{CompiledIteratorEntry, CompiledIteratorSource, CompiledSourceTransform};

/// Composite iterator node: pulls from multiple sources, yields items one by one.
///
/// Each source evaluates `expr` → gets a value, then:
/// - If Iterator: `exec_next` per item
/// - If List/Deque: iterate elements (converted to IterHandle)
/// - If scalar: yield once
///
/// Then applies pagination (start/end) and per-item transform.
///
/// `unordered=false`: sequential — exhaust source A, then B, etc.
/// `unordered=true`: concurrent — yield from whichever source is ready first.
pub struct IteratorNode {
    sources: Vec<CompiledIteratorSource>,
    unordered: bool,
    interner: Interner,
}

impl IteratorNode {
    pub fn new(
        sources: Vec<CompiledIteratorSource>,
        unordered: bool,
        interner: &Interner,
    ) -> Self {
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
    sources: &[CompiledIteratorSource],
    local: &FxHashMap<Astr, Arc<TypedValue>>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<(), RuntimeError> {
    let name_key = interner.intern("name");
    let item_key = interner.intern("item");

    for source in sources {
        // Evaluate pagination bounds
        let start: usize = match &source.start {
            Some(start_script) => {
                let val = eval_script_in_coroutine(
                    interner, &start_script.module, local, handle,
                ).await?;
                match val.value() {
                    Value::Pure(PureValue::Int(n)) => *n as usize,
                    _ => 0,
                }
            }
            None => 0,
        };

        let end: Option<usize> = match &source.end {
            Some(end_script) => {
                let val = eval_script_in_coroutine(
                    interner, &end_script.module, local, handle,
                ).await?;
                match val.to_concrete(interner) {
                    acvus_interpreter::ConcreteValue::Variant { tag, payload } if tag == "Some" => {
                        match payload.as_deref() {
                            Some(acvus_interpreter::ConcreteValue::Int { v }) => Some(*v as usize),
                            _ => None,
                        }
                    }
                    acvus_interpreter::ConcreteValue::Int { v } => Some(v as usize),
                    _ => None,
                }
            }
            None => None,
        };

        // Evaluate source expression
        let source_val = eval_script_in_coroutine(
            interner, &source.expr.module, local, handle,
        ).await?;

        let inner = Arc::unwrap_or_clone(source_val.into_value());

        // Determine if iterable or scalar
        let is_iterable = matches!(
            &inner,
            Value::Lazy(acvus_interpreter::LazyValue::Iterator(_))
            | Value::Lazy(acvus_interpreter::LazyValue::List(_))
            | Value::Lazy(acvus_interpreter::LazyValue::Deque(_))
            | Value::Lazy(acvus_interpreter::LazyValue::Sequence(_))
        );

        if is_iterable {
            let ih = inner.into_iter_handle(Effect::Pure);
            let empty_module = MirModule { main: acvus_mir::ir::MirBody::default(), closures: Default::default() };
            let mut interp = Interpreter::new(interner, empty_module);
            let mut current = Some(ih);

            // Skip `start` items
            for _ in 0..start {
                let Some(ih) = current.take() else { break };
                let result;
                (interp, result) = Interpreter::exec_next(interp, ih, handle).await?;
                match result {
                    Some((_, rest)) => current = Some(rest),
                    None => break,
                }
            }

            // Yield items, stopping at `end`
            let take = end.map(|e| e.saturating_sub(start));
            let mut count = 0usize;
            while let Some(ih) = current.take() {
                if let Some(limit) = take {
                    if count >= limit {
                        break;
                    }
                }

                let result;
                (interp, result) = Interpreter::exec_next(interp, ih, handle).await?;
                let Some((item, rest)) = result else {
                    break;
                };
                current = Some(rest);

                let transformed = apply_entries(
                    interner, &source.entries, item, start + count, local, handle,
                ).await?;

                if let Some(val) = transformed {
                    yield_tagged(handle, name_key, item_key, &source.name, val).await;
                }
                count += 1;
            }
        } else {
            // Scalar: apply entries and yield once
            let transformed = apply_entries(
                interner, &source.entries, inner, 0, local, handle,
            ).await?;

            if let Some(val) = transformed {
                yield_tagged(handle, name_key, item_key, &source.name, val).await;
            }
        }
    }
    Ok(())
}

/// Apply first-match entry processing.
///
/// - If `entries` is empty: pass-through (yield item as-is).
/// - Otherwise: evaluate each entry's condition in order.
///   - `None` condition always matches.
///   - `Some` condition: evaluate with `@item` + `@index`; match if `true`.
///   - First match: apply its transform and return `Some(transformed)`.
///   - No match: return `None` (skip this item).
async fn apply_entries(
    interner: &Interner,
    entries: &[CompiledIteratorEntry],
    item: Value,
    index: usize,
    local: &FxHashMap<Astr, Arc<TypedValue>>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<Option<Value>, RuntimeError> {
    if entries.is_empty() {
        return Ok(Some(item));
    }

    let item_key = interner.intern("item");
    let index_key = interner.intern("index");

    let mut entry_local = local.clone();
    entry_local.insert(
        item_key,
        Arc::new(TypedValue::new(Arc::new(item.clone()), acvus_mir::ty::Ty::Infer)),
    );
    entry_local.insert(
        index_key,
        Arc::new(TypedValue::int(index as i64)),
    );

    for entry in entries {
        // Evaluate condition (None = always matches)
        let matched = match &entry.condition {
            None => true,
            Some(cond_script) => {
                let result = eval_script_in_coroutine(
                    interner, &cond_script.module, &entry_local, handle,
                ).await?;
                matches!(result.value(), Value::Pure(PureValue::Bool(true)))
            }
        };

        if matched {
            let transformed = apply_transform(
                interner, &entry.transform, &entry_local, handle,
            ).await?;
            return Ok(Some(transformed));
        }
    }

    // No entry matched — skip this item
    Ok(None)
}

/// Apply a single transform with the pre-built local context (already contains `@item` + `@index`).
async fn apply_transform(
    interner: &Interner,
    transform: &CompiledSourceTransform,
    local: &FxHashMap<Astr, Arc<TypedValue>>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<Value, RuntimeError> {
    match transform {
        CompiledSourceTransform::Script(script) => {
            let result = eval_script_in_coroutine(
                interner, &script.module, local, handle,
            ).await?;
            Ok(Arc::unwrap_or_clone(result.into_value()))
        }
        CompiledSourceTransform::Template(template) => {
            let rendered = render_block_in_coroutine(
                interner, &template.module, local, handle,
            ).await?;
            Ok(Value::string(rendered))
        }
    }
}

/// Yield a tagged item: `{name: String, item: T}`.
async fn yield_tagged(
    handle: &acvus_utils::YieldHandle<TypedValue>,
    name_key: Astr,
    item_key: Astr,
    name_tag: &str,
    item: Value,
) {
    let mut fields = FxHashMap::default();
    fields.insert(name_key, Value::Pure(PureValue::String(name_tag.to_string())));
    fields.insert(item_key, item);
    let obj = Value::object(fields);
    handle.yield_val(TypedValue::new(
        Arc::new(obj),
        acvus_mir::ty::Ty::Infer,
    )).await;
}
