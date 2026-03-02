use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use acvus_ast::Literal;
use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir_pass::analysis::reachable_context::reachable_context_keys;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};

use crate::compile::CompiledNode;
use crate::dag::Dag;
use crate::error::{OrchError, OrchErrorKind};
use crate::message::{Message, ModelResponse, Output, ToolCall, ToolResult, ToolSpec};
use crate::provider::{Fetch, FetchRequest};
use crate::storage::Storage;

pub struct Executor<F, S>
where
    F: Fetch,
    S: Storage,
{
    nodes: Vec<CompiledNode>,
    dag: Dag,
    storage: S,
    fetch: F,
    mir_registry: ExternRegistry,
    fuel_limit: u64,
}

impl<F, S> Executor<F, S>
where
    F: Fetch + 'static,
    S: Storage,
{
    pub fn new(
        nodes: Vec<CompiledNode>,
        dag: Dag,
        storage: S,
        fetch: F,
        mir_registry: ExternRegistry,
        fuel_limit: u64,
    ) -> Self {
        Self { nodes, dag, storage, fetch, mir_registry, fuel_limit }
    }

    /// Run the full DAG, executing independent nodes in parallel.
    ///
    /// Uses MIR static analysis (`reachable_context_keys`) to determine which
    /// dependencies are actually needed at runtime. Nodes whose live code paths
    /// don't reference a dependency's output can be launched early, even if that
    /// dependency hasn't completed yet.
    ///
    /// Known context values are converted to `Literal` and fed into the analysis
    /// so branch conditions can be evaluated — dead branches are pruned and their
    /// context loads ignored.
    pub async fn run(self) -> Result<S, OrchError> {
        let Executor { nodes, dag, mut storage, fetch, fuel_limit, .. } = self;
        let fetch = Arc::new(fetch);
        let fuel = Arc::new(AtomicU64::new(0));
        let n = nodes.len();

        let mut launched = vec![false; n];
        let mut completed = vec![false; n];

        let mut running: FuturesUnordered<BoxFuture<'static, Result<(usize, Output), OrchError>>> =
            FuturesUnordered::new();

        // Launch initially ready nodes
        for i in 0..n {
            if is_node_ready(i, &nodes, &dag, &storage, &completed) {
                launched[i] = true;
                let context = build_context_for(&nodes[i], &storage);
                running.push(Box::pin(run_node(
                    i,
                    nodes[i].clone(),
                    context,
                    Arc::clone(&fetch),
                    Arc::clone(&fuel),
                    fuel_limit,
                )));
            }
        }

        // Process completions and launch newly ready nodes
        while let Some(result) = running.next().await {
            let (idx, output) = result?;
            storage.set(nodes[idx].name.clone(), output);
            completed[idx] = true;

            // Re-check all pending nodes — with new storage values,
            // the analysis may prune branches and unlock more nodes.
            for i in 0..n {
                if !launched[i] && is_node_ready(i, &nodes, &dag, &storage, &completed) {
                    launched[i] = true;
                    let context = build_context_for(&nodes[i], &storage);
                    running.push(Box::pin(run_node(
                        i,
                        nodes[i].clone(),
                        context,
                        Arc::clone(&fetch),
                        Arc::clone(&fuel),
                        fuel_limit,
                    )));
                }
            }
        }

        Ok(storage)
    }
}

// ---------------------------------------------------------------------------
// Node execution (standalone async, captures everything by value)
// ---------------------------------------------------------------------------

async fn run_node<F>(
    idx: usize,
    node: CompiledNode,
    context: HashMap<String, Output>,
    fetch: Arc<F>,
    fuel: Arc<AtomicU64>,
    fuel_limit: u64,
) -> Result<(usize, Output), OrchError>
where
    F: Fetch,
{
    // Render each block into a message
    let mut messages = Vec::new();
    for block in &node.blocks {
        let context_values: HashMap<String, Value> = context
            .iter()
            .map(|(k, v)| (k.clone(), output_to_value(v)))
            .collect();

        let interp =
            Interpreter::new(block.module.clone(), context_values, ExternFnRegistry::new());
        let output = interp.execute_to_string().await;

        messages.push(Message::text(&block.role, output));
    }

    let tools: Vec<ToolSpec> = node
        .tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: String::new(),
            params: t.params.clone(),
        })
        .collect();

    // Call model
    consume_fuel(&fuel, fuel_limit)?;
    let request = FetchRequest {
        provider: node.provider.clone(),
        model: node.model.clone(),
        messages: messages.clone(),
        tools: tools.clone(),
    };
    let mut response = fetch
        .call(&request)
        .await
        .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;

    // Tool call loop
    let mut all_messages = messages;
    while let ModelResponse::ToolCalls(ref calls) = response {
        consume_fuel(&fuel, fuel_limit)?;

        let tool_results = handle_tool_calls(calls);

        // Assistant message with tool calls
        all_messages.push(Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: calls.clone(),
            tool_call_id: None,
        });

        // Tool result messages
        for result in &tool_results {
            all_messages.push(Message {
                role: "tool".into(),
                content: result.content.clone(),
                tool_calls: Vec::new(),
                tool_call_id: Some(result.call_id.clone()),
            });
        }

        let request = FetchRequest {
            provider: node.provider.clone(),
            model: node.model.clone(),
            messages: all_messages.clone(),
            tools: tools.clone(),
        };
        response = fetch
            .call(&request)
            .await
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
    }

    match response {
        ModelResponse::Text(text) => Ok((idx, Output::Text(text))),
        ModelResponse::ToolCalls(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a node can be launched given the current storage state.
///
/// Uses `reachable_context_keys` to determine which context keys are actually
/// needed on live code paths. A dependency is only blocking if:
/// 1. It hasn't completed yet, AND
/// 2. Its output is referenced on a live path in one of the node's blocks.
fn is_node_ready<S>(
    idx: usize,
    nodes: &[CompiledNode],
    dag: &Dag,
    storage: &S,
    completed: &[bool],
) -> bool
where
    S: Storage,
{
    let node = &nodes[idx];

    // Build known literals from available storage values
    let mut known = HashMap::new();
    for key in &node.all_context_keys {
        if let Some(output) = storage.get(key) {
            if let Some(lit) = output_to_literal(&output) {
                known.insert(key.clone(), lit);
            }
        }
    }

    // Compute actually needed keys via static analysis across all blocks
    let mut needed = HashSet::new();
    for block in &node.blocks {
        needed.extend(reachable_context_keys(&block.module, &known, &block.val_def));
    }

    // Check: for each incomplete dependency, is its output actually needed?
    for &dep_idx in &dag.deps[idx] {
        if completed[dep_idx] {
            continue;
        }
        let dep_name = &nodes[dep_idx].name;
        if needed.contains(dep_name) {
            return false;
        }
    }

    true
}

fn build_context_for<S>(node: &CompiledNode, storage: &S) -> HashMap<String, Output>
where
    S: Storage,
{
    let mut context = HashMap::new();
    for key in &node.all_context_keys {
        if let Some(value) = storage.get(key) {
            context.insert(key.clone(), value);
        }
    }
    context
}

// ---------------------------------------------------------------------------
// Output → Literal conversion (for reachable_context_keys analysis)
// ---------------------------------------------------------------------------

fn output_to_literal(output: &Output) -> Option<Literal> {
    match output {
        Output::Text(s) => Some(Literal::String(s.clone())),
        Output::Json(v) => json_to_literal(v),
        Output::Image(_) => None,
    }
}

fn json_to_literal(v: &serde_json::Value) -> Option<Literal> {
    match v {
        serde_json::Value::String(s) => Some(Literal::String(s.clone())),
        serde_json::Value::Bool(b) => Some(Literal::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Literal::Int(i))
            } else if let Some(f) = n.as_f64() {
                Some(Literal::Float(f))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn consume_fuel(fuel: &AtomicU64, limit: u64) -> Result<(), OrchError> {
    let prev = fuel.fetch_add(1, Ordering::Relaxed);
    if prev + 1 > limit {
        Err(OrchError::new(OrchErrorKind::FuelExhausted))
    } else {
        Ok(())
    }
}

fn handle_tool_calls(calls: &[ToolCall]) -> Vec<ToolResult> {
    calls
        .iter()
        .map(|call| ToolResult {
            call_id: call.id.clone(),
            content: format!("tool '{}' not implemented", call.name),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Output → Value conversion (for interpreter context injection)
// ---------------------------------------------------------------------------

fn output_to_value(output: &Output) -> Value {
    match output {
        Output::Text(s) => Value::String(s.clone()),
        Output::Json(v) => json_to_value(v),
        Output::Image(bytes) => Value::List(bytes.iter().map(|&b| Value::Byte(b)).collect()),
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Unit
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            Value::Object(obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_to_value_text() {
        let v = output_to_value(&Output::Text("hello".into()));
        assert!(matches!(v, Value::String(ref s) if s == "hello"));
    }

    #[test]
    fn output_to_value_json() {
        let v = output_to_value(&Output::Json(serde_json::json!({"name": "alice", "age": 30})));
        match v {
            Value::Object(obj) => {
                assert!(matches!(obj.get("name"), Some(Value::String(s)) if s == "alice"));
                assert!(matches!(obj.get("age"), Some(Value::Int(30))));
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn output_to_value_image() {
        let v = output_to_value(&Output::Image(vec![0xff, 0x00]));
        match v {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], Value::Byte(0xff)));
            }
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn output_to_literal_text() {
        let lit = output_to_literal(&Output::Text("hello".into()));
        assert_eq!(lit, Some(Literal::String("hello".into())));
    }

    #[test]
    fn output_to_literal_json_string() {
        let lit = output_to_literal(&Output::Json(serde_json::json!("world")));
        assert_eq!(lit, Some(Literal::String("world".into())));
    }

    #[test]
    fn output_to_literal_json_int() {
        let lit = output_to_literal(&Output::Json(serde_json::json!(42)));
        assert_eq!(lit, Some(Literal::Int(42)));
    }

    #[test]
    fn output_to_literal_json_bool() {
        let lit = output_to_literal(&Output::Json(serde_json::json!(true)));
        assert_eq!(lit, Some(Literal::Bool(true)));
    }

    #[test]
    fn output_to_literal_image_none() {
        let lit = output_to_literal(&Output::Image(vec![0xff]));
        assert!(lit.is_none());
    }

    #[test]
    fn output_to_literal_json_object_none() {
        let lit = output_to_literal(&Output::Json(serde_json::json!({"key": "val"})));
        assert!(lit.is_none());
    }
}
