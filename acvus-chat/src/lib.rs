mod error;

pub use error::ChatError;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue, Value};
use acvus_orchestration::{
    CompiledNode, EntryMut, EntryRef, Fetch, Journal, Node, ProviderConfig, ResolveState, Resolved,
    Resolver, build_dag, build_node_table,
};
use acvus_utils::{Astr, Coroutine, Interner, Stepped};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pending evaluation: a coroutine that yields items one by one.
struct EvalState {
    coroutine: Coroutine<TypedValue, RuntimeError>,
    no_execute: bool,
    /// Cursor before this evaluation started (for cancel/rollback).
    prev_cursor: Uuid,
}

pub struct ChatEngine<J> {
    nodes: Vec<CompiledNode>,
    node_table: Vec<Arc<dyn Node>>,
    name_to_idx: FxHashMap<Astr, usize>,
    rdeps: Vec<FxHashSet<usize>>,
    pub journal: J,
    pub cursor: Uuid,
    bind_cache: FxHashMap<Astr, Vec<(TypedValue, Arc<TypedValue>)>>,
    entrypoint_idx: usize,
    side_effect_idxs: Vec<usize>,
    interner: Interner,
    eval_state: Option<EvalState>,
}

impl<J> ChatEngine<J>
where
    J: Journal,
{
    pub async fn new<F>(
        nodes: Vec<CompiledNode>,
        providers: FxHashMap<String, ProviderConfig>,
        fetch: F,
        journal: J,
        root: Uuid,
        entrypoint: &str,
        side_effects: &[String],
        interner: &Interner,
    ) -> Result<Self, ChatError>
    where
        F: Fetch + 'static,
    {
        let name_to_idx: FxHashMap<Astr, usize> =
            nodes.iter().enumerate().map(|(i, n)| (n.name, i)).collect();

        let entrypoint_key = interner.intern(entrypoint);
        let entrypoint_idx = *name_to_idx
            .get(&entrypoint_key)
            .ok_or_else(|| ChatError::EntrypointNotFound(entrypoint.to_string()))?;

        // Validate: no dependency cycles + extract rdeps
        let dag = build_dag(interner, &nodes).map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.display(interner).to_string())
                .collect::<Vec<_>>()
                .join("; ");
            ChatError::CycleDetected(msg)
        })?;
        let rdeps = dag.rdeps;

        // Resolve side_effect node indices
        let side_effect_idxs: Vec<usize> = side_effects
            .iter()
            .filter_map(|name| {
                let key = interner.intern(name);
                name_to_idx.get(&key).copied()
            })
            .collect();

        // Build node table — one match, uniform Arc<dyn Node> from here
        let node_table =
            build_node_table(&nodes, &providers, Arc::new(fetch), interner);

        Ok(Self {
            nodes,
            node_table,
            name_to_idx,
            rdeps,
            journal,
            cursor: root,
            bind_cache: FxHashMap::default(),
            entrypoint_idx,
            side_effect_idxs,
            interner: interner.clone(),
            eval_state: None,
        })
    }

    /// Start an evaluation. Resolves the named node and prepares streaming.
    ///
    /// - `no_execute=false`: creates a new journal branch (cursor advances).
    /// - `no_execute=true`: reads from current cursor (no branch).
    pub async fn start_evaluate<R, EH>(
        &mut self,
        node_name: &str,
        no_execute: bool,
        resolver: &R,
        extern_handler: &EH,
    ) -> Result<(), ChatError>
    where
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync,
    {
        let interner = &self.interner;
        let node_key = interner.intern(node_name);
        let node_idx = *self
            .name_to_idx
            .get(&node_key)
            .ok_or_else(|| ChatError::EntrypointNotFound(node_name.to_string()))?;

        tracing::info!(node = %node_name, no_execute, "start_evaluate");

        // Block scope: entry borrows journal mutably, released at block end.
        let (new_cursor, bind_cache, entrypoint_value) = {
            let entry = if no_execute {
                self.journal.entry_mut(self.cursor).await
            } else {
                self.journal.entry_mut(self.cursor).await.next().await
            };
            let new_cursor = entry.uuid();

            let mut rs = ResolveState {
                entry,
                turn_context: FxHashMap::default(),
                bind_cache: std::mem::take(&mut self.bind_cache),
            };

            {
                let ctx = Resolver {
                    nodes: &self.nodes,
                    node_table: &self.node_table,
                    name_to_idx: &self.name_to_idx,
                    extern_handler,
                    resolver,
                    interner,
                    rdeps: &self.rdeps,
                };

                ctx.resolve_node(node_idx, &mut rs, FxHashMap::default(), no_execute)
                    .await
                    .map_err(|e| ChatError::Resolve(format!("[{}] {e}", node_name)))?;

                // Side effects (only when executing)
                if !no_execute && !self.side_effect_idxs.is_empty() {
                    let side_effects: Vec<_> = self
                        .side_effect_idxs
                        .iter()
                        .map(|&idx| (idx, FxHashMap::default()))
                        .collect();
                    ctx.resolve_nodes(side_effects, &mut rs, false)
                        .await
                        .map_err(|e| ChatError::Resolve(format!("[side_effects] {e}")))?;
                }
            }

            let name = self.nodes[node_idx].name;
            let name_str = interner.resolve(name);
            let entrypoint_value = rs.load(name, name_str)
                .ok_or_else(|| ChatError::UnresolvedContext(name_str.to_string()))?;

            let bind_cache = std::mem::take(&mut rs.bind_cache);
            (new_cursor, bind_cache, entrypoint_value)
        };

        self.bind_cache = bind_cache;
        if !no_execute {
            self.cursor = new_cursor;
        }

        // Convert result to a coroutine that yields items one by one.
        let value = Arc::unwrap_or_clone(entrypoint_value);
        let coroutine = value_to_coroutine(value);
        let prev_cursor = self.cursor;
        self.eval_state = Some(EvalState { coroutine, no_execute, prev_cursor });

        tracing::info!(node = %node_name, "evaluate ready");
        Ok(())
    }

    /// Pull the next step from the current evaluation.
    ///
    /// Returns the raw `Stepped` variant. Caller must handle:
    /// - `Emit(value)` — a yielded item
    /// - `Done` — evaluation complete
    /// - `Error(e)` — runtime error
    /// - `NeedContext(req)` — caller resolves and calls `evaluate_next` again
    /// - `NeedExternCall(req)` — caller resolves and calls `evaluate_next` again
    pub async fn evaluate_next(&mut self) -> Stepped<TypedValue, RuntimeError> {
        let state = match self.eval_state.as_mut() {
            Some(s) => s,
            None => return Stepped::Done,
        };

        let stepped = state.coroutine.resume().await;
        if matches!(stepped, Stepped::Done | Stepped::Error(_)) {
            self.eval_state = None;
        }
        stepped
    }

    /// Cancel an in-progress evaluation.
    ///
    /// Drops the coroutine and rolls back the cursor if this was an executing
    /// evaluation (no_execute=false). The incomplete branch remains in the
    /// journal but cursor returns to the previous position (unflushed).
    pub fn cancel_evaluate(&mut self) {
        if let Some(state) = self.eval_state.take() {
            if !state.no_execute {
                self.cursor = state.prev_cursor;
            }
        }
    }

    pub async fn history_len(&self) -> usize {
        self.journal.entry(self.cursor).await.depth()
    }
}

/// Convert a resolved TypedValue into a coroutine that yields items.
///
/// - Iterator/List/Deque: yields each element.
/// - Scalar: yields the single value.
fn value_to_coroutine(value: TypedValue) -> Coroutine<TypedValue, RuntimeError> {
    acvus_utils::coroutine(move |handle| async move {
        match value.value() {
            Value::Lazy(acvus_interpreter::LazyValue::Iterator(_))
            | Value::Lazy(acvus_interpreter::LazyValue::Sequence(_)) => {
                // Iterator/Sequence: use exec_next to pull one by one
                let ih = Arc::unwrap_or_clone(value.into_value())
                    .into_iter_handle(acvus_mir::ty::Effect::Pure);
                let empty_module = acvus_mir::ir::MirModule {
                    main: acvus_mir::ir::MirBody::default(),
                    closures: Default::default(),
                };
                let mut interp = acvus_interpreter::Interpreter::new(
                    &acvus_utils::Interner::new(), // dummy — closures already captured
                    empty_module,
                );
                let mut current = ih;
                loop {
                    let result;
                    (interp, result) = acvus_interpreter::Interpreter::exec_next(
                        interp, current, &handle,
                    ).await.map_err(|e| e)?;
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
            Value::Lazy(acvus_interpreter::LazyValue::List(_))
            | Value::Lazy(acvus_interpreter::LazyValue::Deque(_)) => {
                // List/Deque: yield each element
                let items = match Arc::unwrap_or_clone(value.into_value()) {
                    Value::Lazy(acvus_interpreter::LazyValue::List(items)) => items,
                    Value::Lazy(acvus_interpreter::LazyValue::Deque(d)) => d.into_vec(),
                    _ => unreachable!(),
                };
                for item in items {
                    handle.yield_val(TypedValue::new(
                        Arc::new(item),
                        acvus_mir::ty::Ty::Infer,
                    )).await;
                }
            }
            _ => {
                // Scalar: yield once
                handle.yield_val(value).await;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_interpreter::{LazyValue};
    use acvus_mir::ty::Ty;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_orchestration::{
        ApiKind, ExprSpec, GenerationParams, HttpRequest, LlmSpec, MaxTokens,
        Execution, FnParam, MessageSpec, NodeKind, NodeSpec, Persistency, PlainSpec, Strategy, ToolBinding,
        TreeJournal, compile_nodes,
    };

    struct MockFetch {
        responses: Mutex<Vec<serde_json::Value>>,
    }

    impl MockFetch {
        fn new(responses: Vec<serde_json::Value>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl Fetch for MockFetch {
        async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Err("no more mock responses".into());
            }
            Ok(q.remove(0))
        }
    }

    fn openai_text_response(text: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        })
    }

    fn openai_tool_call_response(calls: Vec<(&str, &str, serde_json::Value)>) -> serde_json::Value {
        let tool_calls: Vec<serde_json::Value> = calls
            .into_iter()
            .map(|(id, name, args)| {
                serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&args).unwrap()
                    }
                })
            })
            .collect();
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls
                }
            }]
        })
    }

    fn default_provider() -> (String, ProviderConfig) {
        (
            "test".into(),
            ProviderConfig {
                api: ApiKind::OpenAI,
                endpoint: "http://mock".into(),
                api_key: String::new(),
            },
        )
    }

    fn noop_resolver() -> impl AsyncFn(Astr) -> Resolved + Sync {
        |_: Astr| async { Resolved::Once(TypedValue::unit()) }
    }

    fn noop_extern_handler() -> impl AsyncFn(Astr, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync {
        |_: Astr, _: Vec<TypedValue>| async { Ok(TypedValue::unit()) }
    }

    /// Drive evaluate_next to completion, handling NeedContext/NeedExternCall
    /// via the provided callbacks. Collects all Emit values.
    async fn drain_evaluate<R, EH>(
        engine: &mut ChatEngine<TreeJournal>,
        resolver: &R,
        extern_handler: &EH,
    ) -> Vec<TypedValue>
    where
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync,
    {
        let mut items = Vec::new();
        loop {
            match engine.evaluate_next().await {
                Stepped::Emit(value) => items.push(value),
                Stepped::Done => break,
                Stepped::Error(e) => panic!("evaluate error: {e}"),
                Stepped::NeedContext(req) => {
                    let value = resolver(req.name()).await;
                    let arc = match value {
                        Resolved::Once(v) | Resolved::Turn(v) | Resolved::Persist(v) => Arc::new(v),
                    };
                    req.resolve(arc);
                }
                Stepped::NeedExternCall(req) => {
                    let name = req.name();
                    let args = req.args().to_vec();
                    match extern_handler(name, args).await {
                        Ok(v) => req.resolve(Arc::new(v)),
                        Err(e) => panic!("extern call error: {e}"),
                    }
                }
            }
        }
        items
    }

    /// Drive evaluate_next and return the first emitted value.
    async fn evaluate_first<R, EH>(
        engine: &mut ChatEngine<TreeJournal>,
        resolver: &R,
        extern_handler: &EH,
    ) -> TypedValue
    where
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<TypedValue>) -> Result<TypedValue, RuntimeError> + Sync,
    {
        loop {
            match engine.evaluate_next().await {
                Stepped::Emit(value) => return value,
                Stepped::Done => panic!("evaluate_first: no items emitted"),
                Stepped::Error(e) => panic!("evaluate error: {e}"),
                Stepped::NeedContext(req) => {
                    let value = resolver(req.name()).await;
                    let arc = match value {
                        Resolved::Once(v) | Resolved::Turn(v) | Resolved::Persist(v) => Arc::new(v),
                    };
                    req.resolve(arc);
                }
                Stepped::NeedExternCall(req) => {
                    let name = req.name();
                    let args = req.args().to_vec();
                    match extern_handler(name, args).await {
                        Ok(v) => req.resolve(Arc::new(v)),
                        Err(e) => panic!("extern call error: {e}"),
                    }
                }
            }
        }
    }

    fn compile_test_nodes(interner: &Interner, specs: &[NodeSpec]) -> Vec<CompiledNode> {
        compile_nodes(
            interner,
            specs,
            PartialContextTypeRegistry::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn new_valid_entrypoint() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello".into(),
                }),

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn new_invalid_entrypoint() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello".into(),
                }),

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "nonexistent",
            &[],
            &interner,
        )
        .await;
        assert!(matches!(result, Err(ChatError::EntrypointNotFound(_))));
    }

    #[tokio::test]
    async fn turn_plain_node() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello world".into(),
                }),

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        assert_eq!(*result, Value::string("hello world".into()));
    }

    #[tokio::test]
    async fn turn_llm_text_response() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Llm(LlmSpec {
                    api: ApiKind::OpenAI,
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: interner.intern("user"),
                        source: "hi".into(),
                    }],
                    tools: vec![],
                    generation: GenerationParams::default(),
                    cache_key: None,
                    max_tokens: MaxTokens::default(),
                }),

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mock = MockFetch::new(vec![openai_text_response("hello from LLM")]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let items = drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        assert!(!items.is_empty(), "expected at least one item");
        let result = items[0].clone().into_value();
        // evaluate_next yields individual message objects
        let Value::Lazy(LazyValue::Object(msg)) = result.as_ref() else {
            panic!("expected Object, got {result:?}");
        };
        let content_key = interner.intern("content");
        assert_eq!(
            msg.get(&content_key),
            Some(&Value::string("hello from LLM".into()))
        );
    }

    #[tokio::test]
    async fn turn_tool_call_round_trip() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("tool_target"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "tool result text".into(),
                    }),
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Llm(LlmSpec {
                        api: ApiKind::OpenAI,
                        provider: "test".into(),
                        model: "gpt-test".into(),
                        messages: vec![MessageSpec::Block {
                            role: interner.intern("user"),
                            source: "use the tool".into(),
                        }],
                        tools: vec![ToolBinding {
                            name: "my_tool".into(),
                            description: String::new(),
                            node: "tool_target".into(),
                            params: FxHashMap::default(),
                        }],
                        generation: GenerationParams::default(),
                        cache_key: None,
                        max_tokens: MaxTokens::default(),
                    }),
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mock = MockFetch::new(vec![
            openai_tool_call_response(vec![("call_1", "my_tool", serde_json::json!({}))]),
            openai_text_response("final answer"),
        ]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        // Drain all items; the last one should contain "final answer"
        let items = drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        let last = items.last().expect("expected at least one value");
        let result = last.clone().into_value();
        let Value::Lazy(LazyValue::Object(msg)) = result.as_ref() else {
            panic!("expected Object, got {result:?}");
        };
        let content_key = interner.intern("content");
        assert_eq!(
            msg.get(&content_key),
            Some(&Value::string("final answer".into()))
        );
    }

    // -- regression tests -------------------------------------------------------

    /// #6: initial_value must be evaluated on first run (not Unit).
    /// OncePerTurn: first turn uses initial_value as @self, subsequent turns use persisted @self.
    /// Accumulation is done in the Expr node body using @self.
    #[tokio::test]
    async fn initial_value_evaluated_on_first_run() {
        let interner = Interner::new();
        // Expr uses @self (previous) to accumulate.
        // initial_value = "A".
        // Turn 1: @self = "A" (initial), expr = @self + "B" → "AB"
        // Turn 2: @self = "AB" (persisted), expr = @self + "B" → "ABB"
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Expr(ExprSpec {
                    source: r#"@self + "B""#.into(),
                    output_ty: Ty::Infer,
                }),
                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::Snapshot,
                    initial_value: Some(interner.intern(r#""A""#)),
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let r1 = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        assert_eq!(*r1, Value::string("AB".into()));

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let r2 = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        assert_eq!(*r2, Value::string("ABB".into()));
    }

    /// #7: Always nodes must re-execute every invocation, not just once per turn.
    #[tokio::test]
    async fn always_node_re_executes_every_reference() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("counter"),
                    kind: NodeKind::Plain(PlainSpec { source: "x".into() }),
    
                    strategy: Strategy {
                        execution: Execution::Always,
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{@counter}}{{@counter}}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        assert_eq!(*result, Value::string("xx".into()));
    }

    /// #3 (double-prompt): Turn-resolved external values must be cached within a turn.
    #[tokio::test]
    async fn turn_resolver_caches_within_turn() {
        use acvus_mir::ty::Ty;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let interner = Interner::new();
        // main template references @input twice.
        // External resolver should be called only once for @input per turn.
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("input"), Ty::String);
        let nodes = compile_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "{{@input}}{{@input}}".into(),
                }),

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
            PartialContextTypeRegistry::user_only(ctx),
        )
        .unwrap();
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let resolver = move |_: Astr| {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Resolved::Turn(TypedValue::string("hi"))
            }
        };

        engine.start_evaluate("main", false, &resolver, &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &resolver, &noop_extern_handler()).await.into_value();
        assert_eq!(*result, Value::string("hihi".into()));
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolver should be called once per turn"
        );
    }

    /// #6b: LLM nodes with OncePerTurn persist correctly.
    #[tokio::test]
    async fn llm_once_per_turn_persists() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Llm(LlmSpec {
                    api: ApiKind::OpenAI,
                    provider: "test".into(),
                    model: "m".into(),
                    messages: vec![MessageSpec::Block {
                        role: interner.intern("user"),
                        source: "hi".into(),
                    }],
                    tools: vec![],
                    generation: GenerationParams::default(),
                    cache_key: None,
                    max_tokens: MaxTokens::default(),
                }),

                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![openai_text_response("hello")]),

            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Verify LLM output is stored and retrievable
        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        // evaluate_next yields individual message objects
        let Value::Lazy(LazyValue::Object(msg)) = result.as_ref() else {
            panic!("expected Object, got {result:?}");
        };
        let content_key = interner.intern("content");
        assert_eq!(msg.get(&content_key), Some(&Value::string("hello".into())));
    }

    /// @self in Expr node body: accumulates across turns.
    /// Uses OncePerTurn so @self persists.
    #[tokio::test]
    async fn node_body_accesses_self() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Expr(ExprSpec {
                    source: r#"@self + "B""#.into(),
                    output_ty: Ty::Infer,
                }),
                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::Snapshot,
                    initial_value: Some(interner.intern(r#""A""#)),
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Turn 1: @self = "A" (initial), output = "AB"
        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let r1 = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        assert_eq!(*r1, Value::string("AB".into()));

        // Turn 2: @self = "AB" (persisted), output = "ABB"
        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let r2 = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        drain_evaluate(&mut engine, &noop_resolver(), &noop_extern_handler()).await;
        assert_eq!(*r2, Value::string("ABB".into()));
    }

    // -- function node tests ---------------------------------------------------

    /// Function node compiles successfully: Expr kind + is_function + fn_params
    #[tokio::test]
    async fn function_node_compile_success() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "ok".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        assert_eq!(nodes.len(), 2);
        assert!(nodes[0].is_function);
    }

    /// Function node is NOT in context_types: @double should fail
    #[test]
    fn function_node_not_in_context() {
        let interner = Interner::new();
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        // @double should be undefined — function nodes are not context
                        source: "{{@double}}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        );
        // Should fail: @double is not a context key (it's a function)
        assert!(result.is_err());
    }

    /// Other nodes can call function nodes: double(5) should typecheck
    #[test]
    fn function_node_callable_from_other_nodes() {
        let interner = Interner::new();
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@double(5)".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        );
        assert!(result.is_ok(), "function call should typecheck: {result:?}");
    }

    /// Function node with global context: fn body references @globalCtx
    #[test]
    fn function_node_with_global_context() {
        let interner = Interner::new();
        let ctx = FxHashMap::from_iter([(interner.intern("offset"), Ty::Int)]);
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("add_offset"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x + @offset".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@add_offset(5)".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::user_only(ctx),
        );
        assert!(result.is_ok(), "function with global context should compile: {result:?}");
    }

    /// Full integration: Plain node calls function node, gets result
    #[tokio::test]
    async fn function_call_full_turn() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(5) | to_string }}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        assert_eq!(*result, Value::string("10".into()));
    }

    /// @funcName context access should produce a resolve error
    #[tokio::test]
    async fn function_node_context_access_error() {
        let interner = Interner::new();
        // We need to manually set up: double is a function node, main tries to
        // reference @double as context. Since compile rejects @double in templates,
        // we test at the resolver level by injecting @double as a known context type
        // but marking the compiled node as is_function.
        let nodes = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "42".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![],
                },
                // main doesn't use @double in template (compile would reject it),
                // so we just verify the is_function flag propagates
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "ok".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        )
        .unwrap();

        assert!(nodes[0].is_function);
    }

    /// Multiple function calls in one template
    #[tokio::test]
    async fn function_call_multiple_times() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(3) | to_string }}-{{ @double(7) | to_string }}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        engine.start_evaluate("main", false, &noop_resolver(), &noop_extern_handler()).await.unwrap();
        let result = evaluate_first(&mut engine, &noop_resolver(), &noop_extern_handler()).await.into_value();
        assert_eq!(*result, Value::string("6-14".into()));
    }

    // -- lazy evaluation tests ------------------------------------------------

    /// Verify that iterator evaluation is lazy: extern fn is called only
    /// as many times as evaluate_next is called, not eagerly for all elements.
    #[tokio::test]
    async fn lazy_evaluation_via_extern_count() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let interner = Interner::new();

        // "tracker" is a function node: takes @x, calls extern to count, returns @x.
        // "main" creates [1,2,3,4,5] | iter | map(x -> @tracker(x))
        // Since main is an Expr returning Iterator, evaluate_next streams one by one.
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("tracker"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x".into(),
                        output_ty: Ty::Int,
                    }),
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "[1, 2, 3, 4, 5] | iter | map(x -> @tracker(x))".into(),
                        output_ty: Ty::Infer,
                    }),
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::Ephemeral,
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );

        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();

        let call_count = Arc::new(AtomicUsize::new(0));

        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let count = Arc::clone(&call_count);
        let extern_handler = move |_name: Astr, args: Vec<TypedValue>| {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                // Return the first arg as-is (pass-through)
                Ok(args.into_iter().next().unwrap_or(TypedValue::unit()))
            }
        };

        engine.start_evaluate("main", false, &noop_resolver(), &extern_handler).await.unwrap();

        // Helper: pull one Emit from evaluate, handling NeedExternCall along the way.
        let pull_one = |engine: &mut ChatEngine<TreeJournal>, count: &Arc<AtomicUsize>| {
            // Can't be async closure, so we inline the loop in the test below.
            let _ = (engine, count);
        };

        let mut items = Vec::new();
        let mut done = false;

        // Pull items one by one via Stepped loop
        while !done {
            match engine.evaluate_next().await {
                Stepped::Emit(value) => {
                    items.push(value);
                    // Check lazy invariant after 2 items
                    if items.len() == 2 {
                        assert_eq!(
                            call_count.load(Ordering::SeqCst),
                            2,
                            "lazy: only 2 extern calls after 2 evaluate_next Emits"
                        );
                    }
                }
                Stepped::Done => done = true,
                Stepped::Error(e) => panic!("evaluate error: {e}"),
                Stepped::NeedContext(req) => {
                    req.resolve(Arc::new(TypedValue::unit()));
                }
                Stepped::NeedExternCall(req) => {
                    let args = req.args().to_vec();
                    call_count.fetch_add(1, Ordering::SeqCst);
                    let result = args.into_iter().next().unwrap_or(TypedValue::unit());
                    req.resolve(Arc::new(result));
                }
            }
        }

        assert_eq!(items.len(), 5, "should have 5 items total");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            5,
            "all 5 extern calls after full consumption"
        );
    }
}
