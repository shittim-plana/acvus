mod error;

pub use error::ChatError;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Value};
use acvus_orchestration::{
    CompiledNode, CompiledNodeKind, CompiledStrategy, Fetch, Node, ProviderConfig, ResolveState,
    Resolved, Resolver, State, Storage, build_dag, build_node_table,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ChatEngine<S> {
    nodes: Vec<CompiledNode>,
    node_table: Vec<Arc<dyn Node>>,
    name_to_idx: HashMap<String, usize>,
    extern_fns: ExternFnRegistry,
    pub state: State<S>,
    bind_cache: HashMap<String, Vec<(Value, Arc<Value>)>>,
    entrypoint_idx: usize,
    history_nodes: Vec<String>,
    side_effect_idxs: Vec<usize>,
}

impl<S> ChatEngine<S>
where
    S: Storage,
{
    pub async fn new<F>(
        nodes: Vec<CompiledNode>,
        providers: HashMap<String, ProviderConfig>,
        fetch: F,
        extern_fns: ExternFnRegistry,
        mut storage: S,
        entrypoint: &str,
        side_effects: &[String],
    ) -> Result<Self, ChatError>
    where
        F: Fetch + 'static,
    {
        let name_to_idx: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.clone(), i))
            .collect();

        let entrypoint_idx = *name_to_idx
            .get(entrypoint)
            .ok_or_else(|| ChatError::EntrypointNotFound(entrypoint.to_string()))?;

        // Collect history nodes
        let history_nodes: Vec<String> = nodes
            .iter()
            .filter_map(|n| match &n.strategy {
                CompiledStrategy::History { .. } => Some(n.name.clone()),
                _ => None,
            })
            .collect();

        // Validate: no dependency cycles
        build_dag(&nodes).map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            ChatError::CycleDetected(msg)
        })?;

        // Validate: history nodes must be reachable from entrypoint via BFS
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(entrypoint_idx);
        reachable.insert(entrypoint_idx);
        while let Some(idx) = queue.pop_front() {
            for key in &nodes[idx].all_context_keys {
                if let Some(&dep_idx) = name_to_idx.get(key)
                    && reachable.insert(dep_idx)
                {
                    queue.push_back(dep_idx);
                }
            }
            if let CompiledNodeKind::Llm(llm) = &nodes[idx].kind {
                for tool in &llm.tools {
                    if let Some(&dep_idx) = name_to_idx.get(&tool.node)
                        && reachable.insert(dep_idx)
                    {
                        queue.push_back(dep_idx);
                    }
                }
            }
        }
        for name in &history_nodes {
            if !reachable.contains(&name_to_idx[name.as_str()]) {
                return Err(ChatError::HistoryNodeUnreachable(name.clone()));
            }
        }

        // Seed context metadata
        let mut context_obj: BTreeMap<String, Value> = BTreeMap::new();
        for node in &nodes {
            if let CompiledNodeKind::Llm(llm) = &node.kind {
                context_obj.insert(
                    node.name.clone(),
                    Value::Object(BTreeMap::from([
                        ("model".into(), Value::String(llm.model.clone())),
                        ("provider".into(), Value::String(llm.provider.clone())),
                    ])),
                );
            }
        }
        if !context_obj.is_empty() {
            storage.set("context".into(), Value::Object(context_obj));
        }


        // Resolve side_effect node indices
        let side_effect_idxs: Vec<usize> = side_effects
            .iter()
            .filter_map(|name| name_to_idx.get(name.as_str()).copied())
            .collect();

        // Build node table — one match, uniform Arc<dyn Node> from here
        let node_table = build_node_table(&nodes, &providers, Arc::new(fetch), &extern_fns);

        Ok(Self {
            nodes,
            node_table,
            name_to_idx,
            extern_fns,
            state: State::new(storage, 0),
            bind_cache: HashMap::new(),
            entrypoint_idx,
            history_nodes,
            side_effect_idxs,
        })
    }

    pub async fn turn<R>(&mut self, resolver: &R) -> Result<Value, ChatError>
    where
        S: Default,
        R: AsyncFn(String) -> Resolved + Sync,
    {
        let entrypoint = &self.nodes[self.entrypoint_idx].name;
        tracing::info!(entrypoint = %entrypoint, "turn start");

        // Ensure @turn exists; compute next index
        let turn_index = if let Some(arc) = self.state.storage.get("turn") {
            let Value::Object(ref turn) = *arc else {
                panic!("@turn must be an Object");
            };
            let Value::Int(i) = turn.get("index").expect("@turn.index missing") else {
                panic!("@turn.index must be Int");
            };
            *i + 1
        } else {
            self.state.storage.set(
                "turn".into(),
                Value::Object(BTreeMap::from([
                    ("index".into(), Value::Int(0)),
                    ("history".into(), Value::List(Vec::new())),
                ])),
            );
            0
        };

        // Always-strategy nodes re-resolve every turn
        for node in &self.nodes {
            if matches!(node.strategy, CompiledStrategy::Always) {
                self.state.storage.remove(&node.name);
            }
        }

        // Build ResolveState for this turn
        let mut rs = ResolveState {
            storage: std::mem::take(&mut self.state.storage),
            turn_context: HashMap::new(),
            bind_cache: std::mem::take(&mut self.bind_cache),
            history_entries: BTreeMap::new(),
        };

        let ctx = Resolver {
            nodes: &self.nodes,
            node_table: &self.node_table,
            name_to_idx: &self.name_to_idx,
            extern_fns: &self.extern_fns,
            resolver,
        };

        ctx.resolve_node(self.entrypoint_idx, &mut rs, HashMap::new())
            .await
            .map_err(|e| ChatError::Resolve(e.to_string()))?;

        // Merge Always node results from turn_context into storage
        for node in &self.nodes {
            if matches!(node.strategy, CompiledStrategy::Always)
                && let Some(v) = rs.turn_context.get(&node.name)
            {
                rs.storage.set(node.name.clone(), Value::clone(v));
            }
        }

        // Flush history + update turn index
        {
            let mut turn_val = rs.storage.get("turn")
                .map(|arc| Value::clone(&arc))
                .unwrap_or_else(|| Value::Object(BTreeMap::new()));

            if let Value::Object(ref mut turn) = turn_val {
                turn.insert("index".into(), Value::Int(turn_index));
                if !rs.history_entries.is_empty() {
                    let history = turn.entry("history".into())
                        .or_insert_with(|| Value::List(Vec::new()));
                    if let Value::List(list) = history {
                        list.push(Value::Object(std::mem::take(&mut rs.history_entries)));
                    }
                }
            }
            rs.storage.set("turn".into(), turn_val);
        }

        self.state.turn = turn_index as usize;

        // Resolve side-effect nodes after history flush + turn increment
        for &idx in &self.side_effect_idxs {
            ctx.resolve_node(idx, &mut rs, HashMap::new())
                .await
                .map_err(|e| ChatError::Resolve(e.to_string()))?;
        }

        // Restore persistent state
        self.state.storage = rs.storage;
        self.bind_cache = rs.bind_cache;

        tracing::info!(turn = self.state.turn, "turn complete");

        let name = &self.nodes[self.entrypoint_idx].name;
        let result = self
            .state
            .storage
            .get(name)
            .ok_or_else(|| ChatError::UnresolvedContext(name.clone()))?;
        Ok(Value::clone(&result))
    }

    pub fn extern_fns(&self) -> &ExternFnRegistry {
        &self.extern_fns
    }

    pub fn history_len(&self) -> usize {
        self.state.turn
    }

    pub fn history_pop(&mut self) {
        if self.state.turn == 0 {
            return;
        }
        self.state.turn -= 1;
        if let Some(arc) = self.state.storage.get("turn") {
            let mut turn_val = Value::clone(&arc);
            if let Value::Object(ref mut turn) = turn_val
                && let Some(Value::List(history)) = turn.get_mut("history")
            {
                history.pop();
            }
            self.state.storage.set("turn".into(), turn_val);
        }
    }

    pub async fn re_execute<R>(&mut self, index: usize, resolver: &R) -> Result<Value, ChatError>
    where
        S: Default,
        R: AsyncFn(String) -> Resolved + Sync,
    {
        assert!(
            index <= self.state.turn,
            "re_execute index out of bounds: {index} > {}",
            self.state.turn,
        );
        if let Some(arc) = self.state.storage.get("turn") {
            let mut turn_val = Value::clone(&arc);
            if let Value::Object(ref mut turn) = turn_val
                && let Some(Value::List(history)) = turn.get_mut("history")
            {
                history.truncate(index);
            }
            self.state.storage.set("turn".into(), turn_val);
        }
        self.state.turn = index;
        self.turn(resolver).await
    }
}

/// Drive an interpreter coroutine to a single value, resolving contexts from storage + local.
async fn drive_script<S>(
    coroutine: &mut acvus_coroutine::Coroutine<Value, acvus_interpreter::RuntimeError>,
    mut key: acvus_coroutine::ResumeKey<Value>,
    storage: &S,
    local: &HashMap<String, Arc<Value>>,
) -> Value
where
    S: Storage,
{
    loop {
        match coroutine.resume(key).await {
            acvus_coroutine::Stepped::Emit(emit) => {
                let (value, _) = emit.into_parts();
                return value;
            }
            acvus_coroutine::Stepped::NeedContext(need) => {
                let name = need.name().to_string();
                if let Some(arc) = local.get(&name) {
                    key = need.into_key(Arc::clone(arc));
                } else if let Some(arc) = storage.get(&name) {
                    key = need.into_key(arc);
                } else {
                    return Value::Unit;
                }
            }
            acvus_coroutine::Stepped::Done => return Value::Unit,
            acvus_coroutine::Stepped::Error(e) => panic!("runtime error: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::extern_module::ExternRegistry;
    use acvus_orchestration::{
        ApiKind, GenerationParams, HashMapStorage, HttpRequest, LlmSpec, MaxTokens, MessageSpec,
        NodeKind, NodeSpec, PlainSpec, SelfSpec, Strategy, ToolBinding, compile_nodes,
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

    fn noop_resolver() -> impl AsyncFn(String) -> Resolved + Sync {
        |_: String| async { Resolved::Once(Value::Unit) }
    }

    fn compile_test_nodes(specs: &[NodeSpec]) -> Vec<CompiledNode> {
        compile_nodes(specs, &HashMap::new(), &ExternRegistry::default()).unwrap()
    }

    fn plain_self_spec() -> SelfSpec {
        SelfSpec {
            initial_value: None,
        }
    }

    fn llm_self_spec() -> SelfSpec {
        SelfSpec {
            initial_value: None,
        }
    }

    #[tokio::test]
    async fn new_valid_entrypoint() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain(PlainSpec {
                source: "hello".into(),
            }),
            self_spec: plain_self_spec(),
            strategy: Strategy::default(),
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn new_invalid_entrypoint() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain(PlainSpec {
                source: "hello".into(),
            }),
            self_spec: plain_self_spec(),
            strategy: Strategy::default(),
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "nonexistent",
            &[],
        )
        .await;
        assert!(matches!(result, Err(ChatError::EntrypointNotFound(_))));
    }

    #[tokio::test]
    async fn turn_plain_node() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain(PlainSpec {
                source: "hello world".into(),
            }),
            self_spec: plain_self_spec(),
            strategy: Strategy::default(),
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(result, Value::String("hello world".into()));
    }

    #[tokio::test]
    async fn turn_llm_text_response() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Llm(LlmSpec {
                api: ApiKind::OpenAI,
                provider: "test".into(),
                model: "gpt-test".into(),
                messages: vec![MessageSpec::Block {
                    role: "user".into(),
                    source: "hi".into(),
                }],
                tools: vec![],
                generation: GenerationParams::default(),
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            self_spec: llm_self_spec(),
            strategy: Strategy::default(),
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![openai_text_response("hello from LLM")]);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        // stored value = raw output (List of messages)
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        assert_eq!(msgs.len(), 1);
        let Value::Object(msg) = &msgs[0] else {
            panic!("expected Object");
        };
        assert_eq!(msg.get("content"), Some(&Value::String("hello from LLM".into())));
    }

    #[tokio::test]
    async fn turn_tool_call_round_trip() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain(PlainSpec {
                    source: "tool result text".into(),
                }),
                strategy: Strategy::default(),
                self_spec: plain_self_spec(),
                retry: 0,
                assert: None,
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm(LlmSpec {
                    api: ApiKind::OpenAI,
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: "user".into(),
                        source: "use the tool".into(),
                    }],
                    tools: vec![ToolBinding {
                        name: "my_tool".into(),
                        description: String::new(),
                        node: "tool_target".into(),
                        params: HashMap::new(),
                    }],
                    generation: GenerationParams::default(),
                    cache_key: None,
                    max_tokens: MaxTokens::default(),
                }),
                strategy: Strategy::default(),
                self_spec: llm_self_spec(),
                retry: 0,
                assert: None,
            },
        ]);
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![
            openai_tool_call_response(vec![("call_1", "my_tool", serde_json::json!({}))]),
            openai_text_response("final answer"),
        ]);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        // stored value = raw output (List of messages)
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        let Value::Object(msg) = msgs.last().unwrap() else {
            panic!("expected Object");
        };
        assert_eq!(msg.get("content"), Some(&Value::String("final answer".into())));
    }

    // -- regression tests -------------------------------------------------------

    /// #6: initial_value must be evaluated on first run (not Unit).
    /// OncePerTurn: first turn uses initial_value as @self, subsequent turns use persisted @self.
    /// With self_bind removed, accumulation is done in the node body template using @self.
    #[tokio::test]
    async fn initial_value_evaluated_on_first_run() {
        // Template uses @self (previous) to accumulate.
        // initial_value = "A".
        // Turn 1: @self = "A" (initial), template = "{{@self}}B" → "AB"
        // Turn 2: @self = "AB" (persisted), template = "{{@self}}B" → "ABB"
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain(PlainSpec {
                source: "{{@self}}B".into(),
            }),
            self_spec: SelfSpec {
                initial_value: Some(r#""A""#.into()),
            },
            strategy: Strategy::OncePerTurn,
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let r1 = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(r1, Value::String("AB".into()));

        let r2 = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(r2, Value::String("ABB".into()));
    }

    /// #7: Always nodes must re-execute every invocation, not just once per turn.
    #[tokio::test]
    async fn always_node_re_executes_every_reference() {
        // Two plain nodes: "counter" (Always) referenced by "main".
        // "main" references @counter twice via template.
        // Each @counter resolve should re-execute the node.
        // counter template = "x", self_bind = @raw.
        // main template = "{{@counter}}{{@counter}}" → "xx"
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "counter".into(),
                kind: NodeKind::Plain(PlainSpec { source: "x".into() }),
                self_spec: plain_self_spec(),
                strategy: Strategy::Always,
                retry: 0,
                assert: None,
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Plain(PlainSpec {
                    source: "{{@counter}}{{@counter}}".into(),
                }),
                self_spec: plain_self_spec(),
                strategy: Strategy::default(),
                retry: 0,
                assert: None,
            },
        ]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(result, Value::String("xx".into()));
    }

    /// #3 (double-prompt): Turn-resolved external values must be cached within a turn.
    #[tokio::test]
    async fn turn_resolver_caches_within_turn() {
        use acvus_mir::ty::Ty;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // main template references @input twice.
        // External resolver should be called only once for @input per turn.
        let mut ctx = HashMap::new();
        ctx.insert("input".into(), Ty::String);
        let nodes = compile_nodes(
            &[NodeSpec {
                name: "main".into(),
                kind: NodeKind::Plain(PlainSpec {
                    source: "{{@input}}{{@input}}".into(),
                }),
                self_spec: plain_self_spec(),
                strategy: Strategy::default(),
                retry: 0,
            assert: None,
            }],
            &ctx,
            &ExternRegistry::default(),
        )
        .unwrap();
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let resolver = move |_: String| {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Resolved::Turn(Value::String("hi".into()))
            }
        };

        let result = engine.turn(&resolver).await.unwrap();
        assert_eq!(result, Value::String("hihi".into()));
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolver should be called once per turn"
        );
    }

    /// #6b (history_bind @self): history_bind must have access to @self without panic.
    #[tokio::test]
    async fn history_bind_accesses_self() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Llm(LlmSpec {
                api: ApiKind::OpenAI,
                provider: "test".into(),
                model: "m".into(),
                messages: vec![MessageSpec::Block {
                    role: "user".into(),
                    source: "hi".into(),
                }],
                tools: vec![],
                generation: GenerationParams::default(),
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            self_spec: llm_self_spec(),
            // history_bind accesses @self (= raw output = List) and extracts content
            strategy: Strategy::History {
                history_bind: r#"@self | map(x -> x.content) | join("")"#.into(),
            },
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![openai_text_response("hello")]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        // Should not panic — @self = raw output = List<{role, content, content_type}>
        let result = engine.turn(&noop_resolver()).await.unwrap();
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        let Value::Object(msg) = &msgs[0] else {
            panic!("expected Object");
        };
        assert_eq!(msg.get("content"), Some(&Value::String("hello".into())));
    }

    /// @self in node body: accumulates across turns.
    /// Uses OncePerTurn so @self persists.
    #[tokio::test]
    async fn node_body_accesses_self() {
        // Template concatenates @self (previous) + "B".
        // Turn 1: @self = initial "A", template = "{{@self}}B" → stored = "AB"
        // Turn 2: @self = "AB", template = "{{@self}}B" → stored = "ABB"
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain(PlainSpec {
                source: "{{@self}}B".into(),
            }),
            self_spec: SelfSpec {
                initial_value: Some(r#""A""#.into()),
            },
            strategy: Strategy::OncePerTurn,
            retry: 0,
            assert: None,
        }]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
            &[],
        )
        .await
        .unwrap();

        // Turn 1: @self = "A" (initial), output = "AB"
        let r1 = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(r1, Value::String("AB".into()));

        // Turn 2: @self = "AB" (persisted), output = "ABB"
        let r2 = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(r2, Value::String("ABB".into()));
    }
}
