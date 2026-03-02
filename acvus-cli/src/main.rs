mod project;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use acvus_interpreter::{ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    build_dag, build_request, compile_nodes, output_to_value, parse_response,
    ApiKind, CompiledBlock, CompiledMessage, Executor, Fetch, HashMapStorage, HttpRequest,
    Message, ModelResponse, NodeSpec, Output, ProviderConfig, Storage, ToolSpec,
};
use futures::future::BoxFuture;

use project::{is_type_decl, toml_to_output, toml_to_ty, ProjectSpec};

#[derive(Clone)]
struct HttpFetch {
    client: reqwest::Client,
}

impl Fetch for HttpFetch {
    fn fetch<'a>(
        &'a self,
        request: &'a HttpRequest,
    ) -> BoxFuture<'a, Result<serde_json::Value, String>> {
        Box::pin(async move {
            let mut builder = self.client.post(&request.url);
            for (k, v) in &request.headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            let resp = builder
                .json(&request.body)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            resp.json().await.map_err(|e| e.to_string())
        })
    }
}

fn print_outputs<S>(nodes: &[acvus_orchestration::CompiledNode], storage: &S)
where
    S: Storage,
{
    for node in nodes {
        if let Some(output) = storage.get(&node.name) {
            println!("[{}]", node.name);
            match output {
                Output::Text(t) => println!("{t}"),
                Output::Json(v) => println!("{v}"),
                Output::Image(_) => println!("<image>"),
            }
            println!();
        }
    }
}

// ---------------------------------------------------------------------------
// Coroutine-driven rendering with demand-driven dependency resolution
// ---------------------------------------------------------------------------

struct RenderCtx<'a> {
    nodes: &'a [acvus_orchestration::CompiledNode],
    name_to_idx: &'a HashMap<&'a str, usize>,
    providers: &'a HashMap<String, ProviderConfig>,
    fetch: &'a HttpFetch,
}

enum BlockDriveResult {
    Done(String),
    NeedContext(NeedContextStepped),
}

/// Drive a coroutine until it completes or needs an unavailable context.
fn drive_block(
    coroutine: &mut acvus_interpreter::Coroutine,
    mut key: ResumeKey,
    output: &mut String,
    storage: &HashMapStorage,
    local: &HashMap<String, Value>,
) -> BlockDriveResult {
    loop {
        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => output.push_str(&s),
                    other => panic!("expected String, got {other:?}"),
                }
                key = next_key;
            }
            Stepped::NeedContext(need) => {
                if !need.bindings().is_empty() {
                    return BlockDriveResult::NeedContext(need);
                }
                let name = need.name().to_string();
                if let Some(value) = local.get(&name) {
                    key = need.into_key(value.clone());
                } else if let Some(out) = storage.get(&name) {
                    key = need.into_key(output_to_value(&out));
                } else {
                    return BlockDriveResult::NeedContext(need);
                }
            }
            Stepped::Done => {
                return BlockDriveResult::Done(std::mem::take(output));
            }
        }
    }
}

/// Render a block, resolving missing dependencies on-demand.
///
/// `local` provides per-invocation context (e.g. iterator @type/@text)
/// that takes priority over storage.
fn render_with_deps<'a>(
    block: &'a CompiledBlock,
    storage: &'a mut HashMapStorage,
    local: HashMap<String, Value>,
    ctx: &'a RenderCtx<'a>,
) -> BoxFuture<'a, String> {
    Box::pin(async move {
        let interp = Interpreter::new(block.module.clone(), ExternFnRegistry::new());
        let (mut coroutine, key) = interp.execute();
        let mut output = String::new();

        let mut result = drive_block(&mut coroutine, key, &mut output, storage, &local);
        loop {
            match result {
                BlockDriveResult::Done(text) => return text,
                BlockDriveResult::NeedContext(need) => {
                    let name = need.name().to_string();
                    let bindings = need.bindings().clone();
                    if !bindings.is_empty() {
                        if let Some(&idx) = ctx.name_to_idx.get(name.as_str()) {
                            resolve_node(idx, storage, ctx, bindings).await;
                        }
                        let value = storage
                            .get(&name)
                            .map(|o| output_to_value(&o))
                            .unwrap_or_else(|| panic!("unresolved context: @{name}"));
                        let key = need.into_key(value);
                        result = drive_block(&mut coroutine, key, &mut output, storage, &local);
                    } else {
                        if let Some(&idx) = ctx.name_to_idx.get(name.as_str()) {
                            if storage.get(&name).is_none() {
                                resolve_node(idx, storage, ctx, HashMap::new()).await;
                            }
                        }
                        let value = storage
                            .get(&name)
                            .map(|o| output_to_value(&o))
                            .unwrap_or_else(|| panic!("unresolved context: @{name}"));
                        let key = need.into_key(value);
                        result = drive_block(&mut coroutine, key, &mut output, storage, &local);
                    }
                }
            }
        }
    })
}

/// Run a dependency node: render all blocks + call model, store result.
///
/// `local` provides per-invocation context bindings (from context call).
fn resolve_node<'a>(
    idx: usize,
    storage: &'a mut HashMapStorage,
    ctx: &'a RenderCtx<'a>,
    local: HashMap<String, Value>,
) -> BoxFuture<'a, ()> {
    Box::pin(async move {
        let node = &ctx.nodes[idx];

        let mut messages = Vec::new();
        for msg in &node.messages {
            let block = match msg {
                CompiledMessage::Block(block) => block,
                CompiledMessage::Iterator { .. } => continue,
            };
            let text = render_with_deps(block, storage, local.clone(), ctx).await;
            messages.push(Message::text(&block.role, text));
        }

        let provider_config = ctx
            .providers
            .get(&node.provider)
            .unwrap_or_else(|| panic!("unknown provider: {}", node.provider))
            .clone();

        let tools: Vec<ToolSpec> = node
            .tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name.clone(),
                description: String::new(),
                params: t.params.clone(),
            })
            .collect();

        let request = build_request(&provider_config, &node.model, &messages, &tools);
        let json = ctx
            .fetch
            .fetch(&request)
            .await
            .unwrap_or_else(|e| panic!("fetch error for node {}: {e}", node.name));
        let response = parse_response(&provider_config.api, &json)
            .unwrap_or_else(|e| panic!("parse error for node {}: {e}", node.name));

        match response {
            ModelResponse::Text(text) => {
                storage.set(node.name.clone(), Output::Text(text));
            }
            ModelResponse::ToolCalls(_) => {
                panic!("tool calls in dependency node {} not supported", node.name);
            }
        }
    })
}

/// Expand an iterator entry into messages.
///
/// If the iterator has a template, each item's `@type` and `@text` are injected
/// as local context and rendered through the template with dep resolution.
async fn expand_iterator(
    key: &str,
    block: Option<&CompiledBlock>,
    storage: &mut HashMapStorage,
    ctx: &RenderCtx<'_>,
) -> Vec<Message> {
    let items = match storage.get(key) {
        Some(Output::Json(serde_json::Value::Array(arr))) => arr,
        _ => return Vec::new(),
    };

    let mut messages = Vec::new();
    for item in &items {
        let role = item["type"].as_str().unwrap_or("user");
        let text = item["text"].as_str().unwrap_or("");

        if let Some(block) = block {
            let local = HashMap::from([
                ("type".into(), Value::String(role.to_string())),
                ("text".into(), Value::String(text.to_string())),
            ]);
            let rendered = render_with_deps(block, storage, local, ctx).await;
            messages.push(Message::text(role, rendered));
        } else {
            messages.push(Message::text(role, text));
        }
    }
    messages
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let interactive = args.iter().any(|a| a == "--interactive");
    let project_dir_arg = args.iter().skip(1).find(|a| !a.starts_with("--"));

    let project_dir = match project_dir_arg {
        Some(dir) => PathBuf::from(dir),
        None => {
            eprintln!("usage: acvus-cli [--interactive] <project-dir>");
            process::exit(1);
        }
    };

    let project_toml = project_dir.join("project.toml");

    let project_src = std::fs::read_to_string(&project_toml).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", project_toml.display());
        process::exit(1);
    });

    let spec: ProjectSpec = toml::from_str(&project_src).unwrap_or_else(|e| {
        eprintln!("failed to parse project.toml: {e}");
        process::exit(1);
    });

    // Build context types from TOML declarations
    let context_types: HashMap<String, Ty> = spec
        .context
        .iter()
        .map(|(k, v)| (k.clone(), toml_to_ty(v)))
        .collect();

    // Load node specs
    let mut node_specs = Vec::new();
    for node_file in &spec.nodes {
        let node_src = std::fs::read_to_string(project_dir.join(node_file)).unwrap_or_else(|e| {
            eprintln!("failed to read {node_file}: {e}");
            process::exit(1);
        });
        let node_spec: NodeSpec = toml::from_str(&node_src).unwrap_or_else(|e| {
            eprintln!("failed to parse {node_file}: {e}");
            process::exit(1);
        });
        node_specs.push(node_spec);
    }

    // Compile nodes (node output types merged internally)
    let registry = ExternRegistry::new();
    let compiled_nodes = match compile_nodes(&node_specs, &project_dir, &context_types, &registry) {
        Ok(nodes) => nodes,
        Err(errors) => {
            for e in &errors {
                eprintln!("compile error: {e}");
            }
            process::exit(1);
        }
    };

    // Build DAG
    let dag = match build_dag(&compiled_nodes) {
        Ok(dag) => dag,
        Err(errors) => {
            for e in &errors {
                eprintln!("dag error: {e}");
            }
            process::exit(1);
        }
    };

    // Seed storage with context values (skip type-only declarations)
    let mut storage = HashMapStorage::new();
    for (k, v) in &spec.context {
        if !is_type_decl(v) {
            storage.set(k.clone(), toml_to_output(v));
        }
    }

    // Build providers from project spec
    let mut providers = HashMap::new();
    for (name, config) in &spec.providers {
        let api_key = std::env::var(&config.api_key_env).unwrap_or_else(|_| {
            eprintln!(
                "environment variable {} not set (provider: {name})",
                config.api_key_env
            );
            process::exit(1);
        });
        let api = match config.api.as_str() {
            "openai" => ApiKind::OpenAI,
            "anthropic" => ApiKind::Anthropic,
            "google" => ApiKind::Google,
            other => {
                eprintln!("unknown api kind: {other}");
                process::exit(1);
            }
        };
        providers.insert(
            name.clone(),
            ProviderConfig { api, endpoint: config.endpoint.clone(), api_key },
        );
    }

    let fetch = HttpFetch { client: reqwest::Client::new() };

    if interactive {
        let name_to_idx: HashMap<&str, usize> = compiled_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.as_str(), i))
            .collect();

        let ctx = RenderCtx {
            nodes: &compiled_nodes,
            name_to_idx: &name_to_idx,
            providers: &providers,
            fetch: &fetch,
        };

        let node = &compiled_nodes[0];

        let has_iterator = node
            .messages
            .iter()
            .any(|m| matches!(m, CompiledMessage::Iterator { .. }));
        if !has_iterator {
            eprintln!("interactive mode requires at least one iterator in messages");
            process::exit(1);
        }

        let provider_config = providers
            .get(&node.provider)
            .unwrap_or_else(|| {
                eprintln!("unknown provider: {}", node.provider);
                process::exit(1);
            })
            .clone();

        let tools: Vec<ToolSpec> = node
            .tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name.clone(),
                description: String::new(),
                params: t.params.clone(),
            })
            .collect();

        // Split messages: static prefix (before first iterator) vs dynamic tail
        let first_iter = node
            .messages
            .iter()
            .position(|m| matches!(m, CompiledMessage::Iterator { .. }))
            .unwrap();

        // Render static prefix once (deps resolved on-demand)
        let mut static_messages: Vec<Message> = Vec::new();
        for msg in &node.messages[..first_iter] {
            if let CompiledMessage::Block(block) = msg {
                let text = render_with_deps(block, &mut storage, HashMap::new(), &ctx).await;
                static_messages.push(Message::text(&block.role, text));
            }
        }

        let dynamic_msgs = &node.messages[first_iter..];

        // Auto-detect per-turn keys: context keys not in initial storage, not node names
        let per_turn_keys: Vec<String> = node
            .all_context_keys
            .iter()
            .filter(|k| !name_to_idx.contains_key(k.as_str()) && storage.get(k).is_none())
            .cloned()
            .collect();

        // Collect iterator keys for history management
        let iterator_keys: Vec<String> = dynamic_msgs
            .iter()
            .filter_map(|m| match m {
                CompiledMessage::Iterator { key, .. } => Some(key.clone()),
                _ => None,
            })
            .collect();

        loop {
            // Prompt for per-turn keys
            for key in &per_turn_keys {
                eprint!("{key}: ");
                std::io::stderr().flush().ok();
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).unwrap() == 0 {
                    return; // EOF
                }
                storage.set(key.clone(), Output::Text(input.trim_end().to_string()));
            }

            // Build messages: static prefix + dynamic expansion
            let mut messages = static_messages.clone();
            let mut new_messages: Vec<Message> = Vec::new();

            for msg in dynamic_msgs {
                match msg {
                    CompiledMessage::Iterator { key, block } => {
                        let expanded =
                            expand_iterator(key, block.as_ref(), &mut storage, &ctx).await;
                        messages.extend(expanded);
                    }
                    CompiledMessage::Block(block) => {
                        let text =
                            render_with_deps(block, &mut storage, HashMap::new(), &ctx).await;
                        let message = Message::text(&block.role, text);
                        messages.push(message.clone());
                        new_messages.push(message);
                    }
                }
            }

            // Call model with full message history
            let request = build_request(&provider_config, &node.model, &messages, &tools);
            let json = fetch.fetch(&request).await.unwrap_or_else(|e| {
                eprintln!("fetch error: {e}");
                process::exit(1);
            });
            let response = parse_response(&provider_config.api, &json).unwrap_or_else(|e| {
                eprintln!("parse error: {e}");
                process::exit(1);
            });

            let response_text = match response {
                ModelResponse::Text(text) => {
                    println!("{text}");
                    text
                }
                ModelResponse::ToolCalls(_) => {
                    eprintln!("tool calls not supported in interactive mode yet");
                    process::exit(1);
                }
            };

            // Append new messages + assistant response to history
            for iter_key in &iterator_keys {
                let mut history = match storage.get(iter_key) {
                    Some(Output::Json(serde_json::Value::Array(arr))) => arr,
                    _ => Vec::new(),
                };
                for msg in &new_messages {
                    history.push(
                        serde_json::json!({"type": msg.role, "text": msg.content}),
                    );
                }
                history
                    .push(serde_json::json!({"type": "assistant", "text": response_text}));
                storage.set(
                    iter_key.clone(),
                    Output::Json(serde_json::Value::Array(history)),
                );
            }

            // Clear per-turn keys for next iteration
            for key in &per_turn_keys {
                storage.remove(key);
            }
        }
    } else {
        let executor = Executor::new(
            compiled_nodes.clone(),
            dag,
            storage,
            fetch,
            providers,
            registry,
            spec.fuel_limit,
        );

        match executor.run().await {
            Ok(storage) => print_outputs(&compiled_nodes, &storage),
            Err(e) => {
                eprintln!("execution error: {e}");
                process::exit(1);
            }
        }
    }
}
