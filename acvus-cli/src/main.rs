mod project;

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process;

use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    build_dag, build_request, compile_node, output_to_value, parse_response, ApiKind,
    CompiledBlock, CompiledMessage, Executor, Fetch, HashMapStorage, HttpRequest, Message,
    ModelResponse, NodeSpec, Output, ProviderConfig, Storage, ToolSpec,
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

async fn render_block(
    block: &CompiledBlock,
    all_context_keys: &HashSet<String>,
    storage: &HashMapStorage,
) -> String {
    let context_values: HashMap<String, Value> = all_context_keys
        .iter()
        .filter_map(|k| storage.get(k).map(|v| (k.clone(), output_to_value(&v))))
        .collect();
    let interp = Interpreter::new(block.module.clone(), ExternFnRegistry::new());
    interp.execute_to_string(context_values).await
}

/// Expand an iterator entry into messages.
///
/// If the iterator has a template, each item's `@type` and `@text` are injected
/// as context and rendered through the template. Otherwise items pass through as-is.
async fn expand_iterator(
    key: &str,
    block: Option<&CompiledBlock>,
    storage: &HashMapStorage,
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
            let mut context_values: HashMap<String, Value> = HashMap::new();
            context_values.insert("type".into(), Value::String(role.to_string()));
            context_values.insert("text".into(), Value::String(text.to_string()));
            let interp = Interpreter::new(
                block.module.clone(),
                ExternFnRegistry::new(),
            );
            let rendered = interp.execute_to_string(context_values).await;
            messages.push(Message::text(role, rendered));
        } else {
            messages.push(Message::text(role, text));
        }
    }
    messages
}

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

    // Build context types: TOML context values + node names as String
    let mut context_types: HashMap<String, Ty> = spec
        .context
        .iter()
        .map(|(k, v)| (k.clone(), toml_to_ty(v)))
        .collect();

    // First pass: collect node names into context_types
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
        context_types.insert(node_spec.name.clone(), Ty::String);
        node_specs.push(node_spec);
    }

    // Compile nodes
    let registry = ExternRegistry::new();
    let mut compiled_nodes = Vec::new();

    for node_spec in &node_specs {
        match compile_node(node_spec, &project_dir, &context_types, &registry) {
            Ok(node) => compiled_nodes.push(node),
            Err(errors) => {
                for e in &errors {
                    eprintln!("compile error: {e}");
                }
                process::exit(1);
            }
        }
    }

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

        // Render static prefix once
        let mut static_messages: Vec<Message> = Vec::new();
        for msg in &node.messages[..first_iter] {
            if let CompiledMessage::Block(block) = msg {
                let text = render_block(block, &node.all_context_keys, &storage).await;
                static_messages.push(Message::text(&block.role, text));
            }
        }

        let dynamic_msgs = &node.messages[first_iter..];

        // Auto-detect per-turn keys: context keys not in initial storage, not node names
        let node_names: HashSet<&str> =
            compiled_nodes.iter().map(|n| n.name.as_str()).collect();
        let per_turn_keys: Vec<String> = node
            .all_context_keys
            .iter()
            .filter(|k| !node_names.contains(k.as_str()) && storage.get(k).is_none())
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
                            expand_iterator(key, block.as_ref(), &storage).await;
                        messages.extend(expanded);
                    }
                    CompiledMessage::Block(block) => {
                        let text =
                            render_block(block, &node.all_context_keys, &storage).await;
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
