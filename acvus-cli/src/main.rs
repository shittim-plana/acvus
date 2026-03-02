mod project;
mod provider;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_orchestration::{
    build_dag, compile_node, Executor, HashMapStorage, NodeSpec, Output, Storage,
};

use project::{toml_to_output, toml_to_ty, ProjectSpec};
use provider::HttpFetch;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: acvus-cli <project-dir>");
        process::exit(1);
    }

    let project_dir = PathBuf::from(&args[1]);
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
    }

    // Load and compile nodes
    let registry = ExternRegistry::new();
    let mut compiled_nodes = Vec::new();

    for node_file in &spec.nodes {
        let node_src = std::fs::read_to_string(project_dir.join(node_file)).unwrap_or_else(|e| {
            eprintln!("failed to read {node_file}: {e}");
            process::exit(1);
        });
        let node_spec: NodeSpec = toml::from_str(&node_src).unwrap_or_else(|e| {
            eprintln!("failed to parse {node_file}: {e}");
            process::exit(1);
        });

        match compile_node(&node_spec, &project_dir, &context_types, &registry) {
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

    // Seed storage with context values
    let mut storage = HashMapStorage::new();
    for (k, v) in &spec.context {
        storage.set(k.clone(), toml_to_output(v));
    }

    // Build HttpFetch from provider configs
    let mut provider_map = HashMap::new();
    for (name, config) in &spec.providers {
        let api_key = std::env::var(&config.api_key_env).unwrap_or_else(|_| {
            eprintln!(
                "environment variable {} not set (provider: {name})",
                config.api_key_env
            );
            process::exit(1);
        });
        provider_map.insert(name.clone(), (config.api.clone(), config.endpoint.clone(), api_key));
    }
    let fetch = HttpFetch::new(provider_map);

    // Run
    let executor = Executor::new(
        compiled_nodes,
        dag,
        storage,
        fetch,
        registry,
        spec.fuel_limit,
    );

    match executor.run().await {
        Ok(storage) => {
            // Print all node outputs
            for node_file in &spec.nodes {
                let node_src = std::fs::read_to_string(project_dir.join(node_file)).ok();
                if let Some(src) = node_src {
                    if let Ok(node_spec) = toml::from_str::<NodeSpec>(&src) {
                        if let Some(output) = storage.get(&node_spec.name) {
                            println!("[{}]", node_spec.name);
                            match output {
                                Output::Text(t) => println!("{t}"),
                                Output::Json(v) => println!("{v}"),
                                Output::Image(_) => println!("<image>"),
                            }
                            println!();
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("execution error: {e}");
            process::exit(1);
        }
    }
}
