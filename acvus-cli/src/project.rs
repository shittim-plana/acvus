use std::collections::HashMap;

use acvus_mir::ty::Ty;
use acvus_orchestration::Output;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ProjectSpec {
    pub name: String,
    #[serde(default = "default_fuel_limit")]
    pub fuel_limit: u64,
    pub nodes: Vec<String>,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub context: toml::Table,
}

fn default_fuel_limit() -> u64 {
    50
}

#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    pub api: String,
    pub endpoint: String,
    pub api_key_env: String,
}

/// Check if a TOML string value is a type declaration (e.g. "string", "int").
pub fn is_type_decl(value: &toml::Value) -> bool {
    match value {
        toml::Value::String(s) => {
            matches!(s.as_str(), "string" | "int" | "float" | "bool")
        }
        _ => false,
    }
}

/// Resolve a `Ty` from a TOML value.
///
/// If the value is a type name string ("string", "int", "float", "bool"),
/// returns that type directly. Otherwise infers from the value.
pub fn toml_to_ty(value: &toml::Value) -> Ty {
    match value {
        toml::Value::String(s) => match s.as_str() {
            "string" => Ty::String,
            "int" => Ty::Int,
            "float" => Ty::Float,
            "bool" => Ty::Bool,
            _ => Ty::String,
        },
        toml::Value::Integer(_) => Ty::Int,
        toml::Value::Float(_) => Ty::Float,
        toml::Value::Boolean(_) => Ty::Bool,
        toml::Value::Array(arr) => {
            let elem_ty = arr.first().map(toml_to_ty).unwrap_or(Ty::Unit);
            Ty::List(Box::new(elem_ty))
        }
        toml::Value::Table(table) => {
            let fields = table.iter().map(|(k, v)| (k.clone(), toml_to_ty(v))).collect();
            Ty::Object(fields)
        }
        toml::Value::Datetime(_) => Ty::String,
    }
}

/// Convert a TOML value to an `Output`.
pub fn toml_to_output(value: &toml::Value) -> Output {
    match value {
        toml::Value::String(s) => Output::Text(s.clone()),
        _ => Output::Json(toml_to_json(value)),
    }
}

fn toml_to_json(value: &toml::Value) -> serde_json::Value {
    match value {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::json!(*i),
        toml::Value::Float(f) => serde_json::json!(*f),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_to_json).collect())
        }
        toml::Value::Table(table) => {
            let map = table.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn toml_to_ty_primitives() {
        assert_eq!(toml_to_ty(&toml::Value::String("hi".into())), Ty::String);
        assert_eq!(toml_to_ty(&toml::Value::Integer(42)), Ty::Int);
        assert_eq!(toml_to_ty(&toml::Value::Float(3.14)), Ty::Float);
        assert_eq!(toml_to_ty(&toml::Value::Boolean(true)), Ty::Bool);
    }

    #[test]
    fn toml_to_ty_array() {
        let arr = toml::Value::Array(vec![toml::Value::Integer(1), toml::Value::Integer(2)]);
        assert_eq!(toml_to_ty(&arr), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn toml_to_ty_empty_array() {
        let arr = toml::Value::Array(vec![]);
        assert_eq!(toml_to_ty(&arr), Ty::List(Box::new(Ty::Unit)));
    }

    #[test]
    fn toml_to_ty_table() {
        let mut table = toml::Table::new();
        table.insert("name".into(), toml::Value::String("alice".into()));
        table.insert("age".into(), toml::Value::Integer(30));
        let ty = toml_to_ty(&toml::Value::Table(table));
        let expected = Ty::Object(BTreeMap::from([
            ("age".into(), Ty::Int),
            ("name".into(), Ty::String),
        ]));
        assert_eq!(ty, expected);
    }

    #[test]
    fn toml_to_output_string() {
        let out = toml_to_output(&toml::Value::String("hello".into()));
        assert!(matches!(out, Output::Text(ref s) if s == "hello"));
    }

    #[test]
    fn toml_to_output_int() {
        let out = toml_to_output(&toml::Value::Integer(42));
        assert!(matches!(out, Output::Json(ref v) if v == &serde_json::json!(42)));
    }

    #[test]
    fn toml_to_output_table() {
        let mut table = toml::Table::new();
        table.insert("x".into(), toml::Value::Integer(1));
        let out = toml_to_output(&toml::Value::Table(table));
        assert!(matches!(out, Output::Json(ref v) if v == &serde_json::json!({"x": 1})));
    }

    #[test]
    fn parse_project_spec() {
        let toml_str = r#"
name = "test-pipeline"
fuel_limit = 10
nodes = ["a.toml", "b.toml"]

[providers.openai]
api = "openai"
endpoint = "https://api.openai.com"
api_key_env = "OPENAI_API_KEY"

[context]
topic = "Rust"
count = 5
"#;
        let spec: ProjectSpec = toml::from_str(toml_str).unwrap();
        assert_eq!(spec.name, "test-pipeline");
        assert_eq!(spec.fuel_limit, 10);
        assert_eq!(spec.nodes.len(), 2);
        assert!(spec.providers.contains_key("openai"));
        assert_eq!(spec.context.get("topic").unwrap().as_str(), Some("Rust"));
    }

    #[test]
    fn parse_project_spec_defaults() {
        let toml_str = r#"
name = "minimal"
nodes = []
"#;
        let spec: ProjectSpec = toml::from_str(toml_str).unwrap();
        assert_eq!(spec.fuel_limit, 50);
        assert!(spec.providers.is_empty());
        assert!(spec.context.is_empty());
    }
}
