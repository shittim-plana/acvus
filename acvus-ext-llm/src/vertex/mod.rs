//! Google Vertex AI provider — ExternFn handler for `generateContent`.
//!
//! Vertex AI serves the same Gemini request/response schema as the
//! Generative Language API, so this module reuses `crate::google::schema`.
//! Differences from the `google` provider:
//! - URL: `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent`
//!   (host is `aiplatform.googleapis.com` when region is `"global"`)
//! - Auth: OAuth2 access token in `Authorization: Bearer` header, not an
//!   API key in the URL. Token acquisition/refresh is the caller's concern
//!   (e.g. pomollu-tauri's PKCE flow) — this handler only consumes a token.

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::ty::{Effect, Hint, ParamTerm, Poly, Ty, TyTerm, lift_effect_to_poly, lift_to_poly};
use acvus_utils::Interner;

use crate::extract::{obj_get_decimal, obj_get_str, obj_get_u32, split_system, values_to_messages};
use crate::google::schema;
use crate::google::{convert_message, parse_response};
use crate::http::{Fetch, HttpRequest};
use crate::message::{Content, ModelResponse};

/// Build the Vertex AI `generateContent` endpoint URL.
///
/// `region == "global"` uses the global host without a region prefix.
pub fn build_endpoint(project_id: &str, region: &str, model: &str) -> String {
    let host = if region == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{region}-aiplatform.googleapis.com")
    };
    format!(
        "https://{host}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:generateContent"
    )
}

// ── Value helpers ───────────────────────────────────────────────────

/// Build the response `Value::Object` from a `ModelResponse`.
///
/// Same single-message shape as the `google` provider: `{role, content, content_type}`.
fn response_to_value(resp: &ModelResponse, interner: &Interner) -> Value {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");

    let (role, text) = match resp {
        ModelResponse::Content(parts) => parts
            .first()
            .map(|item| {
                let text = match &item.content {
                    Content::Text(t) => t.clone(),
                    Content::Blob { data, .. } => data.clone(),
                };
                (item.role.clone(), text)
            })
            .unwrap_or_else(|| ("model".to_string(), String::new())),
        ModelResponse::ToolCalls(_) => ("model".to_string(), String::new()),
    };

    Value::object(
        [
            (role_key, Value::string(role)),
            (content_key, Value::string(text)),
            (content_type_key, Value::string("text")),
        ]
        .into_iter()
        .collect(),
    )
}

// ── Registry ────────────────────────────────────────────────────────

/// Create an `ExternRegistry` for the Vertex AI chat handler.
///
/// The registered function `vertex_llm` takes `(messages, config)` where:
/// - `messages`: list of objects with `{role, content}` fields
/// - `config`: object with `{access_token, project_id, region, model,
///   temperature?, top_p?, top_k?, max_tokens?}`
///
/// `access_token` is a valid (non-expired) OAuth2 access token with the
/// `cloud-platform` scope. Refreshing expired tokens happens outside this
/// handler — pass a fresh token per call.
pub fn vertex_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let role_key = interner.intern("role");
        let content_key = interner.intern("content");
        let content_type_key = interner.intern("content_type");
        let access_token_key = interner.intern("access_token");
        let project_id_key = interner.intern("project_id");
        let region_key = interner.intern("region");
        let model_key = interner.intern("model");
        let temperature_key = interner.intern("temperature");
        let top_p_key = interner.intern("top_p");
        let top_k_key = interner.intern("top_k");
        let max_tokens_key = interner.intern("max_tokens");

        let msg_ty = Ty::Object(
            [
                (role_key, Ty::String),
                (content_key, Ty::String),
                (content_type_key, Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let config_ty = Ty::Object(
            [
                (access_token_key, Ty::String),
                (project_id_key, Ty::String),
                (region_key, Ty::String),
                (model_key, Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let fetch = Arc::clone(&fetch);

        let params = vec![
            Ty::List(Box::new(Ty::Object(
                [(role_key, Ty::String), (content_key, Ty::String)]
                    .into_iter()
                    .collect(),
            ))),
            config_ty,
        ];
        let named: Vec<ParamTerm<Poly>> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty))
            })
            .collect();
        let ty = TyTerm::Fn {
            params: named,
            ret: Box::new(lift_to_poly(&msg_ty)),
            captures: vec![],
            effect: lift_effect_to_poly(&Effect::pure()),
            hint: Some(Hint::Io),
        };

        vec![
            ExternFnBuilder::new("vertex_llm", ty).handler_async(
                move |interner: Interner,
                      (messages, config): (Value, Value),
                      Uses(()): Uses<()>| {
                    let fetch = Arc::clone(&fetch);
                    async move {
                        let messages_list = match &messages {
                            Value::List(l) => l.as_slice(),
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "vertex_llm: expected List for messages, got {:?}",
                                    other.kind()
                                )));
                            }
                        };
                        let msgs = values_to_messages(messages_list, &interner, "vertex_llm")?;
                        let (system, rest) = split_system(&msgs);

                        let config_obj = match &config {
                            Value::Object(o) => o,
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "vertex_llm: expected Object for config, got {:?}",
                                    other.kind()
                                )));
                            }
                        };

                        let access_token =
                            obj_get_str(config_obj, access_token_key).ok_or_else(|| {
                                RuntimeError::fetch("vertex_llm: missing 'access_token' in config")
                            })?;
                        let project_id =
                            obj_get_str(config_obj, project_id_key).ok_or_else(|| {
                                RuntimeError::fetch("vertex_llm: missing 'project_id' in config")
                            })?;
                        let region = obj_get_str(config_obj, region_key).ok_or_else(|| {
                            RuntimeError::fetch("vertex_llm: missing 'region' in config")
                        })?;
                        let model = obj_get_str(config_obj, model_key).ok_or_else(|| {
                            RuntimeError::fetch("vertex_llm: missing 'model' in config")
                        })?;
                        let temperature = obj_get_decimal(config_obj, temperature_key);
                        let top_p = obj_get_decimal(config_obj, top_p_key);
                        let top_k = obj_get_u32(config_obj, top_k_key);
                        let max_tokens = obj_get_u32(config_obj, max_tokens_key);

                        let request_body = schema::Request {
                            contents: rest.iter().map(|m| convert_message(m)).collect(),
                            system_instruction: system.map(|s| schema::SystemInstruction {
                                parts: vec![schema::TextPart { text: s }],
                            }),
                            tools: None,
                            generation_config: Some(schema::GenerationConfig {
                                temperature,
                                top_p,
                                top_k,
                                max_output_tokens: max_tokens,
                                thinking_config: None,
                            }),
                        };

                        let url = build_endpoint(&project_id, &region, &model);

                        let body = serde_json::to_value(&request_body).map_err(|e| {
                            RuntimeError::fetch(format!("vertex_llm: serialization failed: {e}"))
                        })?;

                        let http_request = HttpRequest {
                            url,
                            headers: vec![
                                (
                                    "Authorization".into(),
                                    format!("Bearer {access_token}"),
                                ),
                                ("Content-Type".into(), "application/json".into()),
                            ],
                            body,
                        };

                        let response_json = fetch
                            .fetch(&http_request)
                            .await
                            .map_err(RuntimeError::fetch)?;

                        let (response, _usage) = parse_response(response_json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;

                        let result = response_to_value(&response, &interner);
                        Ok((result, Defs(())))
                    }
                },
            ),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockFetch {
        response: serde_json::Value,
    }

    impl Fetch for MockFetch {
        async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn endpoint_regional() {
        let url = build_endpoint("my-project", "us-central1", "gemini-2.5-flash");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn endpoint_global_has_no_region_prefix() {
        let url = build_endpoint("proj", "global", "gemini-3.0-flash-preview");
        assert!(url.starts_with("https://aiplatform.googleapis.com/"));
        assert!(url.contains("locations/global"));
    }

    #[test]
    fn registry_produces_function() {
        let fetch = Arc::new(MockFetch {
            response: serde_json::json!({}),
        });
        let interner = Interner::new();
        let registry = vertex_registry(fetch);
        let registered = registry.register(&interner);
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "vertex_llm");
    }
}
