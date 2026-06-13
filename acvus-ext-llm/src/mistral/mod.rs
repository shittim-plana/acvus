//! Mistral AI provider — ExternFn handler for chat completions.
//!
//! Mistral's chat API is OpenAI-compatible, so this module reuses
//! `crate::openai::schema` for request/response shapes. Differences:
//! - default endpoint `https://api.mistral.ai/v1/chat/completions`
//! - optional `reasoning_effort` config field (magistral reasoning models)

use std::sync::Arc;

use acvus_interpreter::{Defs, ExternFnBuilder, ExternRegistry, RuntimeError, Uses, Value};
use acvus_mir::ty::{Effect, Hint, ParamTerm, Poly, Ty, TyTerm, lift_effect_to_poly, lift_to_poly};
use acvus_utils::Interner;

use crate::extract::{obj_get_decimal, obj_get_str, obj_get_u32, values_to_messages};
use crate::http::{Fetch, HttpRequest};
use crate::openai::schema;
use crate::openai::{convert_message, parse_response, response_to_value};

/// Default chat completions endpoint for the hosted Mistral API.
pub const DEFAULT_ENDPOINT: &str = "https://api.mistral.ai/v1/chat/completions";

// ── Registry ────────────────────────────────────────────────────────

/// Create an ExternRegistry for the Mistral chat completion handler.
///
/// The registered function `mistral_chat` takes `(messages, config)` where:
/// - `messages`: list of objects with `{role, content}` fields
/// - `config`: object with `{api_key, model, endpoint?, temperature?, top_p?,
///   max_tokens?, reasoning_effort?}`
///
/// When `endpoint` is absent at runtime, [`DEFAULT_ENDPOINT`] is used.
/// Returns the same `{content, tool_calls, usage}` object shape as `openai_chat`.
pub fn mistral_registry<F: Fetch + Send + Sync + 'static>(fetch: Arc<F>) -> ExternRegistry {
    ExternRegistry::new(move |interner| {
        let endpoint_key = interner.intern("endpoint");
        let api_key_key = interner.intern("api_key");
        let model_key = interner.intern("model");
        let temperature_key = interner.intern("temperature");
        let top_p_key = interner.intern("top_p");
        let max_tokens_key = interner.intern("max_tokens");
        let reasoning_effort_key = interner.intern("reasoning_effort");

        let fetch = Arc::clone(&fetch);

        let input_msg_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let config_ty = Ty::Object(
            [
                (interner.intern("endpoint"), Ty::String),
                (interner.intern("api_key"), Ty::String),
                (interner.intern("model"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let msg_elem_ty = Ty::Object(
            [
                (interner.intern("role"), Ty::String),
                (interner.intern("content"), Ty::String),
                (interner.intern("content_type"), Ty::String),
            ]
            .into_iter()
            .collect(),
        );

        let params = vec![Ty::List(Box::new(input_msg_ty)), config_ty];
        let ret = Ty::List(Box::new(msg_elem_ty));
        let named: Vec<ParamTerm<Poly>> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty))
            })
            .collect();
        let ty = TyTerm::Fn {
            params: named,
            ret: Box::new(lift_to_poly(&ret)),
            captures: vec![],
            effect: lift_effect_to_poly(&Effect::pure()),
            hint: Some(Hint::Io),
        };

        vec![
            ExternFnBuilder::new("mistral_chat", ty).handler_async(
                move |interner: Interner,
                      (messages_val, config_val): (Value, Value),
                      Uses(()): Uses<()>| {
                    let fetch = Arc::clone(&fetch);
                    async move {
                        let messages_list = match &messages_val {
                            Value::List(l) => l.as_slice(),
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "mistral_chat: expected List for messages, got {:?}",
                                    other.kind()
                                )));
                            }
                        };
                        let messages =
                            values_to_messages(messages_list, &interner, "mistral_chat")?;

                        let config_obj = match &config_val {
                            Value::Object(o) => o,
                            other => {
                                return Err(RuntimeError::fetch(format!(
                                    "mistral_chat: expected Object for config, got {:?}",
                                    other.kind()
                                )));
                            }
                        };

                        let endpoint = obj_get_str(config_obj, endpoint_key)
                            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
                        let api_key = obj_get_str(config_obj, api_key_key).ok_or_else(|| {
                            RuntimeError::fetch("mistral_chat: missing 'api_key' in config")
                        })?;
                        let model = obj_get_str(config_obj, model_key).ok_or_else(|| {
                            RuntimeError::fetch("mistral_chat: missing 'model' in config")
                        })?;
                        let temperature = obj_get_decimal(config_obj, temperature_key);
                        let top_p = obj_get_decimal(config_obj, top_p_key);
                        let max_tokens = obj_get_u32(config_obj, max_tokens_key);
                        let reasoning_effort = obj_get_str(config_obj, reasoning_effort_key);

                        let request_body = schema::Request {
                            model,
                            messages: messages.iter().map(convert_message).collect(),
                            tools: None,
                            temperature,
                            top_p,
                            max_tokens,
                            reasoning_effort,
                        };

                        let http_request = HttpRequest {
                            url: endpoint,
                            headers: vec![
                                ("Authorization".into(), format!("Bearer {api_key}")),
                                ("Content-Type".into(), "application/json".into()),
                            ],
                            body: serde_json::to_value(&request_body).map_err(|e| {
                                RuntimeError::fetch(format!(
                                    "mistral_chat: serialization failed: {e}"
                                ))
                            })?,
                        };

                        let response_json = fetch
                            .fetch(&http_request)
                            .await
                            .map_err(RuntimeError::fetch)?;

                        let (response, usage) = parse_response(response_json)
                            .map_err(|e| RuntimeError::fetch(e.to_string()))?;

                        let result = response_to_value(&response, &usage, &interner);
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
    use std::sync::Mutex;

    struct CapturingFetch {
        captured: Mutex<Option<(String, Vec<(String, String)>, serde_json::Value)>>,
        response: serde_json::Value,
    }

    impl Fetch for CapturingFetch {
        async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
            *self.captured.lock().unwrap() = Some((
                request.url.clone(),
                request.headers.clone(),
                request.body.clone(),
            ));
            Ok(self.response.clone())
        }
    }

    #[test]
    fn registry_produces_function() {
        let fetch = Arc::new(CapturingFetch {
            captured: Mutex::new(None),
            response: serde_json::json!({}),
        });
        let interner = Interner::new();
        let registry = mistral_registry(fetch);
        let registered = registry.register(&interner);
        assert_eq!(registered.functions.len(), 1);
        assert_eq!(registered.executables.len(), 1);

        let func = &registered.functions[0];
        assert_eq!(interner.resolve(func.qref.name), "mistral_chat");
    }

    #[test]
    fn default_endpoint_is_mistral_chat_completions() {
        assert_eq!(
            DEFAULT_ENDPOINT,
            "https://api.mistral.ai/v1/chat/completions"
        );
    }
}
