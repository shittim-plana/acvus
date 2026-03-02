use crate::dsl::GenerationParams;
use crate::message::{Message, ModelResponse, ToolCall, ToolSpec};

use super::{HttpRequest, ProviderConfig};

pub fn build_request(
    config: &ProviderConfig,
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    generation: &GenerationParams,
) -> HttpRequest {
    let body = format_body(model, messages, tools, generation);
    let url = format!("{}/v1/chat/completions", config.endpoint);
    HttpRequest {
        url,
        headers: vec![
            ("Authorization".into(), format!("Bearer {}", config.api_key)),
        ],
        body,
    }
}

fn format_message(m: &Message) -> serde_json::Value {
    let mut msg = serde_json::json!({
        "role": m.role,
    });

    if !m.content.is_empty() {
        msg["content"] = serde_json::Value::String(m.content.clone());
    }

    if !m.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = m
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments.to_string(),
                    }
                })
            })
            .collect();
        msg["tool_calls"] = serde_json::Value::Array(calls);
    }

    if let Some(ref id) = m.tool_call_id {
        msg["tool_call_id"] = serde_json::Value::String(id.clone());
    }

    msg
}

fn format_body(model: &str, messages: &[Message], tools: &[ToolSpec], generation: &GenerationParams) -> serde_json::Value {
    let msgs: Vec<serde_json::Value> = messages.iter().map(format_message).collect();

    let mut body = serde_json::json!({
        "model": model,
        "messages": msgs,
    });

    if let Some(t) = generation.temperature { body["temperature"] = serde_json::json!(t); }
    if let Some(p) = generation.top_p { body["top_p"] = serde_json::json!(p); }
    if let Some(m) = generation.max_tokens { body["max_tokens"] = serde_json::json!(m); }

    if !tools.is_empty() {
        let tool_specs: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                let properties: serde_json::Map<String, serde_json::Value> = t
                    .params
                    .iter()
                    .map(|(name, type_name)| {
                        (name.clone(), serde_json::json!({ "type": type_name }))
                    })
                    .collect();

                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                        }
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
}

pub fn parse_response(json: &serde_json::Value) -> Result<ModelResponse, String> {
    let choices = json
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or("missing 'choices' in response")?;

    let choice = choices.first().ok_or("empty choices array")?;
    let message = choice.get("message").ok_or("missing 'message' in choice")?;

    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        let calls: Result<Vec<ToolCall>, String> = tool_calls
            .iter()
            .map(|tc| {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool call id")?
                    .to_string();
                let func = tc.get("function").ok_or("missing function")?;
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing function name")?
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(serde_json::Value::Object(Default::default()));

                Ok(ToolCall { id, name, arguments })
            })
            .collect();

        return Ok(ModelResponse::ToolCalls(calls?));
    }

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ModelResponse::Text(content))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::dsl::GenerationParams;
    use crate::message::{Message, ToolSpec};

    use super::*;

    #[test]
    fn format_basic_messages() {
        let body = format_body(
            "gpt-4o",
            &[
                Message::text("system", "You are helpful."),
                Message::text("user", "Hello"),
            ],
            &[],
            &GenerationParams::default(),
        );
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn format_with_tools() {
        let body = format_body(
            "gpt-4o",
            &[Message::text("user", "hi")],
            &[ToolSpec {
                name: "search".into(),
                description: "Search the web".into(),
                params: HashMap::from([("query".into(), "string".into())]),
            }],
            &GenerationParams::default(),
        );
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "search");
    }

    #[test]
    fn format_tool_call_message() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "rust"}),
            }],
            tool_call_id: None,
        };
        let formatted = format_message(&msg);
        let calls = formatted["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "search");
    }

    #[test]
    fn format_tool_result_message() {
        let msg = Message {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: Vec::new(),
            tool_call_id: Some("call_1".into()),
        };
        let formatted = format_message(&msg);
        assert_eq!(formatted["role"], "tool");
        assert_eq!(formatted["tool_call_id"], "call_1");
        assert_eq!(formatted["content"], "result data");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                }
            }]
        });
        let resp = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
    }

    #[test]
    fn parse_tool_call_response() {
        let json = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\": \"hello\"}"
                        }
                    }]
                }
            }]
        });
        let resp = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }
}
