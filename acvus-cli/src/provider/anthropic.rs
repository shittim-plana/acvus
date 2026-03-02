use acvus_orchestration::{FetchRequest, Message, ModelResponse, ToolCall};

pub async fn call(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    request: &FetchRequest,
) -> Result<ModelResponse, String> {
    let body = format_request(request);
    let url = format!("{endpoint}/v1/messages");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("anthropic request failed: {e}"))?;

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("anthropic response parse failed: {e}"))?;

    parse_response(&json)
}

fn format_request(request: &FetchRequest) -> serde_json::Value {
    // Separate system messages from the rest
    let mut system_text = String::new();
    let mut msgs = Vec::new();

    for m in &request.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
        } else {
            msgs.push(format_message(m));
        }
    }

    let mut body = serde_json::json!({
        "model": request.model,
        "messages": msgs,
        "max_tokens": 4096,
    });

    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }

    if !request.tools.is_empty() {
        let tool_specs: Vec<serde_json::Value> = request
            .tools
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
                    "name": t.name,
                    "description": t.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::Value::Array(tool_specs);
    }

    body
}

fn format_message(m: &Message) -> serde_json::Value {
    // Tool call results → user message with tool_result content blocks
    if let Some(ref tool_call_id) = m.tool_call_id {
        return serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": m.content,
            }]
        });
    }

    // Assistant message with tool calls → tool_use content blocks
    if !m.tool_calls.is_empty() {
        let mut content: Vec<serde_json::Value> = Vec::new();

        if !m.content.is_empty() {
            content.push(serde_json::json!({
                "type": "text",
                "text": m.content,
            }));
        }

        for tc in &m.tool_calls {
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }));
        }

        return serde_json::json!({
            "role": "assistant",
            "content": content,
        });
    }

    // Plain text message
    serde_json::json!({
        "role": m.role,
        "content": m.content,
    })
}

fn parse_response(json: &serde_json::Value) -> Result<ModelResponse, String> {
    let content = json
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or("missing 'content' array in response")?;

    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();

    for block in content {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use id")?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("missing tool_use name")?
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall { id, name, arguments });
            }
            _ => {}
        }
    }

    if !tool_calls.is_empty() {
        Ok(ModelResponse::ToolCalls(tool_calls))
    } else {
        Ok(ModelResponse::Text(text_parts.join("")))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use acvus_orchestration::{Message, ToolSpec};

    use super::*;

    fn make_request(messages: Vec<Message>, tools: Vec<ToolSpec>) -> FetchRequest {
        FetchRequest {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
            messages,
            tools,
        }
    }

    #[test]
    fn format_system_separated() {
        let req = make_request(
            vec![
                Message::text("system", "You are helpful."),
                Message::text("user", "Hello"),
            ],
            vec![],
        );
        let body = format_request(&req);
        assert_eq!(body["system"], "You are helpful.");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn format_with_tools() {
        let req = make_request(
            vec![Message::text("user", "hi")],
            vec![ToolSpec {
                name: "search".into(),
                description: "Search".into(),
                params: HashMap::from([("query".into(), "string".into())]),
            }],
        );
        let body = format_request(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "search");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn format_tool_use_message() {
        let msg = Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "toolu_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "rust"}),
            }],
            tool_call_id: None,
        };
        let formatted = format_message(&msg);
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "toolu_1");
    }

    #[test]
    fn format_tool_result_message() {
        let msg = Message {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: Vec::new(),
            tool_call_id: Some("toolu_1".into()),
        };
        let formatted = format_message(&msg);
        assert_eq!(formatted["role"], "user");
        let content = formatted["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn"
        });
        let resp = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
    }

    #[test]
    fn parse_tool_use_response() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "Let me search." },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"query": "hello"}
                }
            ],
            "stop_reason": "tool_use"
        });
        let resp = parse_response(&json).unwrap();
        match resp {
            ModelResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].id, "toolu_123");
                assert_eq!(calls[0].arguments["query"], "hello");
            }
            _ => panic!("expected ToolCalls"),
        }
    }
}
