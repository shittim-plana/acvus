use acvus_orchestration::{FetchRequest, Message, ModelResponse, ToolCall};

pub async fn call(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    request: &FetchRequest,
) -> Result<ModelResponse, String> {
    let body = format_request(request);
    let url = format!(
        "{endpoint}/v1beta/models/{}:generateContent",
        request.model
    );

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("google request failed: {e}"))?;

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("google response parse failed: {e}"))?;

    parse_response(&json)
}

fn format_request(request: &FetchRequest) -> serde_json::Value {
    // Separate system instruction from contents
    let mut system_text = String::new();
    let mut contents = Vec::new();

    for m in &request.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
        } else {
            contents.push(format_content(m));
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
    });

    if !system_text.is_empty() {
        body["system_instruction"] = serde_json::json!({
            "parts": [{ "text": system_text }]
        });
    }

    if !request.tools.is_empty() {
        let declarations: Vec<serde_json::Value> = request
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
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    }
                })
            })
            .collect();

        body["tools"] = serde_json::json!([{
            "function_declarations": declarations,
        }]);
    }

    body
}

fn format_content(m: &Message) -> serde_json::Value {
    // Map roles: "assistant" → "model", "tool" stays special
    let role = match m.role.as_str() {
        "assistant" => "model",
        other => other,
    };

    let mut parts = Vec::new();

    // Tool call results → functionResponse parts
    if let Some(ref tool_call_id) = m.tool_call_id {
        parts.push(serde_json::json!({
            "functionResponse": {
                "name": tool_call_id,
                "response": {
                    "content": m.content,
                }
            }
        }));
        return serde_json::json!({ "role": "function", "parts": parts });
    }

    // Assistant message with tool calls → functionCall parts
    if !m.tool_calls.is_empty() {
        for tc in &m.tool_calls {
            parts.push(serde_json::json!({
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            }));
        }
        return serde_json::json!({ "role": role, "parts": parts });
    }

    // Plain text
    parts.push(serde_json::json!({ "text": m.content }));
    serde_json::json!({ "role": role, "parts": parts })
}

fn parse_response(json: &serde_json::Value) -> Result<ModelResponse, String> {
    let candidates = json
        .get("candidates")
        .and_then(|c| c.as_array())
        .ok_or("missing 'candidates' in response")?;

    let candidate = candidates.first().ok_or("empty candidates array")?;
    let content = candidate.get("content").ok_or("missing 'content' in candidate")?;
    let parts = content
        .get("parts")
        .and_then(|p| p.as_array())
        .ok_or("missing 'parts' in content")?;

    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();

    for part in parts {
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("missing functionCall name")?
                .to_string();
            let arguments = fc.get("args").cloned().unwrap_or(serde_json::json!({}));
            // Gemini doesn't have explicit call IDs; generate from name
            let id = format!("call_{name}");
            tool_calls.push(ToolCall { id, name, arguments });
        } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            text_parts.push(text.to_string());
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
            provider: "google".into(),
            model: "gemini-2.0-flash".into(),
            messages,
            tools,
        }
    }

    #[test]
    fn format_system_as_instruction() {
        let req = make_request(
            vec![
                Message::text("system", "You are helpful."),
                Message::text("user", "Hello"),
            ],
            vec![],
        );
        let body = format_request(&req);
        assert_eq!(
            body["system_instruction"]["parts"][0]["text"],
            "You are helpful."
        );
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn format_assistant_as_model() {
        let msg = Message::text("assistant", "I can help.");
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
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
        let decls = &body["tools"][0]["function_declarations"];
        assert_eq!(decls.as_array().unwrap().len(), 1);
        assert_eq!(decls[0]["name"], "search");
    }

    #[test]
    fn format_function_call_message() {
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
        let content = format_content(&msg);
        assert_eq!(content["role"], "model");
        let parts = content["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "search");
    }

    #[test]
    fn format_function_response_message() {
        let msg = Message {
            role: "tool".into(),
            content: "result data".into(),
            tool_calls: Vec::new(),
            tool_call_id: Some("search".into()),
        };
        let content = format_content(&msg);
        assert_eq!(content["role"], "function");
        let parts = content["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "search");
    }

    #[test]
    fn parse_text_response() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "Hello there!" }]
                }
            }]
        });
        let resp = parse_response(&json).unwrap();
        assert!(matches!(resp, ModelResponse::Text(ref s) if s == "Hello there!"));
    }

    #[test]
    fn parse_function_call_response() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "hello"}
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
