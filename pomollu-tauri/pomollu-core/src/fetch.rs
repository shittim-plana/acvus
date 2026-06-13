//! Native `Fetch` transport for `acvus-ext-llm` providers.
//!
//! This is the engine hook: when the acvus orchestration runtime gains turn
//! execution, `mistral_registry(Arc::new(ReqwestFetch::new()))` etc. plug the
//! engine's LLM calls straight into reqwest — the same role `WebFetch`
//! (web-sys) played in the old wasm engine.

use acvus_ext_llm::{Fetch, HttpRequest};

pub struct ReqwestFetch {
    client: reqwest::Client,
}

impl ReqwestFetch {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for ReqwestFetch {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetch for ReqwestFetch {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        let mut req = self.client.post(&request.url).json(&request.body);
        for (k, v) in &request.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("(body read failed: {e})"));
            return Err(format!("HTTP {status}: {body}"));
        }
        resp.json().await.map_err(|e| e.to_string())
    }
}
