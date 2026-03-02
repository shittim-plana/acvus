pub mod openai;
pub mod anthropic;
pub mod google;

use std::collections::HashMap;

use acvus_orchestration::{Fetch, FetchRequest, ModelResponse};
use futures::future::BoxFuture;

pub struct HttpFetch {
    client: reqwest::Client,
    providers: HashMap<String, ProviderImpl>,
}

struct ProviderImpl {
    api: ApiKind,
    endpoint: String,
    api_key: String,
}

enum ApiKind {
    OpenAI,
    Anthropic,
    Google,
}

impl HttpFetch {
    pub fn new(providers: HashMap<String, (String, String, String)>) -> Self {
        let client = reqwest::Client::new();
        let providers = providers
            .into_iter()
            .map(|(name, (api, endpoint, api_key))| {
                let api = match api.as_str() {
                    "openai" => ApiKind::OpenAI,
                    "anthropic" => ApiKind::Anthropic,
                    "google" => ApiKind::Google,
                    other => panic!("unknown api kind: {other}"),
                };
                (name, ProviderImpl { api, endpoint, api_key })
            })
            .collect();
        Self { client, providers }
    }
}

impl Fetch for HttpFetch {
    fn call<'a>(
        &'a self,
        request: &'a FetchRequest,
    ) -> BoxFuture<'a, Result<ModelResponse, String>> {
        Box::pin(async move {
            let provider = self
                .providers
                .get(&request.provider)
                .ok_or_else(|| format!("unknown provider: {}", request.provider))?;

            match provider.api {
                ApiKind::OpenAI => {
                    openai::call(&self.client, &provider.endpoint, &provider.api_key, request).await
                }
                ApiKind::Anthropic => {
                    anthropic::call(&self.client, &provider.endpoint, &provider.api_key, request)
                        .await
                }
                ApiKind::Google => {
                    google::call(&self.client, &provider.endpoint, &provider.api_key, request).await
                }
            }
        })
    }
}
