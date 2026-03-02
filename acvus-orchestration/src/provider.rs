use futures::future::BoxFuture;

use crate::message::{Message, ModelResponse, ToolSpec};

/// Semantic fetch request — the provider implementation handles
/// wire format serialization and response parsing.
#[derive(Debug, Clone)]
pub struct FetchRequest {
    pub provider: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
}

/// Fetch trait: takes a semantic request and returns a parsed model response.
///
/// Implementors are responsible for:
/// - Serializing messages/tools into the provider's wire format
/// - HTTP transport (endpoint, auth, headers)
/// - Parsing the provider's response into `ModelResponse`
pub trait Fetch: Send + Sync {
    fn call<'a>(
        &'a self,
        request: &'a FetchRequest,
    ) -> BoxFuture<'a, Result<ModelResponse, String>>;
}
