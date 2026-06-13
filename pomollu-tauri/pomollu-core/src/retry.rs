//! Bounded retry with cancellation for provider HTTP requests.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::error::PomolluError;

const MAX_RETRIES: u32 = 2;
const BASE_DELAY: Duration = Duration::from_secs(1);

pub type CancelToken = Arc<AtomicBool>;

pub fn new_cancel_token() -> CancelToken {
    Arc::new(AtomicBool::new(false))
}

pub fn is_cancelled(token: &Option<CancelToken>) -> bool {
    token.as_ref().is_some_and(|t| t.load(Ordering::Relaxed))
}

pub fn is_retryable(err: &PomolluError) -> bool {
    match err {
        PomolluError::Http(_) => true,
        PomolluError::ApiError { status, .. } => *status >= 500 || *status == 429,
        _ => false,
    }
}

pub fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
}

pub async fn retry_request<F, Fut>(
    cancel: &Option<CancelToken>,
    mut make_request: F,
) -> Result<reqwest::Response, PomolluError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<reqwest::Response, PomolluError>>,
{
    let mut last_err = None;

    for attempt in 0..=MAX_RETRIES {
        if is_cancelled(cancel) {
            return Err(PomolluError::Http("cancelled".to_string()));
        }

        match make_request().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if status < 500 && status != 429 {
                    return Ok(resp);
                }
                let delay = retry_after_from_headers(resp.headers())
                    .unwrap_or(BASE_DELAY * 2u32.pow(attempt));
                let body = resp
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("(body read failed: {e})"));
                last_err = Some(PomolluError::ApiError { status, body });

                if attempt < MAX_RETRIES {
                    log::warn!("retry {}/{}: status {}", attempt + 1, MAX_RETRIES, status);
                    tokio::time::sleep(delay).await;
                }
            }
            Err(e) if is_retryable(&e) && attempt < MAX_RETRIES => {
                log::warn!("retry {}/{}: {}", attempt + 1, MAX_RETRIES, e);
                let delay = BASE_DELAY * 2u32.pow(attempt);
                tokio::time::sleep(delay).await;
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_err.unwrap_or_else(|| PomolluError::Http("max retries exceeded".to_string())))
}
