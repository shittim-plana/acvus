use thiserror::Error;

#[derive(Debug, Error)]
pub enum PomolluError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("API error {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("OAuth error: {0}")]
    OAuth(String),

    #[error("invalid_grant: refresh token expired or revoked")]
    InvalidGrant,

    #[error("AES-GCM encryption failed")]
    AesEncrypt,

    #[error("AES-GCM decryption failed")]
    AesDecrypt,

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Tauri commands return `Result<T, String>`; this keeps call sites terse.
impl From<PomolluError> for String {
    fn from(e: PomolluError) -> String {
        e.to_string()
    }
}
