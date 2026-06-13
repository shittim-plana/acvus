//! Pomollu core — Tauri-free modules, host-testable.
//!
//! Local features ported from layream's proven patterns: encrypted token
//! store, atomic JSON persistence, Vertex AI OAuth (PKCE), Mistral/Vertex
//! streaming clients, and the `ReqwestFetch` engine hook for acvus-ext-llm.

pub mod crypto;
pub mod error;
pub mod fetch;
pub mod oauth;
pub mod persistence;
pub mod providers;
pub mod retry;
pub mod store;
