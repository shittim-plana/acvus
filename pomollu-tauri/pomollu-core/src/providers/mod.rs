//! App-level provider clients (streaming, cancellable) — ported from
//! layream-core. The engine-facing non-streaming providers live in
//! `acvus-ext-llm`; these clients serve the interactive chat UI where
//! SSE streaming and cancellation matter.

pub mod gca;
pub mod mistral;
pub mod vertex;
