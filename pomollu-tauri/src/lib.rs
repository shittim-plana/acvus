//! Pomollu Tauri app — native Rust backend for the Pomollu mobile app.
//!
//! Tauri-free logic (OAuth, crypto, persistence, provider clients, the
//! `ReqwestFetch` engine hook for acvus-ext-llm) lives in `pomollu-core`,
//! which is host-testable without webkit/Android toolchains. This crate is
//! the thin Tauri layer: state, commands, plugin wiring.
//!
//! Engine hook: `pomollu_core::fetch::ReqwestFetch` implements
//! `acvus_ext_llm::Fetch`, so the acvus orchestration runtime can register
//! `mistral_registry` / `vertex_registry` / `openai_registry` natively once
//! its turn execution lands (acvus-orchestration `Session` is
//! compile/typecheck-only today).

pub mod commands_auth;
pub mod commands_chat;
pub mod commands_settings;
pub mod commands_workspace;

use std::path::PathBuf;

use tauri::Manager;

pub fn get_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir().map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let result = tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_deep_link::init())
        .manage(commands_auth::AuthState::default())
        .manage(commands_chat::StreamBufferState::default())
        .manage(commands_chat::StreamCancelState::default())
        .setup(|app| {
            let auth_state = app.state::<commands_auth::AuthState>();
            auth_state.load_persisted_tokens(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // auth
            commands_auth::vertex_oauth_start,
            commands_auth::vertex_oauth_callback,
            commands_auth::vertex_oauth_status,
            commands_auth::vertex_oauth_disconnect,
            commands_auth::vertex_list_projects,
            commands_auth::get_pending_oauth,
            // gca auth
            commands_auth::gca_oauth_start,
            commands_auth::gca_oauth_callback,
            commands_auth::gca_oauth_status,
            commands_auth::gca_oauth_disconnect,
            commands_auth::gca_load_project,
            // chat
            commands_chat::chat_mistral,
            commands_chat::chat_vertex,
            commands_chat::chat_gca,
            commands_chat::poll_stream_chunks,
            commands_chat::cancel_chat,
            commands_chat::mistral_list_models,
            commands_chat::vertex_list_models,
            commands_chat::gca_list_models,
            // settings & session
            commands_settings::cmd_save_settings,
            commands_settings::cmd_load_settings,
            commands_settings::cmd_save_session,
            commands_settings::cmd_load_session,
            // workspaces
            commands_workspace::cmd_workspace_create,
            commands_workspace::cmd_workspace_list,
            commands_workspace::cmd_workspace_load,
            commands_workspace::cmd_workspace_update,
            commands_workspace::cmd_workspace_delete,
            commands_workspace::cmd_workspace_save_session,
            commands_workspace::cmd_workspace_load_session,
        ])
        .run(tauri::generate_context!());

    if let Err(e) = result {
        log::error!("Tauri app error: {e}");
    }
}
