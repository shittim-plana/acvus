//! Settings and global session persistence commands.

use serde_json::Value;

use pomollu_core::persistence;

#[tauri::command]
pub fn cmd_save_settings(app: tauri::AppHandle, settings: Value) -> Result<(), String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::save_settings(&data_dir, &settings).map_err(String::from)
}

#[tauri::command]
pub fn cmd_load_settings(app: tauri::AppHandle) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::load_settings(&data_dir).map_err(String::from)
}

#[tauri::command]
pub fn cmd_save_session(app: tauri::AppHandle, session: Value) -> Result<(), String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::write_json_atomic(&data_dir.join("session.json"), &session)
        .map_err(String::from)
}

#[tauri::command]
pub fn cmd_load_session(app: tauri::AppHandle) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    Ok(persistence::read_json(&data_dir.join("session.json"))
        .map_err(String::from)?
        .unwrap_or(Value::Null))
}
