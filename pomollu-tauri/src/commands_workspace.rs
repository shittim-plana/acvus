//! Workspace commands — multi-session isolation, one directory per workspace.

use serde_json::Value;

use pomollu_core::persistence;

#[tauri::command]
pub fn cmd_workspace_create(app: tauri::AppHandle, name: String) -> Result<String, String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::workspace_create(&data_dir, &name).map_err(String::from)
}

#[tauri::command]
pub fn cmd_workspace_list(app: tauri::AppHandle) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    let list = persistence::workspace_list(&data_dir).map_err(String::from)?;
    Ok(Value::Array(list))
}

#[tauri::command]
pub fn cmd_workspace_load(app: tauri::AppHandle, id: String) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::workspace_load(&data_dir, &id)
        .map_err(String::from)?
        .ok_or_else(|| format!("workspace not found: {id}"))
}

#[tauri::command]
pub fn cmd_workspace_update(
    app: tauri::AppHandle,
    id: String,
    data: Value,
) -> Result<(), String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::workspace_update(&data_dir, &id, &data).map_err(String::from)
}

#[tauri::command]
pub fn cmd_workspace_delete(app: tauri::AppHandle, id: String) -> Result<(), String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::workspace_delete(&data_dir, &id).map_err(String::from)
}

#[tauri::command]
pub fn cmd_workspace_save_session(
    app: tauri::AppHandle,
    id: String,
    session: Value,
) -> Result<(), String> {
    let data_dir = crate::get_data_dir(&app)?;
    persistence::workspace_save_session(&data_dir, &id, &session).map_err(String::from)
}

#[tauri::command]
pub fn cmd_workspace_load_session(app: tauri::AppHandle, id: String) -> Result<Value, String> {
    let data_dir = crate::get_data_dir(&app)?;
    Ok(persistence::workspace_load_session(&data_dir, &id)
        .map_err(String::from)?
        .unwrap_or(Value::Null))
}
