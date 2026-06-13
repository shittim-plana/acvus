//! Local persistence — atomic JSON files under the app data directory.
//!
//! Layout (ported from layream, trimmed to Pomollu's needs):
//! ```text
//! $APP_DATA/
//! ├── tokens.json          encrypted (AES-256-GCM) OAuth tokens
//! ├── settings.json        plaintext app settings (serde_json::Value)
//! ├── pkce_verifier.txt    transient PKCE verifier during an OAuth round-trip
//! ├── pending_oauth.txt    deep-link callback handoff (written by Android side)
//! └── workspaces/
//!     ├── {id}.json        workspace metadata
//!     └── {id}/
//!         └── session.json per-workspace session payload
//! ```
//!
//! All writes go through tmp+rename — a crash mid-write never leaves a torn
//! file (atomic within one filesystem).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::crypto;
use crate::error::PomolluError;
use crate::oauth::Tokens;

const TOKEN_STORE_PASSWORD: &str = "pomollu-token-store-v1";

// ── Atomic JSON primitives ──────────────────────────────────────────

pub fn write_json_atomic(path: &Path, value: &Value) -> Result<(), PomolluError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string(value)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json.as_bytes())?;
    fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_json(path: &Path) -> Result<Option<Value>, PomolluError> {
    match fs::read_to_string(path) {
        Ok(s) => Ok(Some(serde_json::from_str(&s)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ── Settings ────────────────────────────────────────────────────────

pub fn save_settings(data_dir: &Path, settings: &Value) -> Result<(), PomolluError> {
    write_json_atomic(&data_dir.join("settings.json"), settings)
}

pub fn load_settings(data_dir: &Path) -> Result<Value, PomolluError> {
    Ok(read_json(&data_dir.join("settings.json"))?.unwrap_or_else(|| Value::Object(Default::default())))
}

// ── Token store (encrypted) ─────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StoredTokens {
    #[serde(default)]
    pub vertex: Option<Tokens>,
    /// `#[serde(default)]` keeps older `tokens.json` (vertex-only) loadable
    /// after GCA was added — forward-compatible schema growth.
    #[serde(default)]
    pub gca: Option<Tokens>,
}

pub fn save_tokens(data_dir: &Path, tokens: &StoredTokens) -> Result<(), PomolluError> {
    fs::create_dir_all(data_dir)?;
    let json = serde_json::to_string(tokens)?;
    let encrypted = crypto::encrypt(json.as_bytes(), TOKEN_STORE_PASSWORD)?;
    let path = data_dir.join("tokens.json");
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &encrypted)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn load_tokens(data_dir: &Path) -> Result<StoredTokens, PomolluError> {
    let path = data_dir.join("tokens.json");
    let encrypted = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(StoredTokens::default()),
        Err(e) => return Err(e.into()),
    };
    let json = crypto::decrypt(&encrypted, TOKEN_STORE_PASSWORD)?;
    Ok(serde_json::from_slice(&json)?)
}

pub fn clear_tokens(data_dir: &Path) -> Result<(), PomolluError> {
    let path = data_dir.join("tokens.json");
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

// ── PKCE verifier (transient) ───────────────────────────────────────

pub fn save_pkce_verifier(data_dir: &Path, verifier: &str) -> Result<(), PomolluError> {
    fs::create_dir_all(data_dir)?;
    fs::write(data_dir.join("pkce_verifier.txt"), verifier)?;
    Ok(())
}

pub fn take_pkce_verifier(data_dir: &Path) -> Option<String> {
    let path = data_dir.join("pkce_verifier.txt");
    let verifier = fs::read_to_string(&path).ok()?;
    let _ = fs::remove_file(&path);
    Some(verifier)
}

// ── Pending OAuth deep-link handoff ─────────────────────────────────

pub fn take_pending_oauth(data_dir: &Path) -> Option<String> {
    let path = data_dir.join("pending_oauth.txt");
    let content = fs::read_to_string(&path).ok()?;
    let _ = fs::remove_file(&path);
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

// ── Workspaces ──────────────────────────────────────────────────────

/// Sortable, collision-resistant ID: millis + sub-second nanos in hex.
pub fn generate_id() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{:x}-{:08x}", now.as_millis() as u64, now.subsec_nanos() as u64)
}

/// Reject anything that could escape the workspaces directory.
pub fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

fn workspaces_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("workspaces")
}

fn workspace_meta_path(data_dir: &Path, id: &str) -> PathBuf {
    workspaces_dir(data_dir).join(format!("{id}.json"))
}

pub fn workspace_create(data_dir: &Path, name: &str) -> Result<String, PomolluError> {
    let id = generate_id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let meta = serde_json::json!({
        "id": id,
        "name": name,
        "created_at": now,
        "updated_at": now,
    });
    write_json_atomic(&workspace_meta_path(data_dir, &id), &meta)?;
    Ok(id)
}

pub fn workspace_list(data_dir: &Path) -> Result<Vec<Value>, PomolluError> {
    let dir = workspaces_dir(data_dir);
    let mut items = Vec::new();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(items),
        Err(e) => return Err(e.into()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Some(v) = read_json(&path)? {
                items.push(v);
            }
        }
    }
    // Hex millis IDs sort lexicographically only at equal length; sort by the
    // recorded timestamp instead.
    items.sort_by_key(|v| v.get("created_at").and_then(|c| c.as_u64()).unwrap_or(0));
    Ok(items)
}

pub fn workspace_load(data_dir: &Path, id: &str) -> Result<Option<Value>, PomolluError> {
    if !is_safe_id(id) {
        return Err(PomolluError::Http(format!("unsafe workspace id: {id}")));
    }
    read_json(&workspace_meta_path(data_dir, id))
}

/// Shallow-merge `patch` into the stored metadata and bump `updated_at`.
pub fn workspace_update(data_dir: &Path, id: &str, patch: &Value) -> Result<(), PomolluError> {
    if !is_safe_id(id) {
        return Err(PomolluError::Http(format!("unsafe workspace id: {id}")));
    }
    let path = workspace_meta_path(data_dir, id);
    let mut meta = read_json(&path)?
        .ok_or_else(|| PomolluError::Http(format!("workspace not found: {id}")))?;
    if let (Some(obj), Some(patch_obj)) = (meta.as_object_mut(), patch.as_object()) {
        for (k, v) in patch_obj {
            if k != "id" && k != "created_at" {
                obj.insert(k.clone(), v.clone());
            }
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        obj.insert("updated_at".into(), serde_json::json!(now));
    }
    write_json_atomic(&path, &meta)
}

pub fn workspace_delete(data_dir: &Path, id: &str) -> Result<(), PomolluError> {
    if !is_safe_id(id) {
        return Err(PomolluError::Http(format!("unsafe workspace id: {id}")));
    }
    let meta = workspace_meta_path(data_dir, id);
    match fs::remove_file(&meta) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let dir = workspaces_dir(data_dir).join(id);
    match fs::remove_dir_all(&dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

pub fn workspace_save_session(
    data_dir: &Path,
    id: &str,
    session: &Value,
) -> Result<(), PomolluError> {
    if !is_safe_id(id) {
        return Err(PomolluError::Http(format!("unsafe workspace id: {id}")));
    }
    write_json_atomic(
        &workspaces_dir(data_dir).join(id).join("session.json"),
        session,
    )
}

pub fn workspace_load_session(data_dir: &Path, id: &str) -> Result<Option<Value>, PomolluError> {
    if !is_safe_id(id) {
        return Err(PomolluError::Http(format!("unsafe workspace id: {id}")));
    }
    read_json(&workspaces_dir(data_dir).join(id).join("session.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("x.json");
        let v = serde_json::json!({"a": 1, "b": [true, "s"]});
        write_json_atomic(&path, &v).unwrap();
        assert_eq!(read_json(&path).unwrap(), Some(v));
        assert_eq!(read_json(&dir.path().join("missing.json")).unwrap(), None);
    }

    #[test]
    fn token_store_roundtrip_encrypted() {
        let dir = tempfile::tempdir().unwrap();
        let stored = StoredTokens {
            vertex: Some(Tokens {
                access_token: "at".into(),
                refresh_token: Some("rt".into()),
                expires_at: 123,
            }),
            gca: None,
        };
        save_tokens(dir.path(), &stored).unwrap();

        // On-disk bytes must not contain the plaintext token.
        let raw = fs::read(dir.path().join("tokens.json")).unwrap();
        assert!(!raw.windows(2).any(|w| w == b"at"));

        let loaded = load_tokens(dir.path()).unwrap();
        assert_eq!(loaded.vertex.as_ref().unwrap().access_token, "at");
        assert_eq!(loaded.vertex.as_ref().unwrap().refresh_token.as_deref(), Some("rt"));

        clear_tokens(dir.path()).unwrap();
        assert!(load_tokens(dir.path()).unwrap().vertex.is_none());
    }

    #[test]
    fn old_vertex_only_token_store_loads_with_gca_absent() {
        // A tokens.json written before GCA existed (no `gca` field) must still
        // deserialize — forward compatibility via #[serde(default)].
        let dir = tempfile::tempdir().unwrap();
        let legacy = serde_json::json!({
            "vertex": { "access_token": "at", "refresh_token": "rt", "expires_at": 1 }
        });
        let encrypted =
            crypto::encrypt(serde_json::to_string(&legacy).unwrap().as_bytes(), TOKEN_STORE_PASSWORD)
                .unwrap();
        fs::write(dir.path().join("tokens.json"), &encrypted).unwrap();

        let loaded = load_tokens(dir.path()).unwrap();
        assert_eq!(loaded.vertex.unwrap().access_token, "at");
        assert!(loaded.gca.is_none());
    }

    #[test]
    fn missing_token_store_is_empty_not_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_tokens(dir.path()).unwrap().vertex.is_none());
    }

    #[test]
    fn pkce_verifier_is_taken_once() {
        let dir = tempfile::tempdir().unwrap();
        save_pkce_verifier(dir.path(), "v123").unwrap();
        assert_eq!(take_pkce_verifier(dir.path()).as_deref(), Some("v123"));
        assert_eq!(take_pkce_verifier(dir.path()), None);
    }

    #[test]
    fn workspace_crud() {
        let dir = tempfile::tempdir().unwrap();
        let id = workspace_create(dir.path(), "test ws").unwrap();
        assert!(is_safe_id(&id));

        let list = workspace_list(dir.path()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["name"], "test ws");

        workspace_update(dir.path(), &id, &serde_json::json!({"provider": "mistral"})).unwrap();
        let meta = workspace_load(dir.path(), &id).unwrap().unwrap();
        assert_eq!(meta["provider"], "mistral");
        assert_eq!(meta["name"], "test ws");

        let session = serde_json::json!({"messages": []});
        workspace_save_session(dir.path(), &id, &session).unwrap();
        assert_eq!(workspace_load_session(dir.path(), &id).unwrap(), Some(session));

        workspace_delete(dir.path(), &id).unwrap();
        assert!(workspace_list(dir.path()).unwrap().is_empty());
        assert_eq!(workspace_load(dir.path(), &id).unwrap(), None);
    }

    #[test]
    fn unsafe_ids_rejected() {
        let dir = tempfile::tempdir().unwrap();
        assert!(workspace_load(dir.path(), "../../etc/passwd").is_err());
        assert!(workspace_load(dir.path(), "").is_err());
        assert!(!is_safe_id("a/b"));
        assert!(!is_safe_id("z"));  // non-hex letter
        assert!(is_safe_id("1a2b-00ff"));
    }

    #[test]
    fn update_preserves_id_and_created_at() {
        let dir = tempfile::tempdir().unwrap();
        let id = workspace_create(dir.path(), "ws").unwrap();
        let before = workspace_load(dir.path(), &id).unwrap().unwrap();
        workspace_update(
            dir.path(),
            &id,
            &serde_json::json!({"id": "evil", "created_at": 0, "name": "renamed"}),
        )
        .unwrap();
        let after = workspace_load(dir.path(), &id).unwrap().unwrap();
        assert_eq!(after["id"], before["id"]);
        assert_eq!(after["created_at"], before["created_at"]);
        assert_eq!(after["name"], "renamed");
    }
}
