use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Data-only value — no functions, no closures.
/// Serializable, cloneable, used at Storage boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PureValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Unit,
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    List(Vec<PureValue>),
    Object(BTreeMap<String, PureValue>),
    Tuple(Vec<PureValue>),
}

/// Hierarchical storage key. Dot-separated, ABI-stable.
///
/// e.g. `"user"`, `"user.address"`, `"user.address.city"`
///
/// The raw string is intentionally not exposed. Accessing it as a flat string
/// would bypass parent-child relationships (e.g. a change to "address" must
/// also invalidate "address.city"). Always use `segments()` to traverse the
/// hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageKey(String);

impl StorageKey {
    pub fn root(name: impl Into<String>) -> Self {
        let name = name.into();
        assert!(
            !name.contains(char::is_whitespace),
            "StorageKey segment must not contain whitespace: {name:?}"
        );
        Self(name)
    }

    pub fn field(&self, name: &str) -> Self {
        assert!(
            !name.contains(char::is_whitespace),
            "StorageKey segment must not contain whitespace: {name:?}"
        );
        Self(format!("{}.{}", self.0, name))
    }

    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.0.split('.')
    }
}

impl fmt::Display for StorageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- StorageKey ABI stability --
    // These tests guard the serialized format. If any of them break,
    // it means the ABI has changed and external consumers may be affected.

    #[test]
    fn root_serializes_as_plain_string() {
        let key = StorageKey::root("user");
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, r#""user""#);
    }

    #[test]
    fn field_serializes_dot_separated() {
        let key = StorageKey::root("user").field("address").field("city");
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, r#""user.address.city""#);
    }

    #[test]
    fn deserialize_roundtrip() {
        let key = StorageKey::root("config").field("db").field("host");
        let json = serde_json::to_string(&key).unwrap();
        let restored: StorageKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, restored);
    }

    #[test]
    fn deserialize_from_raw_string() {
        let key: StorageKey = serde_json::from_str(r#""user.address.city""#).unwrap();
        let segments: Vec<&str> = key.segments().collect();
        assert_eq!(segments, ["user", "address", "city"]);
    }

    #[test]
    fn display_matches_serialized_content() {
        let key = StorageKey::root("a").field("b").field("c");
        assert_eq!(key.to_string(), "a.b.c");
    }

    // -- StorageKey hierarchy --

    #[test]
    fn root_has_single_segment() {
        let key = StorageKey::root("user");
        let segments: Vec<&str> = key.segments().collect();
        assert_eq!(segments, ["user"]);
    }

    #[test]
    fn field_appends_segment() {
        let key = StorageKey::root("user").field("name");
        let segments: Vec<&str> = key.segments().collect();
        assert_eq!(segments, ["user", "name"]);
    }

    #[test]
    fn nested_fields_preserve_order() {
        let key = StorageKey::root("a").field("b").field("c").field("d");
        let segments: Vec<&str> = key.segments().collect();
        assert_eq!(segments, ["a", "b", "c", "d"]);
    }

    #[test]
    fn different_paths_are_not_equal() {
        let a = StorageKey::root("user").field("name");
        let b = StorageKey::root("user").field("age");
        assert_ne!(a, b);
    }

    #[test]
    fn same_paths_are_equal() {
        let a = StorageKey::root("user").field("name");
        let b = StorageKey::root("user").field("name");
        assert_eq!(a, b);
    }

    // -- StorageKey validation --

    #[test]
    #[should_panic(expected = "whitespace")]
    fn root_rejects_whitespace() {
        StorageKey::root("user name");
    }

    #[test]
    #[should_panic(expected = "whitespace")]
    fn field_rejects_whitespace() {
        StorageKey::root("user").field("first name");
    }

    #[test]
    #[should_panic(expected = "whitespace")]
    fn root_rejects_tab() {
        StorageKey::root("user\tname");
    }
}
