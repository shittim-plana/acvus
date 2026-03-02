use std::collections::BTreeMap;

use super::*;

#[test]
fn to_string_int() {
    assert!(matches!(call_pure("to_string", vec![Value::Int(42)]), Value::String(s) if s == "42"));
}

#[test]
fn to_string_float() {
    assert!(matches!(call_pure("to_string", vec![Value::Float(3.14)]), Value::String(s) if s == "3.14"));
}

#[test]
fn to_string_bool() {
    assert!(matches!(call_pure("to_string", vec![Value::Bool(true)]), Value::String(s) if s == "true"));
}

#[test]
fn to_string_string() {
    assert!(
        matches!(call_pure("to_string", vec![Value::String("hi".into())]), Value::String(s) if s == "hi")
    );
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_list_panics() {
    call_pure("to_string", vec![Value::List(vec![Value::Int(1), Value::Int(2)])]);
}

#[test]
#[should_panic(expected = "to_string: expected scalar or Unit, got")]
fn to_string_object_panics() {
    call_pure("to_string", vec![Value::Object(BTreeMap::from([("a".into(), Value::Int(1))]))]);
}

#[test]
fn to_int_float() {
    assert!(matches!(call_pure("to_int", vec![Value::Float(3.7)]), Value::Int(3)));
}

#[test]
fn to_float_int() {
    assert!(matches!(call_pure("to_float", vec![Value::Int(5)]), Value::Float(f) if f == 5.0));
}
