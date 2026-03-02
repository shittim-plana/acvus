use crate::value::Value;

pub fn is_builtin(name: &str) -> bool {
    acvus_mir::builtins::builtins().iter().any(|b| b.name() == name)
}

// -- FromValue / IntoValue ------------------------------------------------

trait FromValue: Sized {
    fn from_value(v: Value) -> Self;
}

trait IntoValue {
    fn into_value(self) -> Value;
}

impl FromValue for i64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Int(n) => n,
            _ => unreachable!("FromValue<i64>: expected Int, got {v:?}"),
        }
    }
}

impl FromValue for f64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Float(f) => f,
            _ => unreachable!("FromValue<f64>: expected Float, got {v:?}"),
        }
    }
}

impl FromValue for String {
    fn from_value(v: Value) -> Self {
        match v {
            Value::String(s) => s,
            _ => unreachable!("FromValue<String>: expected String, got {v:?}"),
        }
    }
}

impl FromValue for bool {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Bool(b) => b,
            _ => unreachable!("FromValue<bool>: expected Bool, got {v:?}"),
        }
    }
}

impl FromValue for u8 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Byte(b) => b,
            _ => unreachable!("FromValue<u8>: expected Byte, got {v:?}"),
        }
    }
}

impl FromValue for Vec<Value> {
    fn from_value(v: Value) -> Self {
        match v {
            Value::List(items) => items,
            _ => unreachable!("FromValue<Vec<Value>>: expected List, got {v:?}"),
        }
    }
}

impl FromValue for Value {
    fn from_value(v: Value) -> Self { v }
}

impl IntoValue for i64 {
    fn into_value(self) -> Value { Value::Int(self) }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value { Value::Float(self) }
}

impl IntoValue for String {
    fn into_value(self) -> Value { Value::String(self) }
}

impl IntoValue for bool {
    fn into_value(self) -> Value { Value::Bool(self) }
}

impl IntoValue for u8 {
    fn into_value(self) -> Value { Value::Byte(self) }
}

impl IntoValue for Value {
    fn into_value(self) -> Value { self }
}

// -- PureBuiltin trait (Axum Handler pattern) -----------------------------

trait PureBuiltin<Args> {
    fn call(self, args: Vec<Value>) -> Value;
}

impl<F, R, A> PureBuiltin<(A,)> for F
where
    F: Fn(A) -> R,
    A: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Value {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        self(a).into_value()
    }
}

impl<F, R, A, B> PureBuiltin<(A, B)> for F
where
    F: Fn(A, B) -> R,
    A: FromValue,
    B: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Value {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        let b = B::from_value(it.next().unwrap());
        self(a, b).into_value()
    }
}

impl<F, R, A, B, C> PureBuiltin<(A, B, C)> for F
where
    F: Fn(A, B, C) -> R,
    A: FromValue,
    B: FromValue,
    C: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Value {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        let b = B::from_value(it.next().unwrap());
        let c = C::from_value(it.next().unwrap());
        self(a, b, c).into_value()
    }
}

// -- pure builtin functions -----------------------------------------------

fn builtin_to_string(v: Value) -> Value {
    Value::String(value_to_string(v))
}

fn builtin_to_int(v: Value) -> Value {
    match v {
        Value::Float(f) => Value::Int(f as i64),
        Value::Byte(b) => Value::Int(b as i64),
        _ => unreachable!("to_int: expected Float or Byte, got {v:?}"),
    }
}

fn builtin_to_float(n: i64) -> f64 {
    n as f64
}

fn builtin_char_to_int(s: String) -> i64 {
    s.chars().next().expect("char_to_int: empty string") as i64
}

fn builtin_int_to_char(n: i64) -> String {
    char::from_u32(n as u32).expect("int_to_char: invalid codepoint").to_string()
}

fn builtin_len(items: Vec<Value>) -> i64 {
    items.len() as i64
}

fn builtin_reverse(items: Vec<Value>) -> Value {
    let mut items = items;
    items.reverse();
    Value::List(items)
}

fn builtin_join(items: Vec<Value>, sep: String) -> String {
    let strs: Vec<String> = items
        .into_iter()
        .map(|v| match v {
            Value::String(s) => s,
            v => unreachable!("join: expected List<String>, got element {v:?}"),
        })
        .collect();
    strs.join(&sep)
}

fn builtin_contains(items: Vec<Value>, target: Value) -> bool {
    items.iter().any(|v| values_equal(v, &target))
}

fn builtin_substring(s: String, start: i64, len: i64) -> String {
    s.chars().skip(start.max(0) as usize).take(len.max(0) as usize).collect()
}

fn builtin_len_str(s: String) -> i64 {
    s.chars().count() as i64
}

fn builtin_to_bytes(s: String) -> Value {
    Value::List(s.into_bytes().into_iter().map(Value::Byte).collect())
}

fn builtin_to_utf8(items: Vec<Value>) -> String {
    let bytes: Vec<u8> = items
        .into_iter()
        .map(|v| match v {
            Value::Byte(b) => b,
            v => unreachable!("to_utf8: expected List<Byte>, got element {v:?}"),
        })
        .collect();
    String::from_utf8(bytes).unwrap()
}

fn builtin_to_utf8_lossy(items: Vec<Value>) -> String {
    let bytes: Vec<u8> = items
        .into_iter()
        .map(|v| match v {
            Value::Byte(b) => b,
            v => unreachable!("to_utf8_lossy: expected List<Byte>, got element {v:?}"),
        })
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// -- dispatch -------------------------------------------------------------

/// Dispatch a pure (non-HOF) builtin.
pub fn call_pure(name: &str, args: Vec<Value>) -> Value {
    match name {
        "to_string"    => PureBuiltin::call(builtin_to_string, args),
        "to_int"       => PureBuiltin::call(builtin_to_int, args),
        "to_float"     => PureBuiltin::call(builtin_to_float, args),
        "char_to_int"  => PureBuiltin::call(builtin_char_to_int, args),
        "int_to_char"  => PureBuiltin::call(builtin_int_to_char, args),
        "len"          => PureBuiltin::call(builtin_len, args),
        "reverse"      => PureBuiltin::call(builtin_reverse, args),
        "join"         => PureBuiltin::call(builtin_join, args),
        "contains"     => PureBuiltin::call(builtin_contains, args),
        "substring"    => PureBuiltin::call(builtin_substring, args),
        "len_str"      => PureBuiltin::call(builtin_len_str, args),
        "to_bytes"     => PureBuiltin::call(builtin_to_bytes, args),
        "to_utf8"      => PureBuiltin::call(builtin_to_utf8, args),
        "to_utf8_lossy" => PureBuiltin::call(builtin_to_utf8_lossy, args),
        _ => panic!("not a pure builtin: {name}"),
    }
}

// -- display ------------------------------------------------------------------

fn value_to_string(v: Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s,
        Value::Bool(b) => b.to_string(),
        Value::Byte(b) => b.to_string(),
        Value::Unit => "()".to_string(),
        _ => unreachable!("to_string: expected scalar or Unit, got {v:?}"),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Byte(a), Value::Byte(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => unreachable!("values_equal: unsupported types ({a:?}, {b:?})"),
    }
}

#[cfg(test)]
mod tests {
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
}
