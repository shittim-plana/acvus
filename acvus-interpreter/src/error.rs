use std::fmt;

/// Runtime error during template/script execution.
///
/// Propagated through the coroutine as `Stepped::Error`.
/// NOT recoverable by retry — indicates a bug or invalid data.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
}

#[derive(Debug, Clone)]
pub enum RuntimeErrorKind {
    /// Type mismatch at runtime (should have been caught by typeck).
    TypeMismatch {
        operation: String,
        expected: String,
        got: String,
    },
    /// Division by zero.
    DivisionByZero,
    /// Index out of bounds.
    IndexOutOfBounds { index: i64, len: usize },
    /// Empty collection operation (e.g. reduce on empty list).
    EmptyCollection { operation: String },
    /// Missing object field.
    MissingField { field: String },
    /// External function call failed.
    ExternCall { name: String, error: String },
    /// LLM fetch / parse failed (retryable at a higher level).
    Fetch { error: String },
    /// Generic runtime error.
    Other(String),
}

impl RuntimeError {
    pub fn type_mismatch(operation: &str, expected: &str, got: &str) -> Self {
        Self {
            kind: RuntimeErrorKind::TypeMismatch {
                operation: operation.into(),
                expected: expected.into(),
                got: got.into(),
            },
        }
    }

    pub fn division_by_zero() -> Self {
        Self {
            kind: RuntimeErrorKind::DivisionByZero,
        }
    }

    pub fn index_out_of_bounds(index: i64, len: usize) -> Self {
        Self {
            kind: RuntimeErrorKind::IndexOutOfBounds { index, len },
        }
    }

    pub fn empty_collection(operation: &str) -> Self {
        Self {
            kind: RuntimeErrorKind::EmptyCollection {
                operation: operation.into(),
            },
        }
    }

    pub fn missing_field(field: &str) -> Self {
        Self {
            kind: RuntimeErrorKind::MissingField {
                field: field.into(),
            },
        }
    }

    pub fn extern_call(name: &str, error: String) -> Self {
        Self {
            kind: RuntimeErrorKind::ExternCall {
                name: name.into(),
                error,
            },
        }
    }

    pub fn fetch(error: String) -> Self {
        Self {
            kind: RuntimeErrorKind::Fetch { error },
        }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self {
            kind: RuntimeErrorKind::Other(msg.into()),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            RuntimeErrorKind::TypeMismatch {
                operation,
                expected,
                got,
            } => write!(f, "{operation}: expected {expected}, got {got}"),
            RuntimeErrorKind::DivisionByZero => write!(f, "division by zero"),
            RuntimeErrorKind::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
            RuntimeErrorKind::EmptyCollection { operation } => {
                write!(f, "{operation}: empty collection")
            }
            RuntimeErrorKind::MissingField { field } => {
                write!(f, "missing field: {field}")
            }
            RuntimeErrorKind::ExternCall { name, error } => {
                write!(f, "extern call '{name}' failed: {error}")
            }
            RuntimeErrorKind::Fetch { error } => write!(f, "fetch failed: {error}"),
            RuntimeErrorKind::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}
