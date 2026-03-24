/// A bound placeholder: name + the AST subtree to substitute.
pub struct BoundPlaceholder<T> {
    pub name: &'static str,
    pub start: usize,
    pub end: usize,
    pub value: T,
}

/// Compile-time validated acvus template with bound placeholders.
pub struct AcvusTemplate<T> {
    pub source: &'static str,
    pub bindings: Vec<BoundPlaceholder<T>>,
}
