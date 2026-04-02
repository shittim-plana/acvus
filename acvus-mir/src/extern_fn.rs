//! ExternFn type system — interpreter-independent trait definitions.
//!
//! This module defines the foundation traits for the ExternFn redesign.
//! All traits live in `acvus-mir` because ExternFn signatures are expressed
//! in terms of `Ty` and `Effect` — no interpreter dependency.
//!
//! Interpreters provide `FromRepr<R>`/`IntoRepr<R>` impls for their own
//! value type `R`. The `Registrar` trait is the late-binding point where
//! type erasure happens at interpreter binding time, not at definition time.

use std::ops::Deref;

use smallvec::SmallVec;

use crate::ty::Ty;

/// Inline capacity for argument/return value vectors.
/// Most ExternFns have 1–4 parameters — avoids heap allocation.
const ARGS_INLINE: usize = 4;

// ── AcvusTy: Rust type → acvus Ty mapping ──────────────────────────

/// Maps a Rust type to its acvus `Ty` representation.
/// Used by the `#[extern_fn]` macro to derive `FnConstraint` from signatures.
pub trait AcvusTy {
    fn ty() -> Ty;
}

impl AcvusTy for i64 {
    fn ty() -> Ty {
        Ty::Int
    }
}
impl AcvusTy for f64 {
    fn ty() -> Ty {
        Ty::Float
    }
}
impl AcvusTy for String {
    fn ty() -> Ty {
        Ty::String
    }
}
impl AcvusTy for bool {
    fn ty() -> Ty {
        Ty::Bool
    }
}
impl AcvusTy for u8 {
    fn ty() -> Ty {
        Ty::Byte
    }
}
impl AcvusTy for () {
    fn ty() -> Ty {
        Ty::Unit
    }
}

impl<T: AcvusTy> AcvusTy for Vec<T> {
    fn ty() -> Ty {
        Ty::List(Box::new(T::ty()))
    }
}

impl<T: AcvusTy> AcvusTy for Option<T> {
    fn ty() -> Ty {
        Ty::Option(Box::new(T::ty()))
    }
}

// ── FromRepr / IntoRepr: interpreter ↔ ExternFn bridge ─────────────

/// Convert an interpreter's value representation `R` into a concrete Rust type.
///
/// Each interpreter implements this for its own `R`:
/// - treewalk: `impl FromRepr<Value> for i64 { ... }`
/// - kovac: `impl FromRepr<KovacSlot> for i64 { ... }`
///
/// `type Output` allows non-identity conversions (e.g. `Any`).
pub trait FromRepr<R>: Sized {
    type Output;
    fn from_repr(repr: R) -> Result<Self::Output, ExternError>;
}

/// Convert a concrete Rust type back into an interpreter's value representation.
pub trait IntoRepr<R> {
    fn into_repr(self) -> R;
}

// ── ArgPack / RetPack: tuple extensions ─────────────────────────────

/// Inline vector for ExternFn argument/return values.
pub type ReprVec<R> = SmallVec<[R; ARGS_INLINE]>;

/// Extract a tuple of concrete types from a sequence of interpreter values.
pub trait ArgPack<R>: Sized {
    fn from_reprs(reprs: ReprVec<R>) -> Result<Self, ExternError>;
}

/// Convert a return tuple back into interpreter values.
///
/// `type Output` tells the acvus type system what the call site sees.
pub trait RetPack<R> {
    type Output: AcvusTy;
    fn into_reprs(self) -> ReprVec<R>;
}

// ── ArgPack tuple impls ─────────────────────────────────────────────

impl<R> ArgPack<R> for () {
    fn from_reprs(_reprs: ReprVec<R>) -> Result<Self, ExternError> {
        Ok(())
    }
}

macro_rules! impl_arg_pack {
    ($($idx:tt $T:ident),+) => {
        impl<R, $($T: FromRepr<R, Output = $T>),+> ArgPack<R> for ($($T,)+) {
            fn from_reprs(reprs: ReprVec<R>) -> Result<Self, ExternError> {
                let mut iter = reprs.into_iter();
                let result = ( $( { let _idx = $idx; $T::from_repr(iter.next().ok_or(ExternError::ArgCount)?)?  }, )+ );
                Ok(result)
            }
        }
    }
}

impl_arg_pack!(0 T0);
impl_arg_pack!(0 T0, 1 T1);
impl_arg_pack!(0 T0, 1 T1, 2 T2);
impl_arg_pack!(0 T0, 1 T1, 2 T2, 3 T3);
impl_arg_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4);
impl_arg_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5);
impl_arg_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6);
impl_arg_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7);

// ── RetPack tuple impls ─────────────────────────────────────────────

macro_rules! impl_ret_pack {
    ($($idx:tt $T:ident),+) => {
        impl<R, $($T: IntoRepr<R> + AcvusTy),+> RetPack<R> for ($($T,)+) {
            type Output = ($($T,)+);
            fn into_reprs(self) -> ReprVec<R> {
                smallvec::smallvec![ $( self.$idx.into_repr(), )+ ]
            }
        }
    }
}

impl_ret_pack!(0 T0);
impl_ret_pack!(0 T0, 1 T1);
impl_ret_pack!(0 T0, 1 T1, 2 T2);
impl_ret_pack!(0 T0, 1 T1, 2 T2, 3 T3);
impl_ret_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4);
impl_ret_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5);
impl_ret_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6);
impl_ret_pack!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7);

// AcvusTy for tuples (needed by RetPack::Output bound)
macro_rules! impl_acvus_ty_tuple {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: AcvusTy),+> AcvusTy for ($($T,)+) {
            fn ty() -> Ty {
                Ty::Tuple(vec![ $( $T::ty(), )+ ])
            }
        }
    }
}

impl_acvus_ty_tuple!(0 T0);
impl_acvus_ty_tuple!(0 T0, 1 T1);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2, 3 T3);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6);
impl_acvus_ty_tuple!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7);

// ── Context + Use/Def ───────────────────────────────────────────────

/// Declares an ExternFn-visible context (shared state).
///
/// Implement on a marker struct. The macro derives read/write effects from
/// `Use<C>` / `Def<C>` parameters.
///
/// ```ignore
/// struct Offset;
/// impl Context for Offset {
///     const NAMESPACE: &'static str = "";
///     const NAME: &'static str = "offset";
///     type Value = u64;
/// }
/// ```
pub trait Context {
    const NAMESPACE: &'static str;
    const NAME: &'static str;
    type Value: AcvusTy;
}

/// Context read — immutable value captured at spawn time.
/// Dereferences to `C::Value`.
pub struct Use<C: Context>(pub C::Value);

impl<C: Context> Deref for Use<C> {
    type Target = C::Value;
    fn deref(&self) -> &C::Value {
        &self.0
    }
}

/// Context write — new value returned from handler.
pub struct Def<C: Context>(C::Value);

impl<C: Context> Def<C> {
    pub fn new(value: C::Value) -> Self {
        Def(value)
    }
    pub fn into_inner(self) -> C::Value {
        self.0
    }
}

// ── Any: materializable values ──────────────────────────────────────

/// Runtime value for ExternFns that return dynamically-typed data.
///
/// Contains only **materializable** types — no Identity-bearing types
/// (Deque, Iterator, Handle, FnValue). This is enforced structurally:
/// the enum simply doesn't have variants for those types.
///
/// When an ExternFn returns `Any`, the system validates it against the
/// call site's inferred type and wraps in `Option<T>`:
/// - Structure matches → `Some(converted_value)`
/// - Mismatch → `None`
///
/// The handler cannot influence this decision. System-enforced.
#[derive(Debug, Clone, PartialEq)]
pub enum Any {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Unit,
    Byte(u8),
    List(Vec<Any>),
    /// Ordered key-value pairs. Keys are strings.
    Object(Vec<(String, Any)>),
    Tuple(Vec<Any>),
    Option(Option<Box<Any>>),
}

// ── ExternError ─────────────────────────────────────────────────────

/// Lightweight error for ExternFn boundary conversions.
/// Separate from `RuntimeError` (which lives in acvus-interpreter).
#[derive(Debug, Clone)]
pub enum ExternError {
    /// Type mismatch during FromRepr conversion.
    TypeMismatch {
        expected: &'static str,
        got: String,
    },
    /// Wrong number of arguments.
    ArgCount,
    /// Custom error from handler.
    Custom(String),
}

impl std::fmt::Display for ExternError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::ArgCount => write!(f, "wrong number of arguments"),
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExternError {}

// ── Registrar + ExternFnSet ─────────────────────────────────────────

/// Late-binding point for ExternFn registration.
///
/// Each interpreter implements this with its own `Repr` type.
/// Type erasure happens here, not at ExternFn definition time.
pub trait Registrar {
    type Repr;
}

/// A composable set of ExternFn definitions.
///
/// Implemented for individual ExternFnDefs and for tuples of sets.
/// Allows type-level composition:
/// ```ignore
/// type StringFns = (LenStrFn, TrimFn, UpperFn);
/// type StdLib = (StringFns, ListFns, OptionFns);
/// StdLib::register(&mut treewalk_registrar);
/// ```
pub trait ExternFnSet<R> {
    fn register(registrar: &mut impl Registrar<Repr = R>);
}

// ExternFnSet tuple impls — recursive delegation.
macro_rules! impl_extern_fn_set_tuple {
    ($($T:ident),+) => {
        impl<R, $($T: ExternFnSet<R>),+> ExternFnSet<R> for ($($T,)+) {
            fn register(registrar: &mut impl Registrar<Repr = R>) {
                $( $T::register(registrar); )+
            }
        }
    }
}

impl_extern_fn_set_tuple!(A);
impl_extern_fn_set_tuple!(A, B);
impl_extern_fn_set_tuple!(A, B, C);
impl_extern_fn_set_tuple!(A, B, C, D);
impl_extern_fn_set_tuple!(A, B, C, D, E);
impl_extern_fn_set_tuple!(A, B, C, D, E, F);
impl_extern_fn_set_tuple!(A, B, C, D, E, F, G);
impl_extern_fn_set_tuple!(A, B, C, D, E, F, G, H);

// ── Tests: type solver verification ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Dummy interpreter representation for testing trait resolution.
    #[derive(Debug, Clone, PartialEq)]
    enum DummyRepr {
        Int(i64),
        Float(f64),
        Str(String),
        Bool(bool),
        Unit,
        Byte(u8),
        List(Vec<DummyRepr>),
    }

    // FromRepr impls for DummyRepr
    impl FromRepr<DummyRepr> for i64 {
        type Output = i64;
        fn from_repr(repr: DummyRepr) -> Result<i64, ExternError> {
            match repr {
                DummyRepr::Int(v) => Ok(v),
                other => Err(ExternError::TypeMismatch {
                    expected: "Int",
                    got: format!("{other:?}"),
                }),
            }
        }
    }

    impl FromRepr<DummyRepr> for String {
        type Output = String;
        fn from_repr(repr: DummyRepr) -> Result<String, ExternError> {
            match repr {
                DummyRepr::Str(v) => Ok(v),
                other => Err(ExternError::TypeMismatch {
                    expected: "String",
                    got: format!("{other:?}"),
                }),
            }
        }
    }

    impl FromRepr<DummyRepr> for bool {
        type Output = bool;
        fn from_repr(repr: DummyRepr) -> Result<bool, ExternError> {
            match repr {
                DummyRepr::Bool(v) => Ok(v),
                other => Err(ExternError::TypeMismatch {
                    expected: "Bool",
                    got: format!("{other:?}"),
                }),
            }
        }
    }

    // IntoRepr impls for DummyRepr
    impl IntoRepr<DummyRepr> for i64 {
        fn into_repr(self) -> DummyRepr {
            DummyRepr::Int(self)
        }
    }

    impl IntoRepr<DummyRepr> for String {
        fn into_repr(self) -> DummyRepr {
            DummyRepr::Str(self)
        }
    }

    impl IntoRepr<DummyRepr> for bool {
        fn into_repr(self) -> DummyRepr {
            DummyRepr::Bool(self)
        }
    }

    // ── AcvusTy tests ───────────────────────────────────────────────

    #[test]
    fn acvus_ty_primitives() {
        assert_eq!(i64::ty(), Ty::Int);
        assert_eq!(f64::ty(), Ty::Float);
        assert_eq!(String::ty(), Ty::String);
        assert_eq!(bool::ty(), Ty::Bool);
        assert_eq!(u8::ty(), Ty::Byte);
        assert_eq!(<()>::ty(), Ty::Unit);
    }

    #[test]
    fn acvus_ty_generic() {
        assert_eq!(Vec::<i64>::ty(), Ty::List(Box::new(Ty::Int)));
        assert_eq!(
            Option::<String>::ty(),
            Ty::Option(Box::new(Ty::String))
        );
    }

    #[test]
    fn acvus_ty_tuple() {
        assert_eq!(
            <(i64,)>::ty(),
            Ty::Tuple(vec![Ty::Int])
        );
        assert_eq!(
            <(i64, String)>::ty(),
            Ty::Tuple(vec![Ty::Int, Ty::String])
        );
    }

    // ── ArgPack tests ───────────────────────────────────────────────

    #[test]
    fn arg_pack_empty() {
        let reprs: ReprVec<DummyRepr> = smallvec::smallvec![];
        let () = <()>::from_reprs(reprs).unwrap();
    }

    #[test]
    fn arg_pack_single() {
        let reprs: ReprVec<DummyRepr> = smallvec::smallvec![DummyRepr::Int(42)];
        let (v,) = <(i64,)>::from_reprs(reprs).unwrap();
        assert_eq!(v, 42);
    }

    #[test]
    fn arg_pack_multi() {
        let reprs: ReprVec<DummyRepr> = smallvec::smallvec![
            DummyRepr::Int(1),
            DummyRepr::Str("hello".into()),
            DummyRepr::Bool(true),
        ];
        let (a, b, c) = <(i64, String, bool)>::from_reprs(reprs).unwrap();
        assert_eq!(a, 1);
        assert_eq!(b, "hello");
        assert_eq!(c, true);
    }

    #[test]
    fn arg_pack_type_mismatch() {
        let reprs: ReprVec<DummyRepr> = smallvec::smallvec![DummyRepr::Str("not a number".into())];
        let result = <(i64,)>::from_reprs(reprs);
        assert!(result.is_err());
    }

    // ── RetPack tests ───────────────────────────────────────────────

    #[test]
    fn ret_pack_single() {
        let reprs = (42i64,).into_reprs();
        assert_eq!(reprs.as_slice(), &[DummyRepr::Int(42)]);
    }

    #[test]
    fn ret_pack_multi() {
        let reprs = (1i64, "hello".to_string()).into_reprs();
        assert_eq!(
            reprs.as_slice(),
            &[DummyRepr::Int(1), DummyRepr::Str("hello".into())]
        );
    }

    #[test]
    fn ret_pack_output_type() {
        // Verify that Output is the tuple itself — type solver resolves this.
        fn check_output<R, T: RetPack<R>>()
        where
            T::Output: AcvusTy,
        {
        }
        check_output::<DummyRepr, (i64,)>();
        check_output::<DummyRepr, (i64, String)>();
    }

    // ── Context + Use/Def tests ─────────────────────────────────────

    struct Offset;
    impl Context for Offset {
        const NAMESPACE: &'static str = "";
        const NAME: &'static str = "offset";
        type Value = i64;
    }

    struct Name;
    impl Context for Name {
        const NAMESPACE: &'static str = "user";
        const NAME: &'static str = "name";
        type Value = String;
    }

    #[test]
    fn use_deref() {
        let u = Use::<Offset>(42);
        assert_eq!(*u, 42);
    }

    #[test]
    fn def_roundtrip() {
        let d = Def::<Offset>::new(100);
        assert_eq!(d.into_inner(), 100);
    }

    #[test]
    fn context_metadata() {
        assert_eq!(Offset::NAMESPACE, "");
        assert_eq!(Offset::NAME, "offset");
        assert_eq!(<Offset as Context>::Value::ty(), Ty::Int);

        assert_eq!(Name::NAMESPACE, "user");
        assert_eq!(Name::NAME, "name");
        assert_eq!(<Name as Context>::Value::ty(), Ty::String);
    }

    // ── Any tests ───────────────────────────────────────────────────

    #[test]
    fn any_nested_structure() {
        let val = Any::Object(vec![
            ("name".into(), Any::String("acvus".into())),
            ("version".into(), Any::Int(1)),
            (
                "tags".into(),
                Any::List(vec![Any::String("dsl".into()), Any::String("compiler".into())]),
            ),
        ]);
        match &val {
            Any::Object(fields) => assert_eq!(fields.len(), 3),
            _ => panic!("expected Object"),
        }
    }

    // ── ExternFnSet composition test ────────────────────────────────

    struct DummyRegistrar;
    impl Registrar for DummyRegistrar {
        type Repr = DummyRepr;
    }

    // Dummy ExternFn types to test set composition.
    struct FnA;
    struct FnB;
    struct FnC;

    impl ExternFnSet<DummyRepr> for FnA {
        fn register(_: &mut impl Registrar<Repr = DummyRepr>) {}
    }
    impl ExternFnSet<DummyRepr> for FnB {
        fn register(_: &mut impl Registrar<Repr = DummyRepr>) {}
    }
    impl ExternFnSet<DummyRepr> for FnC {
        fn register(_: &mut impl Registrar<Repr = DummyRepr>) {}
    }

    #[test]
    fn extern_fn_set_composition() {
        // Flat tuple
        type Group1 = (FnA, FnB);
        // Nested tuple
        type StdLib = (Group1, FnC);

        // Type solver must resolve these.
        Group1::register(&mut DummyRegistrar);
        StdLib::register(&mut DummyRegistrar);
    }

    // ── Full pipeline type solver test ───────────────────────────────

    /// Simulates what the macro + registrar would do:
    /// 1. Extract args via ArgPack
    /// 2. Call handler with concrete types
    /// 3. Convert result via RetPack
    #[test]
    fn full_pipeline_simulation() {
        // Simulate: fn add(a: i64, b: i64) -> (i64,)
        fn handler(a: i64, b: i64) -> (i64,) {
            (a + b,)
        }

        // 1. ArgPack: DummyRepr → (i64, i64)
        let args: ReprVec<DummyRepr> = smallvec::smallvec![DummyRepr::Int(10), DummyRepr::Int(32)];
        let (a, b) = <(i64, i64)>::from_reprs(args).unwrap();

        // 2. Call handler
        let result = handler(a, b);

        // 3. RetPack: (i64,) → ReprVec<DummyRepr>
        let reprs = result.into_reprs();
        assert_eq!(reprs.as_slice(), &[DummyRepr::Int(42)]);
    }
}
