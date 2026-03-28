//! DateTime extension functions via ExternRegistry.
//!
//! Provides UserDefined<DateTime> with formatting and parsing.
//! All functions are pure — DateTime is immutable.

use acvus_interpreter::{
    Defs, ExternFn, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses, Value,
    ValueKind,
};
use acvus_mir::ty::{Ty, TypeRegistry, UserDefinedDecl, UserDefinedId};
use acvus_utils::Interner;

fn user_defined_ty(id: UserDefinedId) -> Ty {
    Ty::UserDefined {
        id,
        type_args: vec![],
        effect_args: vec![],
    }
}

/// Newtype for `chrono::DateTime<Utc>` — carries its `UserDefinedId`.
struct Dt(chrono::DateTime<chrono::Utc>, UserDefinedId);

impl FromValue for Dt {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let id = o.type_id;
                let dt = o
                    .downcast_ref::<chrono::DateTime<chrono::Utc>>()
                    .ok_or_else(|| {
                        RuntimeError::unexpected_type(
                            "FromValue<Dt>",
                            &[ValueKind::Opaque],
                            ValueKind::Opaque,
                        )
                    })?;
                Ok(Dt(*dt, id))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Dt>",
                &[ValueKind::Opaque],
                other.kind(),
            )),
        }
    }
}

impl IntoValue for Dt {
    fn into_value(self) -> Value {
        Value::opaque(OpaqueValue::new(self.1, self.0))
    }
}

/// Build the datetime ExternRegistry.
/// Registers the `DateTime` UserDefined type into `type_registry`.
pub fn datetime_registry(type_registry: &mut TypeRegistry) -> ExternRegistry {
    let id = UserDefinedId::alloc();
    type_registry.register(UserDefinedDecl {
        id,
        name: "DateTime".into(),
        type_params: vec![],
        effect_params: vec![],
    });

    let ty = user_defined_ty(id);
    ExternRegistry::new(move |_interner| {
        let mut fns = Vec::new();

        // now() -> DateTime  (not available on wasm — requires system clock)
        #[cfg(not(target_arch = "wasm32"))]
        fns.push(
            ExternFn::build("now")
                .params(vec![])
                .ret(ty.clone())
                .io()
                .handler(move |_interner: &Interner, (): (), Uses(()): Uses<()>| {
                    Ok((Dt(chrono::Utc::now(), id), Defs(())))
                }),
        );

        fns.extend([
            // format_date(dt, fmt) -> String
            ExternFn::build("format_date")
                .params(vec![ty.clone(), Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner,
                     (Dt(dt, _), fmt): (Dt, String),
                     Uses(()): Uses<()>| {
                        Ok((dt.format(&fmt).to_string(), Defs(())))
                    },
                ),
            // parse_date(s, fmt) -> DateTime
            ExternFn::build("parse_date")
                .params(vec![Ty::String, Ty::String])
                .ret(ty.clone())
                .pure()
                .handler(
                    move |_interner: &Interner,
                          (s, fmt): (String, String),
                          Uses(()): Uses<()>| {
                        let dt = chrono::NaiveDateTime::parse_from_str(&s, &fmt)
                            .map(|ndt| ndt.and_utc())
                            .unwrap_or_else(|e| {
                                panic!(
                                    "parse_date: invalid input '{s}' with format '{fmt}': {e}"
                                )
                            });
                        Ok((Dt(dt, id), Defs(())))
                    },
                ),
            // timestamp(dt) -> Int  (Unix epoch seconds)
            ExternFn::build("timestamp")
                .params(vec![ty.clone()])
                .ret(Ty::Int)
                .pure()
                .handler(
                    |_interner: &Interner, (Dt(dt, _),): (Dt,), Uses(()): Uses<()>| {
                        Ok((dt.timestamp(), Defs(())))
                    },
                ),
            // from_timestamp(epoch) -> DateTime
            ExternFn::build("from_timestamp")
                .params(vec![Ty::Int])
                .ret(ty.clone())
                .pure()
                .handler(
                    move |_interner: &Interner, (epoch,): (i64,), Uses(()): Uses<()>| {
                        let dt = chrono::DateTime::from_timestamp(epoch, 0)
                            .unwrap_or_else(|| panic!("from_timestamp: invalid epoch {epoch}"));
                        Ok((Dt(dt, id), Defs(())))
                    },
                ),
            // add_days(dt, n) -> DateTime
            ExternFn::build("add_days")
                .params(vec![ty.clone(), Ty::Int])
                .ret(ty.clone())
                .pure()
                .handler(
                    move |_interner: &Interner,
                          (Dt(dt, _), n): (Dt, i64),
                          Uses(()): Uses<()>| {
                        Ok((Dt(dt + chrono::Duration::days(n), id), Defs(())))
                    },
                ),
            // add_hours(dt, n) -> DateTime
            ExternFn::build("add_hours")
                .params(vec![ty.clone(), Ty::Int])
                .ret(ty.clone())
                .pure()
                .handler(
                    move |_interner: &Interner,
                          (Dt(dt, _), n): (Dt, i64),
                          Uses(()): Uses<()>| {
                        Ok((Dt(dt + chrono::Duration::hours(n), id), Defs(())))
                    },
                ),
        ]);

        fns
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let mut tr = TypeRegistry::new();
        let reg = datetime_registry(&mut tr);
        let registered = reg.register(&i);
        // 6 pure functions + now() on non-wasm targets.
        assert!(registered.functions.len() >= 6);
        assert_eq!(registered.functions.len(), registered.executables.len());
    }
}
