//! Regex extension functions via ExternRegistry.

use acvus_interpreter::iter::IterHandle;
use acvus_interpreter::{
    Defs, ExternFn, ExternRegistry, FromValue, IntoValue, OpaqueValue, RuntimeError, Uses, Value,
    ValueKind,
};
use acvus_mir::ty::{Effect, Ty, TypeRegistry, UserDefinedDecl, UserDefinedId};
use acvus_utils::Interner;

fn user_defined_ty(id: UserDefinedId) -> Ty {
    Ty::UserDefined {
        id,
        type_args: vec![],
        effect_args: vec![],
    }
}

/// Newtype for `regex::Regex` — carries its `UserDefinedId` for type-safe conversion.
struct Re(regex::Regex, UserDefinedId);

impl FromValue for Re {
    fn from_value(value: Value) -> Result<Self, RuntimeError> {
        match value {
            Value::Opaque(o) => {
                let id = o.type_id;
                let r = o.downcast_ref::<regex::Regex>().ok_or_else(|| {
                    RuntimeError::unexpected_type(
                        "FromValue<Re>",
                        &[ValueKind::Opaque],
                        ValueKind::Opaque,
                    )
                })?;
                Ok(Re(r.clone(), id))
            }
            other => Err(RuntimeError::unexpected_type(
                "FromValue<Re>",
                &[ValueKind::Opaque],
                other.kind(),
            )),
        }
    }
}

impl IntoValue for Re {
    fn into_value(self) -> Value {
        Value::opaque(OpaqueValue::new(self.1, self.0))
    }
}

/// Build the regex ExternRegistry.
/// Registers the `Regex` UserDefined type into `type_registry`.
pub fn regex_registry(type_registry: &mut TypeRegistry) -> ExternRegistry {
    let id = UserDefinedId::alloc();
    type_registry.register(UserDefinedDecl {
        id,
        name: "Regex".into(),
        type_params: vec![],
        effect_params: vec![],
    });

    let ty = user_defined_ty(id);
    ExternRegistry::new(move |_interner| {
        vec![
            // regex(pattern) -> Regex
            ExternFn::build("regex")
                .params(vec![Ty::String])
                .ret(ty.clone())
                .pure()
                .handler(
                    move |_interner: &Interner, (pattern,): (String,), Uses(()): Uses<()>| {
                        let re = regex::Regex::new(&pattern)
                            .unwrap_or_else(|e| panic!("regex: invalid pattern '{pattern}': {e}"));
                        Ok((Re(re, id), Defs(())))
                    },
                ),
            // regex_match(re, text) -> Bool
            ExternFn::build("regex_match")
                .params(vec![ty.clone(), Ty::String])
                .ret(Ty::Bool)
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                        Ok((re.is_match(&text), Defs(())))
                    },
                ),
            // regex_find(re, text) -> Option<String>
            ExternFn::build("regex_find")
                .params(vec![ty.clone(), Ty::String])
                .ret(Ty::String) // TODO: proper Option<String> return type
                .pure()
                .handler(
                    |interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                        let result = match re.find(&text) {
                            Some(m) => Value::some(interner, Value::string(m.as_str())),
                            None => Value::none(interner),
                        };
                        Ok((result, Defs(())))
                    },
                ),
            // regex_find_all(re, text) -> Iterator<String, SelfModifying>
            ExternFn::build("regex_find_all")
                .params(vec![ty.clone(), Ty::String])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                        let mut start = 0;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
                                let m = re.find_at(&text, start)?;
                                start = m.end();
                                Some(Value::string(m.as_str()))
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
            // regex_replace(text, re, replacement) -> String
            ExternFn::build("regex_replace")
                .params(vec![Ty::String, ty.clone(), Ty::String])
                .ret(Ty::String)
                .pure()
                .handler(
                    |_interner: &Interner,
                     (text, Re(re, _), rep): (String, Re, String),
                     Uses(()): Uses<()>| {
                        Ok((re.replace_all(&text, rep.as_str()).into_owned(), Defs(())))
                    },
                ),
            // regex_split(re, text) -> Iterator<String, SelfModifying>
            ExternFn::build("regex_split")
                .params(vec![ty.clone(), Ty::String])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner, (Re(re, _), text): (Re, String), Uses(()): Uses<()>| {
                        let mut last_end = 0;
                        let mut done = false;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
                                if done {
                                    return None;
                                }
                                match re.find_at(&text, last_end) {
                                    Some(m) => {
                                        let segment = &text[last_end..m.start()];
                                        last_end = m.end();
                                        Some(Value::string(segment))
                                    }
                                    None => {
                                        done = true;
                                        Some(Value::string(&text[last_end..]))
                                    }
                                }
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
            // regex_extract(text, re) -> Iterator<String, SelfModifying>  (capture group 1)
            ExternFn::build("regex_extract")
                .params(vec![Ty::String, ty.clone()])
                .ret(Ty::Iterator(Box::new(Ty::String), Effect::self_modifying()))
                .pure()
                .handler(
                    |_interner: &Interner,
                     (text, Re(re, _)): (String, Re),
                     Uses(()): Uses<()>| {
                        let mut start = 0;
                        let iter = Value::iterator(IterHandle::from_fn(
                            Effect::self_modifying(),
                            move || {
                                loop {
                                    let caps = re.captures_at(&text, start)?;
                                    let full = caps.get(0)?;
                                    start = full.end();
                                    if let Some(group1) = caps.get(1) {
                                        return Some(Value::string(group1.as_str()));
                                    }
                                }
                            },
                        ));
                        Ok((iter, Defs(())))
                    },
                ),
        ]
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
        let reg = regex_registry(&mut tr);
        let registered = reg.register(&i);
        assert_eq!(registered.functions.len(), 7);
        assert_eq!(registered.executables.len(), 7);
    }
}
