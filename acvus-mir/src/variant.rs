use std::collections::HashMap;

use crate::ty::{Ty, TySubst};

/// Payload specification for a variant constructor.
#[derive(Debug, Clone)]
pub enum VariantPayload {
    /// No payload (e.g. `None`).
    None,
    /// Payload type is the i-th type parameter of the parent enum.
    TypeParam(usize),
}

/// Definition of a single variant within an enum.
#[derive(Debug, Clone)]
pub struct VariantDef {
    pub tag: String,
    pub payload: VariantPayload,
}

/// Definition of an enum (tagged union) type.
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: String,
    pub type_param_count: usize,
    pub variants: Vec<VariantDef>,
}

/// Registry of all known enum types, indexed by variant tag.
#[derive(Debug, Clone)]
pub struct VariantRegistry {
    enums: Vec<EnumDef>,
    /// tag → (enum index, variant index)
    tag_index: HashMap<String, (usize, usize)>,
}

impl Default for VariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VariantRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            enums: Vec::new(),
            tag_index: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        self.register(EnumDef {
            name: "Option".into(),
            type_param_count: 1,
            variants: vec![
                VariantDef {
                    tag: "Some".into(),
                    payload: VariantPayload::TypeParam(0),
                },
                VariantDef {
                    tag: "None".into(),
                    payload: VariantPayload::None,
                },
            ],
        });
    }

    pub fn register(&mut self, def: EnumDef) {
        let enum_idx = self.enums.len();
        for (var_idx, variant) in def.variants.iter().enumerate() {
            let prev = self
                .tag_index
                .insert(variant.tag.clone(), (enum_idx, var_idx));
            assert!(prev.is_none(), "duplicate variant tag: {}", variant.tag);
        }
        self.enums.push(def);
    }

    /// Resolve a variant tag to its parent enum and variant definition.
    pub fn resolve(&self, tag: &str) -> Option<(&EnumDef, &VariantDef)> {
        let &(enum_idx, var_idx) = self.tag_index.get(tag)?;
        Some((
            &self.enums[enum_idx],
            &self.enums[enum_idx].variants[var_idx],
        ))
    }
}

/// Construct a `Ty` for the given enum with resolved type parameters.
///
/// Currently only `Option` is supported as a built-in enum.
/// When `Ty::Enum { name, args }` is added, this can be generalized.
pub fn make_enum_ty(enum_name: &str, type_params: &[Ty], subst: &TySubst) -> Ty {
    match enum_name {
        "Option" => Ty::Option(Box::new(subst.resolve(&type_params[0]))),
        _ => panic!("unknown enum type: {enum_name}"),
    }
}
