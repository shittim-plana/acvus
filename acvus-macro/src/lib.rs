use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

/// Placeholder info extracted from acvus source.
struct Placeholder {
    name: String,
    /// Byte offset range in the original source.
    start: usize,
    end: usize,
}

/// Scan acvus source for `%ident` placeholders.
/// Returns (substituted source with placeholders replaced by dummy idents, placeholder list).
fn extract_placeholders(source: &str) -> (String, Vec<Placeholder>) {
    let mut result = String::with_capacity(source.len());
    let mut placeholders = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 1 < bytes.len() && is_ident_start(bytes[i + 1]) {
            let start = i;
            i += 1; // skip '%'
            let name_start = i;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let name = &source[name_start..i];
            let dummy = format!("__acvus_ph_{name}__");
            placeholders.push(Placeholder {
                name: name.to_string(),
                start,
                end: i,
            });
            result.push_str(&dummy);
        } else {
            result.push(source[i..].chars().next().unwrap());
            i += source[i..].chars().next().unwrap().len_utf8();
        }
    }

    (result, placeholders)
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Validate acvus source (with placeholders substituted) as a script.
fn validate_script(source: &str) -> Result<(), String> {
    let interner = acvus_utils::Interner::new();
    match acvus_ast::parse_script(&interner, source) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Validate acvus source (with placeholders substituted) as a template.
fn validate_template(source: &str) -> Result<(), String> {
    let interner = acvus_utils::Interner::new();
    match acvus_ast::parse(&interner, source) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// `acvus_script!("source with %placeholders")`
///
/// Validates acvus script syntax at compile time.
/// Returns a closure that takes `(&Interner, placeholder_exprs...)` and returns a `Script`.
///
/// ```ignore
/// let result_ast: Expr = ...;
/// let make = acvus_script!("@history = append(@history, %result); @history");
/// let script: Script = make(&interner, result_ast);
/// ```
#[proc_macro]
pub fn acvus_script(input: TokenStream) -> TokenStream {
    let source_lit = parse_macro_input!(input as LitStr);
    let source = source_lit.value();

    let (substituted, placeholders) = extract_placeholders(&source);

    if let Err(e) = validate_script(&substituted) {
        return syn::Error::new(source_lit.span(), format!("acvus script parse error: {e}"))
            .to_compile_error()
            .into();
    }

    emit_script_closure(&substituted, &placeholders)
}

/// `acvus_template!("source with %placeholders")`
///
/// Validates acvus template syntax at compile time.
/// Returns a closure that takes `(&Interner, placeholder_exprs...)` and returns a `Template`.
#[proc_macro]
pub fn acvus_template(input: TokenStream) -> TokenStream {
    let source_lit = parse_macro_input!(input as LitStr);
    let source = source_lit.value();

    let (substituted, placeholders) = extract_placeholders(&source);

    if let Err(e) = validate_template(&substituted) {
        return syn::Error::new(source_lit.span(), format!("acvus template parse error: {e}"))
            .to_compile_error()
            .into();
    }

    emit_template_closure(&substituted, &placeholders)
}

fn emit_script_closure(substituted: &str, placeholders: &[Placeholder]) -> TokenStream {
    let ph_idents: Vec<proc_macro2::Ident> = placeholders
        .iter()
        .map(|p| proc_macro2::Ident::new(&p.name, proc_macro2::Span::call_site()))
        .collect();
    let ph_dummy_names: Vec<String> = placeholders
        .iter()
        .map(|p| format!("__acvus_ph_{}__", p.name))
        .collect();

    if placeholders.is_empty() {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner| -> acvus_ast::Script {
                acvus_ast::parse_script(__acvus_interner__, #substituted)
                    .expect("acvus_script!: pre-validated source failed to parse")
            }
        };
        expanded.into()
    } else {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner, #( #ph_idents: acvus_ast::Expr ),*|
                -> acvus_ast::Script
            {
                let __script__ = acvus_ast::parse_script(__acvus_interner__, #substituted)
                    .expect("acvus_script!: pre-validated source failed to parse");
                let mut __subs__ = rustc_hash::FxHashMap::default();
                #(
                    __subs__.insert(
                        __acvus_interner__.intern(#ph_dummy_names),
                        #ph_idents,
                    );
                )*
                acvus_ast::substitute::substitute_script(__script__, &__subs__)
            }
        };
        expanded.into()
    }
}

fn emit_template_closure(substituted: &str, placeholders: &[Placeholder]) -> TokenStream {
    let ph_idents: Vec<proc_macro2::Ident> = placeholders
        .iter()
        .map(|p| proc_macro2::Ident::new(&p.name, proc_macro2::Span::call_site()))
        .collect();
    let ph_dummy_names: Vec<String> = placeholders
        .iter()
        .map(|p| format!("__acvus_ph_{}__", p.name))
        .collect();

    if placeholders.is_empty() {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner| -> acvus_ast::Template {
                acvus_ast::parse_template(__acvus_interner__, #substituted)
                    .expect("acvus_template!: pre-validated source failed to parse")
            }
        };
        expanded.into()
    } else {
        let expanded = quote! {
            |__acvus_interner__: &acvus_utils::Interner, #( #ph_idents: acvus_ast::Expr ),*|
                -> acvus_ast::Template
            {
                let __template__ = acvus_ast::parse_template(__acvus_interner__, #substituted)
                    .expect("acvus_template!: pre-validated source failed to parse");
                let mut __subs__ = rustc_hash::FxHashMap::default();
                #(
                    __subs__.insert(
                        __acvus_interner__.intern(#ph_dummy_names),
                        #ph_idents,
                    );
                )*
                acvus_ast::substitute::substitute_template(__template__, &__subs__)
            }
        };
        expanded.into()
    }
}
