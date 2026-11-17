// Names of the generated items: the Rust identifier of each kernel, its signature type, and its methods.

use std::collections::HashMap;

use proc_macro2::Span;
use quote::format_ident;
use syn::{Error, Ident, Result};

use super::{KernelMetadata, source::is_c_identifier_char};

// The method suffixes every kernel gets, one entry point each.
pub(crate) const ENTRY_POINTS: [&str; 4] = ["_async", "_record", "_node", "_set_node_params"];

// The names `Module` defines besides the kernels' methods.
const MODULE_ITEMS: [&str; 3] = ["create", "module", "SOURCE"];

// Converts a CUDA kernel name to a Rust identifier, reporting an unusable name at `span`.
pub(crate) fn kernel_identifier(name: &str, span: Span) -> Result<Ident> {
    let mut identifier = name
        .chars()
        .map(|ch| if is_c_identifier_char(ch) { ch } else { '_' })
        .collect::<String>();
    if identifier.is_empty() {
        identifier.push_str("arg");
    }

    if identifier.starts_with(|ch: char| ch.is_ascii_digit()) {
        identifier.insert(0, '_');
    }

    // `syn` rejects strict and reserved keywords but not `gen`, which edition 2024 reserves.
    if identifier != "gen" && syn::parse_str::<Ident>(&identifier).is_ok() {
        return Ok(Ident::new(&identifier, span));
    }

    if matches!(
        identifier.as_str(),
        "_" | "self" | "Self" | "crate" | "super"
    ) {
        return Err(Error::new(
            span,
            format!(
                "kernel name `{name}` is a rust keyword that cannot be a raw identifier; export it under an alias with `as`"
            ),
        ));
    }
    Ok(Ident::new_raw(&identifier, span))
}

pub(crate) fn signature_name(name: &Ident) -> Ident {
    let span = name.span();
    let name = name
        .to_string()
        .trim_start_matches("r#")
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map_or_default(|first| first.to_uppercase().to_string() + chars.as_str())
        })
        .collect::<String>();
    format_ident!("{}Signature", name, span = span)
}

pub(crate) fn prepare_name(name: &Ident) -> Ident {
    format_ident!("prepare_{}", name, span = name.span())
}

pub(crate) fn entry_point_name(name: &Ident, suffix: &str) -> Ident {
    format_ident!("{}{}", name, suffix, span = name.span())
}

// Rejects kernels whose generated types or methods collide, pointing at both kernels.
pub(crate) fn validate_generated_names(kernels: &[KernelMetadata]) -> Result<()> {
    fn register(names: &mut HashMap<String, Span>, name: &Ident) -> Result<()> {
        let key = name.to_string().trim_start_matches("r#").to_owned();
        if let Some(previous) = names.insert(key.clone(), name.span()) {
            let mut error = Error::new(
                name.span(),
                format!(
                    "generated name `{key}` conflicts with another item; use distinct export aliases"
                ),
            );
            error.combine(Error::new(previous, "conflicting item is generated here"));
            return Err(error);
        }
        Ok(())
    }

    let mut types = HashMap::new();
    let mut methods = HashMap::new();
    for name in MODULE_ITEMS {
        register(&mut methods, &Ident::new(name, Span::call_site()))?;
    }

    for kernel in kernels {
        register(&mut types, &signature_name(&kernel.rust_name))?;
        register(&mut methods, &prepare_name(&kernel.rust_name))?;
        for suffix in ENTRY_POINTS {
            register(&mut methods, &entry_point_name(&kernel.rust_name, suffix))?;
        }
    }
    Ok(())
}
