//! Emits Rust declarations as JSON for documentation review.
//!
//! Run `cargo run -p xtask --example doc_items -- <source paths...>` to list
//! declarations, their documentation, and their source locations. Macro bodies
//! that contain ordinary Rust items are parsed when possible.
//!
//! Pass `--tokens` before the paths to emit parsed tokens with documentation
//! attributes removed and optional trailing commas in argument and field lists
//! normalized. Comparing these records checks whether an edit changed
//! declarations, values, bodies, or attributes other than documentation.

use std::{env, fs, path::Path};

use anyhow::{Context, Result};
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::ToTokens;
use serde::Serialize;
use syn::{
    Attribute, Expr, Fields, FieldsNamed, FieldsUnnamed, File, ForeignItemFn, ImplItem, ImplItemFn,
    Item, ItemEnum, ItemFn, ItemImpl, ItemMacro, ItemMod, ItemStruct, ItemTrait, Lit, Meta,
    Signature, TraitItem, Variant, parse_file,
    spanned::Spanned,
    visit::{self, Visit},
    visit_mut::{self, VisitMut},
};
use walkdir::WalkDir;

#[derive(Serialize)]
struct Declaration {
    file: String,
    kind: &'static str,
    name: String,
    owner: String,
    line: usize,
    column: usize,
    end_line: usize,
    end_column: usize,
    insert_line: usize,
    docs: Vec<String>,
    declaration: String,
    body: Option<String>,
}

struct Collector {
    file: String,
    owners: Vec<String>,
    items: Vec<Declaration>,
}

impl Collector {
    fn record(
        &mut self,
        kind: &'static str,
        name: impl Into<String>,
        node: &impl ToTokens,
        attrs: &[Attribute],
        body: Option<String>,
    ) {
        let name = name.into();
        let tokens = node.to_token_stream();
        let span = tokens.span();
        let start = span.start();
        let end = span.end();
        let insert_line = attrs
            .first()
            .map_or(start.line, |attr| attr.span().start().line);
        let docs = attrs
            .iter()
            .filter_map(|attr| {
                if attr.path().is_ident("doc")
                    && let Meta::NameValue(meta) = &attr.meta
                    && let Expr::Lit(value) = &meta.value
                    && let Lit::Str(value) = &value.lit
                {
                    Some(value.value())
                } else {
                    None
                }
            })
            .collect();
        let declaration = if kind == "module" {
            format!("mod {name}")
        } else {
            tokens.to_string()
        };
        self.items.push(Declaration {
            file: self.file.clone(),
            kind,
            name,
            owner: self.owners.join("::"),
            line: start.line,
            column: start.column,
            end_line: end.line,
            end_column: end.column,
            insert_line,
            docs,
            declaration,
            body,
        });
    }

    fn fields(&mut self, fields: &Fields) {
        for (index, field) in fields.iter().enumerate() {
            let name = field
                .ident
                .as_ref()
                .map_or_else(|| index.to_string(), ToString::to_string);
            self.record("field", name, field, &field.attrs, None);
        }
    }

    fn function(&mut self, sig: &Signature, attrs: &[Attribute], body: &impl ToTokens) {
        self.record(
            "function",
            sig.ident.to_string(),
            sig,
            attrs,
            Some(body.to_token_stream().to_string()),
        );
    }
}

impl<'ast> Visit<'ast> for Collector {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Const(item) => {
                self.record("constant", item.ident.to_string(), item, &item.attrs, None)
            }
            Item::Static(item) => {
                self.record("static", item.ident.to_string(), item, &item.attrs, None)
            }
            Item::Type(item) => {
                self.record("type", item.ident.to_string(), item, &item.attrs, None)
            }
            Item::Trait(item) => {
                self.record("trait", item.ident.to_string(), item, &item.attrs, None)
            }
            Item::Union(item) => {
                self.record("union", item.ident.to_string(), item, &item.attrs, None);
                self.owners.push(item.ident.to_string());
                for field in &item.fields.named {
                    self.record(
                        "field",
                        field
                            .ident
                            .as_ref()
                            .expect("union fields are named")
                            .to_string(),
                        field,
                        &field.attrs,
                        None,
                    );
                }
                self.owners.pop();
            }
            _ => (),
        }
        visit::visit_item(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast ItemMod) {
        self.record("module", item.ident.to_string(), item, &item.attrs, None);
        self.owners.push(item.ident.to_string());
        visit::visit_item_mod(self, item);
        self.owners.pop();
    }

    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        self.record("struct", item.ident.to_string(), item, &item.attrs, None);
        self.owners.push(item.ident.to_string());
        self.fields(&item.fields);
        self.owners.pop();
    }

    fn visit_item_enum(&mut self, item: &'ast ItemEnum) {
        self.record("enum", item.ident.to_string(), item, &item.attrs, None);
        self.owners.push(item.ident.to_string());
        for variant in &item.variants {
            self.visit_variant(variant);
        }
        self.owners.pop();
    }

    fn visit_variant(&mut self, item: &'ast Variant) {
        self.record("variant", item.ident.to_string(), item, &item.attrs, None);
        self.owners.push(item.ident.to_string());
        self.fields(&item.fields);
        self.owners.pop();
    }

    fn visit_item_fn(&mut self, item: &'ast ItemFn) {
        self.function(&item.sig, &item.attrs, &item.block);
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        self.owners.push(item.self_ty.to_token_stream().to_string());
        visit::visit_item_impl(self, item);
        self.owners.pop();
    }

    fn visit_impl_item_fn(&mut self, item: &'ast ImplItemFn) {
        self.function(&item.sig, &item.attrs, &item.block);
    }

    fn visit_foreign_item_fn(&mut self, item: &'ast ForeignItemFn) {
        self.record(
            "function",
            item.sig.ident.to_string(),
            item,
            &item.attrs,
            None,
        );
    }

    fn visit_impl_item(&mut self, item: &'ast ImplItem) {
        if let ImplItem::Const(item) = item {
            self.record("constant", item.ident.to_string(), item, &item.attrs, None);
        }
        visit::visit_impl_item(self, item);
    }

    fn visit_item_trait(&mut self, item: &'ast ItemTrait) {
        self.owners.push(item.ident.to_string());
        visit::visit_item_trait(self, item);
        self.owners.pop();
    }

    fn visit_trait_item(&mut self, item: &'ast TraitItem) {
        match item {
            TraitItem::Fn(item) => self.record(
                "function",
                item.sig.ident.to_string(),
                &item.sig,
                &item.attrs,
                item.default
                    .as_ref()
                    .map(|body| body.to_token_stream().to_string()),
            ),
            TraitItem::Type(item) => self.record(
                "associated_type",
                item.ident.to_string(),
                item,
                &item.attrs,
                None,
            ),
            TraitItem::Const(item) => {
                self.record("constant", item.ident.to_string(), item, &item.attrs, None)
            }
            _ => (),
        }
        visit::visit_trait_item(self, item);
    }

    fn visit_item_macro(&mut self, item: &'ast ItemMacro) {
        if let Some(ident) = &item.ident {
            self.record("macro", ident.to_string(), item, &item.attrs, None);
        } else if let Ok(file) = syn::parse2::<File>(item.mac.tokens.clone()) {
            self.visit_file(&file);
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let tokens_only = args.first().is_some_and(|arg| arg == "--tokens");
    for path in args.iter().skip(usize::from(tokens_only)) {
        for entry in WalkDir::new(Path::new(&path))
            .into_iter()
            .filter_entry(|entry| {
                !matches!(
                    entry.file_name().to_str(),
                    Some("target" | "thirdparty" | ".git")
                )
            })
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() || path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let source = fs::read_to_string(path)?;
            let file =
                parse_file(&source).with_context(|| format!("cannot parse {}", path.display()))?;
            if tokens_only {
                let code = audit_tokens(file).to_string();
                println!(
                    "{}",
                    serde_json::json!({"file": path.display().to_string(), "code": code})
                );
                continue;
            }
            let mut collector = Collector {
                file: path.display().to_string(),
                owners: Vec::new(),
                items: Vec::new(),
            };
            collector.visit_file(&file);
            for item in collector.items {
                println!("{}", serde_json::to_string(&item)?);
            }
        }
    }
    Ok(())
}

fn audit_tokens(mut file: File) -> TokenStream {
    struct NormalizeLists;

    impl VisitMut for NormalizeLists {
        fn visit_fields_named_mut(&mut self, fields: &mut FieldsNamed) {
            let _ = fields.named.pop_punct();
            visit_mut::visit_fields_named_mut(self, fields);
        }

        fn visit_fields_unnamed_mut(&mut self, fields: &mut FieldsUnnamed) {
            let _ = fields.unnamed.pop_punct();
            visit_mut::visit_fields_unnamed_mut(self, fields);
        }

        fn visit_signature_mut(&mut self, signature: &mut Signature) {
            if signature.variadic.is_none() {
                let _ = signature.inputs.pop_punct();
            }
            visit_mut::visit_signature_mut(self, signature);
        }

        fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
            let _ = call.args.pop_punct();
            visit_mut::visit_expr_call_mut(self, call);
        }
    }

    NormalizeLists.visit_file_mut(&mut file);
    without_docs(file.to_token_stream())
}

fn without_docs(tokens: TokenStream) -> TokenStream {
    let mut output = TokenStream::new();
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        if matches!(&token, TokenTree::Punct(punct) if punct.as_char() == '#') {
            let mut lookahead = tokens.clone();
            let inner =
                matches!(lookahead.peek(), Some(TokenTree::Punct(punct)) if punct.as_char() == '!');
            if inner {
                lookahead.next();
            }
            if matches!(lookahead.next(), Some(TokenTree::Group(group)) if is_prose_attribute(&group))
            {
                if inner {
                    tokens.next();
                }
                tokens.next();
                continue;
            }
        }
        let token = if let TokenTree::Group(group) = token {
            TokenTree::Group(Group::new(group.delimiter(), without_docs(group.stream())))
        } else {
            token
        };
        output.extend([token]);
    }
    output
}

fn is_prose_attribute(group: &Group) -> bool {
    if group.delimiter() != Delimiter::Bracket {
        return false;
    }
    let mut tokens = group.stream().into_iter();
    matches!(tokens.next(), Some(TokenTree::Ident(name)) if name == "doc")
        && matches!(tokens.next(), Some(TokenTree::Punct(punct)) if punct.as_char() == '=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_normalizes_optional_commas_without_changing_tuple_arity() {
        let a = parse_file(
            "struct S { value: u32, } struct T(u32,); fn f(x: u32,) { g(x,); let t = (x,); }",
        )
        .unwrap();
        let b = parse_file(
            "struct S { value: u32 } struct T(u32); fn f(x: u32) { g(x); let t = (x,); }",
        )
        .unwrap();
        let different = parse_file(
            "struct S { value: u32 } struct T(u32); fn f(x: u32) { g(x); let t = (x); }",
        )
        .unwrap();
        assert_eq!(
            audit_tokens(a).to_string(),
            audit_tokens(b.clone()).to_string()
        );
        assert_ne!(
            audit_tokens(b).to_string(),
            audit_tokens(different).to_string()
        );
    }

    #[test]
    fn audit_removes_inner_outer_and_macro_docs_but_keeps_other_attributes() {
        let documented: TokenStream = r#"
            #![doc = "Crate docs"]
            #![allow(non_camel_case_types)]
            #[doc = "A native type."]
            #[doc(hidden)]
            #[doc(alias = "OriginalNative")]
            #[deprecated]
            pub struct Native { #[doc = "Field docs"] pub value: u32 }
            macro_rules! declare { () => { #[doc = "Generated docs"] pub fn call() {} }; }
        "#
        .parse()
        .unwrap();
        let expected: TokenStream = r#"
            #![allow(non_camel_case_types)]
            #[doc(hidden)]
            #[doc(alias = "OriginalNative")]
            #[deprecated]
            pub struct Native { pub value: u32 }
            macro_rules! declare { () => { pub fn call() {} }; }
        "#
        .parse()
        .unwrap();
        assert_eq!(without_docs(documented).to_string(), expected.to_string());
    }
}
