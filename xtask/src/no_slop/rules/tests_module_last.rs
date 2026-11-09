//! Require `mod tests` to end its module, so a reader finds the module's own items first.
//! Only an item that a test attribute gates may follow it, such as a second test-only module.

use syn::{Attribute, File, Item, ItemMod, Meta, visit::Visit};

use super::super::{context::Context, diagnostic::Diagnostic};

pub fn check(file: &File, context: &Context<'_>) -> Vec<Diagnostic> {
    struct Check<'a> {
        context: &'a Context<'a>,
        findings: Vec<Diagnostic>,
    }

    impl Check<'_> {
        fn check_items(&mut self, items: &[Item]) {
            let Some(tests) = items
                .iter()
                .position(|item| matches!(item, Item::Mod(module) if module.ident == "tests"))
            else {
                return;
            };
            if items[tests + 1..].iter().all(is_test_only) {
                return;
            }
            if let Item::Mod(module) = &items[tests] {
                self.findings.push(self.context.report(
                    "require-tests-module-last",
                    module.ident.span(),
                    "`mod tests` must come after the module's other items",
                    "Move `mod tests` to the end of the module.",
                ));
            }
        }
    }

    impl<'ast> Visit<'ast> for Check<'_> {
        fn visit_item_mod(&mut self, module: &'ast ItemMod) {
            if let Some((_, items)) = &module.content {
                self.check_items(items);
                for item in items {
                    self.visit_item(item);
                }
            }
        }
    }

    let mut check = Check {
        context,
        findings: Vec::new(),
    };
    check.check_items(&file.items);
    for item in &file.items {
        check.visit_item(item);
    }
    check.findings
}

fn is_test_only(item: &Item) -> bool {
    let attributes: &[Attribute] = match item {
        Item::Mod(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        _ => return false,
    };
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && matches!(&attribute.meta, Meta::List(list) if list.tokens.to_string().contains("test"))
    })
}
