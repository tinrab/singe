//! Require the `mod` declarations at the head of a module to come before its imports there, so its children are listed first and the `pub use self::child::…` re-exports that follow read them.
//! The head is the leading run of declarations and imports; a declaration after other items, such as one that must follow the macros it uses, is left where it is.
//! A module gated on tests is exempt, since `require-tests-module-last` places it.

use syn::{File, Item, ItemMod, Meta, visit::Visit};

use super::super::{context::Context, diagnostic::Diagnostic};

pub fn check(file: &File, context: &Context<'_>) -> Vec<Diagnostic> {
    struct Check<'a> {
        context: &'a Context<'a>,
        findings: Vec<Diagnostic>,
    }

    impl Check<'_> {
        fn check_items(&mut self, items: &[Item]) {
            let head = items
                .iter()
                .position(|item| {
                    !matches!(item, Item::Use(_))
                        && !matches!(item, Item::Mod(module) if module.content.is_none())
                })
                .unwrap_or(items.len());
            let items = &items[..head];
            let Some(first_import) = items.iter().position(|item| matches!(item, Item::Use(_)))
            else {
                return;
            };
            for item in &items[first_import..] {
                if let Item::Mod(module) = item
                    && module.content.is_none()
                    && !is_test_only(module)
                {
                    self.findings.push(self.context.report(
                        "require-mods-before-imports",
                        module.ident.span(),
                        "`mod` declarations must come before the module's imports",
                        "Move this declaration above the first `use`.",
                    ));
                }
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

fn is_test_only(module: &ItemMod) -> bool {
    module.attrs.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && matches!(&attribute.meta, Meta::List(list) if list.tokens.to_string().contains("test"))
    })
}
