//! Reject `"literal".to_string().into()`, which allocates where `"literal".into()` gives a borrowed `Cow`.
//! A target without a `From<&str>` conversion should name its type instead, so this rule has no exceptions to resolve.

use syn::{
    Expr, ExprLit, ExprMethodCall, File, Lit,
    visit::{self, Visit},
};

use super::super::{context::Context, diagnostic::Diagnostic};

pub fn check(file: &File, context: &Context<'_>) -> Vec<Diagnostic> {
    struct Check<'a> {
        context: &'a Context<'a>,
        findings: Vec<Diagnostic>,
    }

    impl<'ast> Visit<'ast> for Check<'_> {
        fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
            if call.method == "into"
                && let Expr::MethodCall(inner) = &*call.receiver
                && (inner.method == "to_string" || inner.method == "to_owned")
                && matches!(
                    &*inner.receiver,
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(_),
                        ..
                    })
                )
            {
                self.findings.push(self.context.report(
                    "no-literal-to-string-into",
                    inner.method.span(),
                    "converting a string literal to an owned string before `.into()` allocates for nothing",
                    "Call `.into()` on the literal directly.",
                ));
            }

            visit::visit_expr_method_call(self, call);
        }
    }

    let mut check = Check {
        context,
        findings: Vec::new(),
    };
    check.visit_file(file);
    check.findings
}
