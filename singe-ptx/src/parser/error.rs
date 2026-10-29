use thiserror::Error;

use crate::{ast::Span, diagnostic::SourceDiagnostic};

/// Why [`parse_module`](crate::parser::parse_module) rejected its input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ParseErrorKind {
    /// Syntax nesting or constructed expression depth exceeds the parser limit of 128.
    #[error("syntax or expression depth exceeds 128")]
    DepthLimitExceeded,
    /// A token does not match the PTX grammar at this position. Holds up to 20
    /// characters of the input starting at the token.
    #[error("unexpected token: {0}")]
    UnexpectedToken(String),
    /// The input ends before the current PTX construct is complete.
    #[error("unexpected end of input")]
    UnexpectedEof,
    /// Non-whitespace input follows the parsed module.
    #[error("trailing input")]
    TrailingInput,
}

/// PTX syntax error with the byte range of the offending input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind} at bytes {}..{}", span.start, span.end)]
pub struct ParseError {
    /// Reason the parser rejected the source.
    pub kind: ParseErrorKind,
    /// Offending source bytes, or an empty range at the end of truncated input.
    pub span: Span,
}

impl ParseError {
    /// Returns a display of this error that quotes the source line containing its span.
    #[must_use]
    pub fn with_source<'a>(&'a self, source: &'a str) -> SourceDiagnostic<'a> {
        SourceDiagnostic::new(self, source, Some(self.span))
    }
}
