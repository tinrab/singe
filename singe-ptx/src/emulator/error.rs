use std::result::Result as StdResult;

use thiserror::Error;

use crate::{ast::Span, diagnostic::SourceDiagnostic};

/// An error found while validating PTX, with the byte range of the offending source.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind} at bytes {}..{}", span.start, span.end)]
pub struct CompileError {
    /// Source byte range of the rejected declaration or instruction.
    pub span: Span,
    /// What is wrong with it.
    pub kind: CompileErrorKind,
}

/// Why PTX failed validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CompileErrorKind {
    /// The source uses a PTX feature the emulator does not implement.
    #[error("unsupported PTX feature: {0}")]
    Unsupported(String),
    /// An instruction has invalid qualifiers or operands.
    #[error("invalid instruction: {0}")]
    InvalidInstruction(String),
    /// A declaration has an invalid type, shape, storage space, or qualifier.
    #[error("invalid declaration: {0}")]
    InvalidDeclaration(String),
    /// A symbol is declared more than once in the same scope.
    #[error("duplicate symbol `{0}`")]
    DuplicateSymbol(String),
    /// An instruction or initializer refers to an undeclared symbol.
    #[error("undefined symbol `{0}`")]
    UndefinedSymbol(String),
    /// An operand representation is incompatible with its declared or required type.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),
    /// A constant initializer cannot be evaluated for the declared storage.
    #[error("invalid initializer: {0}")]
    InvalidInitializer(String),
    /// Storage size or alignment exceeds the addressable range.
    #[error("layout exceeds the addressable size")]
    LayoutOverflow,
    /// A launch requirement cannot be represented or satisfied.
    #[error("invalid launch requirement: {0}")]
    InvalidLaunch(String),
    /// The requested symbol is not a defined kernel entry point.
    #[error("entry point `{0}` is not a defined kernel")]
    InvalidEntry(String),
    /// Compiled code belongs to a different parsed module.
    #[error("compiled kernel belongs to another parsed module")]
    ModuleMismatch,
}

impl CompileError {
    /// Returns a display of this error that quotes the source line containing its span.
    #[must_use]
    pub fn with_source<'a>(&'a self, source: &'a str) -> SourceDiagnostic<'a> {
        SourceDiagnostic::new(self, source, Some(self.span))
    }

    pub(crate) fn new(span: Span, kind: CompileErrorKind) -> Self {
        Self { span, kind }
    }

    pub(crate) fn unsupported(span: Span, feature: impl Into<String>) -> Self {
        Self::new(span, CompileErrorKind::Unsupported(feature.into()))
    }

    pub(crate) fn instruction(span: Span, reason: impl Into<String>) -> Self {
        Self::new(span, CompileErrorKind::InvalidInstruction(reason.into()))
    }

    pub(crate) fn declaration(span: Span, reason: impl Into<String>) -> Self {
        Self::new(span, CompileErrorKind::InvalidDeclaration(reason.into()))
    }
}

/// Result of PTX validation.
pub type Result<T> = StdResult<T, CompileError>;
