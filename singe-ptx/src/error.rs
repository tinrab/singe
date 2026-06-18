use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseErrorKind {
    #[error("unexpected token: {0}")]
    UnexpectedToken(String),
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("invalid integer literal: {0}")]
    InvalidInteger(String),
    #[error("invalid float literal: {0}")]
    InvalidFloat(String),
    #[error("expected {0}")]
    Expected(String),
    #[error("unsupported directive: {0}")]
    UnsupportedDirective(String),
    #[error("unsupported instruction form: {0}")]
    UnsupportedInstruction(String),
    #[error("trailing input")]
    TrailingInput,
}

#[derive(Debug, Error)]
#[error("{kind} at offset {offset}")]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub offset: usize,
}
