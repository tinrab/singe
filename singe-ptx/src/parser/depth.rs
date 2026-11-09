//! Depth limit for nested syntax.
//! It bounds recursion in the parser and in every later traversal of the syntax tree.

use crate::parser::lexical::{Input, PResult, SyntaxError, SyntaxErrorKind};

pub const MAX_DEPTH: usize = 128;

pub fn check(depth: usize, input: Input<'_>) -> PResult<'_, ()> {
    if depth > MAX_DEPTH {
        Err(SyntaxError::failure(
            input,
            SyntaxErrorKind::DepthLimitExceeded,
        ))
    } else {
        Ok((input, ()))
    }
}
