//! Depth limit for nested syntax. It bounds recursion in the parser and in every later
//! traversal of the syntax tree.

use crate::parser::common::{Input, PResult, SyntaxError, SyntaxErrorKind};

pub(super) const MAX_DEPTH: usize = 128;

pub(super) fn check(depth: usize, input: Input<'_>) -> PResult<'_, ()> {
    if depth > MAX_DEPTH {
        Err(SyntaxError::failure(
            input,
            SyntaxErrorKind::DepthLimitExceeded,
        ))
    } else {
        Ok((input, ()))
    }
}
