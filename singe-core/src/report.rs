// Prints an error with everything it was caused by, since each error's own message names only its own part.

use std::{error::Error, fmt};

// An error and each error it was caused by, once each, as `error: cause: cause`.
#[derive(Debug, Clone, Copy)]
pub struct ErrorChain<'a>(pub &'a (dyn Error + 'static));

impl fmt::Display for ErrorChain<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)?;
        let mut cause = self.0.source();
        while let Some(error) = cause {
            write!(formatter, ": {error}")?;
            cause = error.source();
        }
        Ok(())
    }
}
