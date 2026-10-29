use std::{
    error,
    fmt::{self, Display, Formatter},
    result::Result as StdResult,
};

/// Error returned when CUDA/NVIDIA dependency discovery cannot complete.
#[derive(Debug, Clone)]
pub struct FindError {
    /// Component being searched, such as `"CUDA Runtime"` or `"cuDNN"`.
    pub component: String,
    /// Human-readable reason discovery failed.
    pub message: String,
}

impl FindError {
    pub(crate) fn glob(
        component: impl Into<String>,
        pattern: impl Into<String>,
        error: &glob::PatternError,
    ) -> Self {
        Self {
            component: component.into(),
            message: format!("invalid glob pattern `{}`: {error}", pattern.into()),
        }
    }
}

impl Display for FindError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} discovery failed: {}", self.component, self.message)
    }
}

impl error::Error for FindError {}

/// Result type used by CUDA discovery helpers.
pub type Result<T> = StdResult<T, FindError>;
