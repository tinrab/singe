use std::sync::Arc;

use singe_cuda::context::Context;

use crate::error::Result;

/// Creates and returns a CUDA context for cuRAND tests.
///
/// # Errors
///
/// Returns the CUDA error if device initialization or context creation fails.
pub fn setup_context() -> Result<Arc<Context>> {
    let context = Context::create()?;
    Ok(context)
}
