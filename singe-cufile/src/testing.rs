use std::sync::Arc;

use singe_cuda::context::Context;

use crate::error::Result;

/// Creates and returns a CUDA context for cuFile tests.
///
/// # Errors
///
/// Returns an error if a CUDA context cannot be created.
pub fn setup_context() -> Result<Arc<Context>> {
    let context = Context::create()?;
    Ok(context)
}
