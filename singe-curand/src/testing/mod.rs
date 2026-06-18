use std::sync::Arc;

use singe_cuda::context::Context;

use crate::error::Result;

/// Creates and returns a CUDA context for cuRAND tests.
pub fn setup_context() -> Result<Arc<Context>> {
    let ctx = Context::create()?;
    Ok(ctx)
}
