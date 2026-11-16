use std::sync::Arc;

use singe_core::{checked_arithmetic, checked_conversions};
use singe_cuda::context::Context;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_usize -> usize,
    to_i32 -> i32,
    to_u64 -> u64,
});

checked_arithmetic!(Error {
    checked_byte_len,
    checked_mul
});

pub fn buffer_byte_len<T>(len: usize, name: &str) -> Result<u64> {
    to_u64(checked_byte_len::<T>(len, name)?, name)
}

/// NCCL selects CUDA devices internally and launches in their primary contexts.
pub fn ensure_primary_context(context: &Arc<Context>) -> Result<()> {
    let primary = Context::retain_primary_for_device(context.device())?;
    if context != &primary {
        return Err(Error::UnsupportedContext);
    }
    Ok(())
}
