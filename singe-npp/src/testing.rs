use singe_cuda::{context::Context, stream::Stream};

use crate::{context::StreamContext, error::Result};

/// Creates a CUDA stream and its matching application-managed NPP stream context.
///
/// The returned stream retains the newly created CUDA context.
///
/// # Errors
///
/// Returns an error if CUDA context creation, stream creation, or constructing the
/// NPP stream context fails.
pub fn create_stream_context() -> Result<(Stream, StreamContext)> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let stream_context = StreamContext::create(&stream)?;
    Ok((stream, stream_context))
}
