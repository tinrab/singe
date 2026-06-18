use singe_cuda::{context::Context, stream::Stream};

use crate::{context::StreamContext, error::Result};

pub fn create_stream_context() -> Result<(Stream, StreamContext)> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let stream_context = StreamContext::create(&stream)?;
    Ok((stream, stream_context))
}
