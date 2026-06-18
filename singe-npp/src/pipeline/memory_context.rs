use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::{Error, Result},
};

pub(crate) fn validate_device_memory_stream_context<T>(
    stream_context: &StreamContext,
    memory: &DeviceMemory<T>,
) -> Result<()> {
    if memory.is_empty() {
        return Ok(());
    }

    let attributes = DeviceMemory::pointer_attributes(memory.as_ptr())?;
    if attributes.device != stream_context.device_id() {
        return Err(Error::StreamContextMismatch);
    }

    Ok(())
}
