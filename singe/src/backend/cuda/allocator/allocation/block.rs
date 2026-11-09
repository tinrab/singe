use crate::{
    backend::cuda::allocator::{CudaCachingAllocator, cache::CudaCachedBlock},
    error::Result,
};

impl CudaCachingAllocator {
    pub(crate) fn wait_for_block(&self, block: &CudaCachedBlock) -> Result<()> {
        if let Some(event) = &block.ready_event {
            if let Some(stream) = &self.stream {
                stream.wait_event(event)?;
            } else {
                event.synchronize()?;
            }
        }
        Ok(())
    }
}
