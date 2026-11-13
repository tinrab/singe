use std::sync::Arc;

use singe_cuda::stream::Stream;

use crate::{
    backend::cuda::{
        allocator::{CudaCachingAllocator, cache::CudaCachedBlock},
        runtime::runtime_for_device,
    },
    error::Result,
};

impl CudaCachingAllocator {
    // Orders this allocator's stream after the block's last use: the event a caller's stream recorded, and on any stream but the eager one, the eager work a block returns to the cache without.
    pub(crate) fn wait_for_block(&self, block: &CudaCachedBlock) -> Result<()> {
        if let Some(event) = &block.ready_event {
            self.stream.wait_event(event)?;
        }

        if !self.eager {
            runtime_for_device(self.device)?.order_after_eager(&self.stream)?;
        }
        Ok(())
    }

    // The stream dropped storage records its last use on, which eager storage needs none of.
    pub(crate) fn recycle_stream(&self) -> Option<Arc<Stream>> {
        (!self.eager).then(|| Arc::clone(&self.stream))
    }
}
