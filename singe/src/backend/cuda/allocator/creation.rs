use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use singe_cuda::{context::Context as CudaContext, error::Error as CudaError, stream::Stream};

use crate::{
    backend::cuda::allocator::{
        CACHING_ALLOCATOR_CACHES, CudaCachingAllocator, cache::CudaCachingAllocatorCache,
    },
    device::Device,
    error::Result,
    utility::Locked,
};

impl CudaCachingAllocator {
    pub(crate) fn caches() -> &'static Mutex<HashMap<usize, Arc<CudaCachingAllocatorCache>>> {
        CACHING_ALLOCATOR_CACHES.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Creates an allocator retaining the actual context and optional execution stream.
    /// Cached blocks from another context are never reused, even on the same device.
    ///
    /// # Errors
    ///
    /// Returns an error if the optional stream belongs to another CUDA context or the allocator cache registry mutex is poisoned.
    pub fn create(context: &Arc<CudaContext>, stream: Option<Arc<Stream>>) -> Result<Self> {
        if stream
            .as_ref()
            .is_some_and(|stream| stream.context() != context)
        {
            return Err(CudaError::StreamContextMismatch.into());
        }

        let mut caches = Self::caches().locked("cuda runtime cache")?;
        // An expired owner cannot have live cached allocations.
        // Remove its key before the driver can reuse the same raw context address.
        caches.retain(|_, cache| cache.context.upgrade().is_some());
        let cache = caches
            .entry(context.as_raw().addr())
            .or_insert_with(|| Arc::new(CudaCachingAllocatorCache::create(context)));
        let cache = Arc::clone(cache);
        drop(caches);
        Ok(Self {
            device: Device::Cuda(context.device().id()),
            context: Arc::clone(context),
            cache,
            stream,
        })
    }

    /// Returns the actual CUDA context retained by this allocator.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}
