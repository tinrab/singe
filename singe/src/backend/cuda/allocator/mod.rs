mod allocation;
mod cache;
mod creation;
mod stats;
mod traits;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cuda::{context::Context as CudaContext, stream::Stream};

use self::cache::CudaCachingAllocatorCache;
pub use self::{stats::CudaAllocatorStats, traits::CudaAllocator};
use crate::device::Device;

static CACHING_ALLOCATOR_CACHES: OnceLock<Mutex<HashMap<usize, Arc<CudaCachingAllocatorCache>>>> =
    OnceLock::new();

/// Stream-aware CUDA caching allocator.
///
/// Freed blocks are kept in a per-context cache and reused by exact byte size.
/// When the allocator is associated with a stream, dropped storage records a completion event on that stream and later reuse waits for the event before handing the block out again.
#[derive(Debug, Clone)]
pub struct CudaCachingAllocator {
    device: Device,
    context: Arc<CudaContext>,
    cache: Arc<CudaCachingAllocatorCache>,
    stream: Option<Arc<Stream>>,
}

#[cfg(all(test, feature = "testing"))]
mod tests;
