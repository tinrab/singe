mod allocation;
mod cache;
mod creation;
mod stats;
mod traits;

use std::{
    collections::HashMap,
    mem::MaybeUninit,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cuda::{
    context::Context as CudaContext, error::Result as CudaResult, memory::DeviceMemory,
    stream::Stream,
};

use self::cache::CudaCachingAllocatorCache;
pub use self::{stats::CudaAllocatorStats, traits::CudaAllocator};
use crate::device::Device;

static CACHING_ALLOCATOR_CACHES: OnceLock<Mutex<HashMap<usize, Arc<CudaCachingAllocatorCache>>>> =
    OnceLock::new();

/// Stream-aware CUDA caching allocator.
///
/// Freed blocks are kept in a per-context cache and reused by exact byte size.
/// On the eager stream, a dropped block returns to the cache at once: eager work runs in order on that stream, and every other stream waits for it before reusing the block.
/// When the allocator is associated with another stream, dropped storage records a completion event on that stream and later reuse waits for the event before handing the block out again.
#[derive(Debug, Clone)]
pub struct CudaCachingAllocator {
    device: Device,
    context: Arc<CudaContext>,
    cache: Arc<CudaCachingAllocatorCache>,
    stream: Arc<Stream>,
    // Whether `stream` is the runtime's eager stream rather than a caller's.
    eager: bool,
}

// Scratch memory a library asks for, outside the cache: when the device has no room, the cache's free blocks are released and it is asked once more, as the allocator retries its own blocks.
pub(crate) fn allocate_scratch(
    context: &Arc<CudaContext>,
    bytes: usize,
) -> CudaResult<DeviceMemory<MaybeUninit<u8>>> {
    DeviceMemory::<u8>::create_uninit(context, bytes).or_else(|error| {
        match CudaCachingAllocator::clear_cache_for_device(Device::Cuda(context.device().id())) {
            Ok(()) => DeviceMemory::<u8>::create_uninit(context, bytes),
            Err(_) => Err(error),
        }
    })
}

#[cfg(all(test, feature = "testing"))]
mod tests;
