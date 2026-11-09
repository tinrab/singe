use std::{mem, sync::atomic::Ordering};

use crate::{
    backend::cuda::allocator::{CudaCachingAllocator, cache::CudaCachingAllocatorCache},
    device::Device,
    error::Result,
    storage::cuda::cuda_device_id,
    utility::{Locked, checked_add, checked_mul},
};

impl CudaCachingAllocator {
    /// Releases all free cached blocks for this allocator's context.
    ///
    /// Live allocations are unaffected and may return to the cache when their storage is dropped later.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    pub fn clear_cache(&self) -> Result<()> {
        clear_blocks(&self.cache)
    }

    /// Restarts peak tracking at the bytes currently active.
    pub fn reset_peak(&self) {
        self.cache.peak_bytes.store(
            self.cache.active_bytes.load(Ordering::Acquire),
            Ordering::Release,
        );
    }

    /// Caps active plus cached bytes at `limit`, or removes the cap for `None`.
    ///
    /// Allocations that would exceed the cap release cached blocks first, then fail.
    pub fn set_memory_limit(&self, limit: Option<usize>) {
        self.cache
            .memory_limit
            .store(limit.unwrap_or(0), Ordering::Release);
    }

    /// Releases all free cached blocks for `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not a valid CUDA device or if the global allocator cache or per-context free list is poisoned.
    pub fn clear_cache_for_device(device: Device) -> Result<()> {
        let device_id = cuda_device_id(device)?;

        let caches = Self::caches()
            .locked("cuda runtime cache")?
            .values()
            .filter(|cache| cache.device == Device::Cuda(device_id))
            .cloned()
            .collect::<Vec<_>>();
        for cache in caches {
            clear_blocks(&cache)?;
        }

        Ok(())
    }

    /// Returns the number of free cached blocks with exactly `byte_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a lock is poisoned.
    #[cfg(all(test, feature = "testing"))]
    pub fn cached_block_count_for(&self, byte_size: usize) -> Result<usize> {
        Ok(self
            .cache
            .free
            .locked("cuda runtime cache")?
            .get(&byte_size)
            .map_or(0, Vec::len))
    }
}

fn clear_blocks(cache: &CudaCachingAllocatorCache) -> Result<()> {
    let blocks = {
        let mut free = cache.free.locked("cuda runtime cache")?;
        let blocks = mem::take(&mut *free);
        drop(free);
        cache.cached_bytes.store(0, Ordering::SeqCst);
        blocks
    };
    // CUDA destruction may wait for GPU work.
    // Release host cache locks first.
    drop(blocks);

    Ok(())
}

/// Current CUDA allocator cache and allocation counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaAllocatorStats {
    /// Number of free blocks currently held by the caching allocator.
    pub(crate) cached_blocks: usize,
    /// Total bytes currently held in cached free blocks.
    pub(crate) cached_bytes: usize,
    /// Cumulative bytes allocated from CUDA by this allocator cache after cache misses.
    pub(crate) allocated_bytes: usize,
    /// Bytes currently owned by live tensors or pending cached-block reuse.
    pub(crate) active_bytes: usize,
    /// Maximum observed active bytes for this allocator cache.
    pub(crate) peak_bytes: usize,
    /// Cumulative number of CUDA allocation calls made after cache misses.
    pub(crate) allocation_count: usize,
    /// Cumulative number of allocations served by reusing cached free blocks.
    pub(crate) cache_hit_count: usize,
}

impl CudaAllocatorStats {
    /// Returns the number of free blocks held by the caching allocator.
    #[must_use]
    pub fn cached_blocks(&self) -> usize {
        self.cached_blocks
    }

    /// Returns total bytes held in cached free blocks.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Returns cumulative bytes allocated from CUDA after cache misses.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns bytes currently owned by live tensors or pending cached-block reuse.
    #[must_use]
    pub fn active_bytes(&self) -> usize {
        self.active_bytes
    }

    /// Returns maximum observed active bytes for this allocator cache.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Returns cumulative CUDA allocation calls made after cache misses.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Returns cumulative allocations served by reusing cached free blocks.
    #[must_use]
    pub fn cache_hit_count(&self) -> usize {
        self.cache_hit_count
    }
}

impl CudaCachingAllocator {
    /// Returns current cache and allocation counters for this allocator's device.
    pub fn stats(&self) -> Result<CudaAllocatorStats> {
        let free = self.cache.free.locked("cuda runtime cache")?;
        let cached_blocks = free.values().map(Vec::len).sum();
        let cached_bytes = free
            .iter()
            .try_fold(0_usize, |total, (byte_size, blocks)| {
                let block_bytes = checked_mul(
                    *byte_size,
                    blocks.len(),
                    "cuda allocator cached block bytes",
                )?;
                checked_add(total, block_bytes, "cuda allocator cached bytes")
            })?;
        drop(free);

        Ok(CudaAllocatorStats {
            cached_blocks,
            cached_bytes,
            allocated_bytes: self.cache.allocated_bytes.load(Ordering::Acquire),
            active_bytes: self.cache.active_bytes.load(Ordering::Acquire),
            peak_bytes: self.cache.peak_bytes.load(Ordering::Acquire),
            allocation_count: self.cache.allocation_count.load(Ordering::Acquire),
            cache_hit_count: self.cache.cache_hit_count.load(Ordering::Acquire),
        })
    }
}
