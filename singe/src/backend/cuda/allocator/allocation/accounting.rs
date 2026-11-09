use std::sync::atomic::Ordering;

use crate::{
    backend::cuda::allocator::CudaCachingAllocator,
    error::{Error, Result},
    utility::checked_add,
};

impl CudaCachingAllocator {
    pub(crate) fn record_cache_hit(&self, byte_size: usize) -> Result<()> {
        self.cache
            .cached_bytes
            .try_update(Ordering::AcqRel, Ordering::Acquire, |cached_bytes| {
                cached_bytes.checked_sub(byte_size)
            })
            .map_err(|_| Error::out_of_range("cuda allocator cached bytes"))?;
        Ok(())
    }

    pub(crate) fn record_allocation(&self, byte_size: usize) -> Result<()> {
        self.cache
            .allocated_bytes
            .try_update(Ordering::AcqRel, Ordering::Acquire, |allocated_bytes| {
                allocated_bytes.checked_add(byte_size)
            })
            .map_err(|_| Error::out_of_range("cuda allocator allocated bytes"))?;
        Ok(())
    }

    pub(crate) fn activate_bytes(&self, byte_size: usize) -> Result<()> {
        let previous_active = self
            .cache
            .active_bytes
            .try_update(Ordering::AcqRel, Ordering::Acquire, |active_bytes| {
                active_bytes.checked_add(byte_size)
            })
            .map_err(|_| Error::out_of_range("cuda allocator active bytes"))?;
        let active_bytes = checked_add(previous_active, byte_size, "cuda allocator active bytes")?;

        let mut peak_bytes = self.cache.peak_bytes.load(Ordering::Acquire);
        while active_bytes > peak_bytes {
            match self.cache.peak_bytes.compare_exchange(
                peak_bytes,
                active_bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => peak_bytes = current,
            }
        }

        Ok(())
    }
}
