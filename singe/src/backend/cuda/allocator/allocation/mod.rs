mod accounting;
mod block;
mod interface;

use std::{
    mem::MaybeUninit,
    sync::{Arc, atomic::Ordering},
};

use singe_cuda::{
    memory::{DeviceMemory, transfer},
    view::{DeviceRegionProvider, DeviceViewMut},
};

use crate::{
    backend::cuda::allocator::CudaCachingAllocator,
    data_type::DataType,
    error::{Error, Result},
    storage::cuda::{CudaMemoryRecycler, CudaStorage, map_cuda_allocation_error},
    utility::Locked,
};

#[derive(Clone, Copy)]
enum Initialization<'a> {
    Zero,
    Bytes(&'a [u8]),
}

impl CudaCachingAllocator {
    fn initialize_storage(
        &self,
        data_type: DataType,
        mut memory: DeviceMemory<MaybeUninit<u8>>,
        initialization: Initialization<'_>,
    ) -> Result<CudaStorage> {
        self.context.bind()?;
        match initialization {
            Initialization::Zero => {
                if let Some(stream) = &self.stream {
                    // SAFETY: this unique block is writable; earlier uses precede the cache's wait event.
                    unsafe {
                        DeviceMemory::<u8>::set_async(
                            memory.as_mut_ptr().cast(),
                            0,
                            memory.len(),
                            stream,
                        )?;
                    }

                    stream.synchronize()?;
                } else {
                    memory.set_zeroes()?;
                }
            }
            Initialization::Bytes(bytes) => {
                let mut region = memory.region();
                if let Some(stream) = &self.stream {
                    // SAFETY: cache reuse dependencies and this producer use the allocator's stream.
                    region = unsafe { region.with_stream(stream)? };
                }

                // SAFETY: every MaybeUninit value is valid and this owned block is exclusively borrowed.
                let mut destination = unsafe {
                    DeviceViewMut::<MaybeUninit<u8>>::from_raw_parts(region, memory.len())?
                };
                transfer::copy_from_host(&mut destination, bytes)?;
            }
        }

        // SAFETY: the producer initialized and completed the entire byte range.
        let memory = unsafe { memory.assume_init() };
        // SAFETY: zero initialization or validated input bytes establish the datatype's value validity.
        unsafe {
            CudaStorage::from_memory_recyclable(
                data_type,
                memory,
                Arc::clone(&self.cache) as Arc<dyn CudaMemoryRecycler>,
                self.stream.clone(),
            )
        }
    }

    pub(crate) fn take_or_create(
        &self,
        byte_size: usize,
        data_type: DataType,
        shape: Option<Vec<i64>>,
    ) -> Result<DeviceMemory<MaybeUninit<u8>>> {
        let block = self
            .cache
            .free
            .locked("cuda runtime cache")?
            .get_mut(&byte_size)
            .and_then(Vec::pop);
        if let Some(block) = block {
            self.wait_for_block(&block)?;
            self.record_cache_hit(byte_size)?;
            self.cache.cache_hit_count.fetch_add(1, Ordering::SeqCst);
            self.activate_bytes(byte_size)?;
            return Ok(block.memory);
        }

        self.enforce_memory_limit(byte_size)?;
        let memory = self.allocate_with_retry(byte_size, data_type, shape)?;
        self.cache.allocation_count.fetch_add(1, Ordering::SeqCst);
        self.record_allocation(byte_size)?;
        self.activate_bytes(byte_size)?;
        Ok(memory)
    }

    /// Releases cached blocks when a new allocation would exceed the memory limit, and fails if it still would.
    fn enforce_memory_limit(&self, byte_size: usize) -> Result<()> {
        let limit = self.cache.memory_limit.load(Ordering::Acquire);
        if limit == 0 {
            return Ok(());
        }

        let resident = self.cache.active_bytes.load(Ordering::Acquire)
            + self.cache.cached_bytes.load(Ordering::Acquire);
        if resident.saturating_add(byte_size) > limit {
            self.clear_cache()?;
        }

        let active = self.cache.active_bytes.load(Ordering::Acquire);
        if active.saturating_add(byte_size) > limit {
            return Err(Error::MemoryLimitExceeded {
                device: self.device,
                limit,
                requested: byte_size,
                active,
            });
        }
        Ok(())
    }

    fn allocate_with_retry(
        &self,
        byte_size: usize,
        data_type: DataType,
        shape: Option<Vec<i64>>,
    ) -> Result<DeviceMemory<MaybeUninit<u8>>> {
        match self.allocate_once(byte_size, data_type, shape.clone()) {
            Ok(memory) => Ok(memory),
            Err(Error::AllocationFailed {
                device: _,
                byte_count: _,
                data_type,
                ..
            }) => {
                self.clear_cache()?;
                self.allocate_once(byte_size, data_type, shape)
            }
            Err(error) => Err(error),
        }
    }

    fn allocate_once(
        &self,
        byte_size: usize,
        data_type: DataType,
        shape: Option<Vec<i64>>,
    ) -> Result<DeviceMemory<MaybeUninit<u8>>> {
        let memory =
            DeviceMemory::<u8>::create_uninit(&self.context, byte_size).map_err(|error| {
                map_cuda_allocation_error(error, self.device, byte_size, data_type, shape)
            })?;
        Ok(memory)
    }
}
