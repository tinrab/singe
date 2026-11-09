use std::{
    collections::HashMap,
    mem::MaybeUninit,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use singe_cuda::{context::Context as CudaContext, event::Event, memory::DeviceMemory};

use crate::{device::Device, storage::cuda::CudaMemoryRecycler};

#[derive(Debug, Default)]
pub struct CudaCachingAllocatorCache {
    pub(crate) context: Weak<CudaContext>,
    pub(crate) device: Device,
    pub(crate) free: Mutex<HashMap<usize, Vec<CudaCachedBlock>>>,
    pub(crate) allocated_bytes: AtomicUsize,
    pub(crate) cached_bytes: AtomicUsize,
    pub(crate) active_bytes: AtomicUsize,
    pub(crate) peak_bytes: AtomicUsize,
    pub(crate) allocation_count: AtomicUsize,
    pub(crate) cache_hit_count: AtomicUsize,
    /// Cap on active plus cached bytes, or zero for no cap.
    pub(crate) memory_limit: AtomicUsize,
}

#[derive(Debug)]
pub struct CudaCachedBlock {
    pub(crate) memory: DeviceMemory<MaybeUninit<u8>>,
    pub(crate) ready_event: Option<Event>,
}

impl CudaMemoryRecycler for CudaCachingAllocatorCache {
    fn recycle(&self, memory: DeviceMemory<MaybeUninit<u8>>, ready_event: Option<Event>) {
        if let Ok(mut free) = self.free.lock() {
            let byte_size = memory.len();
            let cached_bytes =
                self.cached_bytes
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |cached_bytes| {
                        cached_bytes.checked_add(byte_size)
                    });
            debug_assert!(cached_bytes.is_ok());
            let active =
                self.active_bytes
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |active_bytes| {
                        active_bytes.checked_sub(byte_size)
                    });
            debug_assert!(active.is_ok());
            free.entry(byte_size).or_default().push(CudaCachedBlock {
                memory,
                ready_event,
            });
        }
    }
}

impl CudaCachingAllocatorCache {
    pub(crate) fn create(context: &Arc<CudaContext>) -> Self {
        Self {
            context: Arc::downgrade(context),
            device: Device::Cuda(context.device().id()),
            ..Self::default()
        }
    }
}
