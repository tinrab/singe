use singe_cuda::{
    event::{EventFlags, EventRecordFlags},
    memory::{DeviceMemory, MemoryCopyKind, PinnedHostMemory},
};

use crate::{
    backend::cuda::allocator::{CudaCachingAllocator, cache::StagingBuffer},
    error::Result,
    utility::Locked,
};

pub const STAGED_BYTES: usize = 1 << 20;

const SMALLEST_STAGING: usize = 4096;

impl CudaCachingAllocator {
    pub(crate) fn copy_staged(&self, destination: *mut u8, bytes: &[u8]) -> Result<()> {
        let free = self.take_staging(bytes.len())?;
        let mut buffer = match free {
            Some(buffer) => buffer,
            None => StagingBuffer {
                memory: PinnedHostMemory::from_slice(
                    &self.context,
                    &vec![0_u8; bytes.len().next_power_of_two().max(SMALLEST_STAGING)],
                )?,
                copied: self
                    .context
                    .create_event_with_flags(EventFlags::DISABLE_TIMING)?,
            },
        };
        buffer.memory.as_slice_mut()[..bytes.len()].copy_from_slice(bytes);
        // SAFETY: `destination` holds `bytes.len()` writable bytes that earlier uses precede on the stream, and the staging buffer stays unused until `copied` says the copy has run.
        unsafe {
            DeviceMemory::<u8>::copy_async(
                destination,
                buffer.memory.as_slice().as_ptr(),
                bytes.len(),
                MemoryCopyKind::HostToDevice,
                &self.stream,
            )?;
        }

        buffer
            .copied
            .record(&self.stream, EventRecordFlags::DEFAULT)?;
        self.cache
            .staging
            .locked("cuda staging buffers")?
            .push(buffer);
        Ok(())
    }

    fn take_staging(&self, bytes: usize) -> Result<Option<StagingBuffer>> {
        let mut staging = self.cache.staging.locked("cuda staging buffers")?;
        let mut found = None;
        for (index, buffer) in staging.iter().enumerate() {
            if buffer.memory.as_slice().len() >= bytes && buffer.copied.query()? {
                found = Some(index);
                break;
            }
        }

        let taken = found.map(|index| staging.swap_remove(index));
        drop(staging);
        Ok(taken)
    }
}
