use std::mem;

use singe_cuda::{
    event::{Event, EventFlags, EventRecordFlags},
    memory::{DeviceMemory, MemoryCopyKind},
};

use crate::{
    backend::cuda::allocator::{CudaCachingAllocator, cache::StagingBuffer},
    error::{Error, Result},
    utility::Locked,
};

pub const STAGED_BYTES: usize = 1 << 20;

impl CudaCachingAllocator {
    pub(crate) fn copy_staged(&self, destination: *mut u8, bytes: &[u8]) -> Result<()> {
        let copied = self.retire_staging()?;
        let memory = self.context.pinned_host().from_slice(bytes)?;
        // SAFETY: `destination` holds `bytes.len()` writable bytes that earlier uses precede on the stream, and the staging buffer stays alive until `copied` says the copy has run.
        unsafe {
            DeviceMemory::<u8>::copy_async(
                destination,
                memory.as_ptr(),
                bytes.len(),
                MemoryCopyKind::HostToDevice,
                &self.stream,
            )?;
        }

        let staged = copied
            .record(&self.stream, EventRecordFlags::DEFAULT)
            .map_err(Error::from)
            .and_then(|()| self.cache.staging.locked("cuda staging buffers"));
        match staged {
            Ok(mut staging) => {
                staging.push(StagingBuffer { memory, copied });
                Ok(())
            }
            Err(error) => {
                // The copy may still read the staging, which would return to the pool on drop.
                if self.stream.synchronize().is_err() {
                    mem::forget(memory);
                }
                Err(error)
            }
        }
    }

    fn retire_staging(&self) -> Result<Event> {
        let mut staging = self.cache.staging.locked("cuda staging buffers")?;
        let mut finished = Vec::new();
        let mut index = 0;
        while let Some(buffer) = staging.get(index) {
            if buffer.copied.query()? {
                finished.push(staging.swap_remove(index));
            } else {
                index += 1;
            }
        }

        drop(staging);

        match finished.pop() {
            Some(buffer) => Ok(buffer.copied),
            None => Ok(self
                .context
                .create_event_with_flags(EventFlags::DISABLE_TIMING)?),
        }
    }
}
