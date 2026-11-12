use std::mem::ManuallyDrop;

use singe_cuda::event::{EventFlags, EventRecordFlags};

use crate::storage::cuda::{CudaStorage, CudaStorageByte};

impl<T: CudaStorageByte> Drop for CudaStorage<T> {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, so `data` is taken exactly once.
        let data = unsafe { ManuallyDrop::take(&mut self.data) };
        if let Some(recycler) = self.recycler.take() {
            let ready_event = self.recycle_stream.as_ref().and_then(|stream| {
                let event = data
                    .context()
                    .create_event_with_flags(EventFlags::DISABLE_TIMING)
                    .ok()?;
                event
                    .record(stream, EventRecordFlags::DEFAULT)
                    .ok()
                    .map(|()| event)
            });
            // A caller's stream whose event could not be recorded waits for the whole context: a failed completion wait must never return a block for reuse.
            // Eager storage has no stream to record on, since its reuse is ordered after the eager stream.
            if self.recycle_stream.is_some()
                && ready_event.is_none()
                && data.context().synchronize().is_err()
            {
                return;
            }

            recycler.recycle(T::into_pending(data), ready_event);
        } else {
            drop(data);
        }
    }
}
