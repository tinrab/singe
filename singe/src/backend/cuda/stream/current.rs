use std::{cell::RefCell, collections::HashMap, sync::Arc};

use singe_cuda::stream::{Stream, StreamCaptureStatus};

use crate::{
    device::Device,
    error::{Error, Result},
    storage::cuda::cuda_device_id,
};

thread_local! {
    pub static CURRENT_STREAMS: RefCell<HashMap<i32, Arc<Stream>>> =
        RefCell::new(HashMap::new());
}

/// Guard that restores the previous thread-local current stream for one CUDA device when dropped.
#[derive(Debug)]
pub struct CudaStreamGuard {
    device_id: i32,
    previous: Option<Arc<Stream>>,
}

impl CudaStreamGuard {
    pub fn new(device_id: i32, previous: Option<Arc<Stream>>) -> Self {
        Self {
            device_id,
            previous,
        }
    }
}

impl Drop for CudaStreamGuard {
    fn drop(&mut self) {
        CURRENT_STREAMS.with(|streams| {
            let mut streams = streams.borrow_mut();
            match self.previous.take() {
                Some(previous) => {
                    streams.insert(self.device_id, previous);
                }
                None => {
                    streams.remove(&self.device_id);
                }
            }
        });
    }
}

pub fn current_stream_for_device(device_id: i32) -> Option<Arc<Stream>> {
    CURRENT_STREAMS.with(|streams| streams.borrow().get(&device_id).cloned())
}

pub fn current_stream(device: Device) -> Result<Option<Arc<Stream>>> {
    let device_id = cuda_device_id(device)?;
    Ok(current_stream_for_device(device_id))
}

pub fn ensure_not_stream_capturing(device: Device, operation: &'static str) -> Result<()> {
    let Some(stream) = current_stream(device)? else {
        return Ok(());
    };
    if stream.capture_status()? == StreamCaptureStatus::Active {
        return Err(Error::unsupported_operation(operation));
    }
    Ok(())
}
