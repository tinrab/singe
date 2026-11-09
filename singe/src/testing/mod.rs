//! Current-thread framework activity counters and CUDA test helpers.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod program;

#[cfg(feature = "cuda")]
pub use self::cuda::{maybe_lock_device, skip_if_stub};
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::export::compiled::{
    CompiledProgram, ProgramError, ProgramInstance, build_generation_loop,
    build_set_generation_loop, program_target,
};
use std::cell::Cell;

/// Framework boundary activity on the calling thread.
/// Storage requests count CPU storage construction and CUDA allocation/import requests, including empty buffers.
/// They do not measure Rust metadata allocations or native-library memory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TensorActivity {
    /// Number of tensor descriptors constructed on the current thread.
    pub descriptors: usize,
    /// Number of checked operation invocations on the current thread.
    pub checked_invocations: usize,
    /// Number of CPU storage construction and CUDA allocation or import requests on the current thread.
    pub storage_requests: usize,
    /// Framework CUDA storage exports to host memory.
    pub cuda_readbacks: usize,
}

thread_local! {
    static ACTIVITY: Cell<TensorActivity> = Cell::default();
}

impl TensorActivity {
    /// Returns cumulative framework boundary activity counters for the current thread.
    #[must_use]
    pub fn current() -> Self {
        ACTIVITY.get()
    }
}

pub(crate) fn record_descriptor() {
    ACTIVITY.with(|activity| {
        activity.update(|mut counts| {
            counts.descriptors += 1;
            counts
        });
    });
}

pub(crate) fn record_invocation() {
    ACTIVITY.with(|activity| {
        activity.update(|mut counts| {
            counts.checked_invocations += 1;
            counts
        });
    });
}

pub(crate) fn record_storage_request() {
    ACTIVITY.with(|activity| {
        activity.update(|mut counts| {
            counts.storage_requests += 1;
            counts
        });
    });
}

#[cfg(feature = "cuda")]
pub(crate) fn record_cuda_readback() {
    ACTIVITY.with(|activity| {
        activity.update(|mut counts| {
            counts.cuda_readbacks += 1;
            counts
        });
    });
}
