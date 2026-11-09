use std::{mem::size_of, process, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    image::Image,
    instance::InstanceRef,
    try_ffi,
    types::{BackendKind, LoadHintPolicy, ProcessingStatus},
    utility::{NativeStruct, to_i32, to_usize},
};

/// Where decoders and encoders run: CPU threads, backends, and the CUDA context for GPU work.
///
/// The default runs on the CPU only.
/// Use [`new`](Self::new) with a CUDA context to run on the GPU.
#[derive(Debug)]
pub struct ExecutionParams {
    pub(crate) raw: sys::nvimgcodecExecutionParams_t,
    backends: Vec<sys::nvimgcodecBackend_t>,
    context: Option<Arc<CudaContext>>,
}

impl Default for ExecutionParams {
    fn default() -> Self {
        let backends = vec![Backend::new(BackendKind::CpuOnly).raw];
        let raw = sys::nvimgcodecExecutionParams_t {
            device_id: sys::NVIMGCODEC_DEVICE_CPU_ONLY,
            num_backends: 1,
            backends: backends.as_ptr(),
            ..sys::nvimgcodecExecutionParams_t::init()
        };
        Self {
            raw,
            backends,
            context: None,
        }
    }
}

impl ExecutionParams {
    /// Creates parameters that run GPU work in `context`, which they keep alive.
    ///
    /// Safe decode and encode calls wait for the streams that produce their inputs before submitting.
    #[must_use]
    pub fn new(context: &Arc<CudaContext>) -> Self {
        let raw = sys::nvimgcodecExecutionParams_t {
            device_id: context.device().id(),
            skip_pre_sync: 1,
            ..sys::nvimgcodecExecutionParams_t::init()
        };
        Self {
            raw,
            backends: Vec::new(),
            context: Some(Arc::clone(context)),
        }
    }

    /// Creates parameters that run on the CPU only, without a CUDA context.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self::default()
    }

    /// Returns the CUDA context, or `None` for CPU-only execution.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<CudaContext>> {
        self.context.as_ref()
    }

    /// Sets the maximum number of CPU threads for the default executor.
    ///
    /// Zero uses one thread per CPU core.
    ///
    /// # Errors
    ///
    /// Returns an error if `thread_count` does not fit an `i32`.
    pub fn set_max_cpu_threads(&mut self, thread_count: usize) -> Result<&mut Self> {
        self.raw.max_num_cpu_threads = to_i32(thread_count, "thread_count")?;
        Ok(self)
    }

    /// Sets whether nvImageCodec initializes the backends when the decoder or encoder is created, instead of on first use.
    pub fn set_pre_init(&mut self, pre_init: bool) -> &mut Self {
        self.raw.pre_init = i32::from(pre_init);
        self
    }

    /// Sets whether nvImageCodec skips waiting for the image streams before it starts work.
    ///
    /// Decoders and encoders created by this crate skip that wait and order the work themselves.
    ///
    /// # Safety
    ///
    /// If the wait is not skipped, every image stream used with these parameters must belong to the CUDA context of nvImageCodec's worker events; nvImageCodec does not ensure this for secondary contexts.
    /// If the wait is skipped, raw submissions must make sure their inputs are ready before enqueueing work.
    /// The safe decode and encode calls do this themselves.
    pub unsafe fn set_skip_pre_sync(&mut self, skip: bool) -> &mut Self {
        self.raw.skip_pre_sync = i32::from(skip);
        self
    }

    /// Sets the backends that may run the work, with their load hints.
    ///
    /// An empty list lets nvImageCodec choose GPU backends; for CPU-only execution it selects the CPU backend.
    ///
    /// # Errors
    ///
    /// Returns an error if there are more backends than fit an `i32`, or a GPU backend is requested without a CUDA context.
    pub fn set_backends(&mut self, mut backends: Vec<Backend>) -> Result<&mut Self> {
        if self.context.is_none() {
            if backends
                .iter()
                .any(|backend| backend.raw.kind != BackendKind::CpuOnly.into())
            {
                return Err(Error::Unsupported {
                    name: "gpu_backend_in_cpu_only_execution".into(),
                });
            }
            if backends.is_empty() {
                backends.push(Backend::new(BackendKind::CpuOnly));
            }
        }

        let count = to_i32(backends.len(), "backends")?;
        self.backends = backends.into_iter().map(|backend| backend.raw).collect();
        self.raw.num_backends = count;
        self.raw.backends = if self.backends.is_empty() {
            ptr::null()
        } else {
            self.backends.as_ptr()
        };
        Ok(self)
    }

    /// Returns a pointer to the native parameters, valid while `self` is.
    #[must_use]
    pub fn as_raw(&self) -> *const sys::nvimgcodecExecutionParams_t {
        &raw const self.raw
    }
}

impl Clone for ExecutionParams {
    fn clone(&self) -> Self {
        let backends = self.backends.clone();
        let mut raw = self.raw;
        raw.backends = if backends.is_empty() {
            ptr::null()
        } else {
            backends.as_ptr()
        };

        Self {
            raw,
            backends,
            context: self.context.as_ref().map(Arc::clone),
        }
    }
}

/// Backend that may run decoding or encoding, with an optional hint for how much of a batch it takes.
#[derive(Debug, Clone, Copy)]
pub struct Backend {
    raw: sys::nvimgcodecBackend_t,
}

impl Backend {
    /// Creates a backend of `kind` with no load hint.
    #[must_use]
    pub fn new(kind: BackendKind) -> Self {
        let params = sys::nvimgcodecBackendParams_t {
            struct_type: sys::nvimgcodecStructureType_t::NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS,
            struct_size: size_of::<sys::nvimgcodecBackendParams_t>() as sys::size_t,
            struct_next: ptr::null_mut(),
            load_hint_policy: LoadHintPolicy::Ignore.into(),
            ..Default::default()
        };

        Self {
            raw: sys::nvimgcodecBackend_t {
                struct_type: sys::nvimgcodecStructureType_t::NVIMGCODEC_STRUCTURE_TYPE_BACKEND,
                struct_size: size_of::<sys::nvimgcodecBackend_t>() as sys::size_t,
                struct_next: ptr::null_mut(),
                kind: kind.into(),
                params,
            },
        }
    }

    /// Sets the fraction of a batch that this backend should take, and how to apply it.
    ///
    /// The hint is passed to nvImageCodec unchecked, and a backend may ignore it.
    pub fn set_load_hint(&mut self, hint: f32, policy: LoadHintPolicy) -> &mut Self {
        self.raw.params.load_hint = hint;
        self.raw.params.load_hint_policy = policy.into();
        self
    }
}

pub(crate) fn prepare_images(
    instance: &InstanceRef,
    default_stream: Option<&Stream>,
    images: &mut [&mut Image<'_>],
    output: bool,
    skip_pre_sync: bool,
) -> Result<Vec<Stream>> {
    if let Some(stream) = default_stream {
        stream.ensure_not_capturing()?;
        stream.context().bind()?;
    }

    let mut streams = Vec::new();
    for image in images {
        instance.require_same(image.instance())?;
        image.prepare(default_stream, output)?;
        if let Some(stream) = image.stream()
            && !streams.contains(stream)
        {
            streams.push(stream.clone());
        }
    }

    if skip_pre_sync {
        for stream in &streams {
            stream.synchronize()?;
        }
    }
    Ok(streams)
}

struct Completion {
    future: sys::nvimgcodecFuture_t,
    streams: Vec<Stream>,
    host_completed: bool,
    device_completed: bool,
}

impl Completion {
    fn wait(&mut self) -> Result<()> {
        if !self.host_completed {
            if !self.future.is_null() {
                // SAFETY: `nvimgcodecFutureWaitForAll` receives only plain values.
                unsafe {
                    try_ffi!(sys::nvimgcodecFutureWaitForAll(self.future))?;
                }
            }

            self.host_completed = true;
        }

        if !self.device_completed {
            for stream in &self.streams {
                stream.synchronize()?;
            }

            self.device_completed = true;
        }
        Ok(())
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        // A failed host wait or device fence cannot release borrowed stack storage.
        if self.wait().is_err() {
            process::abort();
        }
        if !self.future.is_null() {
            // SAFETY: `nvimgcodecFutureDestroy` receives only plain values.
            if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecFutureDestroy(self.future)) } {
                eprintln!(
                    "failed to destroy nvimagecodec future: {}",
                    singe_core::ErrorChain(&error)
                );
            }
        }
    }
}

pub(crate) fn complete_submission(
    streams: Vec<Stream>,
    count: usize,
    submit: impl FnOnce(*mut sys::nvimgcodecFuture_t, i32) -> sys::nvimgcodecStatus_t,
) -> Result<Vec<ProcessingStatus>> {
    let batch_size = to_i32(count, "batch_size")?;
    let mut completion = Completion {
        future: ptr::null_mut(),
        streams,
        host_completed: false,
        device_completed: false,
    };
    let status = submit(&raw mut completion.future, batch_size);
    completion.wait()?;
    try_ffi!(status)?;
    if completion.future.is_null() {
        return Err(Error::NullHandle);
    }

    let mut statuses = vec![0; count];
    let mut actual = count as sys::size_t;
    // SAFETY: `nvimgcodecFutureGetProcessingStatus` receives valid output locations.
    unsafe {
        try_ffi!(sys::nvimgcodecFutureGetProcessingStatus(
            completion.future,
            statuses.as_mut_ptr(),
            &raw mut actual
        ))?;
    }

    if actual != count as sys::size_t {
        return Err(Error::LengthMismatch {
            name: "processing_status_count".into(),
            expected: count,
            actual: to_usize(actual, "result count")?,
        });
    }
    Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BackendKind;

    #[test]
    fn test_execution_params_clone_rebuilds_backend_pointer() -> Result<()> {
        let mut original = ExecutionParams::cpu_only();
        original.set_backends(vec![Backend::new(BackendKind::CpuOnly)])?;

        let cloned = original.clone();

        assert!(!original.raw.backends.is_null());
        assert!(!cloned.raw.backends.is_null());
        assert_ne!(original.raw.backends, cloned.raw.backends);
        assert_eq!(cloned.raw.backends, cloned.backends.as_ptr());
        assert_eq!(cloned.raw.num_backends, 1);
        Ok(())
    }
}
