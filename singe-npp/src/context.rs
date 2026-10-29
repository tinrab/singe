use std::{
    fmt::{self, Debug, Formatter},
    sync::Arc,
};

use singe_cuda::{context::Context as CudaContext, stream::Stream, view::DeviceRegionProvider};
use singe_npp_sys as sys;

use crate::error::{Error, Result};

/// CUDA stream metadata passed to context-aware NPP functions.
pub struct StreamContext {
    stream: Stream,
    raw: sys::NppStreamContext,
}

impl Debug for StreamContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamContext").finish_non_exhaustive()
    }
}

impl StreamContext {
    /// Creates an NPP stream context from a CUDA stream.
    ///
    /// The returned context retains the stream and caches the device properties
    /// required by NPP context-aware entry points.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream's CUDA context cannot be bound, or if CUDA
    /// cannot query the stream device, device properties, or stream flags.
    pub fn create(stream: &Stream) -> Result<Self> {
        stream.context().bind()?;
        let device = stream.device()?;
        let properties = device.properties()?;
        let flags = stream.flags()?;
        let stream = stream.clone();

        Ok(Self {
            raw: sys::NppStreamContext {
                hStream: stream.as_raw(),
                nCudaDeviceId: device.id(),
                nMultiProcessorCount: properties.multi_processor_count,
                nMaxThreadsPerMultiProcessor: properties.max_threads_per_multi_processor,
                nMaxThreadsPerBlock: properties.max_threads_per_block,
                nSharedMemPerBlock: properties.shared_mem_per_block as _,
                nCudaDevAttrComputeCapabilityMajor: properties.major,
                nCudaDevAttrComputeCapabilityMinor: properties.minor,
                nStreamFlags: flags.bits(),
                __bindgen_anon_1: sys::NppStreamContext__bindgen_ty_1 { hWorkspace: 0 },
            },
            stream,
        })
    }

    /// Returns the retained CUDA stream used by NPP execution.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns the CUDA context retained by this stream.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    /// Returns the cached CUDA device ordinal.
    #[must_use]
    pub const fn device_id(&self) -> i32 {
        self.raw.nCudaDeviceId
    }

    /// Returns a native stream-context descriptor with cached device properties.
    ///
    /// The descriptor borrows the retained stream; copying it does not retain native ownership.
    /// Keep the stream and its CUDA context alive through native use of the descriptor.
    #[must_use]
    pub fn as_raw(&self) -> sys::NppStreamContext {
        sys::NppStreamContext {
            hStream: self.stream.as_raw(),
            nCudaDeviceId: self.raw.nCudaDeviceId,
            nMultiProcessorCount: self.raw.nMultiProcessorCount,
            nMaxThreadsPerMultiProcessor: self.raw.nMaxThreadsPerMultiProcessor,
            nMaxThreadsPerBlock: self.raw.nMaxThreadsPerBlock,
            nSharedMemPerBlock: self.raw.nSharedMemPerBlock,
            nCudaDevAttrComputeCapabilityMajor: self.raw.nCudaDevAttrComputeCapabilityMajor,
            nCudaDevAttrComputeCapabilityMinor: self.raw.nCudaDevAttrComputeCapabilityMinor,
            nStreamFlags: self.raw.nStreamFlags,
            __bindgen_anon_1: self.raw.__bindgen_anon_1,
        }
    }

    /// Runs borrowed native work and drains this actual stream on every path.
    /// Capture is rejected before submission; an unquiesced CUDA failure aborts
    /// before borrowed resources can be returned to the caller.
    ///
    /// # Errors
    ///
    /// Returns an error if querying capture state or binding the CUDA context fails, capture is active or invalidated, or stream synchronization fails.
    /// Errors returned by `f` propagate after the stream completes.
    /// A stream error takes precedence over an error returned by `f`.
    ///
    /// # Panics
    ///
    /// Propagates a panic from `f` after cleanup drains the stream.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        let mut result = None;
        self.stream.sync_scope(|_| {
            result = Some(f());
            Ok(())
        })?;
        result.expect("native stream scope ran")
    }

    pub(crate) fn prepare_region(
        &self,
        resource: &(impl DeviceRegionProvider + ?Sized),
    ) -> Result<()> {
        self.validate_region(resource)?;
        if let Some(producer) = resource.region().stream()
            && producer.as_raw() != self.stream.as_raw()
        {
            producer.ensure_not_capturing()?;
            producer.synchronize()?;
        }
        Ok(())
    }

    pub(crate) fn validate_region(
        &self,
        resource: &(impl DeviceRegionProvider + ?Sized),
    ) -> Result<()> {
        if resource.region().context() != self.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(())
    }

    /// Waits for all work submitted to the retained CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or stream synchronization fails, including a previous asynchronous execution error.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Into::into)
    }
}
