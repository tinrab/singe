use singe_cuda::{
    context::Context as CudaContext,
    stream::{BorrowedStream, Stream},
};
use singe_npp_sys as sys;

use crate::error::Result;

/// CUDA stream metadata passed to context-aware NPP functions.
pub struct StreamContext {
    stream: BorrowedStream,
    raw: sys::NppStreamContext,
}

impl StreamContext {
    /// Creates an NPP stream context from a CUDA stream.
    ///
    /// The returned context borrows the stream and caches the device properties
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
        let stream = stream.to_borrowed();

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
                nReserved0: 0,
            },
            stream,
        })
    }

    pub fn stream(&self) -> &BorrowedStream {
        &self.stream
    }

    pub fn cuda_context(&self) -> &CudaContext {
        self.stream.context()
    }

    pub const fn device_id(&self) -> i32 {
        self.raw.nCudaDeviceId
    }

    pub const fn as_raw(&self) -> sys::NppStreamContext {
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
            nReserved0: self.raw.nReserved0,
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Into::into)
    }
}
