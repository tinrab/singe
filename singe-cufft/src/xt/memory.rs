use std::{
    mem::MaybeUninit,
    ptr, slice,
    sync::{Arc, PoisonError},
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataTypeLike,
    device::Device,
    error::Error as CudaError,
    stream::Stream,
    view::{AllocationId, DeviceRegion, DeviceRegionProvider},
};
use singe_cuda_sys::driver;
use singe_cufft_sys::xt as sys;

use super::{Layout, SubFormat, check_buffer, plan::PlanHandle};
use crate::{
    error::{Error, Result},
    try_ffi,
    utility::{checked_byte_len, to_usize},
};

/// Pending distributed storage whose native allocation and plan have shared owners.
///
/// Owned chunks expose allocation metadata without granting initialized or mutable sample access.
#[derive(Debug)]
pub struct DistributedMemory {
    allocation: Arc<DistributedAllocation>,
    format: SubFormat,
}

#[derive(Debug)]
struct ChunkInfo {
    context: Arc<CudaContext>,
    allocation_id: AllocationId,
    pointer: *mut u8,
    bytes: usize,
}

#[derive(Debug)]
struct DistributedAllocation {
    base: Arc<PlanHandle>,
    raw: *mut sys::cudaLibXtDesc,
    chunks: Vec<ChunkInfo>,
    byte_len: usize,
    layout: Layout,
    stream: Option<Stream>,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: cuFFT does not tie a descriptor to its allocating thread; host metadata is immutable after construction, and every native call on the descriptor holds the plan gate.
unsafe impl Send for DistributedAllocation {}

// SAFETY: shared access only reads metadata fixed at construction, and native calls on the descriptor hold the plan gate.
unsafe impl Sync for DistributedAllocation {}

/// Retains one actual device allocation independently of the distributed memory wrapper.
/// Cloning this metadata does not grant sample initialization or exclusive access.
#[derive(Debug, Clone)]
pub struct DistributedChunk {
    allocation: Arc<DistributedAllocation>,
    index: usize,
}

// SAFETY: each chunk's address, context, and byte range were validated when the allocation was described, and the chunk retains the allocation.
unsafe impl DeviceRegionProvider for DistributedChunk {
    fn region(&self) -> DeviceRegion<'_> {
        let chunk = &self.allocation.chunks[self.index];
        // SAFETY: cuFFT described this chunk for the allocation that `self.allocation` keeps alive.
        unsafe {
            DeviceRegion::from_raw_parts(
                &chunk.context,
                chunk.allocation_id,
                chunk.pointer,
                chunk.bytes,
            )
        }
        .expect("validated distributed allocation region")
    }
}

impl DistributedAllocation {
    fn describe_chunks(&mut self) -> Result<()> {
        self.base.base.bind()?;
        // SAFETY: `self.raw` is null or points at the descriptor cuFFT allocated for this live allocation.
        let descriptor = unsafe { self.raw.as_ref().and_then(|raw| raw.descriptor.as_ref()) }
            .ok_or(Error::NullHandle)?;
        let count = to_usize(descriptor.nGPUs, "GPU count")?;
        if count == 0 || count > descriptor.data.len() {
            return Err(Error::out_of_range("GPU count"));
        }

        for index in 0..count {
            let bytes = to_usize(descriptor.size[index], "allocation bytes")?;
            self.byte_len = self
                .byte_len
                .checked_add(bytes)
                .ok_or_else(|| Error::out_of_range("allocation bytes"))?;
            if bytes == 0 {
                continue;
            }

            let pointer = descriptor.data[index].cast::<u8>();
            if pointer.is_null() {
                return Err(Error::NullHandle);
            }

            let address = driver::CUdeviceptr::try_from(pointer.addr())
                .map_err(|_| Error::out_of_range("distributed allocation address"))?;
            let mut raw_context: driver::CUcontext = ptr::null_mut();
            // SAFETY: `raw_context` is a valid output location for the context attribute.
            let status = unsafe {
                driver::cuPointerGetAttribute(
                    (&raw mut raw_context).cast(),
                    driver::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                    address,
                )
            };

            if status != driver::CUresult::CUDA_SUCCESS {
                return Err(CudaError::from(status).into());
            }

            let context = if raw_context == self.base.base.cuda_context().as_raw() {
                Arc::clone(self.base.base.cuda_context())
            } else {
                // Query the pointer's actual context first, then verify the retained primary owner.
                let context = CudaContext::create_for_device(Device::new(descriptor.GPUs[index]))?;
                if context.as_raw() != raw_context {
                    return Err(CudaError::MemoryContextMismatch.into());
                }
                context
            };
            if context.device().id() != descriptor.GPUs[index] {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            let allocation_id = AllocationId::create();
            // SAFETY: cuFFT reported this pointer and size for a live chunk; the region is only validated here.
            unsafe {
                DeviceRegion::from_raw_parts(&context, allocation_id, pointer, bytes)?;
            }

            self.chunks.push(ChunkInfo {
                context,
                allocation_id,
                pointer,
                bytes,
            });
        }
        Ok(())
    }

    fn complete(&self, source: Option<&Self>, submit: impl FnOnce() -> Result<()>) -> Result<()> {
        let _native = self.base.guard()?;
        if let Some(stream) = &self.stream {
            stream.ensure_not_capturing()?;
        }

        let mut contexts = vec![Arc::clone(self.base.base.cuda_context())];
        for chunk in &self.chunks {
            if !contexts.iter().any(|context| context == &chunk.context) {
                contexts.push(Arc::clone(&chunk.context));
            }
        }

        if let Some(source) = source {
            for chunk in &source.chunks {
                if !contexts.iter().any(|context| context == &chunk.context) {
                    contexts.push(Arc::clone(&chunk.context));
                }
            }
        }

        // Preflight before enqueueing so a rejected synchronous operation has no host borrow to drain.
        for context in &contexts {
            context.synchronize()?;
        }

        let mut drain = ContextDrain {
            contexts,
            drained: false,
        };
        self.base.base.bind()?;
        let result = submit();
        drain.wait()?;
        result
    }
}

struct ContextDrain {
    contexts: Vec<Arc<CudaContext>>,
    drained: bool,
}

impl ContextDrain {
    fn wait(&mut self) -> Result<()> {
        for context in &self.contexts {
            context.synchronize()?;
        }

        self.drained = true;
        Ok(())
    }
}

impl Drop for ContextDrain {
    fn drop(&mut self) {
        if !self.drained {
            for context in &self.contexts {
                context.synchronize_or_abort();
            }
        }
    }
}

impl DistributedMemory {
    pub(crate) fn allocate(
        base: &Arc<PlanHandle>,
        layout: &Layout,
        stream: Option<&Stream>,
        format: SubFormat,
    ) -> Result<Self> {
        if let Some(stream) = stream {
            stream.ensure_not_capturing()?;
        }

        let mut raw = ptr::null_mut();
        {
            let _native = base.guard()?;
            base.base.bind()?;
            // SAFETY: the plan handle is live, its context is bound, and `raw` is a valid output location.
            unsafe {
                try_ffi!(sys::cufftXtMalloc(
                    base.base.as_raw(),
                    &raw mut raw,
                    format.into()
                ))?;
            }
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let mut allocation = DistributedAllocation {
            base: Arc::clone(base),
            raw,
            chunks: Vec::new(),
            byte_len: 0,
            layout: layout.clone(),
            stream: stream.cloned(),
        };
        allocation.describe_chunks()?;
        if allocation.byte_len == 0 {
            return Err(Error::out_of_range("distributed allocation bytes"));
        }

        // Publish allocation metadata only after the library's allocation work is ready in its actual contexts.
        allocation.base.base.cuda_context().synchronize()?;
        for chunk in &allocation.chunks {
            chunk.context.synchronize()?;
        }
        Ok(Self {
            allocation: Arc::new(allocation),
            format,
        })
    }

    pub(crate) fn belongs_to(&self, base: &Arc<PlanHandle>) -> bool {
        Arc::ptr_eq(&self.allocation.base, base)
    }

    /// Returns the storage format the allocation was made with.
    #[must_use]
    pub fn format(&self) -> SubFormat {
        self.format
    }

    /// Native allocation bytes, including physical padding, across all nonempty GPU chunks.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.allocation.byte_len
    }

    /// Iterates over owned metadata for each nonempty GPU allocation.
    ///
    /// Each yielded chunk retains the allocation and plan independently of this wrapper.
    /// It does not grant initialized sample access.
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = DistributedChunk> + '_ {
        (0..self.allocation.chunks.len()).map(|index| DistributedChunk {
            allocation: Arc::clone(&self.allocation),
            index,
        })
    }

    fn check_host_storage<T>(&self, len: usize) -> Result<()> {
        let bytes = checked_byte_len::<T>(len, "host storage bytes")?;
        if bytes < self.byte_len() {
            return Err(Error::BufferTooSmall {
                name: "host storage".into(),
                required: self.byte_len(),
                actual: bytes,
            });
        }
        Ok(())
    }

    /// Completes upload of initialized host bytes, including any required native storage padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured representation, the host storage is too short (including physical padding), or a size calculation overflows.
    /// Stream capture, CUDA context operations and synchronization, and cuFFT copy failures are also returned.
    pub fn copy_from_host<T: DataTypeLike>(&mut self, values: &[T]) -> Result<()> {
        check_buffer::<T>(&self.allocation.layout, values.len(), true)?;
        self.check_host_storage::<T>(values.len())?;
        // SAFETY: `self.allocation` retains the live plan and descriptor, `check_host_storage` proved `values` covers `byte_len()` bytes, and `complete` holds the plan gate and drains the copy before the borrow ends.
        self.allocation.complete(None, || unsafe {
            try_ffi!(sys::cufftXtMemcpy(
                self.allocation.base.base.as_raw(),
                self.as_raw().cast(),
                values.as_ptr().cast_mut().cast(),
                sys::cufftXtCopyType::CUFFT_COPY_HOST_TO_DEVICE
            ))
        })
    }

    /// Completes readback into pending host storage without claiming sample or padding validity.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured representation, the host storage is too short (including physical padding), or a size calculation overflows.
    /// Stream capture, CUDA context operations and synchronization, and cuFFT copy failures are also returned.
    pub fn copy_to_host<T: DataTypeLike>(&self, values: &mut [MaybeUninit<T>]) -> Result<()> {
        check_buffer::<T>(&self.allocation.layout, values.len(), false)?;
        self.check_host_storage::<T>(values.len())?;
        // SAFETY: `self.allocation` retains the live plan and descriptor, `check_host_storage` proved `values` covers `byte_len()` writable bytes, and `complete` drains the copy before the borrow ends.
        self.allocation.complete(None, || unsafe {
            try_ffi!(sys::cufftXtMemcpy(
                self.allocation.base.base.as_raw(),
                values.as_mut_ptr().cast(),
                self.as_raw().cast(),
                sys::cufftXtCopyType::CUFFT_COPY_DEVICE_TO_HOST
            ))
        })
    }

    /// Completes readback and exposes only the native copied prefix as initialized values.
    ///
    /// # Safety
    ///
    /// `initialized_elements` must describe a prefix that native readback fully produces as valid `T` values.
    /// This includes any physical padding within that prefix.
    /// Logical FFT sample production alone does not establish initialization for allocation padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the host element type differs from the configured representation, the host storage is too short (including physical padding), or a size calculation overflows.
    /// Stream capture, CUDA context operations and synchronization, and cuFFT copy failures are also returned.
    /// Also returns an error if the initialized prefix exceeds either the host slice or the number of bytes copied, or its byte count overflows.
    pub unsafe fn copy_to_host_unchecked<'host, T: DataTypeLike>(
        &self,
        values: &'host mut [MaybeUninit<T>],
        initialized_elements: usize,
    ) -> Result<&'host mut [T]> {
        let initialized_bytes =
            checked_byte_len::<T>(initialized_elements, "initialized prefix bytes")?;
        if initialized_elements > values.len() {
            return Err(Error::BufferTooSmall {
                name: "host values".into(),
                required: initialized_elements,
                actual: values.len(),
            });
        }

        if initialized_bytes > self.byte_len() {
            return Err(Error::BufferTooSmall {
                name: "distributed allocation".into(),
                required: initialized_bytes,
                actual: self.byte_len(),
            });
        }

        self.copy_to_host(values)?;
        // SAFETY: the prefix fits `values` (checked above), and the caller guarantees the completed readback wrote `initialized_elements` valid `T` values.
        Ok(unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast(), initialized_elements) })
    }

    /// Completes a byte copy between allocations from the same native plan.
    /// The destination remains pending; no interpretation of copied padding is introduced.
    ///
    /// # Errors
    ///
    /// Returns an error if the source belongs to another native plan, the destination has fewer allocation bytes than the source, the stream is being captured, CUDA context operations or synchronization fail, or cuFFT cannot perform the copy.
    pub fn copy_from(&mut self, source: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.allocation.base, &source.allocation.base) {
            return Err(Error::PlanMismatch);
        }

        if self.byte_len() < source.byte_len() {
            return Err(Error::BufferTooSmall {
                name: "distributed destination".into(),
                required: source.byte_len(),
                actual: self.byte_len(),
            });
        }
        self.allocation
            // SAFETY: both descriptors are live and belong to the same plan, the destination is at least as large as the source (checked above), and `complete` drains both allocations' contexts.
            .complete(Some(&source.allocation), || unsafe {
                try_ffi!(sys::cufftXtMemcpy(
                    self.allocation.base.base.as_raw(),
                    self.as_raw().cast(),
                    source.as_raw().cast(),
                    sys::cufftXtCopyType::CUFFT_COPY_DEVICE_TO_DEVICE
                ))
            })
    }

    /// Borrows the native descriptor; external mutations must preserve its tracked allocation and plan metadata.
    #[must_use]
    pub fn as_raw(&self) -> *mut sys::cudaLibXtDesc {
        self.allocation.raw
    }
}

impl Drop for DistributedAllocation {
    fn drop(&mut self) {
        // Cleanup must free the allocation whatever a panic left in this gate, which guards no data of its own.
        let _native = self
            .base
            .gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Err(error) = self.base.base.bind() {
            report_cleanup_failure(
                "bind the cuda context to free",
                "cufft distributed allocation",
                &error,
            );
            return;
        }

        // SAFETY: `self` owns the live descriptor from `cufftXtMalloc`, its plan is retained and gated, and `drop` frees it once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cufftXtFree(self.raw)) {
                report_cleanup_failure("free", "cufft distributed allocation", &error);
            }
        }
    }
}
