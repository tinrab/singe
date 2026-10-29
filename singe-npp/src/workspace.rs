//! Scratch-buffer support for NPP wrappers.
//!
//! `ScratchBuffer` owns temporary device memory used by NPP operations that
//! expose host-side buffer-size queries.

use std::{iter, mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    view::{DeviceRegion, DeviceRegionProvider},
};
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    types::BufferDescriptor,
    utility::to_i32,
};

/// Owned uninitialized device-byte storage for NPP scratch work.
#[derive(Debug)]
pub struct ScratchBuffer {
    memory: DeviceMemory<MaybeUninit<u8>>,
}

impl ScratchBuffer {
    /// Returns the CUDA context that owns the scratch allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.memory.context()
    }

    /// Allocates `bytes` of uninitialized scratch storage in `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or device allocation fails.
    pub fn create(context: &Arc<CudaContext>, bytes: usize) -> Result<Self> {
        Ok(Self {
            memory: DeviceMemory::<u8>::create_uninit(context, bytes)?,
        })
    }

    /// Takes ownership of an existing device-byte allocation as scratch storage.
    #[must_use]
    pub fn from_memory(memory: DeviceMemory<MaybeUninit<u8>>) -> Self {
        Self { memory }
    }

    /// Returns the scratch capacity in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns whether the scratch allocation contains zero bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Checks that the scratch capacity is at least `bytes`.
    ///
    /// This checks capacity only; it does not validate a stream context or synchronize device work.
    ///
    /// # Errors
    ///
    /// Returns a length-mismatch error if the allocation contains fewer than `bytes` bytes.
    pub fn require(&self, bytes: usize) -> Result<()> {
        if self.len() < bytes {
            return Err(Error::LengthMismatch {
                name: "scratch buffer".into(),
                expected: bytes,
                actual: self.len(),
            });
        }

        Ok(())
    }

    /// Returns a writable device pointer while retaining ownership of the allocation.
    ///
    /// The caller of a native operation must keep the allocation alive until native use completes.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.memory.as_mut_ptr().cast()
    }

    /// Transfers ownership of the scratch allocation to the returned device memory.
    #[must_use]
    pub fn into_device_memory(self) -> DeviceMemory<MaybeUninit<u8>> {
        self.memory
    }
}

// SAFETY: the region preserves the owned device allocation's actual context, identity, and readiness.
unsafe impl DeviceRegionProvider for ScratchBuffer {
    fn region(&self) -> DeviceRegion<'_> {
        self.memory.region()
    }
}

#[derive(Debug)]
pub(crate) struct BufferDescriptors {
    memory: DeviceMemory<sys::NppiBufferDescriptor>,
}

impl BufferDescriptors {
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "writable descriptor pointers require an exclusive workspace borrow"
    )]
    pub fn as_mut_ptr(&mut self) -> *mut sys::NppiBufferDescriptor {
        self.memory.as_mut_ptr()
    }
}

pub(crate) fn create_buffer_descriptors(
    allocation_context: &Arc<CudaContext>,
    buffer_sizes: impl IntoIterator<Item = usize>,
) -> Result<(Vec<DeviceMemory<MaybeUninit<u8>>>, BufferDescriptors)> {
    let buffer_sizes: Vec<_> = buffer_sizes.into_iter().collect();
    let mut buffers = Vec::with_capacity(buffer_sizes.len());
    let mut descriptors = Vec::with_capacity(buffer_sizes.len());

    for bytes in buffer_sizes {
        let descriptor_size = to_i32(bytes, "buffer size")?;
        let buffer = DeviceMemory::<u8>::create_uninit(allocation_context, bytes)?;
        descriptors.push(BufferDescriptor {
            data: buffer.as_mut_ptr().cast(),
            size: descriptor_size,
        });
        buffers.push(buffer);
    }

    let raw_descriptors = descriptors
        .into_iter()
        .map(sys::NppiBufferDescriptor::from)
        .collect::<Vec<_>>();

    Ok((
        buffers,
        BufferDescriptors {
            // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
            memory: unsafe {
                DeviceMemory::from_slice_unchecked(allocation_context, &raw_descriptors)
            }?,
        },
    ))
}

pub(crate) fn create_repeated_buffer_descriptors(
    allocation_context: &Arc<CudaContext>,
    count: usize,
    bytes_per_buffer: usize,
) -> Result<(Vec<DeviceMemory<MaybeUninit<u8>>>, BufferDescriptors)> {
    create_buffer_descriptors(allocation_context, iter::repeat_n(bytes_per_buffer, count))
}
