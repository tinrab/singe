//! Temporary GPU workspace for nvCOMP operations.

use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::{DeviceAllocation, DeviceMemory},
    types::DevicePtr,
};

use crate::error::{Error, Result};

/// Temporary GPU workspace for batched nvCOMP operations.
///
/// A zero-byte workspace allocates nothing and passes a null pointer to nvCOMP, which is what nvCOMP expects when an operation needs no workspace.
#[derive(Debug)]
pub struct TemporaryBuffer {
    context: Arc<CudaContext>,
    memory: Option<DeviceMemory<MaybeUninit<u8>>>,
    bytes: usize,
}

impl TemporaryBuffer {
    /// Allocates an uninitialized workspace of `bytes` bytes in `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes` is nonzero and the allocation fails.
    pub fn create(context: &Arc<CudaContext>, bytes: usize) -> Result<Self> {
        Ok(Self {
            context: Arc::clone(context),
            memory: if bytes == 0 {
                None
            } else {
                Some(DeviceMemory::<u8>::create_uninit(context, bytes)?)
            },
            bytes,
        })
    }

    /// Uses an existing device allocation as the workspace.
    #[must_use]
    pub fn from_memory(memory: DeviceMemory<MaybeUninit<u8>>) -> Self {
        let bytes = memory.len();
        Self {
            context: Arc::clone(memory.context()),
            memory: Some(memory),
            bytes,
        }
    }

    /// Returns the CUDA context of the workspace, which an empty workspace also keeps.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns the underlying allocation, or `None` for an empty workspace.
    ///
    /// Its contents are not initialized.
    pub fn allocation(&self) -> Option<&Arc<DeviceAllocation>> {
        self.memory.as_ref().map(DeviceMemory::allocation)
    }

    /// Returns the workspace size in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes
    }

    /// Returns whether the workspace has zero bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Checks that the workspace has at least `bytes` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace is smaller than `bytes`.
    pub fn require(&self, bytes: usize) -> Result<()> {
        if self.bytes < bytes {
            return Err(Error::LengthMismatch {
                name: "temp buffer".into(),
                expected: bytes,
                actual: self.bytes,
            });
        }

        Ok(())
    }

    /// Returns whether the workspace starts at a multiple of `alignment` bytes.
    ///
    /// An empty workspace and alignments of 0 or 1 always return `true`.
    #[must_use]
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        if self.bytes == 0 || alignment <= 1 {
            return true;
        }

        self.memory
            .as_ref()
            .is_none_or(|memory| memory.as_device_ptr().is_aligned_to(alignment))
    }

    /// Returns the device pointer to pass to nvCOMP, which is null for an empty workspace.
    pub fn as_device_ptr(&mut self) -> DevicePtr {
        if self.bytes == 0 {
            DevicePtr::null()
        } else {
            self.memory
                .as_ref()
                .map_or(DevicePtr::null(), DeviceMemory::as_device_ptr)
        }
    }

    /// Returns the underlying allocation, or `None` for an empty workspace.
    #[must_use]
    pub fn into_device_memory(self) -> Option<DeviceMemory<MaybeUninit<u8>>> {
        self.memory
    }
}
