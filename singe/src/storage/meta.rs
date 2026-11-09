use std::ptr;

use crate::{
    error::Result,
    storage::{StorageVersion, cpu::CpuStorage},
};

/// Storage of the meta device: a byte size with no device allocation behind it.
///
/// Meta tensors carry data type, shape, and layout so capture and model construction can trace programs without running kernels.
/// Framework literals, such as a scalar divisor, keep their host value, so a graph traced on meta tensors can still be placed on a device with values.
#[derive(Debug, Default)]
pub struct MetaStorage {
    byte_size: usize,
    value: Option<CpuStorage>,
    version: StorageVersion,
}

impl MetaStorage {
    /// Describes an allocation of `byte_size` bytes without allocating it.
    #[must_use]
    pub fn new(byte_size: usize) -> Self {
        Self {
            byte_size,
            value: None,
            version: StorageVersion::default(),
        }
    }

    /// Describes the host bytes `value`, retaining them so copies off the meta device keep their values.
    #[must_use]
    pub fn with_value(value: CpuStorage) -> Self {
        Self {
            byte_size: value.byte_size(),
            value: Some(value),
            version: StorageVersion::default(),
        }
    }

    /// Returns the described allocation size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Returns the retained host value, which only framework literals have.
    #[must_use]
    pub fn value(&self) -> Option<&CpuStorage> {
        self.value.as_ref()
    }

    /// Returns another description of the same allocation, with a copy of any retained value.
    ///
    /// # Errors
    ///
    /// Returns an error if copying the retained value fails.
    pub fn duplicate(&self) -> Result<Self> {
        Ok(match &self.value {
            Some(value) => Self::with_value(value.clone_on_host()?),
            None => Self::new(self.byte_size),
        })
    }

    /// Returns the number of writes recorded for this allocation.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version.get()
    }

    /// Records a write to this allocation.
    pub fn bump_version(&self) {
        self.version.bump();
    }

    /// Returns a process-local identity for this storage owner.
    #[must_use]
    pub fn storage_identity(&self) -> usize {
        ptr::from_ref(self).addr()
    }
}
