use std::mem::align_of;

use singe_cuda::types::DevicePtr;

use crate::{
    error::{Error, Result},
    scalar::{SCALAR_STORAGE_BYTES, ScalarValue},
};

/// Owns an aligned native scalar independently of the Rust enum's payload layout.
#[derive(Debug, Clone)]
#[repr(C, align(16))]
pub struct ScalarStorage {
    bytes: [u8; SCALAR_STORAGE_BYTES],
    value: ScalarValue,
    aligned: Option<AlignedScalarBytes>,
}

impl ScalarStorage {
    pub(crate) fn new(value: ScalarValue) -> Self {
        let value = value.canonical();
        Self {
            bytes: value.native_bytes(),
            value,
            aligned: None,
        }
    }

    pub(crate) fn create(value: ScalarValue, byte_alignment: usize) -> Result<Self> {
        if !byte_alignment.is_power_of_two() {
            return Err(Error::out_of_range("scalar byte alignment"));
        }
        let mut storage = Self::new(value);
        if byte_alignment > align_of::<Self>() {
            storage.aligned = Some(AlignedScalarBytes::create(storage.bytes, byte_alignment)?);
        }
        Ok(storage)
    }

    pub(crate) fn value(&self) -> ScalarValue {
        self.value
    }

    pub(crate) fn byte_alignment(&self) -> usize {
        self.aligned
            .as_ref()
            .map_or(align_of::<Self>(), |storage| storage.alignment)
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.aligned
            .as_ref()
            .map_or_else(|| self.bytes.as_ptr(), AlignedScalarBytes::as_ptr)
    }

    /// Variant-pack by-value entries use host addresses in the pointer array.
    pub(crate) fn device_ptr(&self) -> DevicePtr {
        // SAFETY: this only records the retained host address; cuDNN consumes its native bytes by value.
        unsafe { DevicePtr::from_raw(self.as_ptr().cast_mut().cast()) }
    }
}

/// Keeps stronger scalar alignments independent of the allocation's base address.
#[derive(Debug)]
struct AlignedScalarBytes {
    bytes: Vec<u8>,
    offset: usize,
    alignment: usize,
}

impl AlignedScalarBytes {
    fn create(value: [u8; SCALAR_STORAGE_BYTES], alignment: usize) -> Result<Self> {
        let allocation_bytes = SCALAR_STORAGE_BYTES
            .checked_add(alignment - 1)
            .ok_or_else(|| Error::out_of_range("scalar storage bytes"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(allocation_bytes)
            .map_err(|source| Error::ScalarStorageAllocation {
                bytes: allocation_bytes,
                source,
            })?;
        bytes.resize(allocation_bytes, 0);
        let offset = bytes.as_ptr().align_offset(alignment);
        bytes[offset..offset + SCALAR_STORAGE_BYTES].copy_from_slice(&value);
        Ok(Self {
            bytes,
            offset,
            alignment,
        })
    }

    fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr().wrapping_add(self.offset)
    }
}

impl Clone for AlignedScalarBytes {
    fn clone(&self) -> Self {
        let mut bytes = self.bytes.clone();
        let offset = bytes.as_ptr().align_offset(self.alignment);
        bytes.copy_within(self.offset..self.offset + SCALAR_STORAGE_BYTES, offset);
        Self {
            bytes,
            offset,
            alignment: self.alignment,
        }
    }
}
