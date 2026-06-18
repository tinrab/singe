//! Scratch-buffer support for NPP wrappers.
//!
//! `ScratchBuffer` owns temporary device memory used by NPP operations that
//! expose host-side buffer-size queries.

use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    types::BufferDescriptor,
    utility::to_i32,
};

#[derive(Debug)]
pub struct ScratchBuffer {
    memory: DeviceMemory<u8>,
}

impl ScratchBuffer {
    pub fn create(bytes: usize) -> Result<Self> {
        Ok(Self {
            memory: DeviceMemory::create(bytes)?,
        })
    }

    pub fn from_memory(memory: DeviceMemory<u8>) -> Self {
        Self { memory }
    }

    pub const fn len(&self) -> usize {
        self.memory.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

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

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.memory.as_mut_ptr()
    }

    pub fn into_device_memory(self) -> DeviceMemory<u8> {
        self.memory
    }
}

#[derive(Debug)]
pub(crate) struct BufferDescriptors {
    memory: DeviceMemory<sys::NppiBufferDescriptor>,
}

impl BufferDescriptors {
    pub fn as_mut_ptr(&mut self) -> *mut sys::NppiBufferDescriptor {
        self.memory.as_mut_ptr()
    }
}

pub(crate) fn create_buffer_descriptors(
    buffer_sizes: impl IntoIterator<Item = usize>,
) -> Result<(Vec<DeviceMemory<u8>>, BufferDescriptors)> {
    let buffer_sizes: Vec<_> = buffer_sizes.into_iter().collect();
    let mut buffers = Vec::with_capacity(buffer_sizes.len());
    let mut descriptors = Vec::with_capacity(buffer_sizes.len());

    for bytes in buffer_sizes {
        let descriptor_size = to_i32(bytes, "buffer size")?;
        let buffer = DeviceMemory::<u8>::create(bytes)?;
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
            memory: DeviceMemory::from_slice(&raw_descriptors)?,
        },
    ))
}

pub(crate) fn create_repeated_buffer_descriptors(
    count: usize,
    bytes_per_buffer: usize,
) -> Result<(Vec<DeviceMemory<u8>>, BufferDescriptors)> {
    create_buffer_descriptors(std::iter::repeat_n(bytes_per_buffer, count))
}
