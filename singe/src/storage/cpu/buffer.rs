use std::{
    fmt::{self, Debug, Formatter},
    mem::ManuallyDrop,
    ptr::NonNull,
    slice,
};

#[cfg(feature = "cuda")]
use singe_cuda::memory::PinnedHostMemory;

/// Allocation unit for zero-filled and copied host bytes, aligned for every tensor element type.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([u8; 64]);

/// Owned host bytes from a vector of any element type, freed as that vector.
pub struct HostBytes {
    pointer: NonNull<u8>,
    length: usize,
    capacity: usize,
    free: unsafe fn(NonNull<u8>, usize),
}

// SAFETY: `HostBytes` uniquely owns its allocation, like the `Vec` it came from.
unsafe impl Send for HostBytes {}

// SAFETY: shared access only reads the bytes.
unsafe impl Sync for HostBytes {}

impl HostBytes {
    /// Adopts `values` without copying; the bytes are `values` in native-endian order.
    pub fn from_vec<T: Copy>(values: Vec<T>) -> Self {
        let mut values = ManuallyDrop::new(values);
        let length = values.len() * size_of::<T>();
        let capacity = values.capacity() * size_of::<T>();
        // SAFETY: `Vec::as_mut_ptr` is never null.
        let pointer = unsafe { NonNull::new_unchecked(values.as_mut_ptr().cast::<u8>()) };
        Self {
            pointer,
            length,
            capacity,
            free: free_vec::<T>,
        }
    }

    /// Allocates `length` zero bytes, aligned for every tensor element type.
    pub fn zeroed(length: usize) -> Self {
        let mut bytes = Self::from_vec(vec![Block([0; 64]); length.div_ceil(64)]);
        bytes.length = length;
        bytes
    }

    fn copied(data: &[u8]) -> Self {
        let mut bytes = Self::zeroed(data.len());
        bytes.as_bytes_mut().copy_from_slice(data);
        bytes
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: the allocation holds `capacity >= length` initialized bytes.
        unsafe { slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation holds `capacity >= length` initialized bytes, borrowed uniquely here.
        unsafe { slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
}

impl Drop for HostBytes {
    fn drop(&mut self) {
        // SAFETY: `free` is the deallocator of the vector this allocation came from.
        unsafe { (self.free)(self.pointer, self.capacity) }
    }
}

/// Frees an allocation adopted from a `Vec<T>` of `capacity` bytes.
///
/// # Safety
///
/// `pointer` and `capacity` must describe a vector allocation of `T`.
unsafe fn free_vec<T>(pointer: NonNull<u8>, capacity: usize) {
    let capacity = capacity / size_of::<T>();
    // SAFETY: the caller passes the pointer and capacity of a `Vec<T>`; its length is irrelevant for `Copy` elements.
    drop(unsafe { Vec::from_raw_parts(pointer.as_ptr().cast::<T>(), 0, capacity) });
}

/// Untyped host bytes behind a CPU allocation.
pub enum CpuBuffer {
    Heap(HostBytes),
    /// Page-locked bytes that CUDA copies read and write without staging.
    #[cfg(feature = "cuda")]
    Pinned(PinnedHostMemory<u8>),
}

impl Debug for CpuBuffer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuBuffer")
            .field("bytes", &self.as_bytes().len())
            .field("pinned", &self.is_pinned())
            .finish()
    }
}

/// Clones into ordinary host memory; a pinned buffer's clone is not page-locked.
impl Clone for CpuBuffer {
    fn clone(&self) -> Self {
        Self::from_bytes(self.as_bytes())
    }
}

impl CpuBuffer {
    pub fn zeroed(length: usize) -> Self {
        Self::Heap(HostBytes::zeroed(length))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::Heap(HostBytes::copied(bytes))
    }

    pub fn allocation_byte_size(&self) -> usize {
        match self {
            Self::Heap(bytes) => bytes.capacity,
            #[cfg(feature = "cuda")]
            Self::Pinned(bytes) => bytes.len(),
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Heap(bytes) => bytes.as_bytes(),
            #[cfg(feature = "cuda")]
            Self::Pinned(bytes) => bytes.as_slice(),
        }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Heap(bytes) => bytes.as_bytes_mut(),
            #[cfg(feature = "cuda")]
            Self::Pinned(bytes) => bytes.as_slice_mut(),
        }
    }

    /// Returns whether the bytes are page-locked for CUDA transfers.
    #[cfg_attr(
        not(feature = "cuda"),
        expect(clippy::unused_self, reason = "only CUDA pins host memory")
    )]
    pub const fn is_pinned(&self) -> bool {
        #[cfg(feature = "cuda")]
        if let Self::Pinned(..) = self {
            return true;
        }
        false
    }

    pub fn fill_zeroes(&mut self) {
        self.as_bytes_mut().fill(0);
    }
}
