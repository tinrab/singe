use std::marker::PhantomData;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};

/// A reference to a matrix stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct MatrixRef<'a, T> {
    pub data: &'a DeviceMemory<T>,
    pub leading_dimension: usize,
}

/// A mutable reference to a matrix stored on the device.
#[derive(Debug)]
pub struct MatrixMut<'a, T> {
    pub data: &'a mut DeviceMemory<T>,
    pub leading_dimension: usize,
}

/// A reference to a strided batched matrix stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct StridedBatchedMatrixRef<'a, T> {
    pub data: &'a DeviceMemory<T>,
    pub leading_dimension: usize,
    pub stride: usize,
}

/// A mutable reference to a strided batched matrix stored on the device.
#[derive(Debug)]
pub struct StridedBatchedMatrixMut<'a, T> {
    pub data: &'a mut DeviceMemory<T>,
    pub leading_dimension: usize,
    pub stride: usize,
}

/// A reference to a vector stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct VectorRef<'a, T> {
    pub data: &'a DeviceMemory<T>,
}

/// A mutable reference to a vector stored on the device.
#[derive(Debug)]
pub struct VectorMut<'a, T> {
    pub data: &'a mut DeviceMemory<T>,
}

/// A reference to a strided batched vector stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct StridedBatchedVectorRef<'a, T> {
    pub data: &'a DeviceMemory<T>,
    pub stride: usize,
}

/// A mutable reference to a strided batched vector stored on the device.
#[derive(Debug)]
pub struct StridedBatchedVectorMut<'a, T> {
    pub data: &'a mut DeviceMemory<T>,
    pub stride: usize,
}

/// A reference to a batched matrix (pointer array) stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct BatchedMatrixRef<'a, T> {
    pub pointers: &'a DeviceMemory<DevicePtr>,
    pub leading_dimension: usize,
    _phantom: PhantomData<T>,
}

/// A reference to a batched vector (pointer array) stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct BatchedVectorRef<'a, T> {
    pub pointers: &'a DeviceMemory<DevicePtr>,
    pub leading_dimension: usize,
    _phantom: PhantomData<T>,
}

/// A mutable reference to a byte workspace buffer on the device.
#[derive(Debug)]
pub struct ByteWorkspaceMut<'a> {
    pub device: &'a mut DeviceMemory<u8>,
    pub host: &'a mut [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceSizes {
    pub device_bytes: usize,
    pub host_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionWorkspaceSizes {
    pub selection_size: usize,
    pub workspace: WorkspaceSizes,
}

impl<'a, T> MatrixRef<'a, T> {
    pub const fn new(data: &'a DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            data,
            leading_dimension,
        }
    }
}

impl<'a, T> MatrixMut<'a, T> {
    pub const fn new(data: &'a mut DeviceMemory<T>, leading_dimension: usize) -> Self {
        Self {
            data,
            leading_dimension,
        }
    }

    pub fn as_ref(&self) -> MatrixRef<'_, T> {
        MatrixRef::new(self.data, self.leading_dimension)
    }
}

impl<'a, T> StridedBatchedMatrixRef<'a, T> {
    pub const fn new(data: &'a DeviceMemory<T>, leading_dimension: usize, stride: usize) -> Self {
        Self {
            data,
            leading_dimension,
            stride,
        }
    }
}

impl<'a, T> StridedBatchedMatrixMut<'a, T> {
    pub const fn new(
        data: &'a mut DeviceMemory<T>,
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data,
            leading_dimension,
            stride,
        }
    }

    pub fn as_ref(&self) -> StridedBatchedMatrixRef<'_, T> {
        StridedBatchedMatrixRef::new(self.data, self.leading_dimension, self.stride)
    }
}

impl<'a, T> VectorRef<'a, T> {
    pub const fn new(data: &'a DeviceMemory<T>) -> Self {
        Self { data }
    }
}

impl<'a, T> VectorMut<'a, T> {
    pub const fn new(data: &'a mut DeviceMemory<T>) -> Self {
        Self { data }
    }

    pub fn as_ref(&self) -> VectorRef<'_, T> {
        VectorRef::new(self.data)
    }
}

impl<'a, T> StridedBatchedVectorRef<'a, T> {
    pub const fn new(data: &'a DeviceMemory<T>, stride: usize) -> Self {
        Self { data, stride }
    }
}

impl<'a, T> StridedBatchedVectorMut<'a, T> {
    pub const fn new(data: &'a mut DeviceMemory<T>, stride: usize) -> Self {
        Self { data, stride }
    }

    pub fn as_ref(&self) -> StridedBatchedVectorRef<'_, T> {
        StridedBatchedVectorRef::new(self.data, self.stride)
    }
}

impl<'a, T> BatchedMatrixRef<'a, T> {
    pub const fn new(pointers: &'a DeviceMemory<DevicePtr>, leading_dimension: usize) -> Self {
        Self {
            pointers,
            leading_dimension,
            _phantom: PhantomData,
        }
    }

    pub const fn len(&self) -> usize {
        self.pointers.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    pub const fn as_mut_ptr(&self) -> *mut *mut T {
        self.pointers.as_ptr().cast::<*mut T>().cast_mut()
    }
}

impl<'a, T> BatchedVectorRef<'a, T> {
    pub const fn new(pointers: &'a DeviceMemory<DevicePtr>, leading_dimension: usize) -> Self {
        Self {
            pointers,
            leading_dimension,
            _phantom: PhantomData,
        }
    }

    pub const fn len(&self) -> usize {
        self.pointers.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    pub const fn as_mut_ptr(&self) -> *mut *mut T {
        self.pointers.as_ptr().cast::<*mut T>().cast_mut()
    }
}

impl<'a> ByteWorkspaceMut<'a> {
    pub const fn new(device: &'a mut DeviceMemory<u8>, host: &'a mut [u8]) -> Self {
        Self { device, host }
    }
}

impl WorkspaceSizes {
    pub const fn new(device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device_bytes,
            host_bytes,
        }
    }
}

impl SelectionWorkspaceSizes {
    pub const fn new(selection_size: usize, device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            selection_size,
            workspace: WorkspaceSizes::new(device_bytes, host_bytes),
        }
    }
}
