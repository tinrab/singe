use std::{marker::PhantomData, mem::MaybeUninit};

use singe_cuda::{
    types::DevicePtr,
    view::{
        DeviceOutput, DeviceOutputView, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView,
        DeviceViewMut,
    },
};

use crate::error::{Error, Result};

/// Column-major storage metadata for allocation-free workspace queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    /// Element distance between successive column starts; validated against the row count by the solver operation.
    pub leading_dimension: usize,
}

impl MatrixLayout {
    /// Records column-major layout metadata without validating a matrix shape.
    ///
    /// Leading dimension is measured in elements, not bytes.
    #[must_use]
    pub const fn new(leading_dimension: usize) -> Self {
        Self { leading_dimension }
    }

    pub(crate) fn validate(self, rows: usize, columns: usize) -> Result<()> {
        if rows == 0 || columns == 0 {
            return Err(Error::InvalidMatrixShape);
        }
        if self.leading_dimension < rows {
            return Err(Error::InvalidLeadingDimension);
        }
        self.leading_dimension
            .checked_mul(columns)
            .ok_or(Error::InvalidMatrixShape)?;
        Ok(())
    }

    pub(crate) fn require_capacity(
        self,
        rows: usize,
        columns: usize,
        elements: usize,
    ) -> Result<()> {
        self.validate(rows, columns)?;
        let required = self
            .leading_dimension
            .checked_mul(columns)
            .ok_or(Error::InvalidMatrixShape)?;
        if elements < required {
            return Err(Error::InvalidMatrixShape);
        }
        Ok(())
    }
}

/// Column-major leading dimension and element stride between batched matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedMatrixLayout {
    /// Element distance between successive column starts; validated against the row count by the solver operation.
    pub leading_dimension: usize,
    /// Element distance between the starts of consecutive matrices.
    pub stride: usize,
}

impl StridedMatrixLayout {
    /// Records column-major layout metadata without validating a matrix shape.
    ///
    /// Leading dimension and batch stride are measured in elements, not bytes.
    #[must_use]
    pub const fn new(leading_dimension: usize, stride: usize) -> Self {
        Self {
            leading_dimension,
            stride,
        }
    }
}

/// A reference to a matrix stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct MatrixRef<'a, T: DeviceRepr> {
    /// Borrowed initialized device storage.
    pub data: DeviceView<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
}

/// A mutable reference to a matrix stored on the device.
#[derive(Debug)]
pub struct MatrixMut<'a, T: DeviceRepr> {
    /// Exclusively borrowed initialized device storage.
    pub data: DeviceViewMut<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
}

/// An exclusive destination for a column-major matrix.
///
/// This view does not permit reading existing elements or establish that a solver
/// writes the entire allocation. Leading-dimension padding remains uninitialized
/// unless a producer explicitly writes it.
#[derive(Debug)]
pub struct MatrixOutput<'a, T: DeviceRepr> {
    /// Exclusive output capacity; this view does not establish initialized elements.
    pub data: DeviceOutputView<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
}

impl<'a, T: DeviceRepr> MatrixOutput<'a, T> {
    /// Borrows initialized or uninitialized storage for a matrix producer.
    pub fn new(data: &'a mut (impl DeviceOutput<T> + ?Sized), leading_dimension: usize) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
            leading_dimension,
        }
    }
}

/// A reference to a strided batched matrix stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct StridedBatchedMatrixRef<'a, T: DeviceRepr> {
    /// Borrowed initialized device storage.
    pub data: DeviceView<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
    /// Element distance between the starts of consecutive matrices.
    pub stride: usize,
}

/// A mutable reference to a strided batched matrix stored on the device.
#[derive(Debug)]
pub struct StridedBatchedMatrixMut<'a, T: DeviceRepr> {
    /// Exclusively borrowed initialized device storage.
    pub data: DeviceViewMut<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
    /// Element distance between the starts of consecutive matrices.
    pub stride: usize,
}

/// Write-only batched matrix storage; stride gaps and row padding are not results.
#[derive(Debug)]
pub struct StridedBatchedMatrixOutput<'a, T: DeviceRepr> {
    /// Exclusive output capacity; this view does not establish initialized elements.
    pub data: DeviceOutputView<'a, T>,
    /// Element distance between successive column starts.
    pub leading_dimension: usize,
    /// Element distance between the starts of consecutive matrices.
    pub stride: usize,
}

impl<'a, T: DeviceRepr> StridedBatchedMatrixOutput<'a, T> {
    /// Borrows initialized or uninitialized output capacity and records column-major matrix batch metadata.
    ///
    /// This constructor does not validate shape or capacity, leading dimension, or stride.
    pub fn new(
        data: &'a mut (impl DeviceOutput<T> + ?Sized),
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
            leading_dimension,
            stride,
        }
    }

    /// Returns the recorded column-major batch layout in element units.
    #[must_use]
    pub const fn layout(&self) -> StridedMatrixLayout {
        StridedMatrixLayout::new(self.leading_dimension, self.stride)
    }
}

/// A reference to a vector stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct VectorRef<'a, T: DeviceRepr> {
    /// Borrowed initialized device storage.
    pub data: DeviceView<'a, T>,
}

/// A mutable reference to a vector stored on the device.
#[derive(Debug)]
pub struct VectorMut<'a, T: DeviceRepr> {
    /// Exclusively borrowed initialized device storage.
    pub data: DeviceViewMut<'a, T>,
}

/// A reference to a strided batched vector stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct StridedBatchedVectorRef<'a, T: DeviceRepr> {
    /// Borrowed initialized device storage.
    pub data: DeviceView<'a, T>,
    /// Element distance between the starts of consecutive vectors.
    pub stride: usize,
}

/// A mutable reference to a strided batched vector stored on the device.
#[derive(Debug)]
pub struct StridedBatchedVectorMut<'a, T: DeviceRepr> {
    /// Exclusively borrowed initialized device storage.
    pub data: DeviceViewMut<'a, T>,
    /// Element distance between the starts of consecutive vectors.
    pub stride: usize,
}

/// Write-only batched vectors with an element stride between produced prefixes.
#[derive(Debug)]
pub struct StridedBatchedVectorOutput<'a, T: DeviceRepr> {
    /// Exclusive output capacity; this view does not establish initialized elements.
    pub data: DeviceOutputView<'a, T>,
    /// Element distance between the starts of consecutive vectors.
    pub stride: usize,
}

impl<'a, T: DeviceRepr> StridedBatchedVectorOutput<'a, T> {
    /// Borrows initialized or uninitialized output capacity and records vector batch metadata.
    ///
    /// This constructor does not validate shape or capacity or stride.
    pub fn new(data: &'a mut (impl DeviceOutput<T> + ?Sized), stride: usize) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
            stride,
        }
    }
}

/// A reference to a batched matrix (pointer array) stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct BatchedMatrixRef<'a, T: DeviceRepr> {
    /// Device-resident array of matrix addresses.
    ///
    /// The view retains the pointer table borrow, not the allocations referenced by its entries.
    pub pointers: DeviceView<'a, DevicePtr>,
    /// Leading dimension in elements passed to the native batched operation.
    pub leading_dimension: usize,
    _phantom: PhantomData<T>,
}

/// A reference to a batched vector (pointer array) stored on the device.
#[derive(Debug, Clone, Copy)]
pub struct BatchedVectorRef<'a, T: DeviceRepr> {
    /// Device-resident array of vector addresses.
    ///
    /// The view retains the pointer table borrow, not the allocations referenced by its entries.
    pub pointers: DeviceView<'a, DevicePtr>,
    /// Leading dimension in elements passed to the native batched operation.
    pub leading_dimension: usize,
    _phantom: PhantomData<T>,
}

/// A mutable reference to a byte workspace buffer on the device.
#[derive(Debug)]
pub struct ByteWorkspaceMut<'a> {
    /// Exclusively borrowed device scratch capacity in bytes; scratch contents are not initialized results.
    pub device: DeviceOutputView<'a, u8>,
    /// Exclusively borrowed host scratch storage in bytes; native scratch writes do not establish numerical outputs.
    pub host: &'a mut [MaybeUninit<u8>],
}

/// Required device and host scratch capacities, both measured in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceSizes {
    /// Required device scratch capacity in bytes.
    pub device_bytes: usize,
    /// Required host scratch capacity in bytes.
    pub host_bytes: usize,
}

impl<'a, T: DeviceRepr> MatrixRef<'a, T> {
    /// Returns the recorded column-major layout in element units.
    #[must_use]
    pub const fn layout(&self) -> MatrixLayout {
        MatrixLayout::new(self.leading_dimension)
    }

    /// Borrows initialized storage and records column-major matrix metadata.
    ///
    /// This constructor does not validate shape or capacity or leading dimension.
    pub fn new(data: &'a (impl DeviceSlice<T> + ?Sized), leading_dimension: usize) -> Self {
        Self {
            data: DeviceView::from_buffer(data),
            leading_dimension,
        }
    }
}

impl<'a, T: DeviceRepr> MatrixMut<'a, T> {
    /// Returns the recorded column-major layout in element units.
    #[must_use]
    pub const fn layout(&self) -> MatrixLayout {
        MatrixLayout::new(self.leading_dimension)
    }

    /// Borrows initialized storage and records column-major matrix metadata.
    ///
    /// This constructor does not validate shape or capacity or leading dimension.
    pub fn new(data: &'a mut (impl DeviceSliceMut<T> + ?Sized), leading_dimension: usize) -> Self {
        Self {
            data: DeviceViewMut::from_buffer(data),
            leading_dimension,
        }
    }

    /// Reborrows the initialized storage and layout for shared access.
    #[must_use]
    pub fn as_ref(&self) -> MatrixRef<'_, T> {
        MatrixRef {
            data: self.data.as_view(),
            leading_dimension: self.leading_dimension,
        }
    }
}

impl<'a, T: DeviceRepr> StridedBatchedMatrixRef<'a, T> {
    /// Returns the recorded column-major batch layout in element units.
    #[must_use]
    pub const fn layout(&self) -> StridedMatrixLayout {
        StridedMatrixLayout::new(self.leading_dimension, self.stride)
    }

    /// Borrows initialized storage and records column-major matrix batch metadata.
    ///
    /// This constructor does not validate shape or capacity, leading dimension, or stride.
    pub fn new(
        data: &'a (impl DeviceSlice<T> + ?Sized),
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data: DeviceView::from_buffer(data),
            leading_dimension,
            stride,
        }
    }
}

impl<'a, T: DeviceRepr> StridedBatchedMatrixMut<'a, T> {
    /// Borrows initialized storage and records column-major matrix batch metadata.
    ///
    /// This constructor does not validate shape or capacity, leading dimension, or stride.
    pub fn new(
        data: &'a mut (impl DeviceSliceMut<T> + ?Sized),
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data: DeviceViewMut::from_buffer(data),
            leading_dimension,
            stride,
        }
    }

    /// Reborrows the initialized storage and layout for shared access.
    #[must_use]
    pub fn as_ref(&self) -> StridedBatchedMatrixRef<'_, T> {
        StridedBatchedMatrixRef {
            data: self.data.as_view(),
            leading_dimension: self.leading_dimension,
            stride: self.stride,
        }
    }
}

impl<'a, T: DeviceRepr> VectorRef<'a, T> {
    /// Borrows initialized storage and records vector metadata.
    ///
    /// This constructor does not validate shape or capacity.
    pub fn new(data: &'a (impl DeviceSlice<T> + ?Sized)) -> Self {
        Self {
            data: DeviceView::from_buffer(data),
        }
    }
}

impl<'a, T: DeviceRepr> VectorMut<'a, T> {
    /// Borrows initialized storage and records vector metadata.
    ///
    /// This constructor does not validate shape or capacity.
    pub fn new(data: &'a mut (impl DeviceSliceMut<T> + ?Sized)) -> Self {
        Self {
            data: DeviceViewMut::from_buffer(data),
        }
    }

    /// Reborrows the initialized storage and layout for shared access.
    #[must_use]
    pub fn as_ref(&self) -> VectorRef<'_, T> {
        VectorRef {
            data: self.data.as_view(),
        }
    }
}

impl<'a, T: DeviceRepr> StridedBatchedVectorRef<'a, T> {
    /// Borrows initialized storage and records vector batch metadata.
    ///
    /// This constructor does not validate shape or capacity or stride.
    pub fn new(data: &'a (impl DeviceSlice<T> + ?Sized), stride: usize) -> Self {
        Self {
            data: DeviceView::from_buffer(data),
            stride,
        }
    }
}

impl<'a, T: DeviceRepr> StridedBatchedVectorMut<'a, T> {
    /// Borrows initialized storage and records vector batch metadata.
    ///
    /// This constructor does not validate shape or capacity or stride.
    pub fn new(data: &'a mut (impl DeviceSliceMut<T> + ?Sized), stride: usize) -> Self {
        Self {
            data: DeviceViewMut::from_buffer(data),
            stride,
        }
    }

    /// Reborrows the initialized storage and layout for shared access.
    #[must_use]
    pub fn as_ref(&self) -> StridedBatchedVectorRef<'_, T> {
        StridedBatchedVectorRef {
            data: self.data.as_view(),
            stride: self.stride,
        }
    }
}

impl<'a, T: DeviceRepr> BatchedMatrixRef<'a, T> {
    /// Borrows a device pointer table and records its element leading dimension.
    ///
    /// Does not validate or retain pointees. Callers of unsafe batched operations must establish pointee capacity, context, initialization, and access lifetimes.
    pub fn new(
        pointers: &'a (impl DeviceSlice<DevicePtr> + ?Sized),
        leading_dimension: usize,
    ) -> Self {
        Self {
            pointers: DeviceView::from_buffer(pointers),
            leading_dimension,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of device pointers in the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pointers.len()
    }

    /// Returns whether the device pointer table has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// Returns the native pointer-table address cast for a mutable matrix argument.
    ///
    /// The cast grants no ownership or write permission to the table or its pointees; native use must satisfy the operation safety contract.
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut *mut T {
        self.pointers.as_ptr().cast::<*mut T>().cast_mut()
    }
}

impl<'a, T: DeviceRepr> BatchedVectorRef<'a, T> {
    /// Borrows a device pointer table and records its element leading dimension.
    ///
    /// Does not validate or retain pointees. Callers of unsafe batched operations must establish pointee capacity, context, initialization, and access lifetimes.
    pub fn new(
        pointers: &'a (impl DeviceSlice<DevicePtr> + ?Sized),
        leading_dimension: usize,
    ) -> Self {
        Self {
            pointers: DeviceView::from_buffer(pointers),
            leading_dimension,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of device pointers in the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pointers.len()
    }

    /// Returns whether the device pointer table has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// Returns the native pointer-table address cast for a mutable vector argument.
    ///
    /// The cast grants no ownership or write permission to the table or its pointees; native use must satisfy the operation safety contract.
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut *mut T {
        self.pointers.as_ptr().cast::<*mut T>().cast_mut()
    }
}

impl<'a> ByteWorkspaceMut<'a> {
    /// Borrows device and host scratch storage without validating an operation workspace requirement.
    pub fn new(
        device: &'a mut (impl DeviceOutput<u8> + ?Sized),
        host: &'a mut [MaybeUninit<u8>],
    ) -> Self {
        Self {
            device: DeviceOutputView::from_output(device),
            host,
        }
    }
}

impl WorkspaceSizes {
    /// Records device and host scratch capacities in bytes.
    #[must_use]
    pub const fn new(device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device_bytes,
            host_bytes,
        }
    }
}

/// Write-only vector storage. Only a producer's documented elements become initialized.
#[derive(Debug)]
pub struct VectorOutput<'a, T: DeviceRepr> {
    /// Exclusive output capacity; this view does not establish initialized elements.
    pub data: DeviceOutputView<'a, T>,
}

impl<'a, T: DeviceRepr> VectorOutput<'a, T> {
    /// Borrows initialized or uninitialized output capacity and records vector metadata.
    ///
    /// This constructor does not validate shape or capacity.
    pub fn new(data: &'a mut (impl DeviceOutput<T> + ?Sized)) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
        }
    }
}
