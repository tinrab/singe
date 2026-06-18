use std::{ffi::c_void, marker::PhantomData, mem::ManuallyDrop, ptr};

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudss_sys as sys;

use crate::{
    error::{Error, Result},
    memory::DevicePointerTable,
    try_ffi,
    types::{DataType, DataTypeLike, IndexBase, Layout, MatrixFormat, MatrixType, MatrixViewType},
    utility::{
        checked_element_count, to_i64, to_usize, validate_dense_shape, validate_nonnegative,
    },
};

/// Lightweight cuDSS matrix descriptor.
///
/// cuDSS matrix objects wrap user-provided buffers; destroying the descriptor does not free or modify those buffers.
/// The lifetime parameter tracks backing memory borrowed through the safe constructors.
#[derive(Debug)]
pub struct Matrix<'a> {
    handle: sys::cudssMatrix_t,
    _marker: PhantomData<&'a ()>,
}

/// Shape and layout for a dense matrix descriptor.
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrixDescriptor {
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub columns: i64,
    /// Leading dimension.
    pub leading_dim: i64,
    /// Dense memory layout.
    pub layout: Layout,
}

/// Shape and structural metadata for a CSR matrix descriptor.
#[derive(Debug, Clone, Copy)]
pub struct CsrMatrixDescriptor {
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub columns: i64,
    /// Number of nonzero entries.
    pub nnz: i64,
    /// Matrix structural type.
    pub matrix_type: MatrixType,
    /// Matrix view used for symmetric or Hermitian storage.
    pub view_type: MatrixViewType,
    /// Sparse index base.
    pub index_base: IndexBase,
}

/// Descriptor for a non-uniform batch of dense matrices.
#[derive(Debug, Clone, Copy)]
pub struct BatchDenseMatrixDescriptor<'a, I: BatchIndex = i64> {
    /// Row count for each matrix in the batch.
    pub rows: &'a [I],
    /// Column count for each matrix in the batch.
    pub columns: &'a [I],
    /// Leading dimension for each matrix in the batch.
    pub leading_dim: &'a [I],
    /// Dense memory layout for the batch.
    pub layout: Layout,
}

/// Descriptor for a non-uniform batch of CSR matrices.
#[derive(Debug, Clone, Copy)]
pub struct BatchCsrMatrixDescriptor<'a, I: BatchIndex = i64> {
    /// Row count for each matrix in the batch.
    pub rows: &'a [I],
    /// Column count for each matrix in the batch.
    pub columns: &'a [I],
    /// Nonzero count for each matrix in the batch.
    pub nnz: &'a [I],
    /// Matrix structural type for every matrix in the batch.
    pub matrix_type: MatrixType,
    /// Matrix view for every matrix in the batch.
    pub view_type: MatrixViewType,
    /// Sparse index base.
    pub index_base: IndexBase,
}

/// Integer type accepted by cuDSS for batch shape arrays.
pub trait BatchIndex: private::Sealed + Copy {
    /// cuDSS data type corresponding to this integer type.
    const DATA_TYPE: DataType;

    /// Converts this value to the common signed shape representation.
    fn to_i64(self) -> i64;
}

impl BatchIndex for i32 {
    const DATA_TYPE: DataType = DataType::I32;

    fn to_i64(self) -> i64 {
        i64::from(self)
    }
}

impl BatchIndex for i64 {
    const DATA_TYPE: DataType = DataType::I64;

    fn to_i64(self) -> i64 {
        self
    }
}

/// Integer type accepted by cuDSS for CSR offsets and indices.
pub trait MatrixIndex: private::Sealed + Copy + 'static {
    /// cuDSS data type corresponding to this integer type.
    const DATA_TYPE: DataType;
}

mod private {
    pub trait Sealed {}

    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

impl MatrixIndex for i32 {
    const DATA_TYPE: DataType = DataType::I32;
}

impl MatrixIndex for i64 {
    const DATA_TYPE: DataType = DataType::I64;
}

/// Dense matrix descriptor data returned by cuDSS.
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrixInfo {
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub columns: i64,
    /// Leading dimension.
    pub leading_dim: i64,
    /// Values pointer stored by the descriptor.
    pub values: DevicePtr,
    /// Value data type.
    pub value_type: DataType,
    /// Dense memory layout.
    pub layout: Layout,
}

/// CSR matrix descriptor data returned by cuDSS.
#[derive(Debug, Clone, Copy)]
pub struct CsrMatrixInfo {
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub columns: i64,
    /// Number of nonzero entries.
    pub nnz: i64,
    /// Row start offsets pointer.
    pub row_start: DevicePtr,
    /// Row end offsets pointer.
    pub row_end: DevicePtr,
    /// Column indices pointer.
    pub column_indices: DevicePtr,
    /// Values pointer.
    pub values: DevicePtr,
    /// Row offset data type.
    pub offset_type: DataType,
    /// Column index data type.
    pub index_type: DataType,
    /// Value data type.
    pub value_type: DataType,
    /// Matrix structural type.
    pub matrix_type: MatrixType,
    /// Matrix view.
    pub view_type: MatrixViewType,
    /// Sparse index base.
    pub index_base: IndexBase,
}

/// Batch dense matrix descriptor data returned by cuDSS.
#[derive(Debug, Clone)]
pub struct BatchDenseMatrixInfo {
    /// Row count for each matrix.
    pub rows: Vec<i64>,
    /// Column count for each matrix.
    pub columns: Vec<i64>,
    /// Leading dimension for each matrix.
    pub leading_dim: Vec<i64>,
    /// Pointer table stored by the descriptor.
    pub values: DevicePtr,
    /// Integer type used for shape arrays.
    pub index_type: DataType,
    /// Value data type.
    pub value_type: DataType,
    /// Dense memory layout.
    pub layout: Layout,
}

/// Batch CSR matrix descriptor data returned by cuDSS.
#[derive(Debug, Clone)]
pub struct BatchCsrMatrixInfo {
    /// Row count for each matrix.
    pub rows: Vec<i64>,
    /// Column count for each matrix.
    pub columns: Vec<i64>,
    /// Nonzero count for each matrix.
    pub nnz: Vec<i64>,
    /// Row start pointer table.
    pub row_start: DevicePtr,
    /// Row end pointer table.
    pub row_end: DevicePtr,
    /// Column indices pointer table.
    pub column_indices: DevicePtr,
    /// Values pointer table.
    pub values: DevicePtr,
    /// Row offset data type.
    pub offset_type: DataType,
    /// Column index data type.
    pub index_type: DataType,
    /// Value data type.
    pub value_type: DataType,
    /// Matrix structural type.
    pub matrix_type: MatrixType,
    /// Matrix view.
    pub view_type: MatrixViewType,
    /// Sparse index base.
    pub index_base: IndexBase,
}

// Matrix descriptors borrow caller-owned device buffers and own only the cuDSS descriptor handle.
// Moving the descriptor is sound as long as the lifetime keeps the referenced buffers alive.
unsafe impl Send for Matrix<'_> {}

impl<'a> Matrix<'a> {
    /// Creates a dense matrix descriptor that borrows typed device memory.
    ///
    /// cuDSS stores the values pointer without taking ownership of the device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the matrix descriptor.
    pub fn create_dense<T: DataTypeLike>(
        desc: DenseMatrixDescriptor,
        values: &'a DeviceMemory<T>,
    ) -> Result<Self> {
        validate_dense_descriptor(desc, Some(values.len()))?;
        unsafe { Self::create_dense_raw(desc, device_ptr(values), T::DATA_TYPE) }
    }

    /// Creates a dense matrix descriptor from an untyped pointer.
    ///
    /// cuDSS stores the values pointer without taking ownership of the backing
    /// allocation.
    ///
    /// # Safety
    ///
    /// `values` must be null or point to a device-visible dense matrix buffer
    /// with element type `value_type` and layout described by `desc`. The buffer
    /// must outlive this descriptor and every cuDSS operation that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if `desc` is invalid, if cuDSS cannot create the matrix
    /// descriptor, or if cuDSS returns a null handle.
    pub unsafe fn create_dense_raw(
        desc: DenseMatrixDescriptor,
        values: DevicePtr,
        value_type: DataType,
    ) -> Result<Self> {
        validate_dense_descriptor(desc, None)?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssMatrixCreateDn(
                &raw mut handle,
                desc.rows,
                desc.columns,
                desc.leading_dim,
                values.as_const_ptr().cast(),
                value_type.into(),
                desc.layout.into(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle,
                _marker: PhantomData,
            })
        }
    }

    /// Creates a CSR matrix descriptor that borrows typed device memory.
    ///
    /// cuDSS does not validate CSR consistency at creation time.
    /// The row offsets, column indices, values, index base, matrix type, and view must describe the same matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the matrix descriptor.
    pub fn create_csr<O, I, T>(
        desc: CsrMatrixDescriptor,
        row_start: &'a DeviceMemory<O>,
        row_end: Option<&'a DeviceMemory<O>>,
        column_indices: &'a DeviceMemory<I>,
        values: &'a DeviceMemory<T>,
    ) -> Result<Self>
    where
        O: MatrixIndex,
        I: MatrixIndex,
        T: DataTypeLike,
    {
        validate_csr_descriptor(
            desc,
            Some(row_start.len()),
            row_end.map(DeviceMemory::len),
            Some(column_indices.len()),
            Some(values.len()),
        )?;
        unsafe {
            Self::create_csr_raw(
                desc,
                device_ptr(row_start),
                row_end.map_or(DevicePtr::null(), device_ptr),
                device_ptr(column_indices),
                device_ptr(values),
                O::DATA_TYPE,
                I::DATA_TYPE,
                T::DATA_TYPE,
            )
        }
    }

    /// Creates a CSR matrix descriptor from untyped pointers.
    ///
    /// cuDSS does not validate CSR consistency at creation time.
    ///
    /// # Safety
    ///
    /// `row_start`, `row_end`, `column_indices`, and `values` must be null only
    /// where cuDSS permits null pointers, otherwise they must point to CSR
    /// arrays with types `offset_type`, `index_type`, and `value_type`. The
    /// arrays, [`IndexBase`], [`MatrixType`], and [`MatrixViewType`] in `desc`
    /// must describe the same matrix. The buffers must outlive this descriptor
    /// and every cuDSS operation that uses it.
    ///
    /// Host pointers are only valid for cuDSS modes that explicitly support
    /// them.
    ///
    /// # Errors
    ///
    /// Returns an error if `desc` is invalid, if cuDSS cannot create the matrix
    /// descriptor, or if cuDSS returns a null handle.
    pub unsafe fn create_csr_raw(
        desc: CsrMatrixDescriptor,
        row_start: DevicePtr,
        row_end: DevicePtr,
        column_indices: DevicePtr,
        values: DevicePtr,
        offset_type: DataType,
        index_type: DataType,
        value_type: DataType,
    ) -> Result<Self> {
        validate_csr_descriptor(desc, None, None, None, None)?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssMatrixCreateCsr(
                &raw mut handle,
                desc.rows,
                desc.columns,
                desc.nnz,
                row_start.as_const_ptr().cast(),
                row_end.as_const_ptr().cast(),
                column_indices.as_const_ptr().cast(),
                values.as_const_ptr().cast(),
                offset_type.into(),
                index_type.into(),
                value_type.into(),
                desc.matrix_type.into(),
                desc.view_type.into(),
                desc.index_base.into(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle,
                _marker: PhantomData,
            })
        }
    }

    /// Creates a non-uniform batch dense matrix descriptor.
    ///
    /// The batch value pointer must reference a device-visible pointer table whose entries point to device-visible dense value buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape arrays have mismatched lengths or cuDSS cannot create the descriptor.
    pub fn create_batch_dense<I, T>(
        desc: BatchDenseMatrixDescriptor<'a, I>,
        values: DevicePointerTable<'a, T>,
    ) -> Result<Self>
    where
        I: BatchIndex,
        T: DataTypeLike,
    {
        unsafe { Self::create_batch_dense_raw(desc, values.as_ptr(), T::DATA_TYPE) }
    }

    /// Creates a non-uniform batch dense descriptor from an untyped pointer table.
    ///
    /// # Safety
    ///
    /// `values` must point to a device-visible pointer table whose entries
    /// point to device-visible dense matrix buffers with element type
    /// `value_type`. The table and every referenced buffer must outlive this
    /// descriptor and every cuDSS operation that uses it.
    ///
    /// Matrix batches are not supported in MGMN mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor shape arrays are invalid, if cuDSS
    /// cannot create the matrix descriptor, or if cuDSS returns a null handle.
    pub unsafe fn create_batch_dense_raw<I: BatchIndex>(
        desc: BatchDenseMatrixDescriptor<'a, I>,
        values: DevicePtr,
        value_type: DataType,
    ) -> Result<Self> {
        validate_batch_dense_descriptor(desc)?;
        let batch_count = to_i64(desc.rows.len(), "batch_count")?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssMatrixCreateBatchDn(
                &raw mut handle,
                batch_count,
                desc.rows.as_ptr().cast(),
                desc.columns.as_ptr().cast(),
                desc.leading_dim.as_ptr().cast(),
                values.as_const_ptr().cast(),
                I::DATA_TYPE.into(),
                value_type.into(),
                desc.layout.into(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle,
                _marker: PhantomData,
            })
        }
    }

    /// Creates a non-uniform batch CSR matrix descriptor.
    ///
    /// Pointer fields must reference device-visible pointer tables whose entries point to device-visible CSR arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape arrays have mismatched lengths or cuDSS cannot create the descriptor.
    pub fn create_batch_csr<I, O, J, T>(
        desc: BatchCsrMatrixDescriptor<'a, I>,
        row_start: DevicePointerTable<'a, O>,
        row_end: Option<DevicePointerTable<'a, O>>,
        col_indices: DevicePointerTable<'a, J>,
        values: DevicePointerTable<'a, T>,
    ) -> Result<Self>
    where
        I: BatchIndex,
        O: MatrixIndex,
        J: MatrixIndex,
        T: DataTypeLike,
    {
        unsafe {
            Self::create_batch_csr_raw(
                desc,
                row_start.as_ptr(),
                row_end.map_or(DevicePtr::null(), DevicePointerTable::as_ptr),
                col_indices.as_ptr(),
                values.as_ptr(),
                O::DATA_TYPE,
                J::DATA_TYPE,
                T::DATA_TYPE,
            )
        }
    }

    /// Creates a non-uniform batch CSR descriptor from untyped pointer tables.
    ///
    /// cuDSS supports non-uniform batches with varying row, column, and nonzero
    /// counts. It does not validate CSR consistency at creation time.
    ///
    /// # Safety
    ///
    /// `row_start`, `row_end`, `col_indices`, and `values` must point to
    /// device-visible pointer tables. Each table entry used by cuDSS must point
    /// to a live CSR array with the corresponding data type. The tables and all
    /// referenced arrays must outlive this descriptor and every cuDSS operation
    /// that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor shape arrays are invalid, if cuDSS
    /// cannot create the matrix descriptor, or if cuDSS returns a null handle.
    pub unsafe fn create_batch_csr_raw<I: BatchIndex>(
        desc: BatchCsrMatrixDescriptor<'a, I>,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
        offset_type: DataType,
        index_type: DataType,
        value_type: DataType,
    ) -> Result<Self> {
        validate_batch_csr_descriptor(desc)?;
        let batch_count = to_i64(desc.rows.len(), "batch_count")?;
        unsafe {
            let mut handle = ptr::null_mut();
            try_ffi!(sys::cudssMatrixCreateBatchCsr(
                &raw mut handle,
                batch_count,
                desc.rows.as_ptr().cast(),
                desc.columns.as_ptr().cast(),
                desc.nnz.as_ptr().cast(),
                row_start.as_const_ptr().cast(),
                row_end.as_const_ptr().cast(),
                col_indices.as_const_ptr().cast(),
                values.as_const_ptr().cast(),
                offset_type.into(),
                index_type.into(),
                value_type.into(),
                desc.matrix_type.into(),
                desc.view_type.into(),
                desc.index_base.into(),
            ))?;
            if handle.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                handle,
                _marker: PhantomData,
            })
        }
    }

    /// Returns dense matrix descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata.
    pub fn dense_info(&self) -> Result<DenseMatrixInfo> {
        unsafe {
            let mut rows = 0;
            let mut columns = 0;
            let mut leading_dim = 0;
            let mut values = ptr::null_mut();
            let mut value_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut layout = sys::cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR;
            try_ffi!(sys::cudssMatrixGetDn(
                self.handle,
                &raw mut rows,
                &raw mut columns,
                &raw mut leading_dim,
                &raw mut values,
                &raw mut value_type,
                &raw mut layout,
            ))?;
            Ok(DenseMatrixInfo {
                rows,
                columns,
                leading_dim,
                values: DevicePtr::from(values),
                value_type: value_type.into(),
                layout: layout.into(),
            })
        }
    }

    /// Returns CSR matrix descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata.
    pub fn csr_info(&self) -> Result<CsrMatrixInfo> {
        unsafe {
            let mut rows = 0;
            let mut columns = 0;
            let mut nnz = 0;
            let mut row_start = ptr::null_mut();
            let mut row_end = ptr::null_mut();
            let mut col_indices = ptr::null_mut();
            let mut values = ptr::null_mut();
            let mut offset_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut index_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut value_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut matrix_type = sys::cudssMatrixType_t::CUDSS_MTYPE_GENERAL;
            let mut view_type = sys::cudssMatrixViewType_t::CUDSS_MVIEW_FULL;
            let mut index_base = sys::cudssIndexBase_t::CUDSS_BASE_ZERO;
            try_ffi!(sys::cudssMatrixGetCsr(
                self.handle,
                &raw mut rows,
                &raw mut columns,
                &raw mut nnz,
                &raw mut row_start,
                &raw mut row_end,
                &raw mut col_indices,
                &raw mut values,
                &raw mut offset_type,
                &raw mut index_type,
                &raw mut value_type,
                &raw mut matrix_type,
                &raw mut view_type,
                &raw mut index_base,
            ))?;
            Ok(CsrMatrixInfo {
                rows,
                columns,
                nnz,
                row_start: DevicePtr::from(row_start),
                row_end: DevicePtr::from(row_end),
                column_indices: DevicePtr::from(col_indices),
                values: DevicePtr::from(values),
                offset_type: offset_type.into(),
                index_type: index_type.into(),
                value_type: value_type.into(),
                matrix_type: matrix_type.into(),
                view_type: view_type.into(),
                index_base: index_base.into(),
            })
        }
    }

    /// Returns non-uniform batch dense descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata, if the
    /// reported batch count cannot be represented as `usize`, or if cuDSS
    /// reports an unsupported batch index type.
    pub fn batch_dense_info(&self) -> Result<BatchDenseMatrixInfo> {
        unsafe {
            let mut batch_count = 0;
            let mut rows = ptr::null_mut();
            let mut columns = ptr::null_mut();
            let mut leading_dim = ptr::null_mut();
            let mut values = ptr::null_mut();
            let mut index_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut value_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut layout = sys::cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR;
            try_ffi!(sys::cudssMatrixGetBatchDn(
                self.handle,
                &raw mut batch_count,
                &raw mut rows,
                &raw mut columns,
                &raw mut leading_dim,
                &raw mut values,
                &raw mut index_type,
                &raw mut value_type,
                &raw mut layout,
            ))?;
            let batch_count = to_usize(batch_count, "batch_count")?;
            let index_type = DataType::from(index_type);
            Ok(BatchDenseMatrixInfo {
                rows: copy_index_batch(rows, batch_count, index_type, "rows")?,
                columns: copy_index_batch(columns, batch_count, index_type, "columns")?,
                leading_dim: copy_index_batch(leading_dim, batch_count, index_type, "leading_dim")?,
                values: DevicePtr::from(values.cast::<c_void>()),
                index_type,
                value_type: value_type.into(),
                layout: layout.into(),
            })
        }
    }

    /// Returns non-uniform batch CSR descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata, if the
    /// reported batch count cannot be represented as `usize`, or if cuDSS
    /// reports an unsupported batch index type.
    pub fn batch_csr_info(&self) -> Result<BatchCsrMatrixInfo> {
        unsafe {
            let mut batch_count = 0;
            let mut rows = ptr::null_mut();
            let mut columns = ptr::null_mut();
            let mut nnz = ptr::null_mut();
            let mut row_start = ptr::null_mut();
            let mut row_end = ptr::null_mut();
            let mut col_indices = ptr::null_mut();
            let mut values = ptr::null_mut();
            let mut offset_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut index_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut value_type = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET;
            let mut matrix_type = sys::cudssMatrixType_t::CUDSS_MTYPE_GENERAL;
            let mut view_type = sys::cudssMatrixViewType_t::CUDSS_MVIEW_FULL;
            let mut index_base = sys::cudssIndexBase_t::CUDSS_BASE_ZERO;
            try_ffi!(sys::cudssMatrixGetBatchCsr(
                self.handle,
                &raw mut batch_count,
                &raw mut rows,
                &raw mut columns,
                &raw mut nnz,
                &raw mut row_start,
                &raw mut row_end,
                &raw mut col_indices,
                &raw mut values,
                &raw mut offset_type,
                &raw mut index_type,
                &raw mut value_type,
                &raw mut matrix_type,
                &raw mut view_type,
                &raw mut index_base,
            ))?;
            let batch_count = to_usize(batch_count, "batch_count")?;
            let index_type = DataType::from(index_type);
            Ok(BatchCsrMatrixInfo {
                rows: copy_index_batch(rows, batch_count, index_type, "rows")?,
                columns: copy_index_batch(columns, batch_count, index_type, "columns")?,
                nnz: copy_index_batch(nnz, batch_count, index_type, "nnz")?,
                row_start: DevicePtr::from(row_start.cast::<c_void>()),
                row_end: DevicePtr::from(row_end.cast::<c_void>()),
                column_indices: DevicePtr::from(col_indices.cast::<c_void>()),
                values: DevicePtr::from(values.cast::<c_void>()),
                offset_type: offset_type.into(),
                index_type,
                value_type: value_type.into(),
                matrix_type: matrix_type.into(),
                view_type: view_type.into(),
                index_base: index_base.into(),
            })
        }
    }

    /// Replaces the values pointer with typed device memory.
    ///
    /// The element type must match the matrix descriptor's value type.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer.
    pub fn set_values<T: DataTypeLike>(&mut self, values: &'a DeviceMemory<T>) -> Result<()> {
        unsafe { self.set_values_raw(device_ptr(values)) }
    }

    /// Replaces the values pointer with an untyped pointer.
    ///
    /// # Safety
    ///
    /// `values` must be null or point to a buffer whose type, layout, and
    /// length match this descriptor. The buffer must outlive this descriptor and
    /// every cuDSS operation that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer.
    pub unsafe fn set_values_raw(&mut self, values: DevicePtr) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssMatrixSetValues(
                self.handle,
                values.as_const_ptr().cast()
            ))
        }
    }

    /// Replaces all CSR pointers with typed device memory.
    ///
    /// The offset, index, and value element types must match the matrix descriptor's data types.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointers.
    pub fn set_csr_values<O, I, T>(
        &mut self,
        row_start: &'a DeviceMemory<O>,
        row_end: Option<&'a DeviceMemory<O>>,
        col_indices: &'a DeviceMemory<I>,
        values: &'a DeviceMemory<T>,
    ) -> Result<()>
    where
        O: MatrixIndex,
        I: MatrixIndex,
        T: DataTypeLike,
    {
        unsafe {
            self.set_csr_values_raw(
                device_ptr(row_start),
                row_end.map_or(DevicePtr::null(), device_ptr),
                device_ptr(col_indices),
                device_ptr(values),
            )
        }
    }

    /// Replaces all CSR pointers with untyped pointers.
    ///
    /// # Safety
    ///
    /// The pointers must be null only where cuDSS permits null pointers,
    /// otherwise they must point to buffers whose types, lengths, and CSR
    /// contents match this descriptor. The buffers must outlive this descriptor
    /// and every cuDSS operation that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointers.
    pub unsafe fn set_csr_values_raw(
        &mut self,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssMatrixSetCsrPointers(
                self.handle,
                row_start.as_const_ptr().cast(),
                row_end.as_const_ptr().cast(),
                col_indices.as_const_ptr().cast(),
                values.as_const_ptr().cast(),
            ))
        }
    }

    /// Replaces the batch values pointer table.
    ///
    /// `values` must point to a device-visible pointer table.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer table.
    pub fn set_batch_values<T>(&mut self, values: DevicePointerTable<'a, T>) -> Result<()> {
        unsafe { self.set_batch_values_raw(values.as_ptr()) }
    }

    /// Replaces the batch values pointer table with an untyped pointer.
    ///
    /// # Safety
    ///
    /// `values` must point to a device-visible pointer table whose entries
    /// match this batch descriptor. The table and referenced buffers must
    /// outlive this descriptor and every cuDSS operation that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer table.
    pub unsafe fn set_batch_values_raw(&mut self, values: DevicePtr) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssMatrixSetBatchValues(
                self.handle,
                values.as_const_ptr().cast()
            ))
        }
    }

    /// Replaces the batch CSR pointer tables.
    ///
    /// Each argument must point to a device-visible pointer table matching the descriptor's data types.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer tables.
    pub fn set_batch_csr_values<O, J, T>(
        &mut self,
        row_start: DevicePointerTable<'a, O>,
        row_end: Option<DevicePointerTable<'a, O>>,
        col_indices: DevicePointerTable<'a, J>,
        values: DevicePointerTable<'a, T>,
    ) -> Result<()>
    where
        O: MatrixIndex,
        J: MatrixIndex,
        T: DataTypeLike,
    {
        unsafe {
            self.set_batch_csr_values_raw(
                row_start.as_ptr(),
                row_end.map_or(DevicePtr::null(), DevicePointerTable::as_ptr),
                col_indices.as_ptr(),
                values.as_ptr(),
            )
        }
    }

    /// Replaces the batch CSR pointer tables with untyped pointers.
    ///
    /// # Safety
    ///
    /// Each pointer must point to a device-visible pointer table whose entries
    /// match this batch descriptor. The tables and referenced buffers must
    /// outlive this descriptor and every cuDSS operation that uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the new pointer tables.
    pub unsafe fn set_batch_csr_values_raw(
        &mut self,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssMatrixSetBatchCsrPointers(
                self.handle,
                row_start.as_const_ptr().cast(),
                row_end.as_const_ptr().cast(),
                col_indices.as_const_ptr().cast(),
                values.as_const_ptr().cast(),
            ))
        }
    }

    /// Returns the matrix format flags reported by cuDSS.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the format.
    pub fn format(&self) -> Result<MatrixFormat> {
        unsafe {
            let mut format = 0;
            try_ffi!(sys::cudssMatrixGetFormat(self.handle, &raw mut format))?;
            Ok(MatrixFormat::from_bits_retain(format as _))
        }
    }

    /// Sets the 1D row distribution for MGMN mode.
    ///
    /// `first_row` and `last_row` are zero-based inclusive global row indices
    /// for the local matrix on the calling process. Set `first_row > last_row`
    /// to mark an empty local matrix.
    ///
    /// The system matrix, right-hand side, and solution may use different
    /// distributions. If sparse matrix or right-hand side distributions overlap,
    /// overlapping contributions are summed. If solution distributions overlap,
    /// overlapping solution entries have the same values on the corresponding
    /// processes.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the row distribution.
    pub fn set_distribution_row_1d(&mut self, first_row: i64, last_row: i64) -> Result<()> {
        unsafe {
            try_ffi!(sys::cudssMatrixSetDistributionRow1d(
                self.handle,
                first_row,
                last_row,
            ))
        }
    }

    /// Returns the 1D row distribution boundaries.
    ///
    /// Outside MGMN mode, cuDSS reports `(0, rows - 1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the row distribution.
    pub fn distribution_row_1d(&self) -> Result<(i64, i64)> {
        unsafe {
            let mut first_row = 0;
            let mut last_row = 0;
            try_ffi!(sys::cudssMatrixGetDistributionRow1d(
                self.handle,
                &raw mut first_row,
                &raw mut last_row,
            ))?;
            Ok((first_row, last_row))
        }
    }

    /// Returns the raw cuDSS matrix descriptor handle.
    pub const fn as_raw(&self) -> sys::cudssMatrix_t {
        self.handle
    }

    /// Takes ownership of an existing cuDSS matrix descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuDSS matrix descriptor. The descriptor may
    /// store borrowed pointers, and the caller must ensure those buffers outlive
    /// the returned wrapper's lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(handle: sys::cudssMatrix_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            _marker: PhantomData,
        })
    }

    /// Consumes this wrapper and returns the owned raw cuDSS matrix descriptor.
    ///
    /// The caller becomes responsible for destroying the returned descriptor.
    pub fn into_raw(self) -> sys::cudssMatrix_t {
        let matrix = ManuallyDrop::new(self);
        matrix.handle
    }
}

impl Drop for Matrix<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cudssMatrixDestroy(self.handle);
        }
    }
}

impl DenseMatrixDescriptor {
    /// Creates a column-major dense matrix descriptor.
    pub fn new(rows: i64, columns: i64, leading_dim: i64) -> Self {
        Self {
            rows,
            columns,
            leading_dim,
            layout: Layout::ColumnMajor,
        }
    }

    /// Creates a row-major dense matrix descriptor.
    ///
    /// cuDSS documents row-major layout but does not currently support it for
    /// solve right-hand side and solution matrices.
    pub fn new_row_major(rows: i64, columns: i64, leading_dim: i64) -> Self {
        Self {
            rows,
            columns,
            leading_dim,
            layout: Layout::RowMajor,
        }
    }
}

fn validate_batch_len(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            name: name.into(),
            expected,
            actual,
        })
    }
}

fn validate_dense_descriptor(desc: DenseMatrixDescriptor, values_len: Option<usize>) -> Result<()> {
    validate_dense_shape(desc.rows, desc.columns, desc.leading_dim, desc.layout)?;
    if let Some(values_len) = values_len {
        let required = match desc.layout {
            Layout::ColumnMajor => {
                checked_element_count(&[desc.leading_dim, desc.columns], "dense values")?
            }
            Layout::RowMajor => {
                checked_element_count(&[desc.leading_dim, desc.rows], "dense values")?
            }
        };
        if values_len < required {
            return Err(Error::LengthMismatch {
                name: "values".into(),
                expected: required,
                actual: values_len,
            });
        }
    }
    Ok(())
}

fn validate_csr_descriptor(
    desc: CsrMatrixDescriptor,
    row_start_len: Option<usize>,
    row_end_len: Option<usize>,
    column_indices_len: Option<usize>,
    values_len: Option<usize>,
) -> Result<()> {
    validate_nonnegative(desc.rows, "rows")?;
    validate_nonnegative(desc.columns, "columns")?;
    validate_nonnegative(desc.nnz, "nnz")?;

    let rows = to_usize(desc.rows, "rows")?;
    let nnz = to_usize(desc.nnz, "nnz")?;

    if let Some(row_start_len) = row_start_len {
        let expected = if row_end_len.is_some() {
            rows
        } else {
            rows.checked_add(1).ok_or_else(|| Error::OutOfRange {
                name: "row_start length".into(),
            })?
        };
        validate_min_len("row_start", expected, row_start_len)?;
    }
    if let Some(row_end_len) = row_end_len {
        validate_min_len("row_end", rows, row_end_len)?;
    }
    if let Some(column_indices_len) = column_indices_len {
        validate_min_len("column_indices", nnz, column_indices_len)?;
    }
    if let Some(values_len) = values_len {
        validate_min_len("values", nnz, values_len)?;
    }

    Ok(())
}

fn validate_batch_dense_descriptor<I: BatchIndex>(
    desc: BatchDenseMatrixDescriptor<'_, I>,
) -> Result<()> {
    validate_batch_len("columns", desc.rows.len(), desc.columns.len())?;
    validate_batch_len("leading_dim", desc.rows.len(), desc.leading_dim.len())?;
    for ((row, column), leading_dim) in desc.rows.iter().zip(desc.columns).zip(desc.leading_dim) {
        validate_dense_shape(
            row.to_i64(),
            column.to_i64(),
            leading_dim.to_i64(),
            desc.layout,
        )?;
    }
    Ok(())
}

fn validate_batch_csr_descriptor<I: BatchIndex>(
    desc: BatchCsrMatrixDescriptor<'_, I>,
) -> Result<()> {
    validate_batch_len("columns", desc.rows.len(), desc.columns.len())?;
    validate_batch_len("nnz", desc.rows.len(), desc.nnz.len())?;
    for ((row, column), nnz) in desc.rows.iter().zip(desc.columns).zip(desc.nnz) {
        validate_nonnegative(row.to_i64(), "rows")?;
        validate_nonnegative(column.to_i64(), "columns")?;
        validate_nonnegative(nnz.to_i64(), "nnz")?;
    }
    Ok(())
}

fn validate_min_len(name: &str, expected: usize, actual: usize) -> Result<()> {
    if actual >= expected {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            name: name.into(),
            expected,
            actual,
        })
    }
}

fn device_ptr<T>(memory: &DeviceMemory<T>) -> DevicePtr {
    DevicePtr::from_raw(memory.as_ptr().cast_mut().cast())
}

unsafe fn copy_i64_batch(ptr: *mut c_void, len: usize) -> Vec<i64> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr.cast::<i64>(), len).to_vec() }
    }
}

unsafe fn copy_index_batch(
    ptr: *mut c_void,
    len: usize,
    index_type: DataType,
    name: &str,
) -> Result<Vec<i64>> {
    if ptr.is_null() || len == 0 {
        return Ok(Vec::new());
    }

    match index_type {
        DataType::I32 => Ok(unsafe {
            std::slice::from_raw_parts(ptr.cast::<i32>(), len)
                .iter()
                .copied()
                .map(i64::from)
                .collect()
        }),
        DataType::I64 => Ok(unsafe { copy_i64_batch(ptr, len) }),
        _ => Err(Error::UnsupportedDataType { name: name.into() }),
    }
}
