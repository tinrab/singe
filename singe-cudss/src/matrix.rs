use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr, slice,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    types::DevicePtr,
    view::{DeviceOutputView, DeviceRegion, DeviceRepr, DeviceView},
};
use singe_cudss_sys as sys;

use crate::{
    error::{Error, Result},
    memory::DevicePointerTable,
    try_ffi,
    types::{DataType, DataTypeLike, IndexBase, Layout, MatrixFormat, MatrixType, MatrixViewType},
    utility::{
        checked_element_count, copy_i64_batch, to_i64, to_usize, validate_dense_shape,
        validate_min_len, validate_nonnegative, validate_same_len,
    },
};

/// Lightweight cuDSS matrix descriptor.
///
/// cuDSS matrix objects wrap user-provided buffers; destroying the descriptor does not free or modify those buffers.
/// The lifetime parameter tracks backing memory borrowed through the safe constructors.
#[derive(Debug)]
pub struct Matrix<'a> {
    handle: sys::cudssMatrix_t,
    context: Arc<CudaContext>,
    // Device storage bound through typed views, whose producers `Context::execute_async` joins.
    values: Option<DeviceRegion<'a>>,
    structure: Vec<DeviceRegion<'a>>,
    _marker: PhantomData<&'a ()>,
}

/// A transferred cuDSS matrix descriptor and the CUDA context needed to destroy it.
///
/// Destroy `handle` before dropping this value, or reclaim it with [`Matrix::from_raw`].
#[derive(Debug)]
pub struct RawMatrix {
    /// Uniquely owned native matrix descriptor.
    pub handle: sys::cudssMatrix_t,
    /// CUDA context bound when destroying the descriptor.
    pub context: Arc<CudaContext>,
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
pub trait MatrixIndex: private::Sealed + DeviceRepr {
    /// cuDSS data type corresponding to this integer type.
    const DATA_TYPE: DataType;
}

mod private {
    #[allow(
        unnameable_types,
        reason = "sealed supertrait that callers must not name or implement"
    )]
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

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: a cuDSS matrix descriptor is a host object that cuDSS does not tie to a thread, so it may be used from any thread one call at a time.
// The buffers it references stay borrowed for `'a` wherever it moves; it is not `Sync`, because cuDSS does not document concurrent calls on one descriptor.
unsafe impl Send for Matrix<'_> {}

impl<'a> Matrix<'a> {
    /// Creates a dense producer destination from exclusive write-only storage.
    ///
    /// Descriptor construction does not initialize values.
    /// A completed solve produces logical matrix elements, not pitch padding or allocation tails.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or leading dimension, a required element count that cannot be represented, or insufficient storage.
    /// Context-binding and cuDSS descriptor-creation errors are propagated; a null descriptor returned after successful creation is also rejected.
    pub fn create_dense_output<T: DataTypeLike>(
        desc: DenseMatrixDescriptor,
        mut values: DeviceOutputView<'a, T>,
    ) -> Result<Self> {
        validate_dense_descriptor(desc, Some(values.len()))?;
        let context = Arc::clone(values.context());
        let pointer = values.as_mut_ptr();
        // SAFETY: the view establishes aligned writable storage of the checked size.
        let matrix = unsafe {
            Self::create_dense_native(
                &context,
                desc,
                DevicePtr::from_raw(pointer.cast()),
                T::DATA_TYPE,
            )
        }?;
        Ok(matrix.tracking(Some(values.into_region()), Vec::new()))
    }

    /// Rebinds values to exclusive write-only storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer belongs to another context, its element type does not match the descriptor, or it is too short for the matrix.
    /// Errors querying the existing format and shape, validating dimensions or size conversions, binding the context, or updating the native pointer are also propagated.
    pub fn set_output<T: DataTypeLike>(
        &mut self,
        mut values: DeviceOutputView<'a, T>,
    ) -> Result<()> {
        ensure_context(&self.context, values.context())?;
        self.validate_value_buffer::<T>(values.len())?;
        let pointer = values.as_mut_ptr();
        // SAFETY: the exclusive output view satisfies the checked matrix layout.
        unsafe { self.set_values_native(DevicePtr::from_raw(pointer.cast())) }?;
        self.values = Some(values.into_region());
        Ok(())
    }

    fn validate_value_buffer<T: DataTypeLike>(&self, length: usize) -> Result<()> {
        if self.format()?.contains(MatrixFormat::DENSE) {
            let info = self.dense_info()?;
            if info.value_type != T::DATA_TYPE {
                return Err(Error::UnsupportedDataType {
                    name: "dense replacement values".into(),
                });
            }
            validate_dense_descriptor(
                DenseMatrixDescriptor {
                    rows: info.rows,
                    columns: info.columns,
                    leading_dim: info.leading_dim,
                    layout: info.layout,
                },
                Some(length),
            )
        } else {
            let info = self.csr_info()?;
            if info.value_type != T::DATA_TYPE {
                return Err(Error::UnsupportedDataType {
                    name: "sparse replacement values".into(),
                });
            }
            validate_min_len(to_usize(info.nnz, "nnz")?, length, "values")
        }
    }

    /// CUDA context associated with the descriptor and its device buffers.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Creates a dense matrix descriptor that borrows typed device memory.
    ///
    /// cuDSS stores the values pointer without taking ownership of the device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot create the matrix descriptor.
    pub fn create_dense<T: DataTypeLike>(
        desc: DenseMatrixDescriptor,
        values: DeviceView<'a, T>,
    ) -> Result<Self> {
        validate_dense_descriptor(desc, Some(values.len()))?;
        // SAFETY: `validate_dense_descriptor` checked that `values` covers `desc`, and the view stays borrowed for `'a`, the matrix lifetime.
        let matrix = unsafe {
            Self::create_dense_native(values.context(), desc, device_ptr(values), T::DATA_TYPE)
        }?;
        Ok(matrix.tracking(Some(values.region()), Vec::new()))
    }

    // Records the device storage whose producers execution joins.
    fn tracking(
        mut self,
        values: Option<DeviceRegion<'a>>,
        structure: Vec<DeviceRegion<'a>>,
    ) -> Self {
        self.values = values;
        self.structure = structure;
        self
    }

    pub(crate) fn regions(&self) -> impl Iterator<Item = DeviceRegion<'a>> + '_ {
        self.values
            .into_iter()
            .chain(self.structure.iter().copied())
    }

    // The caller guarantees that the pointers have the declared types, cover `desc`, and outlive every use of the descriptor.
    unsafe fn create_dense_native(
        context: &Arc<CudaContext>,
        desc: DenseMatrixDescriptor,
        values: DevicePtr,
        value_type: DataType,
    ) -> Result<Self> {
        context.bind()?;
        validate_dense_descriptor(desc, None)?;
        // SAFETY: `handle` is a local output checked for null, `desc` passed `validate_dense_descriptor`, and the caller guarantees `values` points to a `value_type` buffer with `desc`'s layout that outlives the descriptor.
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
                context: Arc::clone(context),
                values: None,
                structure: Vec::new(),
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
        row_start: DeviceView<'a, O>,
        row_end: Option<DeviceView<'a, O>>,
        column_indices: DeviceView<'a, I>,
        values: DeviceView<'a, T>,
    ) -> Result<Self>
    where
        O: MatrixIndex,
        I: MatrixIndex,
        T: DataTypeLike,
    {
        ensure_context(values.context(), row_start.context())?;
        ensure_context(values.context(), column_indices.context())?;
        if let Some(row_end) = row_end {
            ensure_context(values.context(), row_end.context())?;
        }

        validate_csr_descriptor(
            desc,
            Some(row_start.len()),
            row_end.map(|values| values.len()),
            Some(column_indices.len()),
            Some(values.len()),
        )?;
        // SAFETY: `validate_csr_descriptor` checked every array length against `desc`, the views share one context, and they stay borrowed for `'a`.
        let matrix = unsafe {
            Self::create_csr_native(
                values.context(),
                desc,
                device_ptr(row_start),
                row_end.map_or(DevicePtr::null(), device_ptr),
                device_ptr(column_indices),
                device_ptr(values),
                O::DATA_TYPE,
                I::DATA_TYPE,
                T::DATA_TYPE,
            )
        }?;
        let structure = [
            Some(row_start.region()),
            row_end.map(|row_end| row_end.region()),
        ]
        .into_iter()
        .flatten()
        .chain([column_indices.region()])
        .collect();
        Ok(matrix.tracking(Some(values.region()), structure))
    }

    /// Creates a dense matrix descriptor over host values, for cuDSS's hybrid execute mode.
    ///
    /// cuDSS stores the values pointer without taking ownership of the host buffer.
    ///
    /// # Safety
    ///
    /// Every solver call that uses this descriptor must run with hybrid execute mode enabled through [`Config::set_hybrid_execute_mode`](crate::config::Config::set_hybrid_execute_mode), since other modes read the values on the device.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or leading dimension, insufficient storage, a failed context bind, or a failed cuDSS descriptor creation that also covers a null handle.
    pub unsafe fn create_dense_host<T: DataTypeLike>(
        context: &Arc<CudaContext>,
        desc: DenseMatrixDescriptor,
        values: &'a [T],
    ) -> Result<Self> {
        validate_dense_descriptor(desc, Some(values.len()))?;
        // SAFETY: `validate_dense_descriptor` checked that `values` covers `desc`, the slice stays borrowed for `'a`, and the caller guarantees hybrid execute mode reads it on the host.
        unsafe { Self::create_dense_native(context, desc, host_ptr(values.as_ptr()), T::DATA_TYPE) }
    }

    /// Creates a dense producer destination over write-only host storage, for cuDSS's hybrid execute mode.
    ///
    /// Descriptor construction does not initialize values.
    ///
    /// # Safety
    ///
    /// Every solver call that uses this descriptor must run with hybrid execute mode enabled through [`Config::set_hybrid_execute_mode`](crate::config::Config::set_hybrid_execute_mode), since other modes write the values on the device.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or leading dimension, insufficient storage, a failed context bind, or a failed cuDSS descriptor creation that also covers a null handle.
    pub unsafe fn create_dense_host_output<T: DataTypeLike>(
        context: &Arc<CudaContext>,
        desc: DenseMatrixDescriptor,
        values: &'a mut [MaybeUninit<T>],
    ) -> Result<Self> {
        validate_dense_descriptor(desc, Some(values.len()))?;
        // SAFETY: `validate_dense_descriptor` checked that `values` covers `desc`, the exclusive slice stays borrowed for `'a`, and the caller guarantees hybrid execute mode writes it on the host.
        unsafe {
            Self::create_dense_native(
                context,
                desc,
                host_ptr(values.as_mut_ptr().cast::<T>()),
                T::DATA_TYPE,
            )
        }
    }

    /// Creates a CSR matrix descriptor over host arrays, for cuDSS's hybrid execute mode.
    ///
    /// cuDSS does not validate CSR consistency at creation time.
    /// The row offsets, column indices, values, index base, matrix type, and view must describe the same matrix.
    ///
    /// # Safety
    ///
    /// Every solver call that uses this descriptor must run with hybrid execute mode enabled through [`Config::set_hybrid_execute_mode`](crate::config::Config::set_hybrid_execute_mode), since other modes read the arrays on the device.
    ///
    /// # Errors
    ///
    /// Returns an error if `desc` is invalid or an array is too short for it, the context cannot be bound, or cuDSS cannot create the descriptor or returns a null handle.
    pub unsafe fn create_csr_host<O, I, T>(
        context: &Arc<CudaContext>,
        desc: CsrMatrixDescriptor,
        row_start: &'a [O],
        row_end: Option<&'a [O]>,
        column_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self>
    where
        O: MatrixIndex,
        I: MatrixIndex,
        T: DataTypeLike,
    {
        validate_csr_descriptor(
            desc,
            Some(row_start.len()),
            row_end.map(<[O]>::len),
            Some(column_indices.len()),
            Some(values.len()),
        )?;
        // SAFETY: `validate_csr_descriptor` checked every array length against `desc`, the slices stay borrowed for `'a`, and the caller guarantees hybrid execute mode reads them on the host.
        unsafe {
            Self::create_csr_native(
                context,
                desc,
                host_ptr(row_start.as_ptr()),
                row_end.map_or(DevicePtr::null(), |row_end| host_ptr(row_end.as_ptr())),
                host_ptr(column_indices.as_ptr()),
                host_ptr(values.as_ptr()),
                O::DATA_TYPE,
                I::DATA_TYPE,
                T::DATA_TYPE,
            )
        }
    }

    // The caller guarantees that the pointers have the declared types, cover `desc`, and outlive every use of the descriptor.
    unsafe fn create_csr_native(
        context: &Arc<CudaContext>,
        desc: CsrMatrixDescriptor,
        row_start: DevicePtr,
        row_end: DevicePtr,
        column_indices: DevicePtr,
        values: DevicePtr,
        offset_type: DataType,
        index_type: DataType,
        value_type: DataType,
    ) -> Result<Self> {
        context.bind()?;
        validate_csr_descriptor(desc, None, None, None, None)?;
        // SAFETY: `handle` is a local output checked for null, `desc` passed `validate_csr_descriptor`, and the caller guarantees the four arrays have the declared types, describe `desc`, and outlive the descriptor.
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
                context: Arc::clone(context),
                values: None,
                structure: Vec::new(),
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
        validate_min_len(desc.rows.len(), values.len(), "values table")?;
        // SAFETY: the table holds an entry per batch matrix, and it and its pointees stay borrowed for `'a`; the raw call validates the shape arrays.
        let matrix = unsafe {
            Self::create_batch_dense_native(values.context(), desc, values.as_ptr(), T::DATA_TYPE)
        }?;
        Ok(matrix.tracking(values.region(), Vec::new()))
    }

    // The caller guarantees that the pointers have the declared types, cover `desc`, and outlive every use of the descriptor.
    unsafe fn create_batch_dense_native<I: BatchIndex>(
        context: &Arc<CudaContext>,
        desc: BatchDenseMatrixDescriptor<'a, I>,
        values: DevicePtr,
        value_type: DataType,
    ) -> Result<Self> {
        context.bind()?;
        validate_batch_dense_descriptor(desc)?;
        let batch_count = to_i64(desc.rows.len(), "batch_count")?;
        // SAFETY: `handle` is a local output checked for null, `validate_batch_dense_descriptor` checked that the shape slices hold `batch_count` entries each, and the caller guarantees `values` is a pointer table of `value_type` buffers that outlive the descriptor.
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
                context: Arc::clone(context),
                values: None,
                structure: Vec::new(),
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
        ensure_context(values.context(), row_start.context())?;
        ensure_context(values.context(), col_indices.context())?;
        for (name, length) in [
            ("row start table", row_start.len()),
            ("column index table", col_indices.len()),
            ("values table", values.len()),
        ] {
            validate_min_len(desc.rows.len(), length, name)?;
        }

        if let Some(row_end) = row_end {
            ensure_context(values.context(), row_end.context())?;
            validate_min_len(desc.rows.len(), row_end.len(), "row end table")?;
        }

        // SAFETY: each table holds an entry per batch matrix and shares one context, and the tables stay borrowed for `'a`; the raw call validates the shape arrays.
        let matrix = unsafe {
            Self::create_batch_csr_native(
                values.context(),
                desc,
                row_start.as_ptr(),
                row_end.map_or(DevicePtr::null(), DevicePointerTable::as_ptr),
                col_indices.as_ptr(),
                values.as_ptr(),
                O::DATA_TYPE,
                J::DATA_TYPE,
                T::DATA_TYPE,
            )
        }?;
        let structure = [
            row_start.region(),
            row_end.and_then(|row_end| row_end.region()),
            col_indices.region(),
        ]
        .into_iter()
        .flatten()
        .collect();
        Ok(matrix.tracking(values.region(), structure))
    }

    // The caller guarantees that the pointers have the declared types, cover `desc`, and outlive every use of the descriptor.
    unsafe fn create_batch_csr_native<I: BatchIndex>(
        context: &Arc<CudaContext>,
        desc: BatchCsrMatrixDescriptor<'a, I>,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
        offset_type: DataType,
        index_type: DataType,
        value_type: DataType,
    ) -> Result<Self> {
        context.bind()?;
        validate_batch_csr_descriptor(desc)?;
        let batch_count = to_i64(desc.rows.len(), "batch_count")?;
        // SAFETY: `handle` is a local output checked for null, `validate_batch_csr_descriptor` checked that the shape slices hold `batch_count` entries each, and the caller guarantees the four pointer tables reference arrays of the declared types that outlive the descriptor.
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
                context: Arc::clone(context),
                values: None,
                structure: Vec::new(),
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
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and every output is a local of the native type.
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
                values: DevicePtr::from_raw(values.cast()),
                value_type: value_type.try_into()?,
                layout: layout.try_into()?,
            })
        }
    }

    /// Returns CSR matrix descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata.
    pub fn csr_info(&self) -> Result<CsrMatrixInfo> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and every output is a local of the native type.
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
                row_start: DevicePtr::from_raw(row_start.cast()),
                row_end: DevicePtr::from_raw(row_end.cast()),
                column_indices: DevicePtr::from_raw(col_indices.cast()),
                values: DevicePtr::from_raw(values.cast()),
                offset_type: offset_type.try_into()?,
                index_type: index_type.try_into()?,
                value_type: value_type.try_into()?,
                matrix_type: matrix_type.try_into()?,
                view_type: view_type.try_into()?,
                index_base: index_base.try_into()?,
            })
        }
    }

    /// Returns non-uniform batch dense descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata, if the reported batch count cannot be represented as `usize`, or if cuDSS reports an unsupported batch index type.
    pub fn batch_dense_info(&self) -> Result<BatchDenseMatrixInfo> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns and every output is a local.
        // cuDSS returns the host shape arrays the descriptor was created with, borrowed for `'a`, each holding `batch_count` entries of `index_type`.
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
            let index_type = DataType::try_from(index_type)?;
            Ok(BatchDenseMatrixInfo {
                rows: copy_index_batch(rows, batch_count, index_type, "rows")?,
                columns: copy_index_batch(columns, batch_count, index_type, "columns")?,
                leading_dim: copy_index_batch(leading_dim, batch_count, index_type, "leading_dim")?,
                values: DevicePtr::from_raw(values.cast::<c_void>().cast()),
                index_type,
                value_type: value_type.try_into()?,
                layout: layout.try_into()?,
            })
        }
    }

    /// Returns non-uniform batch CSR descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS cannot report the descriptor metadata, if the reported batch count cannot be represented as `usize`, or if cuDSS reports an unsupported batch index type.
    pub fn batch_csr_info(&self) -> Result<BatchCsrMatrixInfo> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns and every output is a local.
        // cuDSS returns the host shape arrays the descriptor was created with, borrowed for `'a`, each holding `batch_count` entries of `index_type`.
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
            let index_type = DataType::try_from(index_type)?;
            Ok(BatchCsrMatrixInfo {
                rows: copy_index_batch(rows, batch_count, index_type, "rows")?,
                columns: copy_index_batch(columns, batch_count, index_type, "columns")?,
                nnz: copy_index_batch(nnz, batch_count, index_type, "nnz")?,
                row_start: DevicePtr::from_raw(row_start.cast::<c_void>().cast()),
                row_end: DevicePtr::from_raw(row_end.cast::<c_void>().cast()),
                column_indices: DevicePtr::from_raw(col_indices.cast::<c_void>().cast()),
                values: DevicePtr::from_raw(values.cast::<c_void>().cast()),
                offset_type: offset_type.try_into()?,
                index_type,
                value_type: value_type.try_into()?,
                matrix_type: matrix_type.try_into()?,
                view_type: view_type.try_into()?,
                index_base: index_base.try_into()?,
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
    pub fn set_values<T: DataTypeLike>(&mut self, values: DeviceView<'a, T>) -> Result<()> {
        ensure_context(&self.context, values.context())?;
        self.validate_value_buffer::<T>(values.len())?;
        // SAFETY: `validate_value_buffer` checked the element type and length against this descriptor, and the view stays borrowed for `'a`.
        unsafe { self.set_values_native(device_ptr(values)) }?;
        self.values = Some(values.region());
        Ok(())
    }

    // The caller guarantees that the pointers match this descriptor and outlive every use of it.
    unsafe fn set_values_native(&mut self, values: DevicePtr) -> Result<()> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and the caller guarantees `values` matches it and outlives its uses.
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
        row_start: DeviceView<'a, O>,
        row_end: Option<DeviceView<'a, O>>,
        col_indices: DeviceView<'a, I>,
        values: DeviceView<'a, T>,
    ) -> Result<()>
    where
        O: MatrixIndex,
        I: MatrixIndex,
        T: DataTypeLike,
    {
        ensure_context(&self.context, values.context())?;
        ensure_context(&self.context, row_start.context())?;
        ensure_context(&self.context, col_indices.context())?;
        if let Some(row_end) = row_end {
            ensure_context(&self.context, row_end.context())?;
        }

        let info = self.csr_info()?;
        if info.offset_type != O::DATA_TYPE
            || info.index_type != I::DATA_TYPE
            || info.value_type != T::DATA_TYPE
        {
            return Err(Error::UnsupportedDataType {
                name: "csr replacement buffers".into(),
            });
        }

        validate_csr_descriptor(
            CsrMatrixDescriptor {
                rows: info.rows,
                columns: info.columns,
                nnz: info.nnz,
                matrix_type: info.matrix_type,
                view_type: info.view_type,
                index_base: info.index_base,
            },
            Some(row_start.len()),
            row_end.map(|row| row.len()),
            Some(col_indices.len()),
            Some(values.len()),
        )?;
        // SAFETY: the data types match `csr_info`, `validate_csr_descriptor` checked every length, and the views stay borrowed for `'a`.
        unsafe {
            self.set_csr_values_native(
                device_ptr(row_start),
                row_end.map_or(DevicePtr::null(), device_ptr),
                device_ptr(col_indices),
                device_ptr(values),
            )
        }?;
        self.values = Some(values.region());
        self.structure = [
            Some(row_start.region()),
            row_end.map(|row_end| row_end.region()),
        ]
        .into_iter()
        .flatten()
        .chain([col_indices.region()])
        .collect();
        Ok(())
    }

    // The caller guarantees that the pointers match this descriptor and outlive every use of it.
    unsafe fn set_csr_values_native(
        &mut self,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
    ) -> Result<()> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and the caller guarantees the pointers match it and outlive its uses.
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
    pub fn set_batch_values<T: DataTypeLike>(
        &mut self,
        values: DevicePointerTable<'a, T>,
    ) -> Result<()> {
        ensure_context(&self.context, values.context())?;
        let (data_type, count) = if self.format()?.contains(MatrixFormat::CSR) {
            let info = self.batch_csr_info()?;
            (info.value_type, info.rows.len())
        } else {
            let info = self.batch_dense_info()?;
            (info.value_type, info.rows.len())
        };
        if data_type != T::DATA_TYPE {
            return Err(Error::UnsupportedDataType {
                name: "batch replacement values".into(),
            });
        }

        validate_min_len(count, values.len(), "values table")?;
        // SAFETY: the value type matches the descriptor, the table holds an entry per batch matrix, and it stays borrowed for `'a`.
        unsafe { self.set_batch_values_native(values.as_ptr()) }?;
        self.values = values.region();
        Ok(())
    }

    // The caller guarantees that the pointers match this descriptor and outlive every use of it.
    unsafe fn set_batch_values_native(&mut self, values: DevicePtr) -> Result<()> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and the caller guarantees the pointer table matches it and outlives its uses.
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
        let info = self.batch_csr_info()?;
        if info.offset_type != O::DATA_TYPE
            || info.index_type != J::DATA_TYPE
            || info.value_type != T::DATA_TYPE
        {
            return Err(Error::UnsupportedDataType {
                name: "batch csr replacement buffers".into(),
            });
        }

        for (name, table_context, count) in [
            ("row start table", row_start.context(), row_start.len()),
            (
                "column index table",
                col_indices.context(),
                col_indices.len(),
            ),
            ("values table", values.context(), values.len()),
        ] {
            ensure_context(&self.context, table_context)?;
            validate_min_len(info.rows.len(), count, name)?;
        }

        if let Some(row_end) = row_end {
            ensure_context(&self.context, row_end.context())?;
            validate_min_len(info.rows.len(), row_end.len(), "row end table")?;
        }

        // SAFETY: the data types match `batch_csr_info`, each table holds an entry per batch matrix, and the tables stay borrowed for `'a`.
        unsafe {
            self.set_batch_csr_values_native(
                row_start.as_ptr(),
                row_end.map_or(DevicePtr::null(), DevicePointerTable::as_ptr),
                col_indices.as_ptr(),
                values.as_ptr(),
            )
        }?;
        self.values = values.region();
        self.structure = [
            row_start.region(),
            row_end.and_then(|row_end| row_end.region()),
            col_indices.region(),
        ]
        .into_iter()
        .flatten()
        .collect();
        Ok(())
    }

    // The caller guarantees that the pointers match this descriptor and outlive every use of it.
    unsafe fn set_batch_csr_values_native(
        &mut self,
        row_start: DevicePtr,
        row_end: DevicePtr,
        col_indices: DevicePtr,
        values: DevicePtr,
    ) -> Result<()> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and the caller guarantees the pointer tables match it and outlive their uses.
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
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and `format` is a local `c_int` output.
        unsafe {
            let mut format = 0;
            try_ffi!(sys::cudssMatrixGetFormat(self.handle, &raw mut format))?;
            Ok(MatrixFormat::from_bits_retain(format.cast_unsigned()))
        }
    }

    /// Sets the 1D row distribution for MGMN mode.
    ///
    /// `first_row` and `last_row` are zero-based inclusive global row indices for the local matrix on the calling process.
    /// Set `first_row > last_row` to mark an empty local matrix.
    ///
    /// The system matrix, right-hand side, and solution may use different distributions.
    /// If sparse matrix or right-hand side distributions overlap, overlapping contributions are summed.
    /// If solution distributions overlap, overlapping solution entries have the same values on the corresponding processes.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDSS rejects the row distribution.
    pub fn set_distribution_row_1d(&mut self, first_row: i64, last_row: i64) -> Result<()> {
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and the row bounds are passed by value.
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
        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and both bounds are local `i64` outputs.
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
    #[must_use]
    pub const fn as_raw(&self) -> sys::cudssMatrix_t {
        self.handle
    }

    /// Takes ownership of an existing cuDSS matrix descriptor and its CUDA context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid, uniquely owned cuDSS matrix descriptor created in `raw.context`.
    /// The descriptor may store borrowed pointers, and the caller must ensure those buffers outlive the returned wrapper's lifetime.
    /// The returned wrapper tracks none of those buffers, so the caller orders their producers before execution.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawMatrix) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            values: None,
            structure: Vec::new(),
            _marker: PhantomData,
        })
    }

    /// Consumes this wrapper and returns the owned raw cuDSS matrix descriptor with its CUDA context.
    ///
    /// The caller becomes responsible for destroying the returned descriptor.
    #[must_use]
    pub fn into_raw(self) -> RawMatrix {
        let matrix = ManuallyDrop::new(self);
        // SAFETY: `matrix` is never dropped, so the retained context is read out exactly once.
        let context = unsafe { ptr::read(&raw const matrix.context) };
        // SAFETY: as above; the tracked regions are released here.
        drop(unsafe { ptr::read(&raw const matrix.structure) });
        RawMatrix {
            handle: matrix.handle,
            context,
        }
    }
}

impl Drop for Matrix<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure("bind the cuda context to destroy", "cudss matrix", &error);
            return;
        }

        // SAFETY: `self.handle` is the live descriptor this wrapper owns, its CUDA context was bound above, and `into_raw` skips this drop when ownership leaves the wrapper.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudssMatrixDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cudss matrix", &error);
            }
        }
    }
}

impl DenseMatrixDescriptor {
    /// Creates a column-major dense matrix descriptor.
    #[must_use]
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
    /// cuDSS documents row-major layout but does not currently support it for solve right-hand side and solution matrices.
    #[must_use]
    pub fn new_row_major(rows: i64, columns: i64, leading_dim: i64) -> Self {
        Self {
            rows,
            columns,
            leading_dim,
            layout: Layout::RowMajor,
        }
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
            return Err(Error::BufferTooSmall {
                name: "values".into(),
                required,
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
        validate_min_len(expected, row_start_len, "row_start")?;
    }

    if let Some(row_end_len) = row_end_len {
        validate_min_len(rows, row_end_len, "row_end")?;
    }

    if let Some(column_indices_len) = column_indices_len {
        validate_min_len(nnz, column_indices_len, "column_indices")?;
    }

    if let Some(values_len) = values_len {
        validate_min_len(nnz, values_len, "values")?;
    }

    Ok(())
}

fn validate_batch_dense_descriptor<I: BatchIndex>(
    desc: BatchDenseMatrixDescriptor<'_, I>,
) -> Result<()> {
    validate_same_len(desc.rows.len(), desc.columns.len(), "columns")?;
    validate_same_len(desc.rows.len(), desc.leading_dim.len(), "leading_dim")?;
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
    validate_same_len(desc.rows.len(), desc.columns.len(), "columns")?;
    validate_same_len(desc.rows.len(), desc.nnz.len(), "nnz")?;
    for ((row, column), nnz) in desc.rows.iter().zip(desc.columns).zip(desc.nnz) {
        validate_nonnegative(row.to_i64(), "rows")?;
        validate_nonnegative(column.to_i64(), "columns")?;
        validate_nonnegative(nnz.to_i64(), "nnz")?;
    }
    Ok(())
}

fn device_ptr<T: DeviceRepr>(memory: DeviceView<'_, T>) -> DevicePtr {
    // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
    unsafe { DevicePtr::from_raw(memory.as_ptr().cast_mut().cast()) }
}

fn host_ptr<T>(values: *const T) -> DevicePtr {
    // SAFETY: the address belongs to a live host slice that stays borrowed for the operation that uses it.
    unsafe { DevicePtr::from_raw(values.cast_mut().cast()) }
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
        // SAFETY: the caller guarantees `ptr` points to `len` readable values of `index_type`, here `i32`.
        DataType::I32 => Ok(unsafe {
            slice::from_raw_parts(ptr.cast::<i32>(), len)
                .iter()
                .copied()
                .map(i64::from)
                .collect()
        }),
        // SAFETY: the caller guarantees `ptr` points to `len` readable values of `index_type`, here `i64`.
        DataType::I64 => Ok(unsafe { copy_i64_batch(ptr, len) }),
        _ => Err(Error::UnsupportedDataType {
            name: name.to_owned().into(),
        }),
    }
}

fn ensure_context(expected: &Arc<CudaContext>, actual: &Arc<CudaContext>) -> Result<()> {
    if expected != actual {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    Ok(())
}
