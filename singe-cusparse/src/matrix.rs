use std::{
    marker::PhantomData,
    mem,
    mem::{ManuallyDrop, MaybeUninit, size_of},
    ptr,
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataTypeLike,
    types::DevicePtr,
    view::{DeviceInput, DeviceOutput, DeviceOutputView, DeviceRegion},
};

use crate::{
    context::Context,
    descriptor::{
        RawDescriptor, checked_add, checked_mul, input_region, matrix_extent, output_region,
        output_region_for, workspace,
    },
    error::{Error, Result},
    sys, try_ffi,
    types::{
        Action, CsrToCscAlgorithm, DenseToSparseAlgorithm, DiagonalType, FillMode, Format,
        IndexBase, IndexTypeLike, MatrixType, Order, SparseMatrixAttribute, SparseToDenseAlgorithm,
    },
    utility::{to_i32, to_i64, to_usize},
};

/// Logical matrix dimensions and stored sparse-entry count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMatrixShape {
    /// Logical number of matrix rows.
    pub rows: usize,
    /// Logical number of matrix columns.
    pub cols: usize,
    /// Number of stored sparse entries.
    pub nonzero_count: usize,
}

/// Dimensions, storage order, stride, and value address reported by a dense matrix descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMatrixInfo {
    /// Logical number of matrix rows.
    pub rows: usize,
    /// Logical number of matrix columns.
    pub cols: usize,
    /// Stride between major rows or columns, measured in elements.
    pub leading_dimension: usize,
    /// Device address of the matrix values.
    pub values: DevicePtr,
    /// Row-major or column-major storage order.
    pub order: Order,
}

/// Owns a legacy cuSPARSE matrix-property descriptor.
#[derive(Debug)]
pub struct MatrixDescriptor {
    handle: sys::cusparseMatDescr_t,
}

/// Owns a sparse-matrix descriptor tied to a CUDA context and borrowed device storage.
///
/// Storage metadata does not establish that pending device values are initialized.
#[derive(Debug)]
pub struct SparseMatrixDescriptor<'a> {
    handle: sys::cusparseSpMatDescr_t,
    cuda_context: Arc<CudaContext>,
    regions: Vec<Option<DeviceRegion<'a>>>,
    _buffers: PhantomData<&'a mut ()>,
}

/// Owns a dense-matrix descriptor tied to a CUDA context and borrowed device storage.
#[derive(Debug)]
pub struct DenseMatrixDescriptor<'a> {
    handle: sys::cusparseDnMatDescr_t,
    cuda_context: Arc<CudaContext>,
    regions: Vec<Option<DeviceRegion<'a>>>,
    _buffers: PhantomData<&'a mut ()>,
}

// SAFETY: legacy matrix descriptors contain descriptor attributes only, so immutable
// sharing follows the cuSPARSE descriptor contract. Data-bearing descriptors
// can rebind borrowed pointers and therefore require exclusive access.
unsafe impl Send for MatrixDescriptor {}

unsafe impl Sync for MatrixDescriptor {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for SparseMatrixDescriptor<'_> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for DenseMatrixDescriptor<'_> {}

impl MatrixDescriptor {
    /// Initializes a matrix descriptor.
    /// It sets [`MatrixType`] and [`IndexBase`] to [`MatrixType::General`] and
    /// [`IndexBase::Zero`], respectively, with lower fill mode and non-unit diagonal.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot create the descriptor or if it returns
    /// a null handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseCreateMatDescr` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseCreateMatDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut descriptor = Self { handle };
        descriptor.set_fill_mode(FillMode::Lower)?;
        descriptor.set_diagonal_type(DiagonalType::NonUnit)?;
        Ok(descriptor)
    }

    /// Returns the [`MatrixType`] field of this matrix descriptor.
    #[must_use]
    pub fn matrix_type(&self) -> MatrixType {
        // SAFETY: `cusparseGetMatType` receives live handles owned by the wrappers.
        unsafe { sys::cusparseGetMatType(self.as_raw()).into() }
    }

    /// Sets the [`MatrixType`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `matrix_type`.
    pub fn set_matrix_type(&mut self, matrix_type: MatrixType) -> Result<()> {
        // SAFETY: `cusparseSetMatType` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseSetMatType(self.as_raw(), matrix_type.into()))?;
        }
        Ok(())
    }

    /// Returns the [`FillMode`] field of this matrix descriptor.
    #[must_use]
    pub fn fill_mode(&self) -> FillMode {
        // SAFETY: `cusparseGetMatFillMode` receives live handles owned by the wrappers.
        unsafe { sys::cusparseGetMatFillMode(self.as_raw()).into() }
    }

    /// Sets the [`FillMode`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `fill_mode`.
    pub fn set_fill_mode(&mut self, fill_mode: FillMode) -> Result<()> {
        // SAFETY: `cusparseSetMatFillMode` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseSetMatFillMode(self.as_raw(), fill_mode.into()))?;
        }
        Ok(())
    }

    /// Returns the [`DiagonalType`] field of this matrix descriptor.
    #[must_use]
    pub fn diagonal_type(&self) -> DiagonalType {
        // SAFETY: `cusparseGetMatDiagType` receives live handles owned by the wrappers.
        unsafe { sys::cusparseGetMatDiagType(self.as_raw()).into() }
    }

    /// Sets the [`DiagonalType`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `diagonal_type`.
    pub fn set_diagonal_type(&mut self, diagonal_type: DiagonalType) -> Result<()> {
        // SAFETY: `cusparseSetMatDiagType` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseSetMatDiagType(
                self.as_raw(),
                diagonal_type.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the [`IndexBase`] field of this matrix descriptor.
    #[must_use]
    pub fn index_base(&self) -> IndexBase {
        // SAFETY: `cusparseGetMatIndexBase` receives live handles owned by the wrappers.
        unsafe { sys::cusparseGetMatIndexBase(self.as_raw()).into() }
    }

    /// Sets the [`IndexBase`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `index_base`.
    pub fn set_index_base(&mut self, index_base: IndexBase) -> Result<()> {
        // SAFETY: `cusparseSetMatIndexBase` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseSetMatIndexBase(
                self.as_raw(),
                index_base.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the native matrix-property descriptor handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseMatDescr_t {
        self.handle
    }

    /// Wraps an existing cuSPARSE legacy matrix descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseMatDescr_t`. Ownership of `handle` is
    /// transferred to the returned descriptor, and the handle must not be
    /// destroyed elsewhere after calling this function.
    /// Every descriptor attribute must have an initialized native value.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(handle: sys::cusparseMatDescr_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseDestroyMatDescr`.
    #[must_use]
    pub fn into_raw(self) -> sys::cusparseMatDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for MatrixDescriptor {
    fn drop(&mut self) {
        // SAFETY: `cusparseDestroyMatDescr` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroyMatDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse matrix descriptor: {error}");
            }
        }
    }
}

impl<'a> SparseMatrixDescriptor<'a> {
    /// Initializes a sparse matrix descriptor in the CSR format.
    ///
    /// The descriptor borrows the row-offset, column-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_csr<Offsets: IndexTypeLike, Columns: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        row_offsets: Option<DeviceOutputView<'a, Offsets>>,
        column_indices: Option<DeviceOutputView<'a, Columns>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = Offsets::LAYOUT;
        let () = Columns::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        let row_offsets = row_offsets
            .map(|view| {
                output_region(
                    context,
                    view,
                    checked_add(rows, 1, "row offsets")?,
                    "row_offsets",
                )
            })
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region(context, view, nonzero_count, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, nonzero_count, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateCsr` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateCsr(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                row_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                Offsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![row_offsets, column_indices, values])
    }

    /// Initializes a sparse matrix descriptor in the CSC format.
    ///
    /// The descriptor borrows the column-offset, row-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_csc<Offsets: IndexTypeLike, Rows: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        column_offsets: Option<DeviceOutputView<'a, Offsets>>,
        row_indices: Option<DeviceOutputView<'a, Rows>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = Offsets::LAYOUT;
        let () = Rows::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        let column_offsets = column_offsets
            .map(|view| {
                output_region(
                    context,
                    view,
                    checked_add(cols, 1, "column offsets")?,
                    "column_offsets",
                )
            })
            .transpose()?;
        let row_indices = row_indices
            .map(|view| output_region(context, view, nonzero_count, "row_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, nonzero_count, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateCsc` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateCsc(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                column_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                row_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                Offsets::index_type().into(),
                Rows::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![column_offsets, row_indices, values])
    }

    /// Initializes a sparse matrix descriptor in the COO format (Structure of Arrays layout).
    ///
    /// The descriptor borrows the row-index, column-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_coo<I: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        row_indices: Option<DeviceOutputView<'a, I>>,
        column_indices: Option<DeviceOutputView<'a, I>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = I::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        let row_indices = row_indices
            .map(|view| output_region(context, view, nonzero_count, "row_indices"))
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region(context, view, nonzero_count, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, nonzero_count, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateCoo` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateCoo(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                row_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                I::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![row_indices, column_indices, values])
    }

    /// Initializes a sparse matrix descriptor for the Block Compressed Row (BSR) format.
    ///
    /// The descriptor borrows the block row-offset, column-index, and value
    /// buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. Block value-count multiplication must not overflow. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_bsr<Offsets: IndexTypeLike, Columns: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        block_rows: usize,
        block_cols: usize,
        block_nnz: usize,
        row_block_size: usize,
        col_block_size: usize,
        row_offsets: Option<DeviceOutputView<'a, Offsets>>,
        column_indices: Option<DeviceOutputView<'a, Columns>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
        order: Order,
    ) -> Result<Self> {
        let () = Offsets::LAYOUT;
        let () = Columns::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        let value_count = checked_mul(
            checked_mul(block_nnz, row_block_size, "block values")?,
            col_block_size,
            "block values",
        )?;
        let row_offsets = row_offsets
            .map(|view| {
                output_region(
                    context,
                    view,
                    checked_add(block_rows, 1, "row offsets")?,
                    "row_offsets",
                )
            })
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region(context, view, block_nnz, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, value_count, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateBsr` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateBsr(
                &raw mut raw,
                to_i64(block_rows, "block_rows")?,
                to_i64(block_cols, "block_cols")?,
                to_i64(block_nnz, "block_nnz")?,
                to_i64(row_block_size, "row_block_size")?,
                to_i64(col_block_size, "col_block_size")?,
                row_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                Offsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
                order.into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![row_offsets, column_indices, values])
    }

    /// Initializes a sparse matrix descriptor for the Blocked-Ellpack (ELL) format.
    ///
    /// Blocked-ELL column indices are in the range `0..cols / ell_block_size`.
    /// The array can contain `-1` values for indicating empty blocks.
    /// The descriptor borrows the column-index and value buffers by pointer;
    /// those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. The block size must be nonzero and divide both the row count and padded column count. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_blocked_ell<I: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        rows: usize,
        cols: usize,
        block_size: usize,
        ell_cols: usize,
        column_indices: Option<DeviceOutputView<'a, I>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = I::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        if block_size == 0
            || !rows.is_multiple_of(block_size)
            || !ell_cols.is_multiple_of(block_size)
        {
            return Err(Error::out_of_range("blocked ELL geometry"));
        }
        let index_count = checked_mul(rows / block_size, ell_cols / block_size, "ELL indices")?;
        let value_count = checked_mul(rows, ell_cols, "ELL values")?;
        let column_indices = column_indices
            .map(|view| output_region(context, view, index_count, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, value_count, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateBlockedEll` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateBlockedEll(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(block_size, "block_size")?,
                to_i64(ell_cols, "ell_cols")?,
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                I::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![column_indices, values])
    }

    /// Initializes a sparse matrix descriptor for the Sliced Ellpack (SELL) format.
    ///
    /// The descriptor borrows the slice-offset, column-index, and value buffers
    /// by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if supplied arrays belong to another CUDA context or are smaller than the required storage. Rejects count conversion and storage-size overflow. The slice size must be nonzero. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create_sliced_ell<
        SliceOffsets: IndexTypeLike,
        Columns: IndexTypeLike,
        T: DataTypeLike,
    >(
        context: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        values_size: usize,
        slice_size: usize,
        slice_offsets: Option<DeviceOutputView<'a, SliceOffsets>>,
        column_indices: Option<DeviceOutputView<'a, Columns>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = SliceOffsets::LAYOUT;
        let () = Columns::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        if slice_size == 0 {
            return Err(Error::out_of_range("slice size"));
        }
        let offset_count = checked_add(rows.div_ceil(slice_size), 1, "slice offsets")?;
        let slice_offsets = slice_offsets
            .map(|view| output_region(context, view, offset_count, "slice_offsets"))
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region(context, view, values_size, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, values_size, "values"))
            .transpose()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `cusparseCreateSlicedEll` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateSlicedEll(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                to_i64(values_size, "values_size")?,
                to_i64(slice_size, "slice_size")?,
                slice_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                SliceOffsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, context, vec![slice_offsets, column_indices, values])
    }

    /// Returns the format of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix format.
    pub fn format(&self) -> Result<Format> {
        self.cuda_context.bind()?;
        let mut format = sys::cusparseFormat_t::CUSPARSE_FORMAT_CSR;
        // SAFETY: `cusparseSpMatGetFormat` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetFormat(
                self.as_raw_const(),
                &raw mut format,
            ))?;
        }
        Ok(format.into())
    }

    /// Returns the index base of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix index base.
    pub fn index_base(&self) -> Result<IndexBase> {
        self.cuda_context.bind()?;
        let mut index_base = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        // SAFETY: `cusparseSpMatGetIndexBase` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetIndexBase(
                self.as_raw_const(),
                &raw mut index_base,
            ))?;
        }
        Ok(index_base.into())
    }

    /// Returns the values pointer of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        self.cuda_context.bind()?;
        let mut values = ptr::null_mut();
        // SAFETY: `cusparseSpMatGetValues` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this sparse matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    /// On failure the consumed descriptor is destroyed while both sets of regions remain borrowed.
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch
    /// extents, and preprocessing protocol. Complete previous accesses before rebinding.
    pub unsafe fn rebind_values<'b, T: DataTypeLike>(
        self,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<SparseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { SparseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `cusparseSpMatSetValues` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseSpMatSetValues(
                rebound.as_raw(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
            ))?;
        }
        let incoming = rebound.regions.pop().unwrap_or(None);
        if let Some(last) = rebound.regions.last_mut() {
            *last = incoming;
        } else {
            rebound.regions.push(incoming);
        }
        Ok(rebound)
    }

    /// Returns the shape of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix shape or if
    /// the reported values cannot be represented as `usize`.
    pub fn shape(&self) -> Result<SparseMatrixShape> {
        self.cuda_context.bind()?;
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut nnz = 0_i64;
        // SAFETY: `cusparseSpMatGetSize` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetSize(
                self.as_raw_const(),
                &raw mut rows,
                &raw mut cols,
                &raw mut nnz,
            ))?;
        }
        Ok(SparseMatrixShape {
            rows: to_usize(rows, "rows")?,
            cols: to_usize(cols, "cols")?,
            nonzero_count: to_usize(nnz, "nonzero_count")?,
        })
    }

    /// Returns the batch count of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the strided batch settings.
    pub fn strided_batch_count(&self) -> Result<i32> {
        self.cuda_context.bind()?;
        let mut batch_count = 0;
        // SAFETY: `cusparseSpMatGetStridedBatch` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetStridedBatch(
                self.as_raw_const(),
                &raw mut batch_count,
            ))?;
        }
        Ok(batch_count)
    }

    /// Sets the batch count and batch stride fields of this COO sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if either argument cannot be represented by cuSPARSE or
    /// if cuSPARSE rejects the strided batch settings.
    pub fn set_coo_strided_batch(&mut self, batch_count: usize, batch_stride: usize) -> Result<()> {
        // SAFETY: `cusparseCooSetStridedBatch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseCooSetStridedBatch(
                self.as_raw(),
                to_i32(batch_count, "batch_count")?,
                to_i64(batch_stride, "batch_stride")?,
            ))?;
        }
        Ok(())
    }

    /// Sets the batch count and batch stride fields of this CSR sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch count or strides do not fit the native integer types, or cuSPARSE rejects the batching configuration.
    pub fn set_csr_strided_batch(
        &mut self,
        batch_count: usize,
        offsets_batch_stride: usize,
        columns_values_batch_stride: usize,
    ) -> Result<()> {
        // SAFETY: `cusparseCsrSetStridedBatch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseCsrSetStridedBatch(
                self.as_raw(),
                to_i32(batch_count, "batch_count")?,
                to_i64(offsets_batch_stride, "offsets_batch_stride")?,
                to_i64(columns_values_batch_stride, "columns_values_batch_stride")?,
            ))?;
        }
        Ok(())
    }

    /// Sets the batch count and batch stride fields of this BSR sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch count or strides do not fit the native integer types, or cuSPARSE rejects the batching configuration.
    pub fn set_bsr_strided_batch(
        &mut self,
        batch_count: usize,
        offsets_batch_stride: usize,
        columns_batch_stride: usize,
        values_batch_stride: usize,
    ) -> Result<()> {
        // SAFETY: `cusparseBsrSetStridedBatch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseBsrSetStridedBatch(
                self.as_raw(),
                to_i32(batch_count, "batch_count")?,
                to_i64(offsets_batch_stride, "offsets_batch_stride")?,
                to_i64(columns_batch_stride, "columns_batch_stride")?,
                to_i64(values_batch_stride, "values_batch_stride")?,
            ))?;
        }
        Ok(())
    }

    /// Queries which triangular part of the sparse matrix is stored.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and native attribute-query errors. Also rejects an unrecognized fill-mode value or an attribute size that cannot fit the native size type.
    pub fn fill_mode(&self) -> Result<FillMode> {
        let value = self.raw_attribute(SparseMatrixAttribute::FillMode)?;
        FillMode::try_from(value).map_err(|_| Error::InvalidAttributeValue {
            name: "fill mode".into(),
            value,
        })
    }

    /// Sets which triangular part of the sparse matrix an operation uses.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and native attribute-setting errors. Also rejects an attribute size that cannot fit the native size type.
    pub fn set_fill_mode(&mut self, fill_mode: FillMode) -> Result<()> {
        self.set_attribute_raw::<sys::cusparseFillMode_t>(
            SparseMatrixAttribute::FillMode,
            fill_mode.into(),
        )
    }

    /// Queries whether the sparse matrix diagonal is explicit or implicitly unit-valued.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and native attribute-query errors. Also rejects an unrecognized diagonal-type value or an attribute size that cannot fit the native size type.
    pub fn diagonal_type(&self) -> Result<DiagonalType> {
        let value = self.raw_attribute(SparseMatrixAttribute::DiagonalType)?;
        DiagonalType::try_from(value).map_err(|_| Error::InvalidAttributeValue {
            name: "diagonal type".into(),
            value,
        })
    }

    /// Sets whether the sparse matrix diagonal is explicit or implicitly unit-valued.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and native attribute-setting errors. Also rejects an attribute size that cannot fit the native size type.
    pub fn set_diagonal_type(&mut self, diagonal_type: DiagonalType) -> Result<()> {
        self.set_attribute_raw::<sys::cusparseDiagType_t>(
            SparseMatrixAttribute::DiagonalType,
            diagonal_type.into(),
        )
    }

    /// Consumes and rebinds the pointers of this CSR sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch
    /// extents, and preprocessing protocol. Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context. Also propagates context-binding and native pointer-update errors.
    pub unsafe fn rebind_csr<'b, I: IndexTypeLike, J: IndexTypeLike, T: DataTypeLike>(
        self,
        row_offsets: Option<DeviceOutputView<'b, I>>,
        column_indices: Option<DeviceOutputView<'b, J>>,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<SparseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = I::LAYOUT;
        let () = J::LAYOUT;

        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { SparseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let row_offsets = row_offsets
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "row_offsets"))
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound
            .regions
            .extend([row_offsets, column_indices, values]);
        // SAFETY: `cusparseCsrSetPointers` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCsrSetPointers(
                rebound.as_raw(),
                row_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
            ))?;
        }
        rebound.regions = vec![row_offsets, column_indices, values];
        Ok(rebound)
    }

    /// Consumes and rebinds the pointers of this CSC sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch
    /// extents, and preprocessing protocol. Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context. Also propagates context-binding and native pointer-update errors.
    pub unsafe fn rebind_csc<'b, I: IndexTypeLike, J: IndexTypeLike, T: DataTypeLike>(
        self,
        column_offsets: Option<DeviceOutputView<'b, I>>,
        row_indices: Option<DeviceOutputView<'b, J>>,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<SparseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = I::LAYOUT;
        let () = J::LAYOUT;

        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { SparseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let column_offsets = column_offsets
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "column_offsets"))
            .transpose()?;
        let row_indices = row_indices
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "row_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound
            .regions
            .extend([column_offsets, row_indices, values]);
        // SAFETY: `cusparseCscSetPointers` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCscSetPointers(
                rebound.as_raw(),
                column_offsets
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                row_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
            ))?;
        }
        rebound.regions = vec![column_offsets, row_indices, values];
        Ok(rebound)
    }

    /// Consumes and rebinds the pointers of this COO sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch
    /// extents, and preprocessing protocol. Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context. Also propagates context-binding and native pointer-update errors.
    pub unsafe fn rebind_coo<'b, I: IndexTypeLike, J: IndexTypeLike, T: DataTypeLike>(
        self,
        row_indices: Option<DeviceOutputView<'b, I>>,
        column_indices: Option<DeviceOutputView<'b, J>>,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<SparseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = I::LAYOUT;
        let () = J::LAYOUT;

        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { SparseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let row_indices = row_indices
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "row_indices"))
            .transpose()?;
        let column_indices = column_indices
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "column_indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound
            .regions
            .extend([row_indices, column_indices, values]);
        // SAFETY: `cusparseCooSetPointers` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCooSetPointers(
                rebound.as_raw(),
                row_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                column_indices
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                    .cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
            ))?;
        }
        rebound.regions = vec![row_indices, column_indices, values];
        Ok(rebound)
    }

    /// Returns the native descriptor handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpMatDescr_t {
        self.handle
    }

    /// Returns the native handle in its const-descriptor form.
    #[must_use]
    pub fn as_raw_const(&self) -> sys::cusparseConstSpMatDescr_t {
        self.handle.cast_const()
    }

    /// Available borrowed buffer regions, with `None` for unbound pointers.
    /// Regions do not establish initialized values, valid indices, or producer ordering.
    /// Raw adoption without metadata returns an empty list.
    #[must_use]
    pub fn regions(&self) -> &[Option<DeviceRegion<'a>>] {
        &self.regions
    }

    /// Returns the actual CUDA context retained by the descriptor.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    pub(crate) fn ensure_context(&self, context: &Context) -> Result<()> {
        if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE sparse-matrix descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned, valid `cusparseSpMatDescr_t` associated
    /// with `parts.cuda_context`. Its pointer bindings and regions must describe
    /// live allocations in that context with the required exclusive borrows for
    /// lifetime `'a`. An empty region list leaves those raw obligations to the caller.
    /// No other owner may destroy the handle after transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawDescriptor<'a, sys::cusparseSpMatDescr_t>) -> Result<Self> {
        let RawDescriptor {
            handle,
            cuda_context,
            regions,
        } = parts;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context,
            regions,
            _buffers: PhantomData,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseDestroySpMat`.
    #[must_use]
    pub fn into_raw(mut self) -> RawDescriptor<'a, sys::cusparseSpMatDescr_t> {
        RawDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
            regions: mem::take(&mut self.regions),
        }
    }

    fn raw_attribute(&self, attribute: SparseMatrixAttribute) -> Result<u32> {
        self.cuda_context.bind()?;
        let mut value = 0_u32;
        // SAFETY: `cusparseSpMatGetAttribute` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetAttribute(
                self.as_raw_const(),
                attribute.into(),
                ptr::from_mut(&mut value).cast(),
                u64::try_from(size_of::<u32>())
                    .map_err(|_| Error::out_of_range("attribute size"))?,
            ))?;
        }
        Ok(value)
    }

    fn set_attribute_raw<T>(&mut self, attribute: SparseMatrixAttribute, value: T) -> Result<()> {
        self.cuda_context.bind()?;
        let mut value = value;
        // SAFETY: `cusparseSpMatSetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMatSetAttribute(
                self.as_raw(),
                attribute.into(),
                (&raw mut value).cast(),
                u64::try_from(size_of::<T>()).map_err(|_| Error::out_of_range("attribute size"))?,
            ))?;
        }
        Ok(())
    }
}

impl Drop for SparseMatrixDescriptor<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if self.cuda_context.bind().is_err() {
            return;
        }

        // SAFETY: `cusparseDestroySpMat` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroySpMat(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse sparse matrix descriptor: {error}");
            }
        }
    }
}

impl<'a> DenseMatrixDescriptor<'a> {
    /// Initializes a dense matrix descriptor.
    ///
    /// The descriptor borrows the device value buffer by pointer; the buffer must
    /// remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid leading dimension, storage-extent overflow, native dimension conversion failure, a different buffer context, or insufficient value storage. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create<T: DataTypeLike>(
        context: &Context,
        rows: usize,
        cols: usize,
        leading_dimension: usize,
        values: DeviceOutputView<'a, T>,
        order: Order,
    ) -> Result<Self> {
        let () = T::LAYOUT;

        context.bind()?;
        let extent = matrix_extent(rows, cols, leading_dimension, matches!(order, Order::Row))?;
        let values = output_region(context, values, extent, "values")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseCreateDnMat` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateDnMat(
                &raw mut handle,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(leading_dimension, "leading_dimension")?,
                values.as_ptr().cast(),
                T::data_type().into(),
                order.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
            regions: vec![Some(values)],
            _buffers: PhantomData,
        })
    }

    /// Returns the values pointer of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        self.cuda_context.bind()?;
        let mut values = ptr::null_mut();
        // SAFETY: `cusparseDnMatGetValues` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseDnMatGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this dense matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    /// On failure the consumed descriptor is destroyed while both sets of regions remain borrowed.
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch
    /// extents, and preprocessing protocol. Complete previous accesses before rebinding.
    pub unsafe fn rebind_values<'b, T: DataTypeLike>(
        self,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<DenseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { DenseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `cusparseDnMatSetValues` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseDnMatSetValues(
                rebound.as_raw(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
            ))?;
        }
        let incoming = rebound.regions.pop().unwrap_or(None);
        if let Some(last) = rebound.regions.last_mut() {
            *last = incoming;
        } else {
            rebound.regions.push(incoming);
        }
        Ok(rebound)
    }

    /// Returns the fields of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if
    /// reported dimensions cannot be represented as `usize`.
    pub fn info(&self) -> Result<DenseMatrixInfo> {
        self.cuda_context.bind()?;
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut leading_dimension = 0_i64;
        let mut values = ptr::null_mut();
        let mut value_type = MaybeUninit::uninit();
        let mut order = sys::cusparseOrder_t::CUSPARSE_ORDER_ROW;
        // SAFETY: `cusparseDnMatGet` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseDnMatGet(
                self.as_raw(),
                &raw mut rows,
                &raw mut cols,
                &raw mut leading_dimension,
                &raw mut values,
                value_type.as_mut_ptr(),
                &raw mut order,
            ))?;
        }
        Ok(DenseMatrixInfo {
            rows: to_usize(rows, "rows")?,
            cols: to_usize(cols, "cols")?,
            leading_dimension: to_usize(leading_dimension, "leading_dimension")?,
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            values: unsafe { DevicePtr::from_raw(values.cast()) },
            order: order.into(),
        })
    }

    /// Sets the number of batches and the batch stride of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if either argument cannot be represented by cuSPARSE or
    /// if cuSPARSE rejects the strided batch settings.
    pub fn set_strided_batch(&mut self, batch_count: usize, batch_stride: usize) -> Result<()> {
        // SAFETY: `cusparseDnMatSetStridedBatch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseDnMatSetStridedBatch(
                self.as_raw(),
                to_i32(batch_count, "batch_count")?,
                to_i64(batch_stride, "batch_stride")?,
            ))?;
        }
        Ok(())
    }

    /// Returns the number of batches and the batch stride of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the strided batch settings or
    /// if the reported stride cannot be represented as `usize`.
    pub fn strided_batch(&self) -> Result<(i32, usize)> {
        self.cuda_context.bind()?;
        let mut batch_count = 0;
        let mut batch_stride = 0_i64;
        // SAFETY: `cusparseDnMatGetStridedBatch` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseDnMatGetStridedBatch(
                self.as_raw_const(),
                &raw mut batch_count,
                &raw mut batch_stride,
            ))?;
        }
        Ok((batch_count, to_usize(batch_stride, "batch_stride")?))
    }

    /// Returns the native descriptor handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseDnMatDescr_t {
        self.handle
    }

    /// Returns the native handle in its const-descriptor form.
    #[must_use]
    pub fn as_raw_const(&self) -> sys::cusparseConstDnMatDescr_t {
        self.handle.cast_const()
    }

    /// Available borrowed buffer regions, with `None` for unbound pointers.
    /// Regions do not establish initialized values, valid indices, or producer ordering.
    /// Raw adoption without metadata returns an empty list.
    #[must_use]
    pub fn regions(&self) -> &[Option<DeviceRegion<'a>>] {
        &self.regions
    }

    /// Returns the actual CUDA context retained by the descriptor.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    pub(crate) fn ensure_context(&self, context: &Context) -> Result<()> {
        if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE dense-matrix descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned, valid `cusparseDnMatDescr_t` associated
    /// with `parts.cuda_context`. Its pointer bindings and regions must describe
    /// live allocations in that context with the required exclusive borrows for
    /// lifetime `'a`. An empty region list leaves those raw obligations to the caller.
    /// No other owner may destroy the handle after transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawDescriptor<'a, sys::cusparseDnMatDescr_t>) -> Result<Self> {
        let RawDescriptor {
            handle,
            cuda_context,
            regions,
        } = parts;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context,
            regions,
            _buffers: PhantomData,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseDestroyDnMat`.
    #[must_use]
    pub fn into_raw(mut self) -> RawDescriptor<'a, sys::cusparseDnMatDescr_t> {
        RawDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
            regions: mem::take(&mut self.regions),
        }
    }
}

impl Drop for DenseMatrixDescriptor<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if self.cuda_context.bind().is_err() {
            return;
        }

        // SAFETY: `cusparseDestroyDnMat` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroyDnMat(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse dense matrix descriptor: {error}");
            }
        }
    }
}

fn ensure_spmat<'a>(
    handle: sys::cusparseSpMatDescr_t,
    context: &Context,
    regions: Vec<Option<DeviceRegion<'a>>>,
) -> Result<SparseMatrixDescriptor<'a>> {
    if handle.is_null() {
        Err(Error::NullHandle)
    } else {
        Ok(SparseMatrixDescriptor {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
            regions,
            _buffers: PhantomData,
        })
    }
}

/// Converts a sparse matrix in CSR format into a sparse matrix in CSC format.
///
/// The resulting matrix can also be seen as the transpose of the original sparse matrix.
/// This can also convert a matrix in CSC format into a matrix in CSR format.
///
/// This operation requires extra storage proportional to `nonzero_count`.
/// It always produces the same matrix values in the destination format.
///
/// The operation executes asynchronously with respect to the host and may return before the result is ready.
///
/// [`csr_to_csc_buffer_size`] returns the size of the workspace needed by [`csr_to_csc`].
/// Pass a buffer of this size to [`csr_to_csc`].
///
/// If `nonzero_count` is `0`, then the CSR column indices, CSR values, CSC
/// values, and CSC row indices may be zero-length buffers.
/// In this case, all CSC column pointers are initialized to the configured index base.
///
/// If `m` or `n` is `0`, the pointers are not checked and the operation succeeds with `Ok`.
///
/// [`csr_to_csc`] supports the following data types:
///
/// | `X`/`Y` | Notes |
/// | --- | --- |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |  |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |  |
/// | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | Deprecated. |
/// | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) | Deprecated. |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |  |
/// | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |  |
///
/// [`csr_to_csc`] supports the following algorithms ([`CsrToCscAlgorithm`]):
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`CsrToCscAlgorithm::Default`] | Default algorithm. |
///
/// | Action | Notes |
/// | --- | --- |
/// | [`Action::Symbolic`] | Compute the structure of the CSC output matrix (offset, row indices). |
/// | [`Action::Numeric`] | Compute the structure of the CSC output matrix and copy the values. |
///
/// [`csr_to_csc`] has the following properties:
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
///
/// [`csr_to_csc`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error for a buffer from another CUDA context, insufficient exposed storage, offset-count overflow, or counts that cannot fit in the native integer types. Also propagates context-binding and native conversion errors.
pub unsafe fn csr_to_csc<T: DataTypeLike>(
    context: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &(impl DeviceInput<T> + ?Sized),
    csr_row_offsets: &(impl DeviceInput<i32> + ?Sized),
    csr_column_indices: &(impl DeviceInput<i32> + ?Sized),
    column_ordered_values: &mut (impl DeviceOutput<T> + ?Sized),
    csc_column_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    csc_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    action: Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
    external_buffer: &mut (impl DeviceOutput<u8> + ?Sized),
) -> Result<()> {
    let row_offsets = checked_add(row_count, 1, "CSR row offsets")?;
    let column_offsets = checked_add(col_count, 1, "CSC column offsets")?;
    input_region(context, csr_values, nonzero_count, "csr_values")?;
    input_region(context, csr_row_offsets, row_offsets, "csr_row_offsets")?;
    input_region(
        context,
        csr_column_indices,
        nonzero_count,
        "csr_column_indices",
    )?;
    input_region(
        context,
        column_ordered_values,
        nonzero_count,
        "column_ordered_values",
    )?;
    input_region(
        context,
        csc_column_offsets,
        column_offsets,
        "csc_column_offsets",
    )?;
    input_region(context, csc_row_indices, nonzero_count, "csc_row_indices")?;
    input_region(context, external_buffer, 0, "workspace")?;
    context.bind()?;
    // SAFETY: `cusparseCsr2cscEx2` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2(
            context.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_device_ptr().cast(),
            csr_row_offsets.as_device_ptr().cast(),
            csr_column_indices.as_device_ptr().cast(),
            column_ordered_values.as_mut_ptr().cast(),
            csc_column_offsets.as_mut_ptr(),
            csc_row_indices.as_mut_ptr(),
            T::data_type().into(),
            action.into(),
            index_base.into(),
            algorithm.into(),
            external_buffer.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Converts `matrix_a` in CSR, CSC, or COO format into its dense representation `matrix_b`.
/// Blocked-ELL is not currently supported.
///
/// [`sparse_to_dense_buffer_size`] returns the size of the workspace needed by [`sparse_to_dense`].
///
/// [`sparse_to_dense`] supports the following index type for representing `matrix_a`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// [`sparse_to_dense`] supports the following data types:
///
/// | `A`/`B` | Notes |
/// | --- | --- |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |  |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |  |
/// | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | Deprecated. |
/// | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) | Deprecated. |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |  |
/// | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |  |
///
/// [`sparse_to_dense`] supports the following algorithm:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SparseToDenseAlgorithm::Default`] | Default algorithm. |
///
/// [`sparse_to_dense`] has the following properties:
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run.
/// * Allows the indices of `matrix_a` to be unsorted.
///
/// [`sparse_to_dense`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn sparse_to_dense(
    context: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    // SAFETY: `cusparseSparseToDense` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSparseToDense(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn dense_to_sparse_buffer_size(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: `cusparseDenseToSparse_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_bufferSize(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "dense-to-sparse buffer size")
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn dense_to_sparse_analysis(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    // SAFETY: `cusparseDenseToSparse_analysis` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_analysis(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn dense_to_sparse_convert(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    // SAFETY: `cusparseDenseToSparse_convert` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_convert(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error for a buffer from another CUDA context, insufficient exposed storage, offset-count overflow, or counts that cannot fit in the native integer types. Also propagates context-binding and native conversion errors. Rejects a returned workspace size that cannot fit in `usize`.
pub unsafe fn csr_to_csc_buffer_size<T: DataTypeLike>(
    context: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &(impl DeviceInput<T> + ?Sized),
    csr_row_offsets: &(impl DeviceInput<i32> + ?Sized),
    csr_column_indices: &(impl DeviceInput<i32> + ?Sized),
    column_ordered_values: &mut (impl DeviceOutput<T> + ?Sized),
    csc_column_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    csc_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    action: Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
) -> Result<usize> {
    let row_offsets = checked_add(row_count, 1, "CSR row offsets")?;
    let column_offsets = checked_add(col_count, 1, "CSC column offsets")?;
    input_region(context, csr_values, nonzero_count, "csr_values")?;
    input_region(context, csr_row_offsets, row_offsets, "csr_row_offsets")?;
    input_region(
        context,
        csr_column_indices,
        nonzero_count,
        "csr_column_indices",
    )?;
    input_region(
        context,
        column_ordered_values,
        nonzero_count,
        "column_ordered_values",
    )?;
    input_region(
        context,
        csc_column_offsets,
        column_offsets,
        "csc_column_offsets",
    )?;
    input_region(context, csc_row_indices, nonzero_count, "csc_row_indices")?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: `cusparseCsr2cscEx2_bufferSize` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2_bufferSize(
            context.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_device_ptr().cast(),
            csr_row_offsets.as_device_ptr().cast(),
            csr_column_indices.as_device_ptr().cast(),
            column_ordered_values.as_mut_ptr().cast(),
            csc_column_offsets.as_mut_ptr(),
            csc_row_indices.as_mut_ptr(),
            T::data_type().into(),
            action.into(),
            index_base.into(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "csr-to-csc buffer size")
}

/// Converts compressed row pointers in CSR format into uncompressed row indices in COO format.
///
/// It can also be used to convert the array containing the compressed column indices (corresponding to CSC format) into an array of uncompressed column indices (corresponding to COO format).
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Supports CUDA graph capture.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error for a buffer from another CUDA context, insufficient exposed storage, offset-count overflow, or counts that cannot fit in the native integer types. Also propagates context-binding and native conversion errors.
pub unsafe fn csr_to_coo(
    context: &Context,
    csr_row_offsets: &(impl DeviceInput<i32> + ?Sized),
    nonzero_count: usize,
    row_count: usize,
    coo_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    index_base: IndexBase,
) -> Result<()> {
    context.bind()?;
    input_region(
        context,
        csr_row_offsets,
        checked_add(row_count, 1, "CSR row offsets")?,
        "csr_row_offsets",
    )?;
    input_region(context, coo_row_indices, nonzero_count, "coo_row_indices")?;
    // SAFETY: `cusparseXcsr2coo` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseXcsr2coo(
            context.as_raw(),
            csr_row_offsets.as_device_ptr().cast(),
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            coo_row_indices.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn sparse_to_dense_buffer_size(
    context: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: `cusparseSparseToDense_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseSparseToDense_bufferSize(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "sparse-to-dense buffer size")
}

/// Converts uncompressed row indices in COO format into compressed row pointers in CSR format.
///
/// It can also be used to convert the array containing the uncompressed column indices (corresponding to COO format) into an array of column pointers (corresponding to CSC format).
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Supports CUDA graph capture.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index
/// bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error for a buffer from another CUDA context, insufficient exposed storage, offset-count overflow, or counts that cannot fit in the native integer types. Also propagates context-binding and native conversion errors.
pub unsafe fn coo_to_csr(
    context: &Context,
    coo_row_indices: &(impl DeviceInput<i32> + ?Sized),
    row_count: usize,
    csr_row_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    index_base: IndexBase,
) -> Result<()> {
    context.bind()?;
    let nonzero_count = coo_row_indices.len();
    input_region(context, coo_row_indices, nonzero_count, "coo_row_indices")?;
    input_region(
        context,
        csr_row_offsets,
        checked_add(row_count, 1, "CSR row offsets")?,
        "csr_row_offsets",
    )?;
    // SAFETY: `cusparseXcoo2csr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseXcoo2csr(
            context.as_raw(),
            coo_row_indices.as_device_ptr().cast(),
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            csr_row_offsets.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}
