mod conversion;

use std::{
    marker::PhantomData,
    mem,
    mem::{MaybeUninit, size_of},
    ptr,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataTypeLike,
    types::DevicePtr,
    view::{DeviceOutputView, DeviceRegion},
};

pub use self::conversion::{
    coo_to_csr_async, csr_to_coo_async, csr_to_csc_async, csr_to_csc_workspace_size,
    dense_to_sparse_analysis_async, dense_to_sparse_convert_async, dense_to_sparse_workspace_size,
    sparse_to_dense_async, sparse_to_dense_workspace_size,
};
use crate::{
    context::Context,
    descriptor::RawDescriptor,
    error::{Error, Result},
    sys, try_ffi,
    types::{
        DiagonalType, FillMode, Format, IndexBase, IndexTypeLike, Order, SparseMatrixAttribute,
    },
    utility::{
        checked_add, checked_mul, matrix_extent, output_region, output_region_for, to_i32, to_i64,
        to_u64, to_usize,
    },
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

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: cuSPARSE lets any thread use and destroy the descriptor one call at a time, and its storage regions are borrows that move with it; it is not `Sync`, since rebinding mutates native state.
unsafe impl Send for SparseMatrixDescriptor<'_> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
// SAFETY: cuSPARSE lets any thread use and destroy the descriptor one call at a time, and its storage regions are borrows that move with it; it is not `Sync`, since rebinding mutates native state.
unsafe impl Send for DenseMatrixDescriptor<'_> {}

impl<'a> SparseMatrixDescriptor<'a> {
    /// Initializes a sparse matrix descriptor in the CSR format.
    ///
    /// The descriptor borrows the row-offset, column-index, and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if a supplied array belongs to another CUDA context or is smaller than the required storage, a count or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
    /// The descriptor borrows the column-offset, row-index, and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if a supplied array belongs to another CUDA context or is smaller than the required storage, a count or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
    /// The descriptor borrows the row-index, column-index, and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if a supplied array belongs to another CUDA context or is smaller than the required storage, a count or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
    /// The descriptor borrows the block row-offset, column-index, and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if a supplied array belongs to another CUDA context or is smaller than the required storage, a count, block value count, or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
    /// The descriptor borrows the column-index and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if `block_size` is zero or does not divide both `rows` and `ell_cols`, a supplied array belongs to another CUDA context or is smaller than the required storage, a count or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
    /// The descriptor borrows the slice-offset, column-index, and value buffers by pointer; those buffers must remain valid while the descriptor uses them.
    ///
    /// # Errors
    ///
    /// Returns an error if `slice_size` is zero, a supplied array belongs to another CUDA context or is smaller than the required storage, a count or storage size overflows or does not fit in `i64`, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        // SAFETY: `raw` is a writable local, and each supplied array was checked above to belong to this context and hold the storage the shape needs; the descriptor keeps borrowing them for `'a`.
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
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetFormat(
                self.as_raw_const(),
                &raw mut format,
            ))?;
        }
        Ok(format.try_into()?)
    }

    /// Returns the index base of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix index base.
    pub fn index_base(&self) -> Result<IndexBase> {
        self.cuda_context.bind()?;
        let mut index_base = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetIndexBase(
                self.as_raw_const(),
                &raw mut index_base,
            ))?;
        }
        Ok(index_base.try_into()?)
    }

    /// Returns the values pointer of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        self.cuda_context.bind()?;
        let mut values = ptr::null_mut();
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: cuSPARSE reports the address bound to the descriptor, which is null or in storage the descriptor borrows; the value is only recorded.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this sparse matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch extents, and preprocessing protocol.
    /// Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` belongs to another CUDA context, the context cannot be bound, or cuSPARSE rejects the values pointer.
    /// On failure the consumed descriptor is destroyed while both sets of regions remain borrowed.
    pub unsafe fn rebind_values<'b, T: DataTypeLike>(
        self,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<SparseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: `into_raw` hands over the live, uniquely owned handle and its context, and `'a: 'b` keeps the old regions borrowed for the new lifetime.
        let mut rebound = unsafe { SparseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `rebound` owns the live descriptor with its context bound, and each replacement array was checked to belong to that context; the caller completed earlier accesses and matches the element types.
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
    /// Returns an error if cuSPARSE cannot report the sparse matrix shape or if the reported values cannot be represented as `usize`.
    pub fn shape(&self) -> Result<SparseMatrixShape> {
        self.cuda_context.bind()?;
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut nnz = 0_i64;
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
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
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
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
    /// The storage is checked for the first batch only; the [raw execution contract](crate#raw-execution) requires it to cover every batch.
    ///
    /// # Errors
    ///
    /// Returns an error if either argument cannot be represented by cuSPARSE or if cuSPARSE rejects the strided batch settings.
    pub fn set_coo_strided_batch(&mut self, batch_count: usize, batch_stride: usize) -> Result<()> {
        // SAFETY: `&mut self` owns the live descriptor exclusively, and the call only records the batch fields on the host.
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
    /// The storage is checked for the first batch only; the [raw execution contract](crate#raw-execution) requires it to cover every batch.
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
        // SAFETY: `&mut self` owns the live descriptor exclusively, and the call only records the batch fields on the host.
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
    /// The storage is checked for the first batch only; the [raw execution contract](crate#raw-execution) requires it to cover every batch.
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
        // SAFETY: `&mut self` owns the live descriptor exclusively, and the call only records the batch fields on the host.
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
    /// Returns an error if the context cannot be bound, cuSPARSE rejects the query, or it reports an unknown fill mode.
    pub fn fill_mode(&self) -> Result<FillMode> {
        let value = self.raw_attribute(SparseMatrixAttribute::FillMode)?;
        FillMode::try_from(value).map_err(|_| Error::UnknownEnumValue {
            name: "fill mode".into(),
            value: value.into(),
        })
    }

    /// Sets which triangular part of the sparse matrix an operation uses.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or cuSPARSE rejects the attribute.
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
    /// Returns an error if the context cannot be bound, cuSPARSE rejects the query, or it reports an unknown diagonal type.
    pub fn diagonal_type(&self) -> Result<DiagonalType> {
        let value = self.raw_attribute(SparseMatrixAttribute::DiagonalType)?;
        DiagonalType::try_from(value).map_err(|_| Error::UnknownEnumValue {
            name: "diagonal type".into(),
            value: value.into(),
        })
    }

    /// Sets whether the sparse matrix diagonal is explicit or implicitly unit-valued.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or cuSPARSE rejects the attribute.
    pub fn set_diagonal_type(&mut self, diagonal_type: DiagonalType) -> Result<()> {
        self.set_attribute_raw::<sys::cusparseDiagType_t>(
            SparseMatrixAttribute::DiagonalType,
            diagonal_type.into(),
        )
    }

    /// Consumes and rebinds the pointers of this CSR sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch extents, and preprocessing protocol.
    /// Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context, the context cannot be bound, or cuSPARSE rejects the pointers.
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

        // SAFETY: `into_raw` hands over the live, uniquely owned handle and its context, and `'a: 'b` keeps the old regions borrowed for the new lifetime.
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
        // SAFETY: `rebound` owns the live descriptor with its context bound, and each replacement array was checked to belong to that context; the caller completed earlier accesses and matches the element types.
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
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch extents, and preprocessing protocol.
    /// Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context, the context cannot be bound, or cuSPARSE rejects the pointers.
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

        // SAFETY: `into_raw` hands over the live, uniquely owned handle and its context, and `'a: 'b` keeps the old regions borrowed for the new lifetime.
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
        // SAFETY: `rebound` owns the live descriptor with its context bound, and each replacement array was checked to belong to that context; the caller completed earlier accesses and matches the element types.
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
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch extents, and preprocessing protocol.
    /// Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if a replacement array belongs to another CUDA context, the context cannot be bound, or cuSPARSE rejects the pointers.
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

        // SAFETY: `into_raw` hands over the live, uniquely owned handle and its context, and `'a: 'b` keeps the old regions borrowed for the new lifetime.
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
        // SAFETY: `rebound` owns the live descriptor with its context bound, and each replacement array was checked to belong to that context; the caller completed earlier accesses and matches the element types.
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
            return Err(Error::context_mismatch("sparse matrix descriptor"));
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE sparse-matrix descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a live, uniquely owned sparse matrix descriptor created in `parts.cuda_context`.
    /// Its pointer bindings and regions must describe live allocations in that context with the required exclusive borrows for lifetime `'a`.
    /// An empty region list leaves those raw obligations to the caller.
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

    /// Consumes the descriptor and returns the raw cuSPARSE handle without destroying it.
    ///
    /// The caller must destroy the returned handle with [`sys::cusparseDestroySpMat`] or adopt it again with [`SparseMatrixDescriptor::from_raw`].
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
        // SAFETY: `self` owns the live descriptor with its context bound, and `value` is a writable `u32` whose size is passed with it, matching the 32-bit fill-mode and diagonal-type attributes.
        unsafe {
            try_ffi!(sys::cusparseSpMatGetAttribute(
                self.as_raw_const(),
                attribute.into(),
                ptr::from_mut(&mut value).cast(),
                to_u64(size_of::<u32>(), "attribute size")?,
            ))?;
        }
        Ok(value)
    }

    fn set_attribute_raw<T>(&mut self, attribute: SparseMatrixAttribute, value: T) -> Result<()> {
        self.cuda_context.bind()?;
        let mut value = value;
        // SAFETY: `&mut self` owns the live descriptor with its context bound, and `value` is a readable `T` whose size is passed with it; callers pass the attribute's native enum type.
        unsafe {
            try_ffi!(sys::cusparseSpMatSetAttribute(
                self.as_raw(),
                attribute.into(),
                (&raw mut value).cast(),
                to_u64(size_of::<T>(), "attribute size")?,
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

        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cusparse sparse matrix descriptor",
                &error,
            );
            return;
        }

        // SAFETY: the descriptor is live and uniquely owned, its context is bound above, and it is never used again.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroySpMat(self.as_raw_const())) {
                report_cleanup_failure("destroy", "cusparse sparse matrix descriptor", &error);
            }
        }
    }
}

impl<'a> DenseMatrixDescriptor<'a> {
    /// Initializes a dense matrix descriptor.
    ///
    /// The descriptor borrows the device value buffer by pointer; the buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if `leading_dimension` is smaller than the minor dimension, the storage extent overflows, a dimension does not fit in `i64`, `values` belongs to another CUDA context or is too small, the context cannot be bound, or cuSPARSE fails to create the descriptor or returns a null handle.
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
        let extent = matrix_extent(rows, cols, leading_dimension, order)?;
        let values = output_region(context, values, extent, "values")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local, and `values` was checked above to belong to this context and cover the extent of `rows`, `cols`, and `leading_dimension`; the descriptor keeps borrowing it for `'a`.
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
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
        unsafe {
            try_ffi!(sys::cusparseDnMatGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: cuSPARSE reports the address bound to the descriptor, which is null or in storage the descriptor borrows; the value is only recorded.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this dense matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Safety
    ///
    /// Bindings must match the descriptor's native element types, format, batch extents, and preprocessing protocol.
    /// Complete previous accesses before rebinding.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` belongs to another CUDA context, the context cannot be bound, or cuSPARSE rejects the values pointer.
    /// On failure the consumed descriptor is destroyed while both sets of regions remain borrowed.
    pub unsafe fn rebind_values<'b, T: DataTypeLike>(
        self,
        values: Option<DeviceOutputView<'b, T>>,
    ) -> Result<DenseMatrixDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: `into_raw` hands over the live, uniquely owned handle and its context, and `'a: 'b` keeps the old regions borrowed for the new lifetime.
        let mut rebound = unsafe { DenseMatrixDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `rebound` owns the live descriptor with its context bound, and each replacement array was checked to belong to that context; the caller completed earlier accesses and matches the element types.
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
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if reported dimensions cannot be represented as `usize`.
    pub fn info(&self) -> Result<DenseMatrixInfo> {
        self.cuda_context.bind()?;
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut leading_dimension = 0_i64;
        let mut values = ptr::null_mut();
        let mut value_type = MaybeUninit::uninit();
        let mut order = sys::cusparseOrder_t::CUSPARSE_ORDER_ROW;
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
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
            // SAFETY: cuSPARSE reports the address bound to the descriptor, which is null or in storage the descriptor borrows; the value is only recorded.
            values: unsafe { DevicePtr::from_raw(values.cast()) },
            order: order.try_into()?,
        })
    }

    /// Sets the number of batches and the batch stride of this dense matrix descriptor.
    ///
    /// The storage is checked for the first batch only; the [raw execution contract](crate#raw-execution) requires it to cover every batch.
    ///
    /// # Errors
    ///
    /// Returns an error if either argument cannot be represented by cuSPARSE or if cuSPARSE rejects the strided batch settings.
    pub fn set_strided_batch(&mut self, batch_count: usize, batch_stride: usize) -> Result<()> {
        // SAFETY: `&mut self` owns the live descriptor exclusively, and the call only records the batch fields on the host.
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
    /// Returns an error if cuSPARSE cannot report the strided batch settings or if the reported stride cannot be represented as `usize`.
    pub fn strided_batch(&self) -> Result<(i32, usize)> {
        self.cuda_context.bind()?;
        let mut batch_count = 0;
        let mut batch_stride = 0_i64;
        // SAFETY: `self` owns the live descriptor, its context is bound above, and every output is a writable local.
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
            return Err(Error::context_mismatch("dense matrix descriptor"));
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE dense-matrix descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a live, uniquely owned dense matrix descriptor created in `parts.cuda_context`.
    /// Its pointer bindings and regions must describe live allocations in that context with the required exclusive borrows for lifetime `'a`.
    /// An empty region list leaves those raw obligations to the caller.
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

    /// Consumes the descriptor and returns the raw cuSPARSE handle without destroying it.
    ///
    /// The caller must destroy the returned handle with [`sys::cusparseDestroyDnMat`] or adopt it again with [`DenseMatrixDescriptor::from_raw`].
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

        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cusparse dense matrix descriptor",
                &error,
            );
            return;
        }

        // SAFETY: the descriptor is live and uniquely owned, its context is bound above, and it is never used again.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroyDnMat(self.as_raw_const())) {
                report_cleanup_failure("destroy", "cusparse dense matrix descriptor", &error);
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
