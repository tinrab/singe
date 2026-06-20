use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit, size_of},
    ptr,
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext, data_type::DataTypeLike, memory::DeviceMemory,
    types::DevicePtr,
};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    types::{
        CsrToCscAlgorithm, DenseToSparseAlgorithm, DiagonalType, FillMode, Format, IndexBase,
        IndexTypeLike, MatrixType, Order, SparseMatrixAttribute, SparseToDenseAlgorithm,
    },
    utility::{to_i32, to_i64, to_usize},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMatrixShape {
    pub rows: usize,
    pub cols: usize,
    pub nonzero_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMatrixInfo {
    pub rows: usize,
    pub cols: usize,
    pub leading_dimension: usize,
    pub values: DevicePtr,
    pub order: Order,
}

#[derive(Debug)]
pub struct MatrixDescriptor {
    handle: sys::cusparseMatDescr_t,
}

#[derive(Debug)]
pub struct SparseMatrixDescriptor<'a> {
    handle: sys::cusparseSpMatDescr_t,
    cuda_ctx: Arc<CudaContext>,
    _buffers: PhantomData<&'a mut ()>,
}

#[derive(Debug)]
pub struct DenseMatrixDescriptor<'a> {
    handle: sys::cusparseDnMatDescr_t,
    cuda_ctx: Arc<CudaContext>,
    _buffers: PhantomData<&'a mut ()>,
}

// Legacy matrix descriptors contain descriptor attributes only, so immutable
// sharing follows the cuSPARSE descriptor contract. Data-bearing descriptors
// can rebind borrowed pointers and therefore require exclusive access.
unsafe impl Send for MatrixDescriptor {}
unsafe impl Sync for MatrixDescriptor {}
unsafe impl Send for SparseMatrixDescriptor<'_> {}
unsafe impl Send for DenseMatrixDescriptor<'_> {}

impl MatrixDescriptor {
    /// Initializes a matrix descriptor.
    /// It sets [`MatrixType`] and [`IndexBase`] to [`MatrixType::General`] and
    /// [`IndexBase::Zero`], respectively, while leaving other fields uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot create the descriptor or if it returns
    /// a null handle.
    pub fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateMatDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Returns the [`MatrixType`] field of this matrix descriptor.
    pub fn matrix_type(&self) -> MatrixType {
        unsafe { sys::cusparseGetMatType(self.as_raw()).into() }
    }

    /// Sets the [`MatrixType`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `matrix_type`.
    pub fn set_matrix_type(&mut self, matrix_type: MatrixType) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSetMatType(self.as_raw(), matrix_type.into()))?;
        }
        Ok(())
    }

    /// Returns the [`FillMode`] field of this matrix descriptor.
    pub fn fill_mode(&self) -> FillMode {
        unsafe { sys::cusparseGetMatFillMode(self.as_raw()).into() }
    }

    /// Sets the [`FillMode`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `fill_mode`.
    pub fn set_fill_mode(&mut self, fill_mode: FillMode) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSetMatFillMode(self.as_raw(), fill_mode.into()))?;
        }
        Ok(())
    }

    /// Returns the [`DiagonalType`] field of this matrix descriptor.
    pub fn diagonal_type(&self) -> DiagonalType {
        unsafe { sys::cusparseGetMatDiagType(self.as_raw()).into() }
    }

    /// Sets the [`DiagonalType`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `diagonal_type`.
    pub fn set_diagonal_type(&mut self, diagonal_type: DiagonalType) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSetMatDiagType(
                self.as_raw(),
                diagonal_type.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the [`IndexBase`] field of this matrix descriptor.
    pub fn index_base(&self) -> IndexBase {
        unsafe { sys::cusparseGetMatIndexBase(self.as_raw()).into() }
    }

    /// Sets the [`IndexBase`] field of this matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects `index_base`.
    pub fn set_index_base(&mut self, index_base: IndexBase) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSetMatIndexBase(
                self.as_raw(),
                index_base.into(),
            ))?;
        }
        Ok(())
    }

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
    pub fn into_raw(self) -> sys::cusparseMatDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for MatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroyMatDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse matrix descriptor: {err}");
            }
        }
    }
}

impl<'a> SparseMatrixDescriptor<'a> {
    /// Initializes a sparse matrix descriptor in the CSR format.
    ///
    /// The descriptor borrows the row-offset, column-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    pub fn create_csr<Offsets: IndexTypeLike, Columns: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        row_offsets: &'a mut DeviceMemory<Offsets>,
        column_indices: &'a mut DeviceMemory<Columns>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateCsr(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                row_offsets.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                Offsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Initializes a sparse matrix descriptor in the CSC format.
    ///
    /// The descriptor borrows the column-offset, row-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    pub fn create_csc<Offsets: IndexTypeLike, Rows: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        column_offsets: &'a mut DeviceMemory<Offsets>,
        row_indices: &'a mut DeviceMemory<Rows>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateCsc(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                column_offsets.as_mut_ptr().cast(),
                row_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                Offsets::index_type().into(),
                Rows::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Initializes a sparse matrix descriptor in the COO format (Structure of Arrays layout).
    ///
    /// The descriptor borrows the row-index, column-index, and value buffers by
    /// pointer; those buffers must remain valid while the descriptor uses them.
    pub fn create_coo<I: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        row_indices: &'a mut DeviceMemory<I>,
        column_indices: &'a mut DeviceMemory<I>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateCoo(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                row_indices.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                I::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Initializes a sparse matrix descriptor for the Block Compressed Row (BSR) format.
    ///
    /// The descriptor borrows the block row-offset, column-index, and value
    /// buffers by pointer; those buffers must remain valid while the descriptor uses them.
    pub fn create_bsr<Offsets: IndexTypeLike, Columns: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        block_rows: usize,
        block_cols: usize,
        block_nnz: usize,
        row_block_size: usize,
        col_block_size: usize,
        row_offsets: &'a mut DeviceMemory<Offsets>,
        column_indices: &'a mut DeviceMemory<Columns>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
        order: Order,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateBsr(
                &raw mut raw,
                to_i64(block_rows, "block_rows")?,
                to_i64(block_cols, "block_cols")?,
                to_i64(block_nnz, "block_nnz")?,
                to_i64(row_block_size, "row_block_size")?,
                to_i64(col_block_size, "col_block_size")?,
                row_offsets.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                Offsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
                order.into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Initializes a sparse matrix descriptor for the Blocked-Ellpack (ELL) format.
    ///
    /// Blocked-ELL column indices are in the range `0..cols / ell_block_size`.
    /// The array can contain `-1` values for indicating empty blocks.
    /// The descriptor borrows the column-index and value buffers by pointer;
    /// those buffers must remain valid while the descriptor uses them.
    pub fn create_blocked_ell<I: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        rows: usize,
        cols: usize,
        block_size: usize,
        ell_cols: usize,
        column_indices: &'a mut DeviceMemory<I>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateBlockedEll(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(block_size, "block_size")?,
                to_i64(ell_cols, "ell_cols")?,
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                I::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Initializes a sparse matrix descriptor for the Sliced Ellpack (SELL) format.
    ///
    /// The descriptor borrows the slice-offset, column-index, and value buffers
    /// by pointer; those buffers must remain valid while the descriptor uses them.
    pub fn create_sliced_ell<
        SliceOffsets: IndexTypeLike,
        Columns: IndexTypeLike,
        T: DataTypeLike,
    >(
        ctx: &Context,
        rows: usize,
        cols: usize,
        nonzero_count: usize,
        values_size: usize,
        slice_size: usize,
        slice_offsets: &'a mut DeviceMemory<SliceOffsets>,
        column_indices: &'a mut DeviceMemory<Columns>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut raw = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateSlicedEll(
                &raw mut raw,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(nonzero_count, "nonzero_count")?,
                to_i64(values_size, "values_size")?,
                to_i64(slice_size, "slice_size")?,
                slice_offsets.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                SliceOffsets::index_type().into(),
                Columns::index_type().into(),
                index_base.into(),
                T::data_type().into(),
            ))?;
        }
        ensure_spmat(raw, ctx)
    }

    /// Returns the format of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix format.
    pub fn format(&self) -> Result<Format> {
        let mut format = sys::cusparseFormat_t::CUSPARSE_FORMAT_CSR;
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
        let mut index_base = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
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
        let mut values = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseSpMatGetValues(self.as_raw(), &raw mut values))?;
        }
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Sets the values pointer of this sparse matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    pub fn set_values<T: DataTypeLike>(&mut self, values: &'a mut DeviceMemory<T>) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSpMatSetValues(
                self.as_raw(),
                values.as_mut_ptr().cast(),
            ))?;
        }
        Ok(())
    }

    /// Returns the shape of this sparse matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse matrix shape or if
    /// the reported values cannot be represented as `usize`.
    pub fn shape(&self) -> Result<SparseMatrixShape> {
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut nnz = 0_i64;
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
        let mut batch_count = 0;
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
    pub fn set_csr_strided_batch(
        &mut self,
        batch_count: usize,
        offsets_batch_stride: usize,
        columns_values_batch_stride: usize,
    ) -> Result<()> {
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
    pub fn set_bsr_strided_batch(
        &mut self,
        batch_count: usize,
        offsets_batch_stride: usize,
        columns_batch_stride: usize,
        values_batch_stride: usize,
    ) -> Result<()> {
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

    pub fn fill_mode(&self) -> Result<FillMode> {
        let raw: sys::cusparseFillMode_t = self.raw_attribute(SparseMatrixAttribute::FillMode)?;
        Ok(raw.into())
    }

    pub fn set_fill_mode(&mut self, fill_mode: FillMode) -> Result<()> {
        self.set_attribute_raw::<sys::cusparseFillMode_t>(
            SparseMatrixAttribute::FillMode,
            fill_mode.into(),
        )
    }

    pub fn diagonal_type(&self) -> Result<DiagonalType> {
        let raw: sys::cusparseDiagType_t =
            self.raw_attribute(SparseMatrixAttribute::DiagonalType)?;
        Ok(raw.into())
    }

    pub fn set_diagonal_type(&mut self, diagonal_type: DiagonalType) -> Result<()> {
        self.set_attribute_raw::<sys::cusparseDiagType_t>(
            SparseMatrixAttribute::DiagonalType,
            diagonal_type.into(),
        )
    }

    /// Sets the pointers of this CSR sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    pub fn set_csr_pointers(
        &mut self,
        row_offsets: &'a mut DeviceMemory<impl IndexTypeLike>,
        column_indices: &'a mut DeviceMemory<impl IndexTypeLike>,
        values: &'a mut DeviceMemory<impl DataTypeLike>,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseCsrSetPointers(
                self.as_raw(),
                row_offsets.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
            ))?;
        }
        Ok(())
    }

    /// Sets the pointers of this CSC sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    pub fn set_csc_pointers(
        &mut self,
        column_offsets: &'a mut DeviceMemory<impl IndexTypeLike>,
        row_indices: &'a mut DeviceMemory<impl IndexTypeLike>,
        values: &'a mut DeviceMemory<impl DataTypeLike>,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseCscSetPointers(
                self.as_raw(),
                column_offsets.as_mut_ptr().cast(),
                row_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
            ))?;
        }
        Ok(())
    }

    /// Sets the pointers of this COO sparse matrix descriptor.
    ///
    /// The pointed-to device buffers must remain valid while the descriptor uses them.
    pub fn set_coo_pointers(
        &mut self,
        row_indices: &'a mut DeviceMemory<impl IndexTypeLike>,
        column_indices: &'a mut DeviceMemory<impl IndexTypeLike>,
        values: &'a mut DeviceMemory<impl DataTypeLike>,
    ) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseCooSetPointers(
                self.as_raw(),
                row_indices.as_mut_ptr().cast(),
                column_indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
            ))?;
        }
        Ok(())
    }

    pub fn as_raw(&self) -> sys::cusparseSpMatDescr_t {
        self.handle
    }

    pub fn as_raw_const(&self) -> sys::cusparseConstSpMatDescr_t {
        self.handle.cast_const()
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    pub(crate) fn ensure_context(&self, ctx: &Context) -> Result<()> {
        if self.cuda_ctx.as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE sparse-matrix descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpMatDescr_t`. Any device buffers
    /// referenced by the descriptor must remain valid for lifetime `'a`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseSpMatDescr_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            _buffers: PhantomData,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseDestroySpMat`.
    pub fn into_raw(self) -> sys::cusparseSpMatDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }

    fn raw_attribute<T: Copy>(&self, attribute: SparseMatrixAttribute) -> Result<T> {
        let mut value = MaybeUninit::<T>::uninit();
        unsafe {
            try_ffi!(sys::cusparseSpMatGetAttribute(
                self.as_raw_const(),
                attribute.into(),
                value.as_mut_ptr().cast(),
                u64::try_from(size_of::<T>()).map_err(|_| Error::OutOfRange {
                    name: "attribute size".into(),
                })?,
            ))?;
            Ok(value.assume_init())
        }
    }

    fn set_attribute_raw<T>(&mut self, attribute: SparseMatrixAttribute, value: T) -> Result<()> {
        let mut value = value;
        unsafe {
            try_ffi!(sys::cusparseSpMatSetAttribute(
                self.as_raw(),
                attribute.into(),
                &raw mut value as *mut _,
                u64::try_from(size_of::<T>()).map_err(|_| Error::OutOfRange {
                    name: "attribute size".into(),
                })?,
            ))?;
        }
        Ok(())
    }
}

impl Drop for SparseMatrixDescriptor<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroySpMat(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse sparse matrix descriptor: {err}");
            }
        }
    }
}

impl<'a> DenseMatrixDescriptor<'a> {
    /// Initializes a dense matrix descriptor.
    ///
    /// The descriptor borrows the device value buffer by pointer; the buffer must
    /// remain valid while the descriptor uses it.
    pub fn create<T: DataTypeLike>(
        ctx: &Context,
        rows: usize,
        cols: usize,
        leading_dimension: usize,
        values: &'a mut DeviceMemory<T>,
        order: Order,
    ) -> Result<Self> {
        ctx.bind()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateDnMat(
                &raw mut handle,
                to_i64(rows, "rows")?,
                to_i64(cols, "cols")?,
                to_i64(leading_dimension, "leading_dimension")?,
                values.as_mut_ptr().cast(),
                T::data_type().into(),
                order.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            _buffers: PhantomData,
        })
    }

    /// Returns the values pointer of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        let mut values = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseDnMatGetValues(self.as_raw(), &raw mut values))?;
        }
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Sets the values pointer of this dense matrix descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    pub fn set_values<T: DataTypeLike>(&mut self, values: &'a mut DeviceMemory<T>) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseDnMatSetValues(
                self.as_raw(),
                values.as_mut_ptr().cast(),
            ))?;
        }
        Ok(())
    }

    /// Returns the fields of this dense matrix descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if
    /// reported dimensions cannot be represented as `usize`.
    pub fn info(&self) -> Result<DenseMatrixInfo> {
        let mut rows = 0_i64;
        let mut cols = 0_i64;
        let mut leading_dimension = 0_i64;
        let mut values = ptr::null_mut();
        let mut value_type = MaybeUninit::uninit();
        let mut order = sys::cusparseOrder_t::CUSPARSE_ORDER_ROW;
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
        let mut batch_count = 0;
        let mut batch_stride = 0_i64;
        unsafe {
            try_ffi!(sys::cusparseDnMatGetStridedBatch(
                self.as_raw_const(),
                &raw mut batch_count,
                &raw mut batch_stride,
            ))?;
        }
        Ok((batch_count, to_usize(batch_stride, "batch_stride")?))
    }

    pub fn as_raw(&self) -> sys::cusparseDnMatDescr_t {
        self.handle
    }

    pub fn as_raw_const(&self) -> sys::cusparseConstDnMatDescr_t {
        self.handle.cast_const()
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    pub(crate) fn ensure_context(&self, ctx: &Context) -> Result<()> {
        if self.cuda_ctx.as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE dense-matrix descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseDnMatDescr_t`. Any device buffer
    /// referenced by the descriptor must remain valid for lifetime `'a`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseDnMatDescr_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            _buffers: PhantomData,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseDestroyDnMat`.
    pub fn into_raw(self) -> sys::cusparseDnMatDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for DenseMatrixDescriptor<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroyDnMat(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse dense matrix descriptor: {err}");
            }
        }
    }
}

fn ensure_spmat<'a>(
    handle: sys::cusparseSpMatDescr_t,
    ctx: &Context,
) -> Result<SparseMatrixDescriptor<'a>> {
    if handle.is_null() {
        Err(Error::NullHandle)
    } else {
        Ok(SparseMatrixDescriptor {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            _buffers: PhantomData,
        })
    }
}

/// Converts a sparse matrix in CSR format into a sparse matrix in CSC format.
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
/// | [`Action::Symbolic`](crate::types::Action::Symbolic) | Compute the structure of the CSC output matrix (offset, row indices). |
/// | [`Action::Numeric`](crate::types::Action::Numeric) | Compute the structure of the CSC output matrix and copy the values. |
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
pub fn csr_to_csc<T: DataTypeLike>(
    ctx: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &DeviceMemory<T>,
    csr_row_offsets: &DeviceMemory<i32>,
    csr_column_indices: &DeviceMemory<i32>,
    csc_values: &mut DeviceMemory<T>,
    csc_column_offsets: &mut DeviceMemory<i32>,
    csc_row_indices: &mut DeviceMemory<i32>,
    action: crate::types::Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
    external_buffer: &mut DeviceMemory<u8>,
) -> Result<()> {
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2(
            ctx.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_ptr().cast(),
            csr_row_offsets.as_ptr() as _,
            csr_column_indices.as_ptr() as _,
            csc_values.as_mut_ptr().cast(),
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
pub fn sparse_to_dense(
    ctx: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseSparseToDense(
            ctx.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn dense_to_sparse_buffer_size(
    ctx: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    ctx.bind()?;
    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_bufferSize(
            ctx.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "dense-to-sparse buffer size")
}

pub fn dense_to_sparse_analysis(
    ctx: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_analysis(
            ctx.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn dense_to_sparse_convert(
    ctx: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_convert(
            ctx.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn csr_to_csc_buffer_size<T: DataTypeLike>(
    ctx: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &DeviceMemory<T>,
    csr_row_offsets: &DeviceMemory<i32>,
    csr_column_indices: &DeviceMemory<i32>,
    csc_values: &mut DeviceMemory<T>,
    csc_column_offsets: &mut DeviceMemory<i32>,
    csc_row_indices: &mut DeviceMemory<i32>,
    action: crate::types::Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
) -> Result<usize> {
    ctx.bind()?;
    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2_bufferSize(
            ctx.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_ptr().cast(),
            csr_row_offsets.as_ptr() as _,
            csr_column_indices.as_ptr() as _,
            csc_values.as_mut_ptr().cast(),
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
pub fn csr_to_coo(
    ctx: &Context,
    csr_row_offsets: &DeviceMemory<i32>,
    nonzero_count: usize,
    row_count: usize,
    coo_row_indices: &mut DeviceMemory<i32>,
    index_base: IndexBase,
) -> Result<()> {
    ctx.bind()?;
    if csr_row_offsets.len() < row_count.saturating_add(1) || coo_row_indices.len() < nonzero_count
    {
        return Err(Error::OutOfRange {
            name: "csr/coo buffer length".into(),
        });
    }
    unsafe {
        try_ffi!(sys::cusparseXcsr2coo(
            ctx.as_raw(),
            csr_row_offsets.as_ptr() as _,
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            coo_row_indices.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}

pub fn sparse_to_dense_buffer_size(
    ctx: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    ctx.bind()?;
    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSparseToDense_bufferSize(
            ctx.as_raw(),
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
pub fn coo_to_csr(
    ctx: &Context,
    coo_row_indices: &DeviceMemory<i32>,
    row_count: usize,
    csr_row_offsets: &mut DeviceMemory<i32>,
    index_base: IndexBase,
) -> Result<()> {
    ctx.bind()?;
    let nonzero_count = coo_row_indices.len();
    if csr_row_offsets.len() < row_count.saturating_add(1) {
        return Err(Error::OutOfRange {
            name: "csr_row_offsets length".into(),
        });
    }
    unsafe {
        try_ffi!(sys::cusparseXcoo2csr(
            ctx.as_raw(),
            coo_row_indices.as_ptr() as _,
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            csr_row_offsets.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}
