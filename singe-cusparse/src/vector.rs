use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
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
    types::{IndexBase, IndexType, IndexTypeLike},
    utility::{to_i64, to_usize},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseVectorInfo {
    pub size: usize,
    pub nonzero_count: usize,
    pub indices: DevicePtr,
    pub values: DevicePtr,
    pub index_type: IndexType,
    pub index_base: IndexBase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseVectorInfo {
    pub size: usize,
    pub values: DevicePtr,
}

#[derive(Debug)]
pub struct SparseVectorDescriptor<'a> {
    handle: sys::cusparseSpVecDescr_t,
    cuda_ctx: Arc<CudaContext>,
    _buffers: PhantomData<&'a mut ()>,
}

#[derive(Debug)]
pub struct DenseVectorDescriptor<'a> {
    handle: sys::cusparseDnVecDescr_t,
    cuda_ctx: Arc<CudaContext>,
    _buffers: PhantomData<&'a mut ()>,
}

// Vector descriptors borrow mutable device buffers and can rebind their value
// pointers, so they are movable but not shared concurrently.
unsafe impl Send for SparseVectorDescriptor<'_> {}
unsafe impl Send for DenseVectorDescriptor<'_> {}

impl<'a> SparseVectorDescriptor<'a> {
    /// Initializes a sparse vector descriptor.
    ///
    /// The descriptor borrows the device index and value buffers by pointer; the
    /// buffers must outlive descriptor use or be replaced with [`SparseVectorDescriptor::set_values`].
    pub fn create<I: IndexTypeLike, T: DataTypeLike>(
        ctx: &Context,
        size: usize,
        nonzero_count: usize,
        indices: &'a mut DeviceMemory<I>,
        values: &'a mut DeviceMemory<T>,
        index_base: IndexBase,
    ) -> Result<Self> {
        ctx.bind()?;
        let size = to_i64(size, "size")?;
        let nonzero_count = to_i64(nonzero_count, "nonzero_count")?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateSpVec(
                &raw mut handle,
                size,
                nonzero_count,
                indices.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                I::index_type().into(),
                index_base.into(),
                T::data_type().into(),
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

    /// Returns the index base of this sparse vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse vector index base.
    pub fn index_base(&self) -> Result<IndexBase> {
        let mut value = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        unsafe {
            try_ffi!(sys::cusparseSpVecGetIndexBase(
                self.as_raw_const(),
                &raw mut value
            ))?;
        }
        Ok(value.into())
    }

    /// Returns the values pointer of this sparse vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        let mut values = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseSpVecGetValues(self.as_raw(), &raw mut values))?;
        }
        Ok(values.into())
    }

    /// Sets the values pointer of this sparse vector descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    pub fn set_values<T: DataTypeLike>(&mut self, values: &'a mut DeviceMemory<T>) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseSpVecSetValues(
                self.as_raw(),
                values.as_mut_ptr().cast()
            ))?;
        }
        Ok(())
    }

    /// Returns the fields of this sparse vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if
    /// reported sizes cannot be represented as `usize`.
    pub fn info(&self) -> Result<SparseVectorInfo> {
        let mut size = 0_i64;
        let mut nnz = 0_i64;
        let mut indices = ptr::null_mut();
        let mut values = ptr::null_mut();
        let mut index_type = sys::cusparseIndexType_t::CUSPARSE_INDEX_32I;
        let mut index_base = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        let mut value_type = MaybeUninit::uninit();
        unsafe {
            try_ffi!(sys::cusparseSpVecGet(
                self.as_raw(),
                &raw mut size,
                &raw mut nnz,
                &raw mut indices,
                &raw mut values,
                &raw mut index_type,
                &raw mut index_base,
                value_type.as_mut_ptr(),
            ))?;
        }
        Ok(SparseVectorInfo {
            size: to_usize(size, "size")?,
            nonzero_count: to_usize(nnz, "nonzero_count")?,
            indices: indices.into(),
            values: values.into(),
            index_type: index_type.into(),
            index_base: index_base.into(),
        })
    }

    pub fn as_raw(&self) -> sys::cusparseSpVecDescr_t {
        self.handle
    }

    pub fn as_raw_const(&self) -> sys::cusparseConstSpVecDescr_t {
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

    /// Wraps an existing cuSPARSE sparse-vector descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpVecDescr_t`. Any device buffers
    /// referenced by the descriptor must remain valid for lifetime `'a`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseSpVecDescr_t, ctx: &Context) -> Result<Self> {
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
    /// handle with `cusparseDestroySpVec`.
    pub fn into_raw(self) -> sys::cusparseSpVecDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for SparseVectorDescriptor<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroySpVec(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse sparse vector descriptor: {err}");
            }
        }
    }
}

impl<'a> DenseVectorDescriptor<'a> {
    /// Initializes a dense vector descriptor.
    ///
    /// The descriptor borrows the device value buffer by pointer; the buffer must
    /// outlive descriptor use or be replaced with [`DenseVectorDescriptor::set_values`].
    ///
    /// # Errors
    ///
    /// Returns an error if `size` cannot be represented by cuSPARSE, if cuSPARSE
    /// cannot create the descriptor, or if it returns a null handle.
    pub fn create<T: DataTypeLike>(
        ctx: &Context,
        size: usize,
        values: &'a mut DeviceMemory<T>,
    ) -> Result<Self> {
        ctx.bind()?;
        let size = to_i64(size, "size")?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreateDnVec(
                &raw mut handle,
                size,
                values.as_mut_ptr().cast(),
                T::data_type().into(),
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

    /// Returns the values pointer of this dense vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        let mut values = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseDnVecGetValues(self.as_raw(), &raw mut values))?;
        }
        Ok(values.into())
    }

    /// Sets the values pointer of this dense vector descriptor.
    ///
    /// The pointed-to device buffer must remain valid while the descriptor uses it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the values pointer.
    pub fn set_values<T: DataTypeLike>(&mut self, values: &'a mut DeviceMemory<T>) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseDnVecSetValues(
                self.as_raw(),
                values.as_mut_ptr().cast()
            ))?;
        }
        Ok(())
    }

    /// Returns the fields of this dense vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if the
    /// reported size cannot be represented as `usize`.
    pub fn info(&self) -> Result<DenseVectorInfo> {
        let mut size = 0_i64;
        let mut values = ptr::null_mut();
        let mut value_type = MaybeUninit::uninit();
        unsafe {
            try_ffi!(sys::cusparseDnVecGet(
                self.as_raw(),
                &raw mut size,
                &raw mut values,
                value_type.as_mut_ptr(),
            ))?;
        }
        Ok(DenseVectorInfo {
            size: to_usize(size, "size")?,
            values: values.into(),
        })
    }

    pub fn as_raw(&self) -> sys::cusparseDnVecDescr_t {
        self.handle
    }

    pub fn as_raw_const(&self) -> sys::cusparseConstDnVecDescr_t {
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

    /// Wraps an existing cuSPARSE dense-vector descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseDnVecDescr_t`. Any device buffer
    /// referenced by the descriptor must remain valid for lifetime `'a`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseDnVecDescr_t, ctx: &Context) -> Result<Self> {
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
    /// handle with `cusparseDestroyDnVec`.
    pub fn into_raw(self) -> sys::cusparseDnVecDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for DenseVectorDescriptor<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroyDnVec(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse dense vector descriptor: {err}");
            }
        }
    }
}

/// Gathers the elements of `dense_vector` into `sparse_vector`.
///
/// In other words,
///
/// ```text
/// for i=0 to nnz-1
/// sparse_values[i] = dense_values[sparse_indices[i]]
/// ```
///
/// [`gather`] supports the following index type for representing `sparse_vector`:
///
/// * 32-bit indices ([`IndexType::I32`])
/// * 64-bit indices ([`IndexType::I64`])
///
/// [`gather`] supports the following data types:
///
/// | `X`/`Y` | Notes |
/// | --- | --- |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |  |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |  |
/// | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | Deprecated. |
/// | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) | Deprecated. |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |  |
/// | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |  |
///
/// [`gather`] has the following constraints:
///
/// * The arrays representing `sparse_vector` must be aligned to 16 bytes.
///
/// [`gather`] has the following properties:
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run if `sparse_vector` indices are distinct.
/// * Allows the indices of `sparse_vector` to be unsorted.
///
/// [`gather`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
pub fn gather(
    ctx: &Context,
    sparse_vector: &mut SparseVectorDescriptor,
    dense_vector: &DenseVectorDescriptor,
) -> Result<()> {
    sparse_vector.ensure_context(ctx)?;
    dense_vector.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseGather(
            ctx.as_raw(),
            dense_vector.as_raw_const(),
            sparse_vector.as_raw(),
        ))?;
    }
    Ok(())
}

/// Scatters the elements of `sparse_vector` into `dense_vector`.
///
/// In other words,
///
/// ```text
/// for i=0 to nnz-1
/// dense_values[sparse_indices[i]] = sparse_values[i]
/// ```
///
/// [`scatter`] supports the following index type for representing `sparse_vector`:
///
/// * 32-bit indices ([`IndexType::I32`])
/// * 64-bit indices ([`IndexType::I64`])
///
/// [`scatter`] supports the following data types:
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
/// [`scatter`] has the following constraints:
///
/// * The arrays representing `sparse_vector` must be aligned to 16 bytes.
///
/// [`scatter`] has the following properties:
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run if `sparse_vector` indices are distinct.
/// * Allows the indices of `sparse_vector` to be unsorted.
///
/// [`scatter`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
pub fn scatter(
    ctx: &Context,
    sparse_vector: &SparseVectorDescriptor,
    dense_vector: &mut DenseVectorDescriptor,
) -> Result<()> {
    sparse_vector.ensure_context(ctx)?;
    dense_vector.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseScatter(
            ctx.as_raw(),
            sparse_vector.as_raw_const(),
            dense_vector.as_raw(),
        ))?;
    }
    Ok(())
}
