use std::{marker::PhantomData, mem, mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataTypeLike,
    types::DevicePtr,
    view::{DeviceOutputView, DeviceRegion},
};

use crate::{
    context::Context,
    descriptor::{RawDescriptor, output_region, output_region_for},
    error::{Error, Result},
    sys, try_ffi,
    types::{IndexBase, IndexType, IndexTypeLike},
    utility::{to_i64, to_usize},
};

/// Size, storage addresses, and index metadata reported by a sparse vector descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseVectorInfo {
    /// Logical number of vector elements.
    pub size: usize,
    /// Number of stored sparse entries.
    pub nonzero_count: usize,
    /// Device address of the sparse indices.
    pub indices: DevicePtr,
    /// Device address of the stored values.
    pub values: DevicePtr,
    /// Native representation of each index.
    pub index_type: IndexType,
    /// Zero-based or one-based index convention.
    pub index_base: IndexBase,
}

/// Size and value address reported by a dense vector descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseVectorInfo {
    /// Logical number of vector elements.
    pub size: usize,
    /// Device address of the vector values.
    pub values: DevicePtr,
}

/// Owns a sparse-vector descriptor tied to a CUDA context and borrowed device storage.
#[derive(Debug)]
pub struct SparseVectorDescriptor<'a> {
    handle: sys::cusparseSpVecDescr_t,
    cuda_context: Arc<CudaContext>,
    regions: Vec<Option<DeviceRegion<'a>>>,
    _buffers: PhantomData<&'a mut ()>,
}

/// Owns a dense-vector descriptor tied to a CUDA context and borrowed device storage.
#[derive(Debug)]
pub struct DenseVectorDescriptor<'a> {
    handle: sys::cusparseDnVecDescr_t,
    cuda_context: Arc<CudaContext>,
    regions: Vec<Option<DeviceRegion<'a>>>,
    _buffers: PhantomData<&'a mut ()>,
}

// SAFETY: vector descriptors borrow mutable device buffers and can rebind their value
// pointers, so they are movable but not shared concurrently.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for SparseVectorDescriptor<'_> {}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for DenseVectorDescriptor<'_> {}

impl<'a> SparseVectorDescriptor<'a> {
    /// Initializes a sparse vector descriptor.
    ///
    /// The descriptor borrows the device index and value buffers by pointer; the
    /// buffers must outlive descriptor use or be replaced with [`SparseVectorDescriptor::rebind_values`].
    ///
    /// # Errors
    ///
    /// Returns an error if supplied index or value storage is too small or belongs to another CUDA context, or counts cannot fit in native integers. Also propagates context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create<I: IndexTypeLike, T: DataTypeLike>(
        context: &Context,
        size: usize,
        nonzero_count: usize,
        indices: Option<DeviceOutputView<'a, I>>,
        values: Option<DeviceOutputView<'a, T>>,
        index_base: IndexBase,
    ) -> Result<Self> {
        let () = I::LAYOUT;
        let () = T::LAYOUT;

        context.bind()?;
        let indices = indices
            .map(|view| output_region(context, view, nonzero_count, "indices"))
            .transpose()?;
        let values = values
            .map(|view| output_region(context, view, nonzero_count, "values"))
            .transpose()?;

        let size = to_i64(size, "size")?;
        let nonzero_count = to_i64(nonzero_count, "nonzero_count")?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseCreateSpVec` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateSpVec(
                &raw mut handle,
                size,
                nonzero_count,
                indices.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast(),
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
            cuda_context: Arc::clone(context.cuda_context()),
            regions: vec![indices, values],
            _buffers: PhantomData,
        })
    }

    /// Returns the index base of this sparse vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the sparse vector index base.
    pub fn index_base(&self) -> Result<IndexBase> {
        self.cuda_context.bind()?;
        let mut value = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        // SAFETY: `cusparseSpVecGetIndexBase` receives valid output locations.
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
        self.cuda_context.bind()?;
        let mut values = ptr::null_mut();
        // SAFETY: `cusparseSpVecGetValues` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpVecGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this sparse vector descriptor.
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
    ) -> Result<SparseVectorDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { SparseVectorDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `cusparseSpVecSetValues` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseSpVecSetValues(
                rebound.as_raw(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast()
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

    /// Returns the fields of this sparse vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if
    /// reported sizes cannot be represented as `usize`.
    pub fn info(&self) -> Result<SparseVectorInfo> {
        self.cuda_context.bind()?;
        let mut size = 0_i64;
        let mut nnz = 0_i64;
        let mut indices = ptr::null_mut();
        let mut values = ptr::null_mut();
        let mut index_type = sys::cusparseIndexType_t::CUSPARSE_INDEX_32I;
        let mut index_base = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
        let mut value_type = MaybeUninit::uninit();
        // SAFETY: `cusparseSpVecGet` receives live handles owned by the wrappers and valid output locations.
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
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            indices: unsafe { DevicePtr::from_raw(indices.cast()) },
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            values: unsafe { DevicePtr::from_raw(values.cast()) },
            index_type: index_type.into(),
            index_base: index_base.into(),
        })
    }

    /// Returns the native descriptor handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpVecDescr_t {
        self.handle
    }

    /// Returns the native handle in its const-descriptor form.
    #[must_use]
    pub fn as_raw_const(&self) -> sys::cusparseConstSpVecDescr_t {
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

    /// Wraps an existing cuSPARSE sparse-vector descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned, valid `cusparseSpVecDescr_t` associated
    /// with `parts.cuda_context`. Its pointer bindings and regions must describe
    /// live allocations in that context with the required exclusive borrows for
    /// lifetime `'a`. An empty region list leaves those raw obligations to the caller.
    /// No other owner may destroy the handle after transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawDescriptor<'a, sys::cusparseSpVecDescr_t>) -> Result<Self> {
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
    /// handle with `cusparseDestroySpVec`.
    #[must_use]
    pub fn into_raw(mut self) -> RawDescriptor<'a, sys::cusparseSpVecDescr_t> {
        RawDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
            regions: mem::take(&mut self.regions),
        }
    }
}

impl Drop for SparseVectorDescriptor<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if self.cuda_context.bind().is_err() {
            return;
        }

        // SAFETY: `cusparseDestroySpVec` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroySpVec(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse sparse vector descriptor: {error}");
            }
        }
    }
}

impl<'a> DenseVectorDescriptor<'a> {
    /// Initializes a dense vector descriptor.
    ///
    /// The descriptor borrows the device value buffer by pointer; the buffer must
    /// outlive descriptor use or be replaced with [`DenseVectorDescriptor::rebind_values`].
    ///
    /// # Errors
    ///
    /// Returns an error if `size` cannot be represented by cuSPARSE, if cuSPARSE
    /// cannot create the descriptor, or if it returns a null handle.
    pub fn create<T: DataTypeLike>(
        context: &Context,
        size: usize,
        values: DeviceOutputView<'a, T>,
    ) -> Result<Self> {
        let () = T::LAYOUT;

        context.bind()?;
        let values = output_region(context, values, size, "values")?;

        let size = to_i64(size, "size")?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseCreateDnVec` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseCreateDnVec(
                &raw mut handle,
                size,
                values.as_ptr().cast(),
                T::data_type().into(),
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

    /// Returns the values pointer of this dense vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the values pointer.
    pub fn values(&self) -> Result<DevicePtr> {
        self.cuda_context.bind()?;
        let mut values = ptr::null_mut();
        // SAFETY: `cusparseDnVecGetValues` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseDnVecGetValues(self.as_raw(), &raw mut values))?;
        }
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        Ok(unsafe { DevicePtr::from_raw(values.cast()) })
    }

    /// Consumes and rebinds the values pointer of this dense vector descriptor.
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
    ) -> Result<DenseVectorDescriptor<'b>>
    where
        'a: 'b,
    {
        let () = T::LAYOUT;
        // SAFETY: ownership is transferred once; all original borrows outlive the new lifetime.
        let mut rebound = unsafe { DenseVectorDescriptor::from_raw(self.into_raw())? };
        rebound.cuda_context.bind()?;
        let values = values
            .map(|view| output_region_for(&rebound.cuda_context, view, 0, "values"))
            .transpose()?;

        rebound.regions.extend([values]);
        // SAFETY: `cusparseDnVecSetValues` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseDnVecSetValues(
                rebound.as_raw(),
                values.map_or(ptr::null_mut(), DeviceRegion::as_ptr).cast()
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

    /// Returns the fields of this dense vector descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot report the descriptor fields or if the
    /// reported size cannot be represented as `usize`.
    pub fn info(&self) -> Result<DenseVectorInfo> {
        self.cuda_context.bind()?;
        let mut size = 0_i64;
        let mut values = ptr::null_mut();
        let mut value_type = MaybeUninit::uninit();
        // SAFETY: `cusparseDnVecGet` receives live handles owned by the wrappers and valid output locations.
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
            // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
            values: unsafe { DevicePtr::from_raw(values.cast()) },
        })
    }

    /// Returns the native descriptor handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseDnVecDescr_t {
        self.handle
    }

    /// Returns the native handle in its const-descriptor form.
    #[must_use]
    pub fn as_raw_const(&self) -> sys::cusparseConstDnVecDescr_t {
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

    /// Wraps an existing cuSPARSE dense-vector descriptor and takes ownership
    /// of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned, valid `cusparseDnVecDescr_t` associated
    /// with `parts.cuda_context`. Its pointer bindings and regions must describe
    /// live allocations in that context with the required exclusive borrows for
    /// lifetime `'a`. An empty region list leaves those raw obligations to the caller.
    /// No other owner may destroy the handle after transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawDescriptor<'a, sys::cusparseDnVecDescr_t>) -> Result<Self> {
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
    /// handle with `cusparseDestroyDnVec`.
    #[must_use]
    pub fn into_raw(mut self) -> RawDescriptor<'a, sys::cusparseDnVecDescr_t> {
        RawDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
            regions: mem::take(&mut self.regions),
        }
    }
}

impl Drop for DenseVectorDescriptor<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if self.cuda_context.bind().is_err() {
            return;
        }

        // SAFETY: `cusparseDestroyDnVec` receives only plain values.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroyDnVec(self.as_raw_const())) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse dense vector descriptor: {error}");
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
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution).
/// Sparse indices must address initialized dense elements, and their type,
/// alignment, and value layout must match the descriptors.
/// Retain both descriptors and all buffers through completion and replay.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn gather(
    context: &Context,
    sparse_vector: &mut SparseVectorDescriptor,
    dense_vector: &DenseVectorDescriptor,
) -> Result<()> {
    sparse_vector.ensure_context(context)?;
    dense_vector.ensure_context(context)?;
    context.bind()?;
    // SAFETY: `cusparseGather` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseGather(
            context.as_raw(),
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
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution).
/// Sparse indices must be in bounds and values initialized with matching ABI.
/// Only indexed dense elements are produced; other elements remain unchanged.
/// Retain both descriptors and all buffers through completion and replay.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn scatter(
    context: &Context,
    sparse_vector: &SparseVectorDescriptor,
    dense_vector: &mut DenseVectorDescriptor,
) -> Result<()> {
    sparse_vector.ensure_context(context)?;
    dense_vector.ensure_context(context)?;
    context.bind()?;
    // SAFETY: `cusparseScatter` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseScatter(
            context.as_raw(),
            sparse_vector.as_raw_const(),
            dense_vector.as_raw(),
        ))?;
    }
    Ok(())
}
