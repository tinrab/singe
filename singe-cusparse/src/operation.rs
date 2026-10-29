use std::{mem, mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, data_type::DataType, view::DeviceOutputView};

use crate::{
    context::Context,
    descriptor::{RawResource, workspace},
    error::{Error, Result},
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    sys, try_ffi,
    types::{Operation, SpMmOpAlgorithm},
};

/// Owns native analysis state for sparse-sparse matrix multiplication.
#[derive(Debug)]
pub struct SpGemmDescriptor {
    handle: sys::cusparseSpGEMMDescr_t,
    cuda_context: Arc<CudaContext>,
}

/// Owns native sparse-triangular solve analysis state for one right-hand side.
#[derive(Debug)]
pub struct SpSvDescriptor {
    handle: sys::cusparseSpSVDescr_t,
    cuda_context: Arc<CudaContext>,
}

/// Owns native sparse-triangular solve analysis state for multiple right-hand sides.
#[derive(Debug)]
pub struct SpSmDescriptor {
    handle: sys::cusparseSpSMDescr_t,
    cuda_context: Arc<CudaContext>,
}

/// Owns a custom sparse-dense multiplication plan and its cuSPARSE context.
///
/// The plan borrows its matrix descriptors and custom operator code for its lifetime.
#[derive(Debug)]
pub struct SpMmOpPlan<'a> {
    context: Context,
    handle: sys::cusparseSpMMOpPlan_t,
    workspace_size_bytes: usize,
    matrix_a: &'a SparseMatrixDescriptor<'a>,
    matrix_b: &'a DenseMatrixDescriptor<'a>,
    matrix_c: &'a mut DenseMatrixDescriptor<'a>,
    code: Vec<&'a [u8]>,
}

/// Native `SpMM` plan ownership, its library context, matrix borrows, and custom code.
/// Destroy `handle` before releasing these fields, or re-adopt the complete value.
#[derive(Debug)]
pub struct RawSpMmOpPlan<'a> {
    /// cuSPARSE context owned by the plan.
    pub context: Context,
    /// Native custom-operation plan handle.
    pub handle: sys::cusparseSpMMOpPlan_t,
    /// Required execution workspace size in bytes.
    pub workspace_size_bytes: usize,
    /// Borrowed sparse input matrix descriptor.
    pub matrix_a: &'a SparseMatrixDescriptor<'a>,
    /// Borrowed dense input matrix descriptor.
    pub matrix_b: &'a DenseMatrixDescriptor<'a>,
    /// Exclusively borrowed dense output matrix descriptor.
    pub matrix_c: &'a mut DenseMatrixDescriptor<'a>,
    /// Borrowed custom operator code that must remain live with the plan.
    pub code: Vec<&'a [u8]>,
}

// SAFETY: shared descriptor access exposes only metadata and raw handles.
// Unsafe execution and analysis callers must serialize native mutations and
// retain analysis resources through the final dependent operation.
unsafe impl Send for SpGemmDescriptor {}

unsafe impl Sync for SpGemmDescriptor {}

unsafe impl Send for SpSvDescriptor {}

unsafe impl Sync for SpSvDescriptor {}

unsafe impl Send for SpSmDescriptor {}

unsafe impl Sync for SpSmDescriptor {}

// SAFETY: the plan owns its movable library context; input metadata stays immutable
// and its mutable destination borrow remains exclusive across thread transfer.
unsafe impl Send for SpMmOpPlan<'_> {}

impl SpGemmDescriptor {
    /// Returns the actual CUDA context retained by this analysis descriptor.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    /// Creates native analysis state in the supplied cuSPARSE context's CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create(context: &Context) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseSpGEMM_createDescr` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpGEMM_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
        })
    }

    /// # Safety
    /// Complete the native estimation phase before reading its product count.
    ///
    /// # Errors
    ///
    /// Propagates CUDA context-binding and native product-count query errors. Also rejects a negative count or one that cannot fit in `usize`.
    pub unsafe fn num_products(&self) -> Result<usize> {
        self.cuda_context.bind()?;
        let mut value = 0_i64;
        // SAFETY: `cusparseSpGEMM_getNumProducts` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpGEMM_getNumProducts(
                self.as_raw(),
                &raw mut value,
            ))?;
        }

        usize::try_from(value).map_err(|_| Error::out_of_range("num_products"))
    }

    /// Returns the native analysis descriptor without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpGEMMDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, context: &Context) -> Result<()> {
        if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE `SpGEMM` descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned `cusparseSpGEMMDescr_t` associated
    /// with `parts.cuda_context`. Preserve all analysis bindings and workspaces
    /// through their final dependent operation. No other owner may destroy it.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawResource<sys::cusparseSpGEMMDescr_t>) -> Result<Self> {
        let RawResource {
            handle,
            cuda_context,
        } = parts;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpGEMM_destroyDescr`.
    #[must_use]
    pub fn into_raw(mut self) -> RawResource<sys::cusparseSpGEMMDescr_t> {
        RawResource {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
        }
    }
}

impl Drop for SpGemmDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() || self.cuda_context.bind().is_err() {
            return;
        }
        // SAFETY: `cusparseSpGEMM_destroyDescr` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseSpGEMM_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spgemm descriptor: {error}");
            }
        }
    }
}

impl SpSvDescriptor {
    /// Returns the actual CUDA context retained by this analysis descriptor.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    /// Creates native analysis state in the supplied cuSPARSE context's CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create(context: &Context) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseSpSV_createDescr` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpSV_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
        })
    }

    /// Returns the native analysis descriptor without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpSVDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, context: &Context) -> Result<()> {
        if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE `SpSV` descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned `cusparseSpSVDescr_t` associated
    /// with `parts.cuda_context`. Preserve all analysis bindings and workspaces
    /// through their final dependent operation. No other owner may destroy it.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawResource<sys::cusparseSpSVDescr_t>) -> Result<Self> {
        let RawResource {
            handle,
            cuda_context,
        } = parts;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpSV_destroyDescr`.
    #[must_use]
    pub fn into_raw(mut self) -> RawResource<sys::cusparseSpSVDescr_t> {
        RawResource {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
        }
    }
}

impl Drop for SpSvDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() || self.cuda_context.bind().is_err() {
            return;
        }
        // SAFETY: `cusparseSpSV_destroyDescr` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseSpSV_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spsv descriptor: {error}");
            }
        }
    }
}

impl SpSmDescriptor {
    /// Returns the actual CUDA context retained by this analysis descriptor.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    /// Creates native analysis state in the supplied cuSPARSE context's CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA context-binding and native descriptor-creation errors, and rejects a null returned handle.
    pub fn create(context: &Context) -> Result<Self> {
        context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseSpSM_createDescr` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpSM_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
        })
    }

    /// Returns the native analysis descriptor without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpSMDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, context: &Context) -> Result<()> {
        if self.cuda_context.as_ref() != context.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE `SpSM` descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned `cusparseSpSMDescr_t` associated
    /// with `parts.cuda_context`. Preserve all analysis bindings and workspaces
    /// through their final dependent operation. No other owner may destroy it.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawResource<sys::cusparseSpSMDescr_t>) -> Result<Self> {
        let RawResource {
            handle,
            cuda_context,
        } = parts;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_context,
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpSM_destroyDescr`.
    #[must_use]
    pub fn into_raw(mut self) -> RawResource<sys::cusparseSpSMDescr_t> {
        RawResource {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            cuda_context: Arc::clone(&self.cuda_context),
        }
    }
}

impl<'a> SpMmOpPlan<'a> {
    /// # Safety
    ///
    /// Custom operation code and matrix bindings must match the native ABI.
    /// Retain all participants through any native work, including failure paths.
    ///
    /// # Errors
    ///
    /// Returns an error if a matrix belongs to another CUDA context or the reported workspace size cannot fit in `usize`. Also propagates context-binding and native custom-plan creation errors, and rejects a null returned plan handle.
    pub unsafe fn create(
        context: Context,
        op_a: Operation,
        op_b: Operation,
        matrix_a: &'a SparseMatrixDescriptor<'a>,
        matrix_b: &'a DenseMatrixDescriptor<'a>,
        matrix_c: &'a mut DenseMatrixDescriptor<'a>,
        compute_type: DataType,
        algorithm: SpMmOpAlgorithm,
        add_operation_ltoir: Option<&'a [u8]>,
        mul_operation_ltoir: Option<&'a [u8]>,
        epilogue_ltoir: Option<&'a [u8]>,
    ) -> Result<Self> {
        matrix_a.ensure_context(&context)?;
        matrix_b.ensure_context(&context)?;
        matrix_c.ensure_context(&context)?;
        context.bind()?;

        let mut handle = ptr::null_mut();
        let mut workspace_size = 0;
        let (add_ptr, add_len) = ltoir_parts(add_operation_ltoir);
        let (mul_ptr, mul_len) = ltoir_parts(mul_operation_ltoir);
        let (epilogue_ptr, epilogue_len) = ltoir_parts(epilogue_ltoir);
        // SAFETY: `cusparseSpMMOp_createPlan` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseSpMMOp_createPlan(
                context.as_raw(),
                &raw mut handle,
                op_a.into(),
                op_b.into(),
                matrix_a.as_raw_const(),
                matrix_b.as_raw_const(),
                matrix_c.as_raw(),
                compute_type.into(),
                algorithm.into(),
                add_ptr.cast(),
                add_len as _,
                mul_ptr.cast(),
                mul_len as _,
                epilogue_ptr.cast(),
                epilogue_len as _,
                &raw mut workspace_size,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut plan = Self {
            context,
            handle,
            workspace_size_bytes: 0,
            matrix_a,
            matrix_b,
            matrix_c,
            code: [add_operation_ltoir, mul_operation_ltoir, epilogue_ltoir]
                .into_iter()
                .flatten()
                .collect(),
        };
        plan.workspace_size_bytes = usize::try_from(workspace_size)
            .map_err(|_| Error::out_of_range("spmm op workspace size"))?;
        Ok(plan)
    }

    /// Borrows the cuSPARSE context owned by this plan.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Required scratch capacity for this compiled plan, in bytes.
    #[must_use]
    pub const fn required_workspace_size_bytes(&self) -> usize {
        self.workspace_size_bytes
    }

    /// NVRTC and nvJitLink are not currently available on Arm64 Android platforms.
    ///
    /// This operation does not support Android and Tegra platforms except Judy (sm87).
    ///
    /// Experimental: multiplies `matrix_a` and `matrix_b` with custom operators.
    ///
    /// where
    ///
    /// * `op(A)` is a sparse matrix of size $m \times k$.
    /// * `op(B)` is a dense matrix of size $k \times n$.
    /// * `C` is a dense matrix of size $m \times n$.
    /// * $\oplus$, $\otimes$, and $\text{epilogue}$ are custom **add**, **mul**, and **epilogue** operators respectively.
    ///
    /// `op(A)` is selected by `op_a` and may be `A`, `A^T`, or `A^H`.
    /// `op(B)` is selected by `op_b` and may be `B`, `B^T`, or `B^H`.
    ///
    /// Only `op_a == Operation::NonTranspose` is currently supported.
    ///
    /// [`SpMmOpPlan::required_workspace_size_bytes`] reports the scratch capacity needed by this compiled plan.
    ///
    /// The custom add, multiply, and epilogue operators must accept and return the selected compute type.
    /// The compute type may be `float`, `double`, `cuComplex`, `cuDoubleComplex`, or `int`.
    ///
    /// [`SpMmOpPlan::execute`] supports the following sparse matrix formats:
    ///
    /// * [`Format::Csr`](crate::types::Format::Csr)
    ///
    /// [`SpMmOpPlan::execute`] supports the following index type for representing `matrix_a`:
    ///
    /// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
    /// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
    ///
    /// [`SpMmOpPlan::execute`] supports the following data types:
    ///
    /// Uniform-precision computation:
    ///
    /// | `A`/`B`/`C`/`compute_type` |
    /// | --- |
    /// | [`DataType::F32`] |
    /// | [`DataType::F64`] |
    /// | [`DataType::ComplexF32`] |
    /// | [`DataType::ComplexF64`] |
    ///
    /// Mixed-precision computation:
    ///
    /// | `A`/`B` | `C` | `compute_type` |
    /// | --- | --- | --- |
    /// | [`DataType::I8`] | [`DataType::I32`] | [`DataType::I32`] |
    /// | [`DataType::I8`] | [`DataType::F32`] | [`DataType::F32`] |
    /// | [`DataType::F16`] | | |
    /// | [`DataType::Bf16`] | | |
    /// | [`DataType::F16`] | [`DataType::F16`] | |
    /// | [`DataType::Bf16`] | [`DataType::Bf16`] | |
    ///
    /// [`SpMmOpPlan::execute`] supports the following algorithms:
    ///
    /// | Algorithm | Notes |
    /// | --- | --- |
    /// | [`SpMmOpAlgorithm::Default`] | Default algorithm for any sparse matrix format. |
    ///
    /// **Performance notes:**
    ///
    /// * Row-major layout provides higher performance than column-major.
    ///
    /// [`SpMmOpPlan::execute`] has the following properties:
    ///
    /// * Requires extra storage.
    /// * Supports asynchronous execution.
    /// * Provides deterministic (bitwise) results for each run.
    /// * Allows the indices of `matrix_a` to be unsorted.
    ///
    /// [`SpMmOpPlan::execute`] supports the following optimizations:
    ///
    /// * CUDA graph capture.
    /// * Hardware Memory Compression.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// rejects the prepared `SpMM` operation.
    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    /// # Safety
    ///
    /// Retain the plan, descriptors, and correctly sized scratch with exclusive
    /// destination access through completion on every path and captured replay.
    pub unsafe fn execute(&self, external_buffer: Option<DeviceOutputView<'_, u8>>) -> Result<()> {
        let external_buffer = workspace(&self.context, external_buffer)?;
        let actual = external_buffer.map_or(0, singe_cuda::view::DeviceRegion::byte_len);
        if actual < self.workspace_size_bytes {
            return Err(Error::LengthMismatch {
                name: "spmm op workspace".into(),
                expected: self.workspace_size_bytes,
                actual,
            });
        }

        self.context.bind()?;

        // SAFETY: `cusparseSpMMOp` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cusparseSpMMOp(
                self.as_raw(),
                external_buffer
                    .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                    .cast(),
            ))?;
        }
        Ok(())
    }

    /// Returns the native custom-operation plan handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseSpMMOpPlan_t {
        self.handle
    }

    /// Wraps an existing cuSPARSE `SpMM` operation plan and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned plan created with `parts.context`,
    /// the supplied matrix descriptors, and the retained custom code.
    /// `parts.workspace_size_bytes` must be its actual queried scratch capacity.
    /// Preserve all allocation and analysis bindings for lifetime `'a`, with
    /// exclusive destination access. No other owner may destroy the plan.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(parts: RawSpMmOpPlan<'a>) -> Result<Self> {
        if parts.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            context: parts.context,
            handle: parts.handle,
            workspace_size_bytes: parts.workspace_size_bytes,
            matrix_a: parts.matrix_a,
            matrix_b: parts.matrix_b,
            matrix_c: parts.matrix_c,
            code: parts.code,
        })
    }

    /// Consumes the plan and returns the raw cuSPARSE handle without destroying
    /// it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpMMOp_destroyPlan`.
    #[must_use]
    pub fn into_raw(self) -> RawSpMmOpPlan<'a> {
        let plan = ManuallyDrop::new(self);
        // SAFETY: every owned field is moved once and the original plan never drops.
        unsafe {
            RawSpMmOpPlan {
                handle: plan.handle,
                workspace_size_bytes: plan.workspace_size_bytes,
                context: ptr::read(&raw const plan.context),
                matrix_a: plan.matrix_a,
                matrix_b: plan.matrix_b,
                matrix_c: ptr::read(&raw const plan.matrix_c),
                code: ptr::read(&raw const plan.code),
            }
        }
    }
}

impl Drop for SpMmOpPlan<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cusparse spmm op plan context during drop: {error}");
            return;
        }

        // SAFETY: `cusparseSpMMOp_destroyPlan` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseSpMMOp_destroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spmm op plan: {error}");
            }
        }
    }
}

fn ltoir_parts(buffer: Option<&[u8]>) -> (*const u8, usize) {
    buffer.map_or((ptr::null(), 0), |buffer| (buffer.as_ptr(), buffer.len()))
}

impl Drop for SpSmDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() || self.cuda_context.bind().is_err() {
            return;
        }
        // SAFETY: `cusparseSpSM_destroyDescr` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseSpSM_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spsm descriptor: {error}");
            }
        }
    }
}
