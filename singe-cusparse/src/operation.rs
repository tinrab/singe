use std::{marker::PhantomData, mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda::context::Context as CudaContext;
use singe_cuda::data_type::DataType;
use singe_cuda::types::DevicePtr;

use crate::{
    context::Context,
    error::{Error, Result},
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    sys, try_ffi,
    types::{Operation, SpMmOpAlgorithm},
};

#[derive(Debug)]
pub struct SpGemmDescriptor {
    handle: sys::cusparseSpGEMMDescr_t,
    cuda_ctx: Arc<CudaContext>,
}

#[derive(Debug)]
pub struct SpSvDescriptor {
    handle: sys::cusparseSpSVDescr_t,
    cuda_ctx: Arc<CudaContext>,
}

#[derive(Debug)]
pub struct SpSmDescriptor {
    handle: sys::cusparseSpSMDescr_t,
    cuda_ctx: Arc<CudaContext>,
}

#[derive(Debug)]
pub struct SpMmOpPlan<'a> {
    context: &'a Context,
    handle: sys::cusparseSpMMOpPlan_t,
    _matrix_a: PhantomData<&'a SparseMatrixDescriptor<'a>>,
    _matrix_b: PhantomData<&'a DenseMatrixDescriptor<'a>>,
    _matrix_c: PhantomData<&'a DenseMatrixDescriptor<'a>>,
}

// Operation descriptors and plans own cuSPARSE analysis state but do not expose
// pointer rebinding through shared references, so immutable sharing is allowed.
unsafe impl Send for SpGemmDescriptor {}
unsafe impl Sync for SpGemmDescriptor {}
unsafe impl Send for SpSvDescriptor {}
unsafe impl Sync for SpSvDescriptor {}
unsafe impl Send for SpSmDescriptor {}
unsafe impl Sync for SpSmDescriptor {}
unsafe impl Send for SpMmOpPlan<'_> {}
unsafe impl Sync for SpMmOpPlan<'_> {}

impl SpGemmDescriptor {
    pub fn create(ctx: &Context) -> Result<Self> {
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseSpGEMM_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    pub fn num_products(&self) -> Result<usize> {
        let mut value = 0_i64;
        unsafe {
            try_ffi!(sys::cusparseSpGEMM_getNumProducts(
                self.as_raw(),
                &raw mut value,
            ))?;
        }

        usize::try_from(value).map_err(|_| Error::OutOfRange {
            name: "num_products".into(),
        })
    }

    pub fn as_raw(&self) -> sys::cusparseSpGEMMDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, ctx: &Context) -> Result<()> {
        if self.cuda_ctx.as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE SpGEMM descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpGEMMDescr_t` associated with `ctx`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseSpGEMMDescr_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpGEMM_destroyDescr`.
    pub fn into_raw(self) -> sys::cusparseSpGEMMDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for SpGemmDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseSpGEMM_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spgemm descriptor: {err}");
            }
        }
    }
}

impl SpSvDescriptor {
    pub fn create(ctx: &Context) -> Result<Self> {
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseSpSV_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    pub fn as_raw(&self) -> sys::cusparseSpSVDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, ctx: &Context) -> Result<()> {
        if self.cuda_ctx.as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE SpSV descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpSVDescr_t` associated with `ctx`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseSpSVDescr_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpSV_destroyDescr`.
    pub fn into_raw(self) -> sys::cusparseSpSVDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl Drop for SpSvDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseSpSV_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spsv descriptor: {err}");
            }
        }
    }
}

impl SpSmDescriptor {
    pub fn create(ctx: &Context) -> Result<Self> {
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseSpSM_createDescr(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    pub fn as_raw(&self) -> sys::cusparseSpSMDescr_t {
        self.handle
    }

    pub(crate) fn ensure_context(&self, ctx: &Context) -> Result<()> {
        if self.cuda_ctx.as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::PlanContextMismatch);
        }
        Ok(())
    }

    /// Wraps an existing cuSPARSE SpSM descriptor and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpSMDescr_t` associated with `ctx`.
    /// Ownership of `handle` is transferred to the returned descriptor, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cusparseSpSMDescr_t, ctx: &Context) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
        })
    }

    /// Consumes the descriptor and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpSM_destroyDescr`.
    pub fn into_raw(self) -> sys::cusparseSpSMDescr_t {
        let descriptor = ManuallyDrop::new(self);
        descriptor.handle
    }
}

impl<'a> SpMmOpPlan<'a> {
    pub fn create(
        ctx: &'a Context,
        op_a: Operation,
        op_b: Operation,
        matrix_a: &'a SparseMatrixDescriptor<'a>,
        matrix_b: &'a DenseMatrixDescriptor<'a>,
        matrix_c: &'a mut DenseMatrixDescriptor<'a>,
        compute_type: DataType,
        algorithm: SpMmOpAlgorithm,
        add_operation_ltoir: Option<&[u8]>,
        mul_operation_ltoir: Option<&[u8]>,
        epilogue_ltoir: Option<&[u8]>,
    ) -> Result<(Self, usize)> {
        matrix_a.ensure_context(ctx)?;
        matrix_b.ensure_context(ctx)?;
        matrix_c.ensure_context(ctx)?;
        ctx.bind()?;

        let mut handle = ptr::null_mut();
        let mut workspace_size = 0;
        let (add_ptr, add_len) = ltoir_parts(add_operation_ltoir);
        let (mul_ptr, mul_len) = ltoir_parts(mul_operation_ltoir);
        let (epilogue_ptr, epilogue_len) = ltoir_parts(epilogue_ltoir);
        unsafe {
            try_ffi!(sys::cusparseSpMMOp_createPlan(
                ctx.as_raw(),
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

        Ok((
            Self {
                context: ctx,
                handle,
                _matrix_a: PhantomData,
                _matrix_b: PhantomData,
                _matrix_c: PhantomData,
            },
            usize::try_from(workspace_size).map_err(|_| Error::OutOfRange {
                name: "spmm op workspace size".into(),
            })?,
        ))
    }

    pub fn context(&self) -> &Context {
        self.context
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
    /// [`SpMmOpPlan::create`] returns the workspace size together with the compiled kernel state needed by [`SpMmOpPlan::execute`].
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
    /// rejects the prepared SpMM operation.
    #[allow(deprecated)]
    pub fn execute(&self, external_buffer: Option<DevicePtr>) -> Result<()> {
        self.context.bind()?;

        unsafe {
            try_ffi!(sys::cusparseSpMMOp(
                self.as_raw(),
                external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
            ))?;
        }
        Ok(())
    }

    pub fn as_raw(&self) -> sys::cusparseSpMMOpPlan_t {
        self.handle
    }

    /// Wraps an existing cuSPARSE SpMM operation plan and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cusparseSpMMOpPlan_t` associated with `ctx`
    /// and with matrix descriptors that remain valid for lifetime `'a`.
    /// Ownership of `handle` is transferred to the returned plan, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(ctx: &'a Context, handle: sys::cusparseSpMMOpPlan_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            context: ctx,
            handle,
            _matrix_a: PhantomData,
            _matrix_b: PhantomData,
            _matrix_c: PhantomData,
        })
    }

    /// Consumes the plan and returns the raw cuSPARSE handle without destroying
    /// it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with `cusparseSpMMOp_destroyPlan`.
    pub fn into_raw(self) -> sys::cusparseSpMMOpPlan_t {
        let plan = ManuallyDrop::new(self);
        plan.handle
    }
}

impl Drop for SpMmOpPlan<'_> {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cusparse spmm op plan context during drop: {err}");
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseSpMMOp_destroyPlan(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spmm op plan: {err}");
            }
        }
    }
}

fn ltoir_parts(buffer: Option<&[u8]>) -> (*const u8, usize) {
    match buffer {
        Some(buffer) => (buffer.as_ptr(), buffer.len()),
        None => (ptr::null(), 0),
    }
}

impl Drop for SpSmDescriptor {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseSpSM_destroyDescr(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse spsm descriptor: {err}");
            }
        }
    }
}
