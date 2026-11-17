use std::{mem::ManuallyDrop, ptr, slice};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    view::{DeviceOutputView, DeviceRegion},
};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpMmAlgorithm, SpMmOpAlgorithm},
    utility::{to_size_t, to_usize, workspace_region},
};

/// Performs the multiplication of a sparse matrix `matrix_a` and a dense matrix `matrix_b`.
/// Formally, it computes `C = alpha * op(A) * op(B) + beta * C`, where:
///
/// * `op(A)` is a sparse matrix of size $m \times k$.
/// * `op(B)` is a dense matrix of size $k \times n$.
/// * `C` is a dense matrix of size $m \times n$.
/// * $\alpha$ and $\beta$ are scalars.
///
/// This can also multiply a dense matrix and a sparse matrix by switching the dense matrix layouts.
/// `B_C` and `C_C` indicate column-major layout, while `B_R` and `C_R` indicate row-major layout.
///
/// `op(A)` is selected by `op_a` and may be `A`, `A^T`, or `A^H`.
/// `op(B)` is selected by `op_b` and may be `B`, `B^T`, or `B^H`.
///
/// When using the (conjugate) transpose of the sparse matrix `A`, this operation may produce slightly different results during different runs with the same input parameters.
///
/// [`spmm_workspace_size`] returns the size of the workspace needed by [`spmm_async`].
///
/// Calling [`spmm_preprocess_async`] is optional.
/// It may accelerate subsequent calls to [`spmm_async`].
/// Useful when [`spmm_async`] is called multiple times with the same sparsity pattern (`matrix_a`).
/// It provides performance advantages with [`SpMmAlgorithm::Csr1`] or [`SpMmAlgorithm::Csr3`].
/// It has no effect for all other formats and algorithms.
///
/// Calling [`spmm_preprocess_async`] with `buffer` makes that buffer active for `matrix_a` `SpMM` calls.
/// Subsequent calls to [`spmm_async`] with `matrix_a` and the active buffer must use the same values for all parameters as the call to [`spmm_preprocess_async`].
/// The exceptions are: `alpha`, `beta`, `matrix_b`, `matrix_c`, and the values (but not indices) of `matrix_a` may be different.
/// Importantly, the buffer contents must be unmodified since the call to [`spmm_preprocess_async`].
/// When [`spmm_async`] is called with `matrix_a` and its active buffer, it may read acceleration data from the buffer.
///
/// Calling [`spmm_preprocess_async`] again with `matrix_a` and a new buffer makes the new buffer active and makes the previously active buffer inactive.
/// For [`spmm_async`], there can only be one active buffer per sparse matrix at a time.
/// To get the effect of multiple active buffers for a single sparse matrix, create multiple matrix handles that all point to the same index and value buffers, and call [`spmm_preprocess_async`] once per handle with different workspace buffers.
///
/// Calling [`spmm_async`] with an inactive buffer is always permitted.
/// However, there may be no acceleration from the preprocessing in that case.
///
/// For the purposes of thread safety, [`spmm_preprocess_async`] is writing to `matrix_a` internal state.
///
/// [`spmm_async`] supports the following sparse matrix formats:
///
/// * [`Format::Coo`](crate::types::Format::Coo)
/// * [`Format::Csr`](crate::types::Format::Csr)
/// * [`Format::Csc`](crate::types::Format::Csc)
/// * [`Format::Bsr`](crate::types::Format::Bsr)
/// * [`Format::BlockedEll`](crate::types::Format::BlockedEll)
///
/// **(1) COO/CSR/CSC/BSR formats**
///
/// [`spmm_async`] supports the following index type for representing the sparse matrix `matrix_a`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// [`spmm_async`] supports the following data types:
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
/// | `A`/`B` | `C` | `compute_type` | Notes |
/// | --- | --- | --- | --- |
/// | [`DataType::I8`] | [`DataType::I32`] | [`DataType::I32`] |  |
/// | [`DataType::I8`] | [`DataType::F32`] | [`DataType::F32`] |  |
/// | [`DataType::F16`] |  |  |  |
/// | [`DataType::Bf16`] |  |  |  |
/// | [`DataType::F16`] | [`DataType::F16`] |  |  |
/// | [`DataType::Bf16`] | [`DataType::Bf16`] |  |  |
/// | [`DataType::ComplexF16`] | [`DataType::ComplexF16`] | [`DataType::ComplexF32`] | Deprecated. |
/// | [`DataType::ComplexBf16`] | [`DataType::ComplexBf16`] |  | Deprecated. |
///
/// [`DataType::F16`], [`DataType::Bf16`], [`DataType::ComplexF16`], and [`DataType::ComplexBf16`] data types always imply mixed-precision computation.
///
/// [`spmm_async`] supports the following algorithms:
///
/// * [`SpMmAlgorithm::Default`]: default algorithm for any sparse matrix format.
/// * [`SpMmAlgorithm::Coo1`]: algorithm 1 for COO sparse matrix format.
///   May provide better performance for a small number of nonzero values, provides the best performance with column-major layout, supports batched computation, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Coo2`]: algorithm 2 for COO sparse matrix format.
///   Provides deterministic results and the best performance with column-major layout.
///   In general, it is slower than algorithm 1, supports batched computation, requires additional memory, and is identical to [`SpMmAlgorithm::Coo1`] if `op_a` is not [`Operation::NonTranspose`].
/// * [`SpMmAlgorithm::Coo3`]: algorithm 3 for COO sparse matrix format.
///   May provide better performance for a large number of nonzero values and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Coo4`]: algorithm 4 for COO sparse matrix format.
///   Provides better performance with row-major layout, supports batched computation, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr1`]: algorithm 1 for CSR/CSC sparse matrix format.
///   Provides the best performance with column-major layout, supports batched computation, requires additional memory, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr2`]: algorithm 2 for CSR/CSC sparse matrix format.
///   Provides the best performance with row-major layout, supports batched computation, requires additional memory, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr3`]: algorithm 3 for CSR sparse matrix format.
///   Provides deterministic results and requires additional memory.
///   It supports only CSR matrices with `op_a` set to [`Operation::NonTranspose`]; `op_b` cannot be [`Operation::ConjugateTranspose`], and the data type cannot be [`DataType::ComplexF16`] or [`DataType::ComplexBf16`].
/// * [`SpMmAlgorithm::Bsr1`]: algorithm 1 for BSR sparse matrix format.
///   Provides deterministic results and requires no additional memory.
///   It supports only `op_a` set to [`Operation::NonTranspose`]; the data type cannot be [`DataType::ComplexF16`] or [`DataType::ComplexBf16`], and blocks in `A` cannot be column-major.
///
/// When using [`spmm_async`] for mixed-precision computation on COO or CSR matrices, it defaults to algorithms [`SpMmAlgorithm::Coo2`] and [`SpMmAlgorithm::Csr3`], respectively.
/// If the required computation is not supported by those algorithms, the mixed-precision operation fails.
///
/// **Performance notes:**
///
/// * Row-major layout provides higher performance than column-major.
/// * [`SpMmAlgorithm::Coo4`] and [`SpMmAlgorithm::Csr2`] are intended for row-major layout, while [`SpMmAlgorithm::Coo1`], [`SpMmAlgorithm::Coo2`], [`SpMmAlgorithm::Coo3`], and [`SpMmAlgorithm::Csr1`] are intended for column-major layout.
/// * When `beta` is not `1`, most algorithms scale the output matrix before the main computation.
/// * When `n` is `1`, this operation may use [`spmv_async`](crate::spmv::spmv_async).
///
/// [`spmm_async`] with all algorithms support the following batch modes except for [`SpMmAlgorithm::Csr3`]:
///
/// * $C\_{i} = A \cdot B\_{i}$
/// * $C\_{i} = A\_{i} \cdot B$
/// * $C\_{i} = A\_{i} \cdot B\_{i}$
///
/// The number of batches and their strides can be set by using [`SparseMatrixDescriptor::set_coo_strided_batch`], [`SparseMatrixDescriptor::set_csr_strided_batch`], and [`DenseMatrixDescriptor::set_strided_batch`].
/// The maximum number of batches for [`spmm_async`] is 65,535.
///
/// [`spmm_async`] has the following properties:
///
/// * Requires no extra storage for [`SpMmAlgorithm::Coo1`], [`SpMmAlgorithm::Coo3`], [`SpMmAlgorithm::Coo4`], [`SpMmAlgorithm::Bsr1`].
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run only for [`SpMmAlgorithm::Coo2`], [`SpMmAlgorithm::Csr3`], and [`SpMmAlgorithm::Bsr1`] algorithms.
/// * `compute-sanitizer` could report false race conditions for this operation.
///   These reports result from optimization and do not affect computation correctness.
/// * Allows the indices of `matrix_a` to be unsorted.
///
/// [`spmm_async`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
///
/// **(2) Blocked-ELLPACK format**
///
/// [`spmm_async`] supports the following data types for [`Format::BlockedEll`](crate::types::Format::BlockedEll) format and the following GPU architectures for exploiting NVIDIA Tensor Cores:
///
/// | `A`/`B` | `C` | `compute_type` | `op_b` | Compute Capability |
/// | --- | --- | --- | --- | --- |
/// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F16`] | `N`, `T` | `≥ 70` |
/// | [`DataType::F16`] | [`DataType::F16`] | [`DataType::F32`] | `N`, `T` | `≥ 70` |
/// | [`DataType::F16`] | [`DataType::F32`] | [`DataType::F32`] | `N`, `T` | `≥ 70` |
/// | [`DataType::I8`] | [`DataType::I32`] | [`DataType::I32`] | `N` column-major, `T` row-major | `≥ 75` |
/// | [`DataType::Bf16`] | [`DataType::Bf16`] | [`DataType::F32`] | `N`, `T` | `≥ 80` |
/// | [`DataType::Bf16`] | [`DataType::F32`] | [`DataType::F32`] | `N`, `T` | `≥ 80` |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `N`, `T` | `≥ 80` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `N`, `T` | `≥ 80` |
///
/// [`spmm_async`] supports the following algorithms with [`Format::BlockedEll`](crate::types::Format::BlockedEll) format:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SpMmAlgorithm::Default`] | Default algorithm for any sparse matrix format. |
/// | [`SpMmAlgorithm::BlockedEll1`] | Default algorithm for Blocked-ELL format. |
///
/// **Performance notes:**
///
/// * Blocked-ELL `SpMM` provides the best performance with Power-of-2 Block-Sizes.
/// * Large block sizes, such as 64 or greater, provide the best performance.
///
/// This operation has the following limitations:
///
/// * The pointer mode must be [`PointerMode::Host`](crate::types::PointerMode::Host).
/// * Only `op_a` set to [`Operation::NonTranspose`] is supported.
/// * `op_b` set to [`Operation::ConjugateTranspose`] is not supported.
/// * Only [`IndexType::I32`](crate::types::IndexType::I32) is supported.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if the cuSPARSE context cannot be bound, the scalar pointer modes do not match the handle configuration, the descriptors or selected algorithm are incompatible, a required workspace buffer is missing or invalid, the data type combination is unsupported, or cuSPARSE rejects the operation.
pub unsafe fn spmm_async<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region(), beta.region()],
    ])?;

    // SAFETY: descriptors, scalars, and the workspace were checked against the bound `context` and ordered after their producers; the caller retains them through completion per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseSpMM(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            beta.ptr().cast(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Returns the workspace size in bytes that `SpMM` and its preprocessing need.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or device scalar belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spmm_workspace_size<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = 0;
    // SAFETY: every descriptor was checked against `context`, which is bound with the scalars' pointer mode, and `size` is a writable local; the query enqueues nothing.
    unsafe {
        try_ffi!(sys::cusparseSpMM_bufferSize(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            beta.ptr().cast(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "spmm buffer size")
}

/// Enqueues `SpMM` preprocessing, which may accelerate later `SpMM` calls with the same sparsity pattern.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor, device scalar, or workspace belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spmm_preprocess_async<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region(), beta.region()],
    ])?;

    // SAFETY: descriptors, scalars, and the workspace were checked against the bound `context` and ordered after their producers; the caller retains them through completion per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseSpMM_preprocess(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            beta.ptr().cast(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
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

impl<'a> SpMmOpPlan<'a> {
    /// Compiles a custom sparse-dense multiplication plan from LTO-IR operators.
    ///
    /// # Safety
    ///
    /// Custom operation code and matrix bindings must match the native ABI.
    /// Retain all participants through any native work, including failure paths.
    ///
    /// # Errors
    ///
    /// Returns an error if a matrix belongs to another CUDA context, the context cannot be bound, cuSPARSE fails to create the plan or returns a null handle, or the reported workspace size does not fit in `usize`.
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
        // SAFETY: the matrices were checked against the bound `context`, each code pointer comes with its slice length, and `handle` and `workspace_size` are writable locals; the caller guarantees the code matches the native ABI.
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
                to_size_t(add_len, "add operation size")?,
                mul_ptr.cast(),
                to_size_t(mul_len, "mul operation size")?,
                epilogue_ptr.cast(),
                to_size_t(epilogue_len, "epilogue operation size")?,
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
        plan.workspace_size_bytes = to_usize(workspace_size, "spmm op workspace size")?;
        Ok(plan)
    }

    /// Borrows the cuSPARSE context owned by this plan.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Required scratch capacity for this compiled plan, in bytes.
    #[must_use]
    pub const fn workspace_size(&self) -> usize {
        self.workspace_size_bytes
    }

    /// Enqueues the experimental multiplication of `matrix_a` and `matrix_b` with the plan's custom operators.
    ///
    /// NVRTC and nvJitLink are not currently available on Arm64 Android platforms.
    /// This operation does not support Android and Tegra platforms except Judy (sm87).
    ///
    /// The operands are:
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
    /// [`SpMmOpPlan::workspace_size`] reports the scratch capacity needed by this compiled plan.
    ///
    /// The custom add, multiply, and epilogue operators must accept and return the selected compute type.
    /// The compute type may be `float`, `double`, `cuComplex`, `cuDoubleComplex`, or `int`.
    ///
    /// [`SpMmOpPlan::execute_async`] supports the following sparse matrix formats:
    ///
    /// * [`Format::Csr`](crate::types::Format::Csr)
    ///
    /// [`SpMmOpPlan::execute_async`] supports the following index type for representing `matrix_a`:
    ///
    /// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
    /// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
    ///
    /// [`SpMmOpPlan::execute_async`] supports the following data types:
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
    /// [`SpMmOpPlan::execute_async`] supports the following algorithms:
    ///
    /// | Algorithm | Notes |
    /// | --- | --- |
    /// | [`SpMmOpAlgorithm::Default`] | Default algorithm for any sparse matrix format. |
    ///
    /// **Performance notes:**
    ///
    /// * Row-major layout provides higher performance than column-major.
    ///
    /// [`SpMmOpPlan::execute_async`] has the following properties:
    ///
    /// * Requires extra storage.
    /// * Supports asynchronous execution.
    /// * Provides deterministic (bitwise) results for each run.
    /// * Allows the indices of `matrix_a` to be unsorted.
    ///
    /// [`SpMmOpPlan::execute_async`] supports the following optimizations:
    ///
    /// * CUDA graph capture.
    /// * Hardware Memory Compression.
    ///
    /// # Safety
    ///
    /// Uphold the [raw execution contract](crate#raw-execution): retain the plan, descriptors, and correctly sized workspace with exclusive destination access through completion on every path and captured replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace belongs to another CUDA context or is smaller than [`SpMmOpPlan::workspace_size`], the CUDA context cannot be bound, or cuSPARSE rejects the operation.
    #[expect(
        deprecated,
        reason = "wraps deprecated upstream API values that remain part of the supported surface"
    )]
    pub unsafe fn execute_async(&self, workspace: Option<DeviceOutputView<'_, u8>>) -> Result<()> {
        let workspace = workspace_region(&self.context, workspace)?;
        let actual = workspace.map_or(0, DeviceRegion::byte_len);
        if actual < self.workspace_size_bytes {
            return Err(Error::BufferTooSmall {
                name: "spmm op workspace".into(),
                required: self.workspace_size_bytes,
                actual,
            });
        }

        self.context.bind()?;
        self.context.prepare_operands(&[
            self.matrix_a.regions(),
            self.matrix_b.regions(),
            self.matrix_c.regions(),
            slice::from_ref(&workspace),
        ])?;

        // SAFETY: the plan owns its live handle and borrows its matrices and code for `'a`, the workspace was checked against its context and its queried size, and the caller retains them per the raw execution contract.
        unsafe {
            try_ffi!(sys::cusparseSpMMOp(
                self.as_raw(),
                workspace
                    .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
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
    /// `parts.handle` must be a uniquely owned plan created with `parts.context`, the supplied matrix descriptors, and the retained custom code.
    /// `parts.workspace_size_bytes` must be its actual queried scratch capacity.
    /// Preserve all allocation and analysis bindings for lifetime `'a`, with exclusive destination access.
    /// No other owner may destroy the plan.
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

    /// Consumes the plan and returns the raw cuSPARSE handle without destroying it.
    ///
    /// The caller must destroy the returned handle with [`sys::cusparseSpMMOp_destroyPlan`] or adopt it again with [`SpMmOpPlan::from_raw`].
    #[must_use]
    pub fn into_raw(self) -> RawSpMmOpPlan<'a> {
        let plan = ManuallyDrop::new(self);
        // SAFETY: `plan` is never dropped, so each field read here moves out exactly once.
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
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cusparse spmm op plan",
                &error,
            );
            return;
        }

        // SAFETY: the plan handle is live and uniquely owned, its context is bound above, and it is never used again.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseSpMMOp_destroyPlan(self.handle)) {
                report_cleanup_failure("destroy", "cusparse spmm op plan", &error);
            }
        }
    }
}

fn ltoir_parts(buffer: Option<&[u8]>) -> (*const u8, usize) {
    buffer.map_or((ptr::null(), 0), |buffer| (buffer.as_ptr(), buffer.len()))
}
