use std::ptr;

use singe_cuda::{data_type::DataTypeLike, types::DevicePtr};

use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    operation::SpMmOpPlan,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpMmAlgorithm},
    utility::to_usize,
};

/// Performs the multiplication of a sparse matrix `matrix_a` and a dense matrix `matrix_b`.
/// Formally, it computes `C = alpha * op(A) * op(B) + beta * C`, where:
///
/// * `op(A)` is a sparse matrix of size $m \times k$.
/// * `op(B)` is a dense matrix of size $k \times n$.
/// * `C` is a dense matrix of size $m \times n$.
/// * $\alpha$ and $\beta$ are scalars.
///
/// This can also multiply a dense matrix and a sparse matrix by switching the
/// dense matrix layouts. `B_C` and `C_C` indicate column-major layout, while
/// `B_R` and `C_R` indicate row-major layout.
///
/// `op(A)` is selected by `op_a` and may be `A`, `A^T`, or `A^H`.
/// `op(B)` is selected by `op_b` and may be `B`, `B^T`, or `B^H`.
///
/// When using the (conjugate) transpose of the sparse matrix `A`, this operation may produce slightly different results during different runs with the same input parameters.
///
/// [`spmm_buffer_size`] returns the size of the workspace needed by [`spmm`].
///
/// Calling [`spmm_preprocess`] is optional.
/// It may accelerate subsequent calls to [`spmm`].
/// Useful when [`spmm`] is called multiple times with the same sparsity
/// pattern (`matrix_a`).
/// It provides performance advantages with [`SpMmAlgorithm::Csr1`] or [`SpMmAlgorithm::Csr3`].
/// It has no effect for all other formats and algorithms.
///
/// Calling [`spmm_preprocess`] with `buffer` makes that buffer active for `matrix_a` SpMM calls.
/// Subsequent calls to [`spmm`] with `matrix_a` and the active buffer must use the same values for all parameters as the call to [`spmm_preprocess`].
/// The exceptions are: `alpha`, `beta`, `matrix_b`, `matrix_c`, and the values (but not indices) of `matrix_a` may be different.
/// Importantly, the buffer contents must be unmodified since the call to [`spmm_preprocess`].
/// When [`spmm`] is called with `matrix_a` and its active buffer, it may read acceleration data from the buffer.
///
/// Calling [`spmm_preprocess`] again with `matrix_a` and a new buffer makes the
/// new buffer active and makes the previously active buffer inactive.
/// For [`spmm`], there can only be one active buffer per sparse matrix at a time.
/// To get the effect of multiple active buffers for a single sparse matrix, create multiple matrix handles that all point to the same index and value buffers, and call [`spmm_preprocess`] once per handle with different workspace buffers.
///
/// Calling [`spmm`] with an inactive buffer is always permitted.
/// However, there may be no acceleration from the preprocessing in that case.
///
/// For the purposes of thread safety, [`spmm_preprocess`] is writing to `matrix_a` internal state.
///
/// [`spmm`] supports the following sparse matrix formats:
///
/// * [`Format::Coo`](crate::types::Format::Coo)
/// * [`Format::Csr`](crate::types::Format::Csr)
/// * [`Format::Csc`](crate::types::Format::Csc)
/// * [`Format::Bsr`](crate::types::Format::Bsr)
/// * [`Format::BlockedEll`](crate::types::Format::BlockedEll)
///
/// **(1) COO/CSR/CSC/BSR formats**
///
/// [`spmm`] supports the following index type for representing the sparse matrix `matrix_a`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// [`spmm`] supports the following data types:
///
/// Uniform-precision computation:
///
/// | `A`/`B`/`C`/`compute_type` |
/// | --- |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |
/// | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |
///
/// Mixed-precision computation:
///
/// | `A`/`B` | `C` | `compute_type` | Notes |
/// | --- | --- | --- | --- |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) |  |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |  |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |  |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |  |
/// | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) | Deprecated. |
/// | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) |  | Deprecated. |
///
/// [`DataType::F16`](singe_cuda::data_type::DataType::F16), [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16), [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16), and [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) data types always imply mixed-precision computation.
///
/// [`spmm`] supports the following algorithms:
///
/// * [`SpMmAlgorithm::Default`]: default algorithm for any sparse matrix format.
/// * [`SpMmAlgorithm::Coo1`]: algorithm 1 for COO sparse matrix format. May provide better performance for a small number of nonzero values, provides the best performance with column-major layout, supports batched computation, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Coo2`]: algorithm 2 for COO sparse matrix format. Provides deterministic results and the best performance with column-major layout. In general, it is slower than algorithm 1, supports batched computation, requires additional memory, and is identical to [`SpMmAlgorithm::Coo1`] if `op_a` is not [`Operation::NonTranspose`].
/// * [`SpMmAlgorithm::Coo3`]: algorithm 3 for COO sparse matrix format. May provide better performance for a large number of nonzero values and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Coo4`]: algorithm 4 for COO sparse matrix format. Provides better performance with row-major layout, supports batched computation, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr1`]: algorithm 1 for CSR/CSC sparse matrix format. Provides the best performance with column-major layout, supports batched computation, requires additional memory, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr2`]: algorithm 2 for CSR/CSC sparse matrix format. Provides the best performance with row-major layout, supports batched computation, requires additional memory, and may produce slightly different results across runs with the same input parameters.
/// * [`SpMmAlgorithm::Csr3`]: algorithm 3 for CSR sparse matrix format. Provides deterministic results and requires additional memory. It supports only CSR matrices with `op_a` set to [`Operation::NonTranspose`]; `op_b` cannot be [`Operation::ConjugateTranspose`], and the data type cannot be [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) or [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16).
/// * [`SpMmAlgorithm::Bsr1`]: algorithm 1 for BSR sparse matrix format. Provides deterministic results and requires no additional memory. It supports only `op_a` set to [`Operation::NonTranspose`]; the data type cannot be [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) or [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16), and blocks in `A` cannot be column-major.
///
/// When using [`spmm`] for mixed-precision computation on COO or CSR matrices, it defaults to algorithms [`SpMmAlgorithm::Coo2`] and [`SpMmAlgorithm::Csr3`], respectively.
/// If the required computation is not supported by those algorithms, the
/// mixed-precision operation fails.
///
/// **Performance notes:**
///
/// * Row-major layout provides higher performance than column-major.
/// * [`SpMmAlgorithm::Coo4`] and [`SpMmAlgorithm::Csr2`] are intended for row-major layout, while [`SpMmAlgorithm::Coo1`], [`SpMmAlgorithm::Coo2`], [`SpMmAlgorithm::Coo3`], and [`SpMmAlgorithm::Csr1`] are intended for column-major layout.
/// * When `beta` is not `1`, most algorithms scale the output matrix before the main computation.
/// * When `n` is `1`, this operation may use [`spmv`](crate::spmv::spmv).
///
/// [`spmm`] with all algorithms support the following batch modes except for [`SpMmAlgorithm::Csr3`]:
///
/// * $C\_{i} = A \cdot B\_{i}$
/// * $C\_{i} = A\_{i} \cdot B$
/// * $C\_{i} = A\_{i} \cdot B\_{i}$
///
/// The number of batches and their strides can be set by using [`SparseMatrixDescriptor::set_coo_strided_batch`], [`SparseMatrixDescriptor::set_csr_strided_batch`], and [`DenseMatrixDescriptor::set_strided_batch`].
/// The maximum number of batches for [`spmm`] is 65,535.
///
/// [`spmm`] has the following properties:
///
/// * Requires no extra storage for [`SpMmAlgorithm::Coo1`], [`SpMmAlgorithm::Coo3`], [`SpMmAlgorithm::Coo4`], [`SpMmAlgorithm::Bsr1`].
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run only for [`SpMmAlgorithm::Coo2`], [`SpMmAlgorithm::Csr3`], and [`SpMmAlgorithm::Bsr1`] algorithms.
/// * `compute-sanitizer` could report false race conditions for this operation.
///   These reports result from optimization and do not affect computation correctness.
/// * Allows the indices of `matrix_a` to be unsorted.
///
/// [`spmm`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
///
/// **(2) Blocked-ELLPACK format**
///
/// [`spmm`] supports the following data types for [`Format::BlockedEll`](crate::types::Format::BlockedEll) format and the following GPU architectures for exploiting NVIDIA Tensor Cores:
///
/// | `A`/`B` | `C` | `compute_type` | `op_b` | Compute Capability |
/// | --- | --- | --- | --- | --- |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | `N`, `T` | `≥ 70` |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | `N`, `T` | `≥ 70` |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | `N`, `T` | `≥ 70` |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) | `N` column-major, `T` row-major | `≥ 75` |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | `N`, `T` | `≥ 80` |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | `N`, `T` | `≥ 80` |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | `N`, `T` | `≥ 80` |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) | [`DataType::F64`](singe_cuda::data_type::DataType::F64) | [`DataType::F64`](singe_cuda::data_type::DataType::F64) | `N`, `T` | `≥ 80` |
///
/// [`spmm`] supports the following algorithms with [`Format::BlockedEll`](crate::types::Format::BlockedEll) format:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SpMmAlgorithm::Default`] | Default algorithm for any sparse matrix format. |
/// | [`SpMmAlgorithm::BlockedEll1`] | Default algorithm for Blocked-ELL format. |
///
/// **Performance notes:**
///
/// * Blocked-ELL SpMM provides the best performance with Power-of-2 Block-Sizes.
/// * Large block sizes, such as 64 or greater, provide the best performance.
///
/// This operation has the following limitations:
///
/// * The pointer mode must be [`PointerMode::Host`](crate::types::PointerMode::Host).
/// * Only `op_a` set to [`Operation::NonTranspose`] is supported.
/// * `op_b` set to [`Operation::ConjugateTranspose`] is not supported.
/// * Only [`IndexType::I32`](crate::types::IndexType::I32) is supported.
///
/// # Errors
///
/// Returns an error if the cuSPARSE context cannot be bound, the scalar pointer
/// modes do not match the handle configuration, the descriptors or selected
/// algorithm are incompatible, a required workspace buffer is missing or
/// invalid, the data type combination is unsupported, or cuSPARSE rejects the
/// operation.
pub fn spmm<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    unsafe {
        try_ffi!(sys::cusparseSpMM(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            beta.ptr().cast(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn spmm_buffer_size<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpMM_bufferSize(
            ctx.as_raw(),
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

pub fn spmm_preprocess<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpMmAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    unsafe {
        try_ffi!(sys::cusparseSpMM_preprocess(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            beta.ptr().cast(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn spmm_op(
    ctx: &Context,
    plan: &SpMmOpPlan<'_>,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    if !ptr::eq(ctx, plan.context()) {
        return Err(Error::PlanContextMismatch);
    }
    plan.execute(external_buffer)
}
