use std::ptr;

use singe_cuda::{data_type::DataTypeLike, view::DeviceOutputView};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    descriptor::workspace,
    error::Result,
    matrix::SparseMatrixDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpMvAlgorithm},
    utility::to_usize,
    vector::DenseVectorDescriptor,
};

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Input scalar locations must match; device scalars must belong to the same CUDA context.
/// Also propagates scalar pointer-mode query or selection errors.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spmv_buffer_size<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    beta: Scalar<'_, Compute>,
    y: &mut DenseVectorDescriptor,
    algorithm: SpMvAlgorithm,
) -> Result<usize> {
    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = 0;
    // SAFETY: `cusparseSpMV_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseSpMV_bufferSize(
            context.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            beta.ptr().cast(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "spmv buffer size")
}

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Input scalar locations must match; device scalars must belong to the same CUDA context.
/// Also propagates scalar pointer-mode query or selection errors.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spmv_preprocess<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    beta: Scalar<'_, Compute>,
    y: &mut DenseVectorDescriptor,
    algorithm: SpMvAlgorithm,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    // SAFETY: `cusparseSpMV_preprocess` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpMV_preprocess(
            context.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            beta.ptr().cast(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Multiplies sparse matrix `matrix` by dense vector `x`.
/// Formally, it computes `y = alpha * op(A) * x + beta * y`, where:
///
/// * `op(A)` is a sparse matrix of size $m \times k$.
/// * `X` is a dense vector of size $k$.
/// * `Y` is a dense vector of size $m$.
/// * $\alpha$ and $\beta$ are scalars.
///
/// `op(A)` is selected by `operation` and may be `A`, `A^T`, or `A^H`.
///
/// [`spmv_buffer_size`] returns the workspace size needed by [`spmv_preprocess`] and [`spmv`].
///
/// The sparse matrix formats currently supported are listed below:
///
/// * [`Format::Coo`](crate::types::Format::Coo)
/// * [`Format::Csr`](crate::types::Format::Csr)
/// * [`Format::Csc`](crate::types::Format::Csc)
/// * [`Format::Bsr`](crate::types::Format::Bsr)
/// * [`Format::SlicedEllpack`](crate::types::Format::SlicedEllpack)
///
/// [`spmv`] supports the following index type for representing the sparse matrix `matrix`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// [`spmv`] supports the following data types:
///
/// Uniform-precision computation:
///
/// | `A`/`X`/`Y`/`compute_type` |
/// | --- |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |
/// | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |
///
/// Mixed-precision computation:
///
/// | `A`/`X` | `Y` | `compute_type` | Notes |
/// | --- | --- | --- | --- |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) | [`DataType::I32`](singe_cuda::data_type::DataType::I32) |  |
/// | [`DataType::I8`](singe_cuda::data_type::DataType::I8) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |  |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |  |  |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |  |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |  |
/// | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |  |
/// | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) | [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) |  | Deprecated. |
/// | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) | [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) |  | Deprecated. |
///
/// | `A` | `X`/`Y`/`compute_type` |
/// | --- | --- |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F64`](singe_cuda::data_type::DataType::F64) |
///
/// Mixed Regular/Complex computation:
///
/// | `A` | `X`/`Y`/`compute_type` |
/// | --- | --- |
/// | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::ComplexF32`](singe_cuda::data_type::DataType::ComplexF32) |
/// | [`DataType::F64`](singe_cuda::data_type::DataType::F64) | [`DataType::ComplexF64`](singe_cuda::data_type::DataType::ComplexF64) |
///
/// [`DataType::F16`](singe_cuda::data_type::DataType::F16), [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16), [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16), and [`DataType::ComplexBf16`](singe_cuda::data_type::DataType::ComplexBf16) data types always imply mixed-precision computation.
///
/// [`spmv`] supports the following algorithms:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SpMvAlgorithm::Default`] | Default algorithm for any sparse matrix format. |
/// | [`SpMvAlgorithm::Coo1`] | Default algorithm for COO sparse matrix format. May produce slightly different results during different runs with the same input parameters. |
/// | [`SpMvAlgorithm::Coo2`] | Provides deterministic (bitwise) results for each run. If `operation` is not [`Operation::NonTranspose`], it is identical to [`SpMvAlgorithm::Coo1`]. |
/// | [`SpMvAlgorithm::Csr1`] | Default algorithm for CSR/CSC sparse matrix format. May produce slightly different results during different runs with the same input parameters. |
/// | [`SpMvAlgorithm::Csr2`] | Provides deterministic (bitwise) results for each run. If `operation` is not [`Operation::NonTranspose`], it is identical to [`SpMvAlgorithm::Csr1`]. |
/// | [`SpMvAlgorithm::Sell1`] | Default algorithm for Sliced Ellpack sparse matrix format. Provides deterministic (bitwise) results for each run. |
/// | [`SpMvAlgorithm::Bsr1`] | Default algorithm for BSR sparse matrix format. Provides deterministic (bitwise) results for each run. Supports only [`Operation::NonTranspose`]. Supports both row-major and column-major block layouts in `A`. |
///
/// Calling [`spmv_preprocess`] is optional.
/// It may accelerate subsequent calls to [`spmv`].
/// Useful when [`spmv`] is called multiple times with the same sparsity
/// pattern (`matrix`).
///
/// Calling [`spmv_preprocess`] with `buffer` makes that buffer active for `matrix` `SpMV` calls.
/// Subsequent calls to [`spmv`] with `matrix` and the active buffer must use the same values for all parameters as the call to [`spmv_preprocess`].
/// The exceptions are: `alpha`, `beta`, `x`, `y`, and the values (but not indices) of `matrix` may be different.
/// Importantly, the buffer contents must be unmodified since the call to [`spmv_preprocess`].
/// When [`spmv`] is called with `matrix` and its active buffer, it may read acceleration data from the buffer.
///
/// Calling [`spmv_preprocess`] again with `matrix` and a new buffer makes the new
/// buffer active and makes the previously active buffer inactive.
/// For [`spmv`], there can only be one active buffer per sparse matrix at a time.
/// To get the effect of multiple active buffers for a single sparse matrix, create multiple matrix handles that all point to the same index and value buffers, and call [`spmv_preprocess`] once per handle with different workspace buffers.
///
/// Calling [`spmv`] with an inactive buffer is always permitted.
/// However, there may be no acceleration from the preprocessing in that case.
///
/// For the purposes of thread safety, [`spmv_preprocess`] is writing to `matrix` internal state.
///
/// **Performance notes:**
///
/// * [`SpMvAlgorithm::Coo1`] and [`SpMvAlgorithm::Csr1`] provide higher performance than [`SpMvAlgorithm::Coo2`] and [`SpMvAlgorithm::Csr2`].
/// * In general, [`Operation::NonTranspose`] is 3x faster than transpose or conjugate-transpose operations.
/// * Using [`spmv_preprocess`] helps improve performance of [`spmv`] in CSR.
///   It is beneficial when [`spmv`] runs multiple times with the same matrix
///   ([`spmv_preprocess`] is executed only once).
///
/// [`spmv`] has the following properties:
///
/// * Requires extra storage for CSR/CSC format (all algorithms) and for COO format with [`SpMvAlgorithm::Coo2`] algorithm.
/// * Provides deterministic (bitwise) results for each run only for [`SpMvAlgorithm::Coo2`], [`SpMvAlgorithm::Csr2`] and [`SpMvAlgorithm::Bsr1`] algorithms, and only with [`Operation::NonTranspose`].
/// * Supports asynchronous execution.
/// * `compute-sanitizer` could report false race conditions for this operation when `beta` is `0`.
///   These reports result from optimization and do not affect computation correctness.
/// * Allows the indices of `matrix` to be unsorted.
///
/// [`spmv`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Input scalar locations must match; device scalars must belong to the same CUDA context.
/// Also propagates scalar pointer-mode query or selection errors.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spmv<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    beta: Scalar<'_, Compute>,
    y: &mut DenseVectorDescriptor,
    algorithm: SpMvAlgorithm,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    // SAFETY: `cusparseSpMV` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpMV(
            context.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            beta.ptr().cast(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}
