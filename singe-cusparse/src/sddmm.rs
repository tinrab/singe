use singe_cuda::{data_type::DataTypeLike, types::DevicePtr};

use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::Result,
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    scalar::Scalar,
    try_ffi,
    types::{Operation, SddmmAlgorithm},
    utility::to_usize,
};

pub fn sddmm_buffer_size<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SddmmAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSDDMM_bufferSize(
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
    to_usize(size, "sddmm buffer size")
}

pub fn sddmm_preprocess<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SddmmAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    unsafe {
        try_ffi!(sys::cusparseSDDMM_preprocess(
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

/// Multiplies `matrix_a` and `matrix_b`, then applies the sparsity pattern of `matrix_c`.
/// Formally, it computes $C = \alpha (op(A) op(B)) \circ spy(C) + \beta C$,
/// where:
///
/// * `op(A)` is a dense matrix of size $m \times k$.
/// * `op(B)` is a dense matrix of size $k \times n$.
/// * `C` is a sparse matrix of size $m \times n$.
/// * `alpha` and `beta` are scalars.
/// * $\circ$ denotes the Hadamard (entry-wise) matrix product, and `spy(C)` is the structural sparsity pattern of `C`, equal to `1` where `C` stores an entry and `0` elsewhere.
///
/// `op(A)` is selected by `op_a` and may be `A`, `A^T`, or `A^H`.
/// `op(B)` is selected by `op_b` and may be `B`, `B^T`, or `B^H`.
///
/// [`sddmm_buffer_size`] returns the workspace size needed by [`sddmm`] or [`sddmm_preprocess`].
///
/// Calling [`sddmm_preprocess`] is optional.
/// It may accelerate subsequent calls to [`sddmm`].
/// Useful when [`sddmm`] is called multiple times with the same sparsity
/// pattern (`matrix_c`).
///
/// Calling [`sddmm_preprocess`] with `buffer` makes that buffer active for `matrix_c` SDDMM calls.
/// Subsequent calls to [`sddmm`] with `matrix_c` and the active buffer must use the same values for all parameters as the call to [`sddmm_preprocess`].
/// The exceptions are: `alpha`, `beta`, `matrix_a`, `matrix_b`, and the values (but not indices) of `matrix_c` may be different.
/// Importantly, the buffer contents must be unmodified since the call to [`sddmm_preprocess`].
/// When [`sddmm`] is called with `matrix_c` and its active buffer, it may read acceleration data from the buffer.
///
/// Calling [`sddmm_preprocess`] again with `matrix_c` and a new buffer makes the
/// new buffer active and makes the previously active buffer inactive.
/// For [`sddmm`], there can only be one active buffer per sparse matrix at a time.
/// To get the effect of multiple active buffers for a single sparse matrix, create multiple matrix handles that all point to the same index and value buffers, and call [`sddmm_preprocess`] once per handle with different workspace buffers.
///
/// Calling [`sddmm`] with an inactive buffer is always permitted.
/// However, there may be no acceleration from the preprocessing in that case.
///
/// For the purposes of thread safety, [`sddmm_preprocess`] is writing to `matrix_c` internal state.
///
/// Currently supported sparse matrix formats:
///
/// * [`Format::Csr`](crate::types::Format::Csr)
/// * [`Format::Bsr`](crate::types::Format::Bsr)
///
/// [`sddmm`] supports the following index type for representing `matrix_c`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// The data type combinations currently supported for [`sddmm`] are listed below:
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
/// | `A`/`B` | `C` | `compute_type` |
/// | --- | --- | --- |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |
/// | [`DataType::F16`](singe_cuda::data_type::DataType::F16) | [`DataType::F16`](singe_cuda::data_type::DataType::F16) |  |
///
/// [`sddmm`] for [`Format::Bsr`](crate::types::Format::Bsr) also supports the following mixed-precision computation:
///
/// | `A`/`B` | `C` | `compute_type` |
/// | --- | --- | --- |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) | [`DataType::F32`](singe_cuda::data_type::DataType::F32) |
/// | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) | [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) |  |
///
/// [`DataType::F16`](singe_cuda::data_type::DataType::F16) and [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) always imply mixed-precision computation.
///
/// [`sddmm`] for [`Format::Bsr`](crate::types::Format::Bsr) supports block sizes of 2, 4, 8, 16, 32, 64 and 128.
///
/// [`sddmm`] supports the following algorithms:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SddmmAlgorithm::Default`] | Default algorithm. |
///
/// It supports batched computation.
///
/// Performance notes: [`sddmm`] for [`Format::Csr`](crate::types::Format::Csr) provides the best performance when `matrix_a` and `matrix_b` satisfy:
///
/// * For `matrix_a`:
///
///   * `matrix_a` is in row-major order and `op_a` is [`Operation::NonTranspose`], or
///   * `matrix_a` is in col-major order and `op_a` is not [`Operation::NonTranspose`].
/// * For `matrix_b`:
///
///   * `matrix_b` is in col-major order and `op_b` is [`Operation::NonTranspose`], or
///   * `matrix_b` is in row-major order and `op_b` is not [`Operation::NonTranspose`].
///
/// [`sddmm`] for [`Format::Bsr`](crate::types::Format::Bsr) provides the best performance when `matrix_a` and `matrix_b` satisfy:
///
/// * For `matrix_a`:
///
///   * `matrix_a` is in row-major order and `op_a` is [`Operation::NonTranspose`], or
///   * `matrix_a` is in col-major order and `op_a` is not [`Operation::NonTranspose`].
/// * For `matrix_b`:
///
///   * `matrix_b` is in row-major order and `op_b` is [`Operation::NonTranspose`], or
///   * `matrix_b` is in col-major order and `op_b` is not [`Operation::NonTranspose`].
///
/// [`sddmm`] supports the following batch modes:
///
/// * $C\_{i} = (A \cdot B) \circ C\_{i}$
/// * $C\_{i} = \left( A\_{i} \cdot B \right) \circ C\_{i}$
/// * $C\_{i} = \left( A \cdot B\_{i} \right) \circ C\_{i}$
/// * $C\_{i} = \left( A\_{i} \cdot B\_{i} \right) \circ C\_{i}$
///
/// The number of batches and their strides can be set by using [`SparseMatrixDescriptor::set_csr_strided_batch`] and [`DenseMatrixDescriptor::set_strided_batch`].
/// The maximum number of batches for [`sddmm`] is 65,535.
///
/// [`sddmm`] has the following properties:
///
/// * Requires no extra storage.
/// * Provides deterministic (bitwise) results for each run.
/// * Supports asynchronous execution.
/// * Allows the indices of `matrix_c` to be unsorted.
///
/// [`sddmm`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
pub fn sddmm<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SddmmAlgorithm,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    unsafe {
        try_ffi!(sys::cusparseSDDMM(
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
