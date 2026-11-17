use std::{ptr, slice};

use singe_cuda::{
    data_type::DataTypeLike,
    view::{DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegion},
};

use super::{DenseMatrixDescriptor, SparseMatrixDescriptor};
use crate::{
    context::Context,
    error::Result,
    sys, try_ffi,
    types::{Action, CsrToCscAlgorithm, DenseToSparseAlgorithm, IndexBase, SparseToDenseAlgorithm},
    utility::{checked_add, input_region, to_i32, to_usize, workspace_region},
};

/// Converts a sparse matrix in CSR format into a sparse matrix in CSC format.
///
/// The resulting matrix can also be seen as the transpose of the original sparse matrix.
/// This can also convert a matrix in CSC format into a matrix in CSR format.
///
/// This operation requires a workspace whose size grows with `nonzero_count`.
/// It always produces the same matrix values in the destination format.
///
/// The operation executes asynchronously with respect to the host and may return before the result is ready.
///
/// [`csr_to_csc_workspace_size`] returns the size of the workspace needed by [`csr_to_csc_async`].
/// Pass a buffer of this size to [`csr_to_csc_async`].
///
/// If `nonzero_count` is `0`, then the CSR column indices, CSR values, CSC values, and CSC row indices may be zero-length buffers.
/// In this case, all CSC column pointers are initialized to the configured index base.
///
/// If `row_count` or `col_count` is `0`, cuSPARSE does not read the arrays and the operation succeeds.
///
/// [`csr_to_csc_async`] supports the following data types:
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
/// [`csr_to_csc_async`] supports the following algorithms ([`CsrToCscAlgorithm`]):
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`CsrToCscAlgorithm::Default`] | Default algorithm. |
///
/// | Action | Notes |
/// | --- | --- |
/// | [`Action::Symbolic`] | Compute the structure of the CSC output matrix (offset, row indices). |
/// | [`Action::Numeric`] | Compute the structure of the CSC output matrix and copy the values. |
///
/// [`csr_to_csc_async`] supports asynchronous execution.
///
/// [`csr_to_csc_async`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small, an offset count overflows, a count does not fit in `i32`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn csr_to_csc_async<T: DataTypeLike>(
    context: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &(impl DeviceRange<T> + ?Sized),
    csr_row_offsets: &(impl DeviceRange<i32> + ?Sized),
    csr_column_indices: &(impl DeviceRange<i32> + ?Sized),
    column_ordered_values: &mut (impl DeviceOutput<T> + ?Sized),
    csc_column_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    csc_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    action: Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;
    let row_offsets = checked_add(row_count, 1, "CSR row offsets")?;
    let column_offsets = checked_add(col_count, 1, "CSC column offsets")?;
    let regions = [
        input_region(context, csr_values, nonzero_count, "csr_values")?,
        input_region(context, csr_row_offsets, row_offsets, "csr_row_offsets")?,
        input_region(
            context,
            csr_column_indices,
            nonzero_count,
            "csr_column_indices",
        )?,
        input_region(
            context,
            column_ordered_values,
            nonzero_count,
            "column_ordered_values",
        )?,
        input_region(
            context,
            csc_column_offsets,
            column_offsets,
            "csc_column_offsets",
        )?,
        input_region(context, csc_row_indices, nonzero_count, "csc_row_indices")?,
    ];
    context.bind()?;
    context.prepare_regions(&regions)?;
    context.prepare_operands(&[slice::from_ref(&workspace)])?;
    // SAFETY: every array was checked above to belong to this context and hold the counts the shape needs, and they and the workspace are ordered after their producers; the caller retains them per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2(
            context.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_device_ptr().cast(),
            csr_row_offsets.as_device_ptr().cast(),
            csr_column_indices.as_device_ptr().cast(),
            column_ordered_values.as_mut_ptr().cast(),
            csc_column_offsets.as_mut_ptr(),
            csc_row_indices.as_mut_ptr(),
            T::data_type().into(),
            action.into(),
            index_base.into(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Converts `matrix_a` in CSR, CSC, or COO format into its dense representation `matrix_b`.
/// Blocked-ELL is not currently supported.
///
/// [`sparse_to_dense_workspace_size`] returns the size of the workspace needed by [`sparse_to_dense_async`].
///
/// [`sparse_to_dense_async`] supports the following index type for representing `matrix_a`:
///
/// * 32-bit indices ([`IndexType::I32`](crate::types::IndexType::I32))
/// * 64-bit indices ([`IndexType::I64`](crate::types::IndexType::I64))
///
/// [`sparse_to_dense_async`] supports the following data types:
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
/// [`sparse_to_dense_async`] supports the following algorithm:
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`SparseToDenseAlgorithm::Default`] | Default algorithm. |
///
/// [`sparse_to_dense_async`] has the following properties:
///
/// * Requires no extra storage.
/// * Supports asynchronous execution.
/// * Provides deterministic (bitwise) results for each run.
/// * Allows the indices of `matrix_a` to be unsorted.
///
/// [`sparse_to_dense_async`] supports the following optimizations:
///
/// * CUDA graph capture.
/// * Hardware Memory Compression.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn sparse_to_dense_async(
    context: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        slice::from_ref(&workspace),
    ])?;
    // SAFETY: both descriptors and the workspace were checked against the bound `context` and ordered after their producers; the caller retains them per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseSparseToDense(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Returns the workspace size in bytes that a dense-to-sparse conversion needs.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor belongs to another CUDA context, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn dense_to_sparse_workspace_size(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: both descriptors were checked against `context`, which is bound, and `size` is a writable local; the query enqueues nothing.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_bufferSize(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "dense-to-sparse buffer size")
}

/// Enqueues the analysis phase of a dense-to-sparse conversion, which counts the nonzero entries.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn dense_to_sparse_analysis_async(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        slice::from_ref(&workspace),
    ])?;
    // SAFETY: both descriptors and the workspace were checked against the bound `context` and ordered after their producers; the caller retains them per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_analysis(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Enqueues the conversion of `matrix_a` into the sparse matrix `matrix_b`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn dense_to_sparse_convert_async(
    context: &Context,
    matrix_a: &DenseMatrixDescriptor,
    matrix_b: &mut SparseMatrixDescriptor,
    algorithm: DenseToSparseAlgorithm,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        slice::from_ref(&workspace),
    ])?;
    // SAFETY: both descriptors and the workspace were checked against the bound `context` and ordered after their producers; the caller ran the analysis with this workspace and retains every operand per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseDenseToSparse_convert(
            context.as_raw(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw(),
            algorithm.into(),
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Returns the workspace size in bytes that a CSR-to-CSC conversion needs.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small, an offset count overflows, a count does not fit in `i32`, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn csr_to_csc_workspace_size<T: DataTypeLike>(
    context: &Context,
    row_count: usize,
    col_count: usize,
    nonzero_count: usize,
    csr_values: &(impl DeviceRange<T> + ?Sized),
    csr_row_offsets: &(impl DeviceRange<i32> + ?Sized),
    csr_column_indices: &(impl DeviceRange<i32> + ?Sized),
    column_ordered_values: &mut (impl DeviceOutput<T> + ?Sized),
    csc_column_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    csc_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    action: Action,
    index_base: IndexBase,
    algorithm: CsrToCscAlgorithm,
) -> Result<usize> {
    let row_offsets = checked_add(row_count, 1, "CSR row offsets")?;
    let column_offsets = checked_add(col_count, 1, "CSC column offsets")?;
    input_region(context, csr_values, nonzero_count, "csr_values")?;
    input_region(context, csr_row_offsets, row_offsets, "csr_row_offsets")?;
    input_region(
        context,
        csr_column_indices,
        nonzero_count,
        "csr_column_indices",
    )?;
    input_region(
        context,
        column_ordered_values,
        nonzero_count,
        "column_ordered_values",
    )?;
    input_region(
        context,
        csc_column_offsets,
        column_offsets,
        "csc_column_offsets",
    )?;
    input_region(context, csc_row_indices, nonzero_count, "csc_row_indices")?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: every array was checked above to belong to this context and hold the counts the shape needs, the context is bound, and `size` is a writable local; the query enqueues nothing.
    unsafe {
        try_ffi!(sys::cusparseCsr2cscEx2_bufferSize(
            context.as_raw(),
            to_i32(row_count, "row_count")?,
            to_i32(col_count, "col_count")?,
            to_i32(nonzero_count, "nonzero_count")?,
            csr_values.as_device_ptr().cast(),
            csr_row_offsets.as_device_ptr().cast(),
            csr_column_indices.as_device_ptr().cast(),
            column_ordered_values.as_mut_ptr().cast(),
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
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small, an offset count overflows, a count does not fit in `i32`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn csr_to_coo_async(
    context: &Context,
    csr_row_offsets: &(impl DeviceRange<i32> + ?Sized),
    nonzero_count: usize,
    row_count: usize,
    coo_row_indices: &mut (impl DeviceOutput<i32> + ?Sized),
    index_base: IndexBase,
) -> Result<()> {
    let regions = [
        input_region(
            context,
            csr_row_offsets,
            checked_add(row_count, 1, "CSR row offsets")?,
            "csr_row_offsets",
        )?,
        input_region(context, coo_row_indices, nonzero_count, "coo_row_indices")?,
    ];
    context.bind()?;
    context.prepare_regions(&regions)?;
    // SAFETY: the offsets hold `row_count + 1` and the output `nonzero_count` elements of this context, both ordered after their producers; the caller retains them per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseXcsr2coo(
            context.as_raw(),
            csr_row_offsets.as_device_ptr().cast(),
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            coo_row_indices.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}

/// Returns the workspace size in bytes that a sparse-to-dense conversion needs.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor belongs to another CUDA context, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn sparse_to_dense_workspace_size(
    context: &Context,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &mut DenseMatrixDescriptor,
    algorithm: SparseToDenseAlgorithm,
) -> Result<usize> {
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    context.bind()?;
    let mut size = 0;
    // SAFETY: both descriptors were checked against `context`, which is bound, and `size` is a writable local; the query enqueues nothing.
    unsafe {
        try_ffi!(sys::cusparseSparseToDense_bufferSize(
            context.as_raw(),
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
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including index bounds, initialized inputs, produced output ranges, and scratch lifetimes.
///
/// # Errors
///
/// Returns an error if a buffer belongs to another CUDA context or is too small, an offset count overflows, a count does not fit in `i32`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn coo_to_csr_async(
    context: &Context,
    coo_row_indices: &(impl DeviceRange<i32> + ?Sized),
    row_count: usize,
    csr_row_offsets: &mut (impl DeviceOutput<i32> + ?Sized),
    index_base: IndexBase,
) -> Result<()> {
    let nonzero_count = coo_row_indices.len();
    let regions = [
        input_region(context, coo_row_indices, nonzero_count, "coo_row_indices")?,
        input_region(
            context,
            csr_row_offsets,
            checked_add(row_count, 1, "CSR row offsets")?,
            "csr_row_offsets",
        )?,
    ];
    context.bind()?;
    context.prepare_regions(&regions)?;
    // SAFETY: the indices hold `nonzero_count` and the output `row_count + 1` elements of this context, both ordered after their producers; the caller retains them per the raw execution contract.
    unsafe {
        try_ffi!(sys::cusparseXcoo2csr(
            context.as_raw(),
            coo_row_indices.as_device_ptr().cast(),
            to_i32(nonzero_count, "nonzero_count")?,
            to_i32(row_count, "row_count")?,
            csr_row_offsets.as_mut_ptr(),
            index_base.into(),
        ))?;
    }
    Ok(())
}
