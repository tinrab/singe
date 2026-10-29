use std::ptr;

use singe_cuda::{data_type::DataTypeLike, view::DeviceOutputView};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    descriptor::workspace,
    error::Result,
    matrix::SparseMatrixDescriptor,
    operation::SpGemmDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpGemmAlgorithm},
    utility::to_usize,
};

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
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemm_work_estimation<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let external_buffer = workspace(context, external_buffer)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = external_buffer.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    // SAFETY: `cusparseSpGEMM_workEstimation` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_workEstimation(
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
            descriptor.as_raw(),
            &raw mut size,
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemm work estimation buffer size")
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
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemm_estimate_memory<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    chunk_fraction: f32,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<(usize, usize)> {
    let external_buffer = workspace(context, external_buffer)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    let mut buffer_size3 = external_buffer.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    let mut buffer_size2 = 0;
    // SAFETY: `cusparseSpGEMM_estimateMemory` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_estimateMemory(
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
            descriptor.as_raw(),
            chunk_fraction,
            &raw mut buffer_size3,
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
            &raw mut buffer_size2,
        ))?;
    }
    Ok((
        to_usize(buffer_size2, "spgemm compute buffer size")?,
        to_usize(buffer_size3, "spgemm estimate buffer size")?,
    ))
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
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemm_compute<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let external_buffer = workspace(context, external_buffer)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    let mut buffer_size2 = external_buffer.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    // SAFETY: `cusparseSpGEMM_compute` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_compute(
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
            descriptor.as_raw(),
            &raw mut buffer_size2,
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(buffer_size2, "spgemm compute buffer size")
}

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
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemm_copy<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;

    // SAFETY: `cusparseSpGEMM_copy` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_copy(
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
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
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
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemmreuse_work_estimation(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer1: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let external_buffer1 = workspace(context, external_buffer1)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    let mut size = external_buffer1.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    // SAFETY: `cusparseSpGEMMreuse_workEstimation` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_workEstimation(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
            external_buffer1
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemmreuse work estimation buffer size")
}

/// Computes nonzero counts for the reusable sparse matrix-matrix multiplication pipeline.
///
/// * This requires temporary extra storage that is allocated internally.
/// * Supports asynchronous execution if the Stream Ordered Memory Allocator is available.
/// * Supports CUDA graph capture if the Stream Ordered Memory Allocator is available.
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// An optional workspace must belong to the same CUDA context.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemmreuse_nnz(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer2: Option<DeviceOutputView<'_, u8>>,
    external_buffer3: Option<DeviceOutputView<'_, u8>>,
    external_buffer4: Option<DeviceOutputView<'_, u8>>,
) -> Result<(usize, usize, usize)> {
    let external_buffer2 = workspace(context, external_buffer2)?;
    let external_buffer3 = workspace(context, external_buffer3)?;
    let external_buffer4 = workspace(context, external_buffer4)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    let mut size2 = external_buffer2.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    let mut size3 = external_buffer3.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    let mut size4 = external_buffer4.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    // SAFETY: `cusparseSpGEMMreuse_nnz` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_nnz(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size2,
            external_buffer2
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
            &raw mut size3,
            external_buffer3
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
            &raw mut size4,
            external_buffer4
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok((
        to_usize(size2, "spgemmreuse nnz buffer2 size")?,
        to_usize(size3, "spgemmreuse nnz buffer3 size")?,
        to_usize(size4, "spgemmreuse nnz buffer4 size")?,
    ))
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
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemmreuse_copy(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer5: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let external_buffer5 = workspace(context, external_buffer5)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    let mut size = external_buffer5.map_or(0, singe_cuda::view::DeviceRegion::byte_len) as _;
    // SAFETY: `cusparseSpGEMMreuse_copy` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_copy(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
            external_buffer5
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemmreuse copy buffer size")
}

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
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spgemmreuse_compute<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    // SAFETY: `cusparseSpGEMMreuse_compute` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_compute(
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
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
}
