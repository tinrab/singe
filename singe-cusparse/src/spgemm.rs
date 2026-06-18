use singe_cuda::{data_type::DataTypeLike, types::DevicePtr};

use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::Result,
    matrix::SparseMatrixDescriptor,
    operation::SpGemmDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpGemmAlgorithm},
    utility::to_usize,
};

pub fn spgemm_work_estimation<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer: Option<DevicePtr>,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_workEstimation(
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
            descriptor.as_raw(),
            &raw mut size,
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    to_usize(size, "spgemm work estimation buffer size")
}

pub fn spgemm_estimate_memory<Compute: DataTypeLike>(
    ctx: &Context,
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
    external_buffer: Option<DevicePtr>,
) -> Result<(usize, usize)> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    let mut buffer_size3 = 0;
    let mut buffer_size2 = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_estimateMemory(
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
            descriptor.as_raw(),
            chunk_fraction,
            &raw mut buffer_size3,
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
            &raw mut buffer_size2,
        ))?;
    }
    Ok((
        to_usize(buffer_size2, "spgemm compute buffer size")?,
        to_usize(buffer_size3, "spgemm estimate buffer size")?,
    ))
}

pub fn spgemm_compute<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    beta: Scalar<'_, Compute>,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer: Option<DevicePtr>,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    let mut buffer_size2 = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMM_compute(
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
            descriptor.as_raw(),
            &raw mut buffer_size2,
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    to_usize(buffer_size2, "spgemm compute buffer size")
}

pub fn spgemm_copy<Compute: DataTypeLike>(
    ctx: &Context,
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
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;

    unsafe {
        try_ffi!(sys::cusparseSpGEMM_copy(
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
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn spgemmreuse_work_estimation(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer1: Option<DevicePtr>,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_workEstimation(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
            external_buffer1.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    to_usize(size, "spgemmreuse work estimation buffer size")
}

/// Computes nonzero counts for the reusable sparse matrix-matrix multiplication pipeline.
///
/// * This requires temporary extra storage that is allocated internally.
/// * Supports asynchronous execution if the Stream Ordered Memory Allocator is available.
/// * Supports CUDA graph capture if the Stream Ordered Memory Allocator is available.
pub fn spgemmreuse_nnz(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer2: Option<DevicePtr>,
    external_buffer3: Option<DevicePtr>,
    external_buffer4: Option<DevicePtr>,
) -> Result<(usize, usize, usize)> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    let mut size2 = 0;
    let mut size3 = 0;
    let mut size4 = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_nnz(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size2,
            external_buffer2.unwrap_or(DevicePtr::null()).as_ptr() as _,
            &raw mut size3,
            external_buffer3.unwrap_or(DevicePtr::null()).as_ptr() as _,
            &raw mut size4,
            external_buffer4.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok((
        to_usize(size2, "spgemmreuse nnz buffer2 size")?,
        to_usize(size3, "spgemmreuse nnz buffer3 size")?,
        to_usize(size4, "spgemmreuse nnz buffer4 size")?,
    ))
}

pub fn spgemmreuse_copy(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    external_buffer5: Option<DevicePtr>,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_copy(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
            external_buffer5.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    to_usize(size, "spgemmreuse copy buffer size")
}

pub fn spgemmreuse_compute<Compute: DataTypeLike>(
    ctx: &Context,
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
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    ctx.require_scalar_pointer_mode(alpha, beta)?;
    unsafe {
        try_ffi!(sys::cusparseSpGEMMreuse_compute(
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
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
}
