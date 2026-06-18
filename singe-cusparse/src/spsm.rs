use singe_cuda::{data_type::DataTypeLike, types::DevicePtr};

use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::Result,
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    operation::SpSmDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpSMAlgorithm, SpSmUpdate},
    utility::to_usize,
};

pub fn spsm_buffer_size<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpSM_bufferSize(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "spsm buffer size")
}

pub fn spsm_analysis<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    unsafe {
        try_ffi!(sys::cusparseSpSM_analysis(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn spsm_solve<Compute: DataTypeLike>(
    ctx: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    matrix_a.ensure_context(ctx)?;
    matrix_b.ensure_context(ctx)?;
    matrix_c.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    unsafe {
        try_ffi!(sys::cusparseSpSM_solve(
            ctx.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn spsm_update_matrix(
    ctx: &Context,
    descriptor: &SpSmDescriptor,
    new_values: DevicePtr,
    update: SpSmUpdate,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseSpSM_updateMatrix(
            ctx.as_raw(),
            descriptor.as_raw(),
            new_values.as_ptr() as _,
            update.into(),
        ))?;
    }
    Ok(())
}
