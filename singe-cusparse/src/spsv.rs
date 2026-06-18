use singe_cuda::{data_type::DataTypeLike, types::DevicePtr};

use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    error::Result,
    matrix::SparseMatrixDescriptor,
    operation::SpSvDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpSvAlgorithm, SpSvUpdate},
    utility::to_usize,
    vector::DenseVectorDescriptor,
};

pub fn spsv_buffer_size<Compute: DataTypeLike>(
    ctx: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
) -> Result<usize> {
    descriptor.ensure_context(ctx)?;
    matrix.ensure_context(ctx)?;
    x.ensure_context(ctx)?;
    y.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    let mut size = 0;
    unsafe {
        try_ffi!(sys::cusparseSpSV_bufferSize(
            ctx.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size,
        ))?;
    }
    to_usize(size, "spsv buffer size")
}

pub fn spsv_analysis<Compute: DataTypeLike>(
    ctx: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
    external_buffer: Option<DevicePtr>,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    matrix.ensure_context(ctx)?;
    x.ensure_context(ctx)?;
    y.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    unsafe {
        try_ffi!(sys::cusparseSpSV_analysis(
            ctx.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
            external_buffer.unwrap_or(DevicePtr::null()).as_ptr() as _,
        ))?;
    }
    Ok(())
}

pub fn spsv_solve<Compute: DataTypeLike>(
    ctx: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    matrix.ensure_context(ctx)?;
    x.ensure_context(ctx)?;
    y.ensure_context(ctx)?;
    ctx.bind()?;
    if ctx.scalar_pointer_mode()? != alpha.pointer_mode() {
        ctx.set_scalar_pointer_mode(alpha.pointer_mode())?;
    }

    unsafe {
        try_ffi!(sys::cusparseSpSV_solve(
            ctx.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn spsv_update_matrix(
    ctx: &Context,
    descriptor: &SpSvDescriptor,
    new_values: DevicePtr,
    update: SpSvUpdate,
) -> Result<()> {
    descriptor.ensure_context(ctx)?;
    ctx.bind()?;
    unsafe {
        try_ffi!(sys::cusparseSpSV_updateMatrix(
            ctx.as_raw(),
            descriptor.as_raw(),
            new_values.as_ptr() as _,
            update.into(),
        ))?;
    }
    Ok(())
}
