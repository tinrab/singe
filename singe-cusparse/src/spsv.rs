use std::ptr;

use singe_cuda::{
    data_type::DataTypeLike,
    view::{DeviceInputView, DeviceOutputView},
};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    descriptor::workspace,
    error::{Error, Result},
    matrix::SparseMatrixDescriptor,
    operation::SpSvDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpSvAlgorithm, SpSvUpdate},
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
pub unsafe fn spsv_buffer_size<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
) -> Result<usize> {
    descriptor.ensure_context(context)?;
    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    let mut size = 0;
    // SAFETY: `cusparseSpSV_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseSpSV_bufferSize(
            context.as_raw(),
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
pub unsafe fn spsv_analysis<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    descriptor.ensure_context(context)?;
    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    // SAFETY: `cusparseSpSV_analysis` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpSV_analysis(
            context.as_raw(),
            operation.into(),
            alpha.ptr().cast(),
            matrix.as_raw_const(),
            x.as_raw_const(),
            y.as_raw(),
            Compute::data_type().into(),
            algorithm.into(),
            descriptor.as_raw(),
            external_buffer
                .map_or(ptr::null_mut(), singe_cuda::view::DeviceRegion::as_ptr)
                .cast(),
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
/// Input scalar locations must match; device scalars must belong to the same CUDA context.
/// Also propagates scalar pointer-mode query or selection errors.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spsv_solve<Compute: DataTypeLike>(
    context: &Context,
    operation: Operation,
    alpha: Scalar<'_, Compute>,
    matrix: &SparseMatrixDescriptor,
    x: &DenseVectorDescriptor,
    y: &mut DenseVectorDescriptor,
    algorithm: SpSvAlgorithm,
    descriptor: &SpSvDescriptor,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    matrix.ensure_context(context)?;
    x.ensure_context(context)?;
    y.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    // SAFETY: `cusparseSpSV_solve` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseSpSV_solve(
            context.as_raw(),
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

/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the
/// selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if an operand or analysis descriptor belongs to another CUDA context.
/// Replacement values must belong to that CUDA context.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spsv_update_matrix<T: DataTypeLike>(
    context: &Context,
    descriptor: &SpSvDescriptor,
    new_values: DeviceInputView<'_, T>,
    update: SpSvUpdate,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    if new_values.context() != context.cuda_context() {
        return Err(Error::PlanContextMismatch);
    }
    context.bind()?;
    // SAFETY: `cusparseSpSV_updateMatrix` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpSV_updateMatrix(
            context.as_raw(),
            descriptor.as_raw(),
            new_values.as_ptr().cast_mut().cast(),
            update.into(),
        ))?;
    }
    Ok(())
}
