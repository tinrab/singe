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
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    operation::SpSmDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpSMAlgorithm, SpSmUpdate},
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
/// Input scalar locations must match; device scalars must belong to the same CUDA context.
/// Also propagates scalar pointer-mode query or selection errors.
/// Returned workspace sizes must fit in `usize`.
/// Also propagates CUDA context-binding errors and any failure status returned by cuSPARSE.
pub unsafe fn spsm_buffer_size<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<usize> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    let mut size = 0;
    // SAFETY: `cusparseSpSM_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusparseSpSM_bufferSize(
            context.as_raw(),
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
pub unsafe fn spsm_analysis<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
    external_buffer: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let external_buffer = workspace(context, external_buffer)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    // SAFETY: `cusparseSpSM_analysis` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpSM_analysis(
            context.as_raw(),
            op_a.into(),
            op_b.into(),
            alpha.ptr().cast(),
            matrix_a.as_raw_const(),
            matrix_b.as_raw_const(),
            matrix_c.as_raw(),
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
pub unsafe fn spsm_solve<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSMAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    // SAFETY: `cusparseSpSM_solve` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cusparseSpSM_solve(
            context.as_raw(),
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
pub unsafe fn spsm_update_matrix<T: DataTypeLike>(
    context: &Context,
    descriptor: &SpSmDescriptor,
    new_values: DeviceInputView<'_, T>,
    update: SpSmUpdate,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    if new_values.context() != context.cuda_context() {
        return Err(Error::PlanContextMismatch);
    }
    context.bind()?;
    // SAFETY: `cusparseSpSM_updateMatrix` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusparseSpSM_updateMatrix(
            context.as_raw(),
            descriptor.as_raw(),
            new_values.as_ptr().cast_mut().cast(),
            update.into(),
        ))?;
    }
    Ok(())
}
