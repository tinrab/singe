use std::{ptr, slice};

use singe_cuda::{
    data_type::DataTypeLike,
    error::Error as CudaError,
    view::{DeviceInputView, DeviceOutputView, DeviceRegion},
};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    descriptor::analysis_descriptor,
    error::{Error, Result},
    matrix::{DenseMatrixDescriptor, SparseMatrixDescriptor},
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpSmAlgorithm, SpSmUpdate},
    utility::{to_usize, workspace_region},
};

analysis_descriptor! {
    /// Owns native sparse triangular solve analysis state for multiple right-hand sides.
    SpSmDescriptor {
        raw: sys::cusparseSpSMDescr_t,
        object: "cusparse spsm descriptor",
        create: sys::cusparseSpSM_createDescr,
        destroy: sys::cusparseSpSM_destroyDescr,
    }
}

/// Returns the workspace size in bytes that `SpSM` analysis and solve need.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or device scalar belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spsm_workspace_size<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSmAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<usize> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;

    let mut size = 0;
    // SAFETY: every descriptor was checked against `context`, which is bound with the scalars' pointer mode, and `size` is a writable local; the query enqueues nothing.
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

/// Enqueues the analysis phase of a sparse triangular solve with multiple right-hand sides.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor, device scalar, or workspace belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spsm_analysis_async<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSmAlgorithm,
    descriptor: &SpSmDescriptor,
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    let workspace = workspace_region(context, workspace)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region()],
    ])?;

    // SAFETY: descriptors, scalars, and the workspace were checked against the bound `context` and ordered after their producers; the caller retains them through completion per the raw execution contract.
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
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok(())
}

/// Enqueues the solve phase of a sparse triangular solve with multiple right-hand sides.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or device scalar belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spsm_solve_async<Compute: DataTypeLike>(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    alpha: Scalar<'_, Compute>,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &DenseMatrixDescriptor,
    matrix_c: &mut DenseMatrixDescriptor,
    algorithm: SpSmAlgorithm,
    descriptor: &SpSmDescriptor,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, alpha)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        &[alpha.region()],
    ])?;

    // SAFETY: descriptors and scalars were checked against the bound `context` and ordered after their producers; the caller ran the analysis on `descriptor` and retains every operand per the raw execution contract.
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

/// Enqueues an update of the matrix values stored in `SpSM` analysis state.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or replacement value buffer belongs to another CUDA context, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spsm_update_matrix_async<T: DataTypeLike>(
    context: &Context,
    descriptor: &SpSmDescriptor,
    new_values: DeviceInputView<'_, T>,
    update: SpSmUpdate,
) -> Result<()> {
    descriptor.ensure_context(context)?;
    if new_values.context() != context.cuda_context() {
        return Err(Error::Cuda(CudaError::MemoryContextMismatch));
    }

    context.bind()?;
    context.prepare_operands(&[&[Some(new_values.region())]])?;
    // SAFETY: `descriptor` and `new_values` were checked against the bound `context`, and the update is ordered after the values' producers; the caller retains `new_values` per the raw execution contract.
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
