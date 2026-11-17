use std::{ptr, slice};

use singe_cuda::{
    data_type::DataTypeLike,
    view::{DeviceOutputView, DeviceRegion},
};
use singe_cusparse_sys as sys;

use crate::{
    context::Context,
    descriptor::analysis_descriptor,
    error::Result,
    matrix::SparseMatrixDescriptor,
    scalar::Scalar,
    try_ffi,
    types::{Operation, SpGemmAlgorithm},
    utility::{to_size_t, to_usize, workspace_region},
};

analysis_descriptor! {
    /// Owns native analysis state for sparse-sparse matrix multiplication.
    SpGemmDescriptor {
        raw: sys::cusparseSpGEMMDescr_t,
        object: "cusparse spgemm descriptor",
        create: sys::cusparseSpGEMM_createDescr,
        destroy: sys::cusparseSpGEMM_destroyDescr,
    }
}

impl SpGemmDescriptor {
    /// Returns the number of intermediate products the estimation phase counted.
    ///
    /// # Safety
    ///
    /// The estimation phase on this descriptor must have completed.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, cuSPARSE rejects the query, or the count is negative.
    pub unsafe fn product_count(&self) -> Result<usize> {
        self.cuda_context().bind()?;
        let mut value = 0_i64;
        // SAFETY: the descriptor is live, the output location is valid for writes, and the caller completed the estimation phase.
        unsafe {
            try_ffi!(sys::cusparseSpGEMM_getNumProducts(
                self.as_raw(),
                &raw mut value,
            ))?;
        }

        to_usize(value, "spgemm product count")
    }
}

/// Enqueues the work estimation phase of `SpGEMM`, or returns its workspace size in bytes when `workspace` is `None`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor, device scalar, or workspace belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_work_estimation_async<Compute: DataTypeLike>(
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
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let workspace = workspace_region(context, workspace)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region(), beta.region()],
    ])?;

    let mut size = to_size_t(
        workspace.map_or(0, DeviceRegion::byte_len),
        "workspace size",
    )?;
    // SAFETY: descriptors, scalars, and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemm work estimation buffer size")
}

/// Enqueues the memory estimation phase of `SpGEMM`, returning the compute and estimation workspace sizes in bytes.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor, device scalar, or workspace belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_estimate_memory_async<Compute: DataTypeLike>(
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
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<(usize, usize)> {
    let workspace = workspace_region(context, workspace)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region(), beta.region()],
    ])?;

    let mut buffer_size3 = to_size_t(
        workspace.map_or(0, DeviceRegion::byte_len),
        "workspace size",
    )?;
    let mut buffer_size2 = 0;
    // SAFETY: descriptors, scalars, and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
            &raw mut buffer_size2,
        ))?;
    }
    Ok((
        to_usize(buffer_size2, "spgemm compute buffer size")?,
        to_usize(buffer_size3, "spgemm estimate buffer size")?,
    ))
}

/// Enqueues the compute phase of `SpGEMM`, or returns its workspace size in bytes when `workspace` is `None`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor, device scalar, or workspace belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_compute_async<Compute: DataTypeLike>(
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
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let workspace = workspace_region(context, workspace)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.require_scalar_pointer_mode(alpha, beta)?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace),
        &[alpha.region(), beta.region()],
    ])?;

    let mut buffer_size2 = to_size_t(
        workspace.map_or(0, DeviceRegion::byte_len),
        "workspace size",
    )?;
    // SAFETY: descriptors, scalars, and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(buffer_size2, "spgemm compute buffer size")
}

/// Enqueues the copy of the computed `SpGEMM` result into `matrix_c`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or device scalar belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_copy_async<Compute: DataTypeLike>(
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
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        &[alpha.region(), beta.region()],
    ])?;

    // SAFETY: descriptors and scalars were checked against the bound `context` and ordered after their producers; the caller ran the preceding phases on `descriptor` and retains every operand per the raw execution contract.
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

/// Enqueues the work estimation phase of reusable `SpGEMM`, or returns its workspace size in bytes when `workspace1` is `None`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_reuse_work_estimation_async(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    workspace1: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let workspace1 = workspace_region(context, workspace1)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace1),
    ])?;
    let mut size = to_size_t(
        workspace1.map_or(0, DeviceRegion::byte_len),
        "workspace1 size",
    )?;
    // SAFETY: descriptors and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace1
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemm reuse work estimation buffer size")
}

/// Computes nonzero counts for the reusable sparse matrix-matrix multiplication pipeline.
///
/// * This requires temporary extra storage that is allocated internally.
/// * Supports asynchronous execution if the Stream Ordered Memory Allocator is available.
/// * Supports CUDA graph capture if the Stream Ordered Memory Allocator is available.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_reuse_nnz_async(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    workspace2: Option<DeviceOutputView<'_, u8>>,
    workspace3: Option<DeviceOutputView<'_, u8>>,
    workspace4: Option<DeviceOutputView<'_, u8>>,
) -> Result<(usize, usize, usize)> {
    let workspace2 = workspace_region(context, workspace2)?;
    let workspace3 = workspace_region(context, workspace3)?;
    let workspace4 = workspace_region(context, workspace4)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace2),
        slice::from_ref(&workspace3),
        slice::from_ref(&workspace4),
    ])?;
    let mut size2 = to_size_t(
        workspace2.map_or(0, DeviceRegion::byte_len),
        "workspace2 size",
    )?;
    let mut size3 = to_size_t(
        workspace3.map_or(0, DeviceRegion::byte_len),
        "workspace3 size",
    )?;
    let mut size4 = to_size_t(
        workspace4.map_or(0, DeviceRegion::byte_len),
        "workspace4 size",
    )?;
    // SAFETY: descriptors and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace2
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
            &raw mut size3,
            workspace3
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
            &raw mut size4,
            workspace4
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    Ok((
        to_usize(size2, "spgemm reuse nnz buffer2 size")?,
        to_usize(size3, "spgemm reuse nnz buffer3 size")?,
        to_usize(size4, "spgemm reuse nnz buffer4 size")?,
    ))
}

/// Enqueues the structure copy phase of reusable `SpGEMM`, or returns its workspace size in bytes when `workspace5` is `None`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or workspace belongs to another CUDA context, the returned size does not fit in `usize`, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_reuse_copy_async(
    context: &Context,
    op_a: Operation,
    op_b: Operation,
    matrix_a: &SparseMatrixDescriptor,
    matrix_b: &SparseMatrixDescriptor,
    matrix_c: &mut SparseMatrixDescriptor,
    algorithm: SpGemmAlgorithm,
    descriptor: &SpGemmDescriptor,
    workspace5: Option<DeviceOutputView<'_, u8>>,
) -> Result<usize> {
    let workspace5 = workspace_region(context, workspace5)?;

    descriptor.ensure_context(context)?;
    matrix_a.ensure_context(context)?;
    matrix_b.ensure_context(context)?;
    matrix_c.ensure_context(context)?;
    context.bind()?;
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        slice::from_ref(&workspace5),
    ])?;
    let mut size = to_size_t(
        workspace5.map_or(0, DeviceRegion::byte_len),
        "workspace5 size",
    )?;
    // SAFETY: descriptors and workspaces were checked against the bound `context` and ordered after their producers, each size passed in is its workspace's byte length, and the caller retains them per the raw execution contract.
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
            workspace5
                .map_or(ptr::null_mut(), DeviceRegion::as_ptr)
                .cast(),
        ))?;
    }
    to_usize(size, "spgemm reuse copy buffer size")
}

/// Enqueues the numeric compute phase of reusable `SpGEMM`.
///
/// # Safety
///
/// Uphold the [raw execution contract](crate#raw-execution), including the selected operation's descriptor ABI, sparse index bounds, and resource lifetimes.
///
/// # Errors
///
/// Returns an error if a descriptor or device scalar belongs to another CUDA context, the input scalars mix host and device locations, the scalar pointer mode cannot be queried or set, the CUDA context cannot be bound, or cuSPARSE reports a failure.
pub unsafe fn spgemm_reuse_compute_async<Compute: DataTypeLike>(
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
    context.prepare_operands(&[
        matrix_a.regions(),
        matrix_b.regions(),
        matrix_c.regions(),
        &[alpha.region(), beta.region()],
    ])?;
    // SAFETY: descriptors and scalars were checked against the bound `context` and ordered after their producers; the caller ran the preceding phases on `descriptor` and retains every operand per the raw execution contract.
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
