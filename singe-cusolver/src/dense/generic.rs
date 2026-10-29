use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    view::{DeviceOutputView, DeviceView},
};

use crate::{
    context::Context,
    dense::validation::{
        require_host_workspace, require_info_buffer, require_pivot64_buffer,
        require_workspace_bytes, validate_x_matrix, validate_xlarft_inputs,
    },
    error::{Error, Result},
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixRef, VectorRef, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{DiagonalType, DirectMode, FillMode, Operation, StorevMode},
    utility::{to_i64, to_usize},
};

/// Queries device and host workspace capacities in bytes for Cholesky factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xpotrf_buffer_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXpotrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use [`xpotrf_buffer_size`] to calculate the sizes needed for pre-allocated
/// workspace.
///
/// Computes the Cholesky factorization of a Hermitian positive-definite matrix.
///
/// `A` is an $n \times n$ Hermitian matrix; only its lower or upper triangular
/// part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// The operation leaves the other part untouched.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular Cholesky factor `L`.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular Cholesky factor `U`.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xpotrf_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If Cholesky factorization fails, some leading minor of `A` is not positive
/// definite, or equivalently some diagonal element of `L` or `U` is not a real
/// number.
/// `dev_info` reports the smallest leading minor of `A` that is not positive definite.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Currently, [`xpotrf`] supports only the default algorithm.
///
/// **Algorithms supported by [`xpotrf`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// List of input arguments for [`xpotrf_buffer_size`] and [`xpotrf`]:
///
/// The generic cuSOLVER routine separates matrix and compute data types: `data_type_a` is
/// the data type of matrix `A`, and `compute_type` is the operation's compute
/// type.
/// [`xpotrf`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **`data_type_a`** | **`compute_type`** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SPOTRF` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DPOTRF` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CPOTRF` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZPOTRF` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, or if cuSOLVER reports
/// an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xpotrf<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&dev_info)?;
    context.validate_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xpotrf_buffer_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXpotrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Solves a system of linear equations.
///
/// Here `A` is an $n \times n$ Hermitian matrix; only its lower or upper
/// triangular part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// The operation leaves the other part untouched.
///
/// Call [`xpotrf`] first to factorize matrix `A`.
/// If `fill_mode` is [`FillMode::Lower`], `A` is lower triangular Cholesky factor `L` corresponding to $A = L\cdot L^{H}$.
/// If `fill_mode` is [`FillMode::Upper`], `A` is upper triangular Cholesky factor `U` corresponding to $A = U^{H}\cdot U$.
///
/// The operation is in-place, that is, matrix `X` overwrites matrix `B` with the same leading dimension `ldb`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Currently, [`xpotrs`] supports only the default algorithm.
///
/// **Algorithms supported by [`xpotrs`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// List of input arguments for [`xpotrs`]:
///
/// The generic cuSOLVER routine separates matrix data types: `data_type_a` is the data type
/// of matrix `A`, and `data_type_b` is the data type of matrix `B`.
/// [`xpotrs`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **`data_type_a`** | **`data_type_b`** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SPOTRS` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DPOTRS` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CPOTRS` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZPOTRS` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, right-hand-side count, or leading dimensions are
/// invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xpotrs<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    mut b: MatrixMut<'_, TB>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&dev_info)?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        b_type,
    )?;
    require_info_buffer(dev_info.len())?;
    // SAFETY: `cusolverDnXpotrs` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrs(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            b_type.into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device and host workspace capacities in bytes for triangular inversion.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xtrtri_buffer_size<TA: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    a: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXtrtri_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            diagonal_type.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            ptr::null_mut(),
            to_i64(a.leading_dimension, "lda")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// Computes the inverse of a triangular matrix through the generic cuSOLVER routine.
///
/// `A` is an $n \times n$ triangular matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// The other triangular part is left unchanged.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular inverse.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular inverse.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xtrtri_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If matrix inversion fails, `dev_info = i` shows `A(i, i) = 0`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// List of input arguments for [`xtrtri_buffer_size`] and [`xtrtri`]:
///
/// **Valid data types**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | data type | Meaning |
/// | [`DataType::F32`] | `STRTRI` |
/// | [`DataType::F64`] | `DTRTRI` |
/// | [`DataType::ComplexF32`] | `CTRTRI` |
/// | [`DataType::ComplexF64`] | `ZTRTRI` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the data type is not
/// supported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xtrtri<TA: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&dev_info)?;
    context.validate_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xtrtri_buffer_size::<TA>(context, fill_mode, diagonal_type, n, a.layout())?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXtrtri` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri(
            context.as_raw(),
            fill_mode.into(),
            diagonal_type.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device and host workspace capacities in bytes for LU factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgetrf_buffer_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(m, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXgetrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Computes the LU factorization of an $m \times n$ matrix
///
/// where `A` is an $m \times n$ matrix, `P` is a permutation matrix, `L` is a lower triangular matrix with unit diagonal, and `U` is an upper triangular matrix.
///
/// If LU factorization failed, that is, matrix `A` (`U`) is singular, `dev_info = i` indicates `U(i,i) = 0`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// If `pivots` is `None`, no pivoting is performed.
/// The factorization is `A=L*U`, which is not numerically stable.
///
/// Whether LU factorization succeeds or fails, `pivots` contains the pivoting
/// sequence. Row `i` is interchanged with row `pivots[i]`.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgetrf_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// Callers can combine [`xgetrf`] and [`xgetrs`] to complete a linear solver.
///
/// Currently, [`xgetrf`] supports two algorithms.
/// To select the legacy implementation, call [`Params::set_adv_options`].
///
/// **Algorithms supported by [`xgetrf`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Fastest algorithm; requires a large workspace of `m*n` elements. |
/// | [`AlgorithmMode::Algorithm1`](crate::types::AlgorithmMode::Algorithm1) | Legacy implementation. |
///
/// List of input arguments for [`xgetrf_buffer_size`] and [`xgetrf`]:
///
/// The generic cuSOLVER routine has two data types: `data_type_a` is the data type of matrix `A`, and `compute_type` is the operation's compute type.
/// [`xgetrf`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **`data_type_a`** | **`compute_type`** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SGETRF` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DGETRF` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGETRF` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGETRF` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, or if cuSOLVER reports
/// an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xgetrf<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    context.validate_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), m.min(n))?;
    }
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xgetrf_buffer_size::<TA>(context, params, m, n, a.layout(), compute_type)?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXgetrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Solves a linear system of multiple right-hand sides
///
/// where `A` is an $n \times n$ matrix, and was LU-factored by [`xgetrf`], that is, lower triangular part of A is `L`, and upper triangular part (including diagonal elements) of `A` is `U`.
/// `B` is an $n \times {nrhs}$ right-hand side matrix.
///
/// The `operation` argument is described by [`Operation`].
///
/// `pivots` is an output of [`xgetrf`].
/// It contains the pivot indices used to permute the right-hand sides.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Callers can combine [`xgetrf`] and [`xgetrs`] to complete a linear solver.
///
/// Currently, [`xgetrs`] supports only the default algorithm.
///
/// **Algorithms supported by [`xgetrs`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// List of input arguments for [`xgetrs`]:
///
/// The generic cuSOLVER routine has two data types: `data_type_a` is the data type of matrix `A`, and `data_type_b` is the data type of matrix `B`.
/// [`xgetrs`] only supports the following four combinations:
///
/// **Valid combination of data type and compute type**
///
/// | **`data_type_a`** | **`data_type_b`** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SGETRS` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DGETRS` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGETRS` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGETRS` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimensions are invalid, or if cuSOLVER reports
/// an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xgetrs<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    operation: Operation,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: DeviceView<'_, i64>,
    mut b: MatrixMut<'_, TB>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&pivots)?;
    context.validate_region(&dev_info)?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    require_pivot64_buffer(pivots.len(), n)?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        b_type,
    )?;
    require_info_buffer(dev_info.len())?;
    // SAFETY: `cusolverDnXgetrs` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrs(
            context.as_raw(),
            params.as_raw(),
            operation.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.as_ptr().cast(),
            b_type.into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device and host workspace capacities in bytes for symmetric indefinite solving.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsytrs_buffer_size<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, nrhs)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXsytrs_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            TB::data_type().into(),
            ptr::null_mut(),
            to_i64(b.leading_dimension, "ldb")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// Solves a system of linear equations through the generic cuSOLVER routine.
///
/// `A` contains the factorization produced by [`xsytrf`].
/// Only the lower or upper part is meaningful; the other part is left untouched.
///
/// Provide the pivot indices returned by the matching [`xsytrf`] operation, along
/// with device and host workspace through `workspace`.
/// Use [`xsytrs_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
/// To factorize and solve the symmetric system without pivoting, pass `None`
/// for the pivot buffer to both the matching [`xsytrf`] operation and [`xsytrs`].
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// List of input arguments for [`xsytrs_buffer_size`] and [`xsytrs`]:
///
/// The generic cuSOLVER routine has two data types: `data_type_a` is the data type of the
/// matrix `A`, and `data_type_b` is the data type of the matrix `B`.
/// [`xsytrs`] only supports the following four combinations:
///
/// **Valid combination of data type and compute type**
///
/// | **`data_type_a`** | **`data_type_b`** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SSYTRS` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DSYTRS` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CSYTRS` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZSYTRS` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the matrix data type
/// is not supported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xsytrs<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<DeviceView<'_, i64>>,
    mut b: MatrixMut<'_, TB>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    context.validate_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xsytrs_buffer_size::<TA, TB>(context, fill_mode, n, nrhs, a.layout(), b.layout())?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXsytrs` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs(
            context.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null(), |buffer| buffer.as_ptr()),
            TB::data_type().into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device and host workspace capacities in bytes for formation of the triangular factor of a block reflector.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xlarft_buffer_size<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    context: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixLayout,
    t: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    v.validate(n, k)
        .and_then(|()| t.validate(k, k))
        .and_then(|()| {
            if k > n || storev != StorevMode::Columnwise {
                Err(Error::InvalidMatrixShape)
            } else {
                Ok(())
            }
        })?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXlarft_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXlarft_bufferSize(
            context.as_raw(),
            params.as_raw(),
            direct.into(),
            storev.into(),
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            v_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(v.leading_dimension, "ldv")?,
            tau_type.into(),
            ptr::null(),
            t_type.into(),
            ptr::null_mut(),
            to_i64(t.leading_dimension, "ldt")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// Forms the triangular factor `T` of a real block reflector `H` of order `n`,
/// which is defined as a product of `k` elementary reflectors.
///
/// Only [`StorevMode::Columnwise`] storage is supported. This means the vector
/// defining the elementary reflector `H(i)` is stored in the `i`th column of
/// `V`, and $H = I - V \cdot T \cdot V^{T}$ ($H = I - V \cdot T \cdot V^{H}$
/// for complex types).
///
/// Provide device and host workspace through `workspace`.
/// Use [`xlarft_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// Currently, only the `n >= k` scenario is supported.
///
/// The generic cuSOLVER routine has four data types:
///
/// [`xlarft`] only supports the following four combinations.
///
/// **Valid combinations of data types and compute types**
///
/// | **`data_type_v`** | **`data_type_tau`** | **`data_type_t`** | **`compute_type`** | **Meaning** |
/// | --- | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SLARFT` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DLARFT` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CLARFT` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZLARFT` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// reflector dimensions or storage mode are invalid, or if cuSOLVER reports an
/// internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xlarft<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    context: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixRef<'_, TV>,
    tau: VectorRef<'_, TTau>,
    mut t: MatrixMut<'_, TT>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&v.data)?;
    context.validate_region(&tau.data)?;
    context.validate_region(&t.data)?;
    context.validate_region(&workspace.device)?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    validate_xlarft_inputs(
        n,
        k,
        storev,
        v.data.region().byte_len(),
        v.leading_dimension,
        v_type,
        tau.data.region().byte_len(),
        tau_type,
        t.data.region().byte_len(),
        t.leading_dimension,
        t_type,
    )?;
    let workspace_sizes = xlarft_buffer_size::<TV, TTau, TT>(
        context,
        params,
        direct,
        storev,
        n,
        k,
        v.layout(),
        t.layout(),
        compute_type,
    )?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXlarft` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnXlarft(
            context.as_raw(),
            params.as_raw(),
            direct.into(),
            storev.into(),
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            v_type.into(),
            v.data.as_ptr().cast(),
            to_i64(v.leading_dimension, "ldv")?,
            tau_type.into(),
            tau.data.as_ptr().cast(),
            t_type.into(),
            t.data.as_mut_ptr().cast(),
            to_i64(t.leading_dimension, "ldt")?,
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
        ))?;
    }
    Ok(())
}

/// Calculates the sizes needed for pre-allocated device and host workspace.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsytrf_buffer_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXsytrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Computes the factorization of a symmetric indefinite matrix using
/// Bunch-Kaufman diagonal pivoting and the generic API interface.
///
/// Only the lower or upper triangular part of `a` is meaningful. The other
/// triangular part may also be used as workspace. If `pivots` is `None`, no
/// pivoting is performed, which is not numerically stable.
/// Each block of the diagonal factor is either a 1×1 or 2×2 block.
/// `dev_info` reports an invalid parameter as a negative index, or a singular
/// diagonal factor as a positive index. Use [`xsytrf_buffer_size`] for workspace.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsytrf<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    context.validate_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xsytrf_buffer_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXsytrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut buffer| buffer.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Calculates the sizes needed for pre-allocated device and host workspace.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xhetrf_buffer_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXhetrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Computes the factorization of a Hermitian indefinite matrix using
/// Bunch-Kaufman diagonal pivoting and the generic API interface.
///
/// Only the lower or upper triangular part of `a` is meaningful. The other
/// triangular part may also be used as workspace. If `pivots` is `None`, no
/// pivoting is performed, which is not numerically stable.
/// Each block of the diagonal factor is either a 1×1 or 2×2 block.
/// `dev_info` reports an invalid parameter as a negative index, or a singular
/// diagonal factor as a positive index. Use [`xhetrf_buffer_size`] for workspace.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xhetrf<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    context.validate_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xhetrf_buffer_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXhetrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut buffer| buffer.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Calculates the sizes needed for pre-allocated device and host workspace.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xhetrs_buffer_size<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, nrhs)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: `cusolverDnXhetrs_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrs_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            TB::data_type().into(),
            ptr::null_mut(),
            to_i64(b.leading_dimension, "ldb")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Solves a system of linear equations with a Hermitian indefinite coefficient matrix.
///
/// `a` contains the factorization from [`xhetrf`]; only its lower or upper
/// triangular part is meaningful. Provide the pivot indices from that operation,
/// or `None` when both factorization and solve omit pivoting.
/// Supports complex single and double precision. Use [`xhetrs_buffer_size`]
/// to size device and host workspace. Negative `dev_info` identifies an invalid parameter.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xhetrs<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<DeviceView<'_, i64>>,
    mut b: MatrixMut<'_, TB>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    context.validate_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xhetrs_buffer_size::<TA, TB>(context, params, fill_mode, n, nrhs, a.layout(), b.layout())?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXhetrs` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrs(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null(), |buffer| buffer.as_ptr()),
            TB::data_type().into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
