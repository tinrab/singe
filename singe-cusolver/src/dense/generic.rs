use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
};

use crate::{
    context::Context,
    dense::validation::{
        require_host_workspace, require_info_buffer, require_pivot64_buffer,
        require_workspace_bytes, validate_x_matrix, validate_xlarft_inputs,
    },
    error::Result,
    layout::{ByteWorkspaceMut, MatrixMut, MatrixRef, VectorRef, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{DiagonalType, DirectMode, FillMode, Operation, StorevMode},
    utility::{to_i64, to_usize},
};

pub fn xpotrf_buffer_size<TA: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixRef<'_, TA>,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    let a_type = TA::data_type();
    validate_x_matrix(n, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_ptr().cast(),
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
/// | **data_type_a** | **compute_type** | **Meaning** |
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
pub fn xpotrf<TA: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixMut<'_, TA>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    let a_type = TA::data_type();
    validate_x_matrix(n, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xpotrf_buffer_size(ctx, params, fill_mode, n, a.as_ref(), compute_type)?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf(
            ctx.as_raw(),
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
/// | **data_type_a** | **data_type_b** | **Meaning** |
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
pub fn xpotrs<TA: DataTypeLike, TB: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    b: MatrixMut<'_, TB>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(n, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    validate_x_matrix(n, nrhs, b.data.byte_len(), b.leading_dimension, b_type)?;
    require_info_buffer(dev_info)?;
    unsafe {
        try_ffi!(sys::cusolverDnXpotrs(
            ctx.as_raw(),
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

pub fn xtrtri_buffer_size<TA: DataTypeLike>(
    ctx: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    a: MatrixRef<'_, TA>,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    validate_x_matrix(
        n,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            diagonal_type.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_ptr().cast_mut().cast(),
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
pub fn xtrtri<TA: DataTypeLike>(
    ctx: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    a: MatrixMut<'_, TA>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_x_matrix(
        n,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xtrtri_buffer_size(ctx, fill_mode, diagonal_type, n, a.as_ref())?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri(
            ctx.as_raw(),
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

pub fn xgetrf_buffer_size<TA: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixRef<'_, TA>,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    let a_type = TA::data_type();
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_ptr().cast(),
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
/// | **data_type_a** | **compute_type** | **Meaning** |
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
pub fn xgetrf<TA: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixMut<'_, TA>,
    pivots: Option<&mut DeviceMemory<i64>>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    let a_type = TA::data_type();
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots, m.min(n))?;
    }
    require_info_buffer(dev_info)?;
    let workspace_sizes = xgetrf_buffer_size(ctx, params, m, n, a.as_ref(), compute_type)?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf(
            ctx.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
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
/// | **data_type_a** | **data_type_b** | **Meaning** |
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
pub fn xgetrs<TA: DataTypeLike, TB: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    operation: Operation,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: &DeviceMemory<i64>,
    b: MatrixMut<'_, TB>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(n, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    require_pivot64_buffer(pivots, n)?;
    validate_x_matrix(n, nrhs, b.data.byte_len(), b.leading_dimension, b_type)?;
    require_info_buffer(dev_info)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgetrs(
            ctx.as_raw(),
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

pub fn xsytrs_buffer_size<TA: DataTypeLike, TB: DataTypeLike>(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<&DeviceMemory<i64>>,
    b: MatrixRef<'_, TB>,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    validate_x_matrix(
        n,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots, n)?;
    }

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(std::ptr::null(), DeviceMemory::as_ptr),
            TB::data_type().into(),
            b.data.as_ptr().cast_mut().cast(),
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
/// `A` contains the factorization produced by the typed `*sytrf` operations in this module.
/// Only the lower or upper part is meaningful; the other part is left untouched.
///
/// Provide the pivot indices returned by the matching `*sytrf` operation, along
/// with device and host workspace through `workspace`.
/// Use [`xsytrs_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
/// To factorize and solve the symmetric system without pivoting, pass `None`
/// for the pivot buffer to both the matching `*sytrf` operation and [`xsytrs`].
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
/// | **data_type_a** | **data_type_b** | **Meaning** |
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
pub fn xsytrs<TA: DataTypeLike, TB: DataTypeLike>(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<&DeviceMemory<i64>>,
    b: MatrixMut<'_, TB>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_x_matrix(
        n,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots, n)?;
    }
    require_info_buffer(dev_info)?;
    let workspace_sizes = xsytrs_buffer_size(ctx, fill_mode, n, nrhs, a, pivots, b.as_ref())?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs(
            ctx.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(std::ptr::null(), DeviceMemory::as_ptr),
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

pub fn xlarft_buffer_size<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixRef<'_, TV>,
    tau: VectorRef<'_, TTau>,
    t: MatrixRef<'_, TT>,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    validate_xlarft_inputs(
        n,
        k,
        storev,
        v.data.byte_len(),
        v.leading_dimension,
        v_type,
        tau.data.byte_len(),
        tau_type,
        t.data.byte_len(),
        t.leading_dimension,
        t_type,
    )?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXlarft_bufferSize(
            ctx.as_raw(),
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
            t.data.as_ptr().cast_mut().cast(),
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
/// | **data_type_v** | **data_type_tau** | **data_type_t** | **compute_type** | **Meaning** |
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
pub fn xlarft<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixRef<'_, TV>,
    tau: VectorRef<'_, TTau>,
    t: MatrixMut<'_, TT>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
) -> Result<()> {
    ctx.bind()?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    validate_xlarft_inputs(
        n,
        k,
        storev,
        v.data.byte_len(),
        v.leading_dimension,
        v_type,
        tau.data.byte_len(),
        tau_type,
        t.data.byte_len(),
        t.leading_dimension,
        t_type,
    )?;
    let workspace_sizes = xlarft_buffer_size(
        ctx,
        params,
        direct,
        storev,
        n,
        k,
        v,
        tau,
        t.as_ref(),
        compute_type,
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXlarft(
            ctx.as_raw(),
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
