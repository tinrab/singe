use std::ptr;

use singe_cuda::{
    data_type::DataTypeLike,
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    error::{Error, Result},
    layout::{
        ByteWorkspaceMut, MatrixMut, MatrixRef, StridedBatchedMatrixMut, StridedBatchedMatrixRef,
        StridedBatchedVectorMut, StridedBatchedVectorRef, WorkspaceSizes,
    },
    params::Params,
    svd::{
        info::GesvdjInfo,
        validation::{
            matrix_mut_parts, matrix_mut_ref_option, matrix_mut_ref_parts, matrix_ref_parts,
            optional_gesvda_output_mut_ptr, optional_gesvda_output_ptr,
            optional_gesvdj_matrix_mut_ptr, optional_gesvdj_matrix_ptr,
            optional_x_eig_matrix_mut_ptr, optional_x_eig_matrix_ptr, optional_x_matrix_mut_ptr,
            optional_x_matrix_ptr, optional_x_truncated_u_mut_ptr, optional_x_truncated_u_ptr,
            optional_x_truncated_v_mut_ptr, optional_x_truncated_v_ptr, require_host_workspace,
            require_info_buffer, require_info_buffer_len, require_workspace,
            require_workspace_bytes, strided_batched_matrix_mut_parts,
            strided_batched_matrix_mut_ref_option, strided_batched_matrix_ref_parts,
            validate_gesvd_dims, validate_gesvda_strided_batched_inputs,
            validate_gesvdj_batched_inputs, validate_gesvdj_inputs, validate_x_matrix,
            validate_x_svd_output, validate_x_vector, validate_xgesvdp_inputs,
            validate_xgesvdr_inputs,
        },
    },
    sys, try_ffi,
    types::{EigenMode, SvdMode, TruncatedSvdMode},
    utility::{to_i32, to_i64, to_usize},
};

pub fn xgesvd_buffer_size<
    TA: DataTypeLike,
    TS: DataTypeLike,
    TU: DataTypeLike,
    TVT: DataTypeLike,
>(
    ctx: &Context,
    params: &Params,
    job_u: SvdMode,
    job_vt: SvdMode,
    m: usize,
    n: usize,
    a: MatrixRef<'_, TA>,
    s: &DeviceMemory<TS>,
    u: Option<MatrixRef<'_, TU>>,
    vt: Option<MatrixRef<'_, TVT>>,
) -> Result<WorkspaceSizes> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let vt_type = TVT::data_type();
    ctx.bind()?;
    validate_gesvd_dims(m, n)?;
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    validate_x_vector(m.min(n), s.byte_len(), s_type)?;
    validate_x_svd_output(m, m, matrix_ref_parts(u), job_u, u_type)?;
    validate_x_svd_output(n, n, matrix_ref_parts(vt), job_vt, vt_type)?;
    if matches!(job_u, SvdMode::Overwrite) && matches!(job_vt, SvdMode::Overwrite) {
        return Err(Error::InvalidSvdMode);
    }

    let (u_ptr, ldu) = optional_x_matrix_ptr(matrix_ref_parts(u), m, m, job_u, u_type)?;
    let (vt_ptr, ldvt) = optional_x_matrix_ptr(matrix_ref_parts(vt), n, n, job_vt, vt_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvd_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_vt.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            vt_type.into(),
            vt_ptr.cast(),
            ldvt,
            a_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use [`xgesvd_buffer_size`] to calculate the sizes needed for pre-allocated
/// workspace.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix
/// `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an
/// $m \times n$ matrix which is zero except for its `min(m,n)` diagonal
/// elements, `U` is an $m \times m$ unitary matrix, and `V` is an
/// $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgesvd_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid. If `bdsqr` did not converge, `info` specifies how many superdiagonals of an intermediate bidiagonal form did not converge to zero.
///
/// Currently, [`xgesvd`] supports only the default algorithm.
///
/// **Algorithms supported by [`xgesvd`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// `gesvd` only supports `m >= n`.
///
/// Returns $V^H$, not `V`.
///
/// List of input arguments for [`xgesvd_buffer_size`] and [`xgesvd`]:
///
/// The generic cuSOLVER routine separates matrix, singular-value, vector, and compute data
/// types: `data_type_a` is the data type of matrix `A`, `data_type_s` is the
/// data type of vector `S`, `data_type_u` is the data type of matrix `U`,
/// `data_type_vt` is the data type of matrix `VT`, and `compute_type` is the
/// operation's compute type.
/// [`xgesvd`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **data_type_s** | **data_type_u** | **data_type_vt** | **compute_type** | **Meaning** |
/// | --- | --- | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SGESVD` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DGESVD` |
/// | [`DataType::ComplexF32`] | [`DataType::F32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGESVD` |
/// | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGESVD` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, output modes, or output buffers are
/// invalid, or if cuSOLVER reports an internal failure.
pub fn xgesvd<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TVT: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    job_u: SvdMode,
    job_vt: SvdMode,
    m: usize,
    n: usize,
    a: MatrixMut<'_, TA>,
    s: &mut DeviceMemory<TS>,
    u: Option<MatrixMut<'_, TU>>,
    vt: Option<MatrixMut<'_, TVT>>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let vt_type = TVT::data_type();
    ctx.bind()?;
    validate_gesvd_dims(m, n)?;
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    validate_x_vector(m.min(n), s.byte_len(), s_type)?;
    validate_x_svd_output(m, m, matrix_mut_ref_parts(u.as_ref()), job_u, u_type)?;
    validate_x_svd_output(n, n, matrix_mut_ref_parts(vt.as_ref()), job_vt, vt_type)?;
    if matches!(job_u, SvdMode::Overwrite) && matches!(job_vt, SvdMode::Overwrite) {
        return Err(Error::InvalidSvdMode);
    }
    require_info_buffer(dev_info)?;

    let workspace_sizes = xgesvd_buffer_size(
        ctx,
        params,
        job_u,
        job_vt,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(vt.as_ref()),
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;

    let (u_ptr, ldu) = optional_x_matrix_mut_ptr(matrix_mut_parts(u), m, m, job_u, u_type)?;
    let (vt_ptr, ldvt) = optional_x_matrix_mut_ptr(matrix_mut_parts(vt), n, n, job_vt, vt_type)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvd(
            ctx.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_vt.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_mut_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            vt_type.into(),
            vt_ptr.cast(),
            ldvt,
            a_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub fn xgesvdp_buffer_size<
    TA: DataTypeLike,
    TS: DataTypeLike,
    TU: DataTypeLike,
    TV: DataTypeLike,
>(
    ctx: &Context,
    params: &Params,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixRef<'_, TA>,
    s: &DeviceMemory<TS>,
    u: Option<MatrixRef<'_, TU>>,
    v: Option<MatrixRef<'_, TV>>,
) -> Result<WorkspaceSizes> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let v_type = TV::data_type();
    ctx.bind()?;
    validate_xgesvdp_inputs(
        m,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        a_type,
        s.byte_len(),
        s_type,
        jobz,
        econ,
        matrix_ref_parts(u).as_ref(),
        u_type,
        matrix_ref_parts(v).as_ref(),
        v_type,
    )?;
    let (u_ptr, ldu) = optional_x_eig_matrix_ptr(matrix_ref_parts(u), m, n, jobz, econ, u_type)?;
    let (v_ptr, ldv) = optional_x_eig_matrix_ptr(matrix_ref_parts(v), n, n, jobz, econ, v_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdp_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            v_type.into(),
            v_ptr.cast(),
            ldv,
            a_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use [`xgesvdp_buffer_size`] to calculate the sizes needed for pre-allocated
/// workspace.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix
/// `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an
/// $m \times n$ matrix which is zero except for its `min(m,n)` diagonal
/// elements, `U` is an $m \times m$ unitary matrix, and `V` is an
/// $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// [`xgesvdp`] combines polar decomposition in \[14\] and the symmetric
/// eigensolver used by this crate to compute the SVD.
/// It is much faster than [`xgesvd`], which is based on a QR algorithm.
/// However polar decomposition in \[14\] may not deliver a full unitary matrix when the matrix A has a singular value close to zero.
/// To workaround the issue when the singular value is close to zero, we add a small perturbation so polar decomposition can deliver the correct result.
/// The consequence is inaccurate singular values shifted by this perturbation.
/// `residual` stores the magnitude of this perturbation when requested.
/// In other words, it reports the accuracy of the SVD approximation.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgesvdp_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Currently, [`xgesvdp`] supports only the default algorithm.
///
/// **Algorithms supported by [`xgesvdp`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// `gesvdp` also supports `n >= m`.
///
/// Returns `V`, not $V^{H}$.
///
/// List of input arguments for [`xgesvdp_buffer_size`] and [`xgesvdp`]:
///
/// The generic cuSOLVER routine separates matrix, singular-value, vector, and compute data
/// types: `data_type_a` is the data type of matrix `A`, `data_type_s` is the
/// data type of vector `S`, `data_type_u` is the data type of matrix `U`,
/// `data_type_v` is the data type of matrix `V`, and `compute_type` is the
/// operation's compute type.
/// [`xgesvdp`] only supports the following four combinations:
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **data_type_s** | **data_type_u** | **data_type_v** | **compute_type** | **Meaning** |
/// | --- | --- | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SGESVDP` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DGESVDP` |
/// | [`DataType::ComplexF32`] | [`DataType::F32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGESVDP` |
/// | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGESVDP` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimensions are invalid, or if cuSOLVER reports
/// an internal failure.
pub fn xgesvdp<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TV: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixMut<'_, TA>,
    s: &mut DeviceMemory<TS>,
    u: Option<MatrixMut<'_, TU>>,
    v: Option<MatrixMut<'_, TV>>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
    err_sigma: Option<&mut f64>,
) -> Result<()> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let v_type = TV::data_type();
    ctx.bind()?;
    validate_xgesvdp_inputs(
        m,
        n,
        a.data.byte_len(),
        a.leading_dimension,
        a_type,
        s.byte_len(),
        s_type,
        jobz,
        econ,
        matrix_mut_ref_parts(u.as_ref()).as_ref(),
        u_type,
        matrix_mut_ref_parts(v.as_ref()).as_ref(),
        v_type,
    )?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xgesvdp_buffer_size(
        ctx,
        params,
        jobz,
        econ,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;

    let (u_ptr, ldu) =
        optional_x_eig_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, econ, u_type)?;
    let (v_ptr, ldv) =
        optional_x_eig_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, econ, v_type)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdp(
            ctx.as_raw(),
            params.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_mut_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            v_type.into(),
            v_ptr.cast(),
            ldv,
            a_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
            err_sigma.map_or(ptr::null_mut(), |value| value as *mut f64),
        ))?;
    }
    Ok(())
}

pub fn xgesvdr_buffer_size<
    TA: DataTypeLike,
    TS: DataTypeLike,
    TU: DataTypeLike,
    TV: DataTypeLike,
>(
    ctx: &Context,
    params: &Params,
    job_u: TruncatedSvdMode,
    job_v: TruncatedSvdMode,
    m: usize,
    n: usize,
    k: usize,
    p: usize,
    niters: usize,
    a: MatrixRef<'_, TA>,
    s: &DeviceMemory<TS>,
    u: Option<MatrixRef<'_, TU>>,
    v: Option<MatrixRef<'_, TV>>,
) -> Result<WorkspaceSizes> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let v_type = TV::data_type();
    ctx.bind()?;
    validate_xgesvdr_inputs(
        m,
        n,
        k,
        p,
        niters,
        a.data.byte_len(),
        a.leading_dimension,
        a_type,
        s.byte_len(),
        s_type,
        job_u,
        matrix_ref_parts(u).as_ref(),
        u_type,
        job_v,
        matrix_ref_parts(v).as_ref(),
        v_type,
    )?;
    let (u_ptr, ldu) = optional_x_truncated_u_ptr(matrix_ref_parts(u), m, k, job_u, u_type)?;
    let (v_ptr, ldv) = optional_x_truncated_v_ptr(matrix_ref_parts(v), n, k, job_v, v_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdr_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_v.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            to_i64(p, "p")?,
            to_i64(niters, "niters")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            v_type.into(),
            v_ptr.cast(),
            ldv,
            a_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Use [`xgesvdr_buffer_size`] to calculate the sizes needed for pre-allocated
/// workspace.
///
/// Computes the approximate rank-k singular value decomposition (k-SVD) of an
/// $m \times n$ matrix `A` and the corresponding left and/or right singular
/// vectors.
/// The k-SVD is written as
///
/// where `Σ` is a $k \times k$ matrix which is zero except for its diagonal elements, `U` is an $m \times k$ orthonormal matrix, and `V` is an $k \times n$ orthonormal matrix.
/// The diagonal elements of `Σ` are the approximated singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The columns of `U` and `V` are the top-`k` left and right singular vectors of `A`.
///
/// [`xgesvdr`] implements randomized methods described in \[15\] to compute k-SVD that is accurate with high probability if the conditions described in \[15\] hold.
/// [`xgesvdr`] is intended to compute a small portion of the spectrum of `A`
/// quickly and accurately, especially when `k` is much smaller than
/// `min(m,n)` and the matrix dimensions are large.
///
/// The accuracy of the method depends on the spectrum of `A`, the number of power iterations `niters`, the oversampling parameter `p` and the ratio between `p` and the dimensions of the matrix `A`.
/// Larger values of oversampling `p` or more iterations `niters` may produce
/// more accurate approximations, but also increase the run time of
/// [`xgesvdr`].
///
/// Our recommendation is to use two iterations and set the oversampling to at least `2k`.
/// Once the solver provides enough accuracy, adjust the values of `k` and `niters` for better performance.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgesvdr_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Currently, [`xgesvdr`] supports only the default algorithm.
///
/// **Algorithms supported by [`xgesvdr`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// `gesvdr` also supports `n >= m`.
///
/// Returns `V`, not $V^{H}$.
///
/// List of input arguments for [`xgesvdr_buffer_size`] and [`xgesvdr`]:
///
/// The generic cuSOLVER routine separates matrix, singular-value, vector, and compute data
/// types: `data_type_a` is the data type of matrix `A`, `data_type_s` is the
/// data type of vector `S`, `data_type_u` is the data type of matrix `U`,
/// `data_type_v` is the data type of matrix `V`, and `compute_type` is the
/// operation's compute type.
/// [`xgesvdr`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **data_type_s** | **data_type_u** | **data_type_v** | **compute_type** | **Meaning** |
/// | --- | --- | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SGESVDR` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DGESVDR` |
/// | [`DataType::ComplexF32`] | [`DataType::F32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGESVDR` |
/// | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGESVDR` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimensions are invalid, or if cuSOLVER reports
/// an internal failure.
pub fn xgesvdr<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TV: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    job_u: TruncatedSvdMode,
    job_v: TruncatedSvdMode,
    m: usize,
    n: usize,
    k: usize,
    p: usize,
    niters: usize,
    a: MatrixMut<'_, TA>,
    s: &mut DeviceMemory<TS>,
    u: Option<MatrixMut<'_, TU>>,
    v: Option<MatrixMut<'_, TV>>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    let a_type = TA::data_type();
    let s_type = TS::data_type();
    let u_type = TU::data_type();
    let v_type = TV::data_type();
    ctx.bind()?;
    validate_xgesvdr_inputs(
        m,
        n,
        k,
        p,
        niters,
        a.data.byte_len(),
        a.leading_dimension,
        a_type,
        s.byte_len(),
        s_type,
        job_u,
        matrix_mut_ref_parts(u.as_ref()).as_ref(),
        u_type,
        job_v,
        matrix_mut_ref_parts(v.as_ref()).as_ref(),
        v_type,
    )?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xgesvdr_buffer_size(
        ctx,
        params,
        job_u,
        job_v,
        m,
        n,
        k,
        p,
        niters,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    let (u_ptr, ldu) = optional_x_truncated_u_mut_ptr(matrix_mut_parts(u), m, k, job_u, u_type)?;
    let (v_ptr, ldv) = optional_x_truncated_v_mut_ptr(matrix_mut_parts(v), n, k, job_v, v_type)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdr(
            ctx.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_v.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            to_i64(p, "p")?,
            to_i64(niters, "niters")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            s_type.into(),
            s.as_mut_ptr().cast(),
            u_type.into(),
            u_ptr.cast(),
            ldu,
            v_type.into(),
            v_ptr.cast(),
            ldv,
            a_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub fn sgesvdj_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixRef<'_, f32>,
    s: &DeviceMemory<f32>,
    u: Option<MatrixRef<'_, f32>>,
    v: Option<MatrixRef<'_, f32>>,
    params: &GesvdjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, econ)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdj_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dgesvdj_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixRef<'_, f64>,
    s: &DeviceMemory<f64>,
    u: Option<MatrixRef<'_, f64>>,
    v: Option<MatrixRef<'_, f64>>,
    params: &GesvdjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, econ)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdj_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cgesvdj_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixRef<'_, Complex32>,
    s: &DeviceMemory<f32>,
    u: Option<MatrixRef<'_, Complex32>>,
    v: Option<MatrixRef<'_, Complex32>>,
    params: &GesvdjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, econ)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdj_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zgesvdj_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixRef<'_, Complex64>,
    s: &DeviceMemory<f64>,
    u: Option<MatrixRef<'_, Complex64>>,
    v: Option<MatrixRef<'_, Complex64>>,
    params: &GesvdjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, econ)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdj_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
pub fn sgesvdj(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixMut<'_, f32>,
    s: &mut DeviceMemory<f32>,
    u: Option<MatrixMut<'_, f32>>,
    v: Option<MatrixMut<'_, f32>>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
    )?;
    require_info_buffer(dev_info)?;
    let lwork = sgesvdj_buffer_size(
        ctx,
        jobz,
        econ,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, econ)?;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdj(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
pub fn dgesvdj(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixMut<'_, f64>,
    s: &mut DeviceMemory<f64>,
    u: Option<MatrixMut<'_, f64>>,
    v: Option<MatrixMut<'_, f64>>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
    )?;
    require_info_buffer(dev_info)?;
    let lwork = dgesvdj_buffer_size(
        ctx,
        jobz,
        econ,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, econ)?;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdj(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
pub fn cgesvdj(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixMut<'_, Complex32>,
    s: &mut DeviceMemory<f32>,
    u: Option<MatrixMut<'_, Complex32>>,
    v: Option<MatrixMut<'_, Complex32>>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
    )?;
    require_info_buffer(dev_info)?;
    let lwork = cgesvdj_buffer_size(
        ctx,
        jobz,
        econ,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, econ)?;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdj(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
pub fn zgesvdj(
    ctx: &Context,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    a: MatrixMut<'_, Complex64>,
    s: &mut DeviceMemory<f64>,
    u: Option<MatrixMut<'_, Complex64>>,
    v: Option<MatrixMut<'_, Complex64>>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        econ,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
    )?;
    require_info_buffer(dev_info)?;
    let lwork = zgesvdj_buffer_size(
        ctx,
        jobz,
        econ,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, econ)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, econ)?;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdj(
            ctx.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub fn sgesvdj_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixRef<'_, f32>,
    s: &DeviceMemory<f32>,
    u: Option<MatrixRef<'_, f32>>,
    v: Option<MatrixRef<'_, f32>>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, true)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdjBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dgesvdj_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixRef<'_, f64>,
    s: &DeviceMemory<f64>,
    u: Option<MatrixRef<'_, f64>>,
    v: Option<MatrixRef<'_, f64>>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, true)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdjBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cgesvdj_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixRef<'_, Complex32>,
    s: &DeviceMemory<f32>,
    u: Option<MatrixRef<'_, Complex32>>,
    v: Option<MatrixRef<'_, Complex32>>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, true)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdjBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zgesvdj_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixRef<'_, Complex64>,
    s: &DeviceMemory<f64>,
    u: Option<MatrixRef<'_, Complex64>>,
    v: Option<MatrixRef<'_, Complex64>>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_ref_parts(u),
        matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_ptr(matrix_ref_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_ptr(matrix_ref_parts(v), n, n, jobz, true)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdjBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn sgesvdj_batched(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixMut<'_, f32>,
    s: &mut DeviceMemory<f32>,
    u: Option<MatrixMut<'_, f32>>,
    v: Option<MatrixMut<'_, f32>>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = sgesvdj_batched_buffer_size(
        ctx,
        jobz,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, true)?;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdjBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn dgesvdj_batched(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixMut<'_, f64>,
    s: &mut DeviceMemory<f64>,
    u: Option<MatrixMut<'_, f64>>,
    v: Option<MatrixMut<'_, f64>>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = dgesvdj_batched_buffer_size(
        ctx,
        jobz,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, true)?;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdjBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn cgesvdj_batched(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixMut<'_, Complex32>,
    s: &mut DeviceMemory<f32>,
    u: Option<MatrixMut<'_, Complex32>>,
    v: Option<MatrixMut<'_, Complex32>>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = cgesvdj_batched_buffer_size(
        ctx,
        jobz,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, true)?;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdjBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn zgesvdj_batched(
    ctx: &Context,
    jobz: EigenMode,
    m: usize,
    n: usize,
    a: MatrixMut<'_, Complex64>,
    s: &mut DeviceMemory<f64>,
    u: Option<MatrixMut<'_, Complex64>>,
    v: Option<MatrixMut<'_, Complex64>>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvdj_batched_inputs(
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        s.len(),
        jobz,
        matrix_mut_ref_parts(u.as_ref()),
        matrix_mut_ref_parts(v.as_ref()),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = zgesvdj_batched_buffer_size(
        ctx,
        jobz,
        m,
        n,
        a.as_ref(),
        s,
        matrix_mut_ref_option(u.as_ref()),
        matrix_mut_ref_option(v.as_ref()),
        params,
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(u), m, n, jobz, true)?;
    let (v_ptr, ldv) = optional_gesvdj_matrix_mut_ptr(matrix_mut_parts(v), n, n, jobz, true)?;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdjBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr().cast(),
            u_ptr.cast(),
            ldu,
            v_ptr.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

pub fn sgesvda_strided_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f32>,
    s: StridedBatchedVectorRef<'_, f32>,
    u: Option<StridedBatchedMatrixRef<'_, f32>>,
    v: Option<StridedBatchedMatrixRef<'_, f32>>,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_ref_parts(u),
        strided_batched_matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(v), n, rank, jobz)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdaStridedBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dgesvda_strided_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f64>,
    s: StridedBatchedVectorRef<'_, f64>,
    u: Option<StridedBatchedMatrixRef<'_, f64>>,
    v: Option<StridedBatchedMatrixRef<'_, f64>>,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_ref_parts(u),
        strided_batched_matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(v), n, rank, jobz)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdaStridedBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cgesvda_strided_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex32>,
    s: StridedBatchedVectorRef<'_, f32>,
    u: Option<StridedBatchedMatrixRef<'_, Complex32>>,
    v: Option<StridedBatchedMatrixRef<'_, Complex32>>,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_ref_parts(u),
        strided_batched_matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(v), n, rank, jobz)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdaStridedBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zgesvda_strided_batched_buffer_size(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex64>,
    s: StridedBatchedVectorRef<'_, f64>,
    u: Option<StridedBatchedMatrixRef<'_, Complex64>>,
    v: Option<StridedBatchedMatrixRef<'_, Complex64>>,
    batch_size: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_ref_parts(u),
        strided_batched_matrix_ref_parts(v),
        batch_size,
    )?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_ptr(strided_batched_matrix_ref_parts(v), n, rank, jobz)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdaStridedBatched_bufferSize(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives the Frobenius norm of the residual.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ stride_a\cdot k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ stride_s\cdot k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ stride_u\cdot k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ stride_v\cdot k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
pub fn sgesvda_strided_batched(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f32>,
    s: StridedBatchedVectorMut<'_, f32>,
    u: Option<StridedBatchedMatrixMut<'_, f32>>,
    v: Option<StridedBatchedMatrixMut<'_, f32>>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    residual: Option<&mut f64>,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_mut_ref_option(u.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        strided_batched_matrix_mut_ref_option(v.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = sgesvda_strided_batched_buffer_size(
        ctx,
        jobz,
        rank,
        m,
        n,
        a,
        s.as_ref(),
        strided_batched_matrix_mut_ref_option(u.as_ref()),
        strided_batched_matrix_mut_ref_option(v.as_ref()),
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(v), n, rank, jobz)?;
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdaStridedBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            residual.map_or(ptr::null_mut(), |value| value as *mut f64),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives the Frobenius norm of the residual.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ stride_a\cdot k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ stride_s\cdot k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ stride_u\cdot k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ stride_v\cdot k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
pub fn dgesvda_strided_batched(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f64>,
    s: StridedBatchedVectorMut<'_, f64>,
    u: Option<StridedBatchedMatrixMut<'_, f64>>,
    v: Option<StridedBatchedMatrixMut<'_, f64>>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    residual: Option<&mut f64>,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_mut_ref_option(u.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        strided_batched_matrix_mut_ref_option(v.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = dgesvda_strided_batched_buffer_size(
        ctx,
        jobz,
        rank,
        m,
        n,
        a,
        s.as_ref(),
        strided_batched_matrix_mut_ref_option(u.as_ref()),
        strided_batched_matrix_mut_ref_option(v.as_ref()),
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(v), n, rank, jobz)?;
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdaStridedBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            residual.map_or(ptr::null_mut(), |value| value as *mut f64),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives the Frobenius norm of the residual.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ stride_a\cdot k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ stride_s\cdot k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ stride_u\cdot k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ stride_v\cdot k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
pub fn cgesvda_strided_batched(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex32>,
    s: StridedBatchedVectorMut<'_, f32>,
    u: Option<StridedBatchedMatrixMut<'_, Complex32>>,
    v: Option<StridedBatchedMatrixMut<'_, Complex32>>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    residual: Option<&mut f64>,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_mut_ref_option(u.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        strided_batched_matrix_mut_ref_option(v.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = cgesvda_strided_batched_buffer_size(
        ctx,
        jobz,
        rank,
        m,
        n,
        a,
        s.as_ref(),
        strided_batched_matrix_mut_ref_option(u.as_ref()),
        strided_batched_matrix_mut_ref_option(v.as_ref()),
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(v), n, rank, jobz)?;
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdaStridedBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            residual.map_or(ptr::null_mut(), |value| value as *mut f64),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives the Frobenius norm of the residual.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ stride_a\cdot k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ stride_s\cdot k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ stride_u\cdot k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ stride_v\cdot k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
pub fn zgesvda_strided_batched(
    ctx: &Context,
    jobz: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex64>,
    s: StridedBatchedVectorMut<'_, f64>,
    u: Option<StridedBatchedMatrixMut<'_, Complex64>>,
    v: Option<StridedBatchedMatrixMut<'_, Complex64>>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    residual: Option<&mut f64>,
    batch_size: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_gesvda_strided_batched_inputs(
        rank,
        m,
        n,
        a.data.len(),
        a.leading_dimension,
        a.stride,
        s.data.len(),
        s.stride,
        jobz,
        strided_batched_matrix_mut_ref_option(u.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        strided_batched_matrix_mut_ref_option(v.as_ref())
            .map(|m| (m.data, m.leading_dimension, m.stride)),
        batch_size,
    )?;
    require_info_buffer_len(dev_info, batch_size)?;
    let lwork = zgesvda_strided_batched_buffer_size(
        ctx,
        jobz,
        rank,
        m,
        n,
        a,
        s.as_ref(),
        strided_batched_matrix_mut_ref_option(u.as_ref()),
        strided_batched_matrix_mut_ref_option(v.as_ref()),
        batch_size,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (u_ptr, ldu, stride_u) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(u), m, rank, jobz)?;
    let (v_ptr, ldv, stride_v) =
        optional_gesvda_output_mut_ptr(strided_batched_matrix_mut_parts(v), n, rank, jobz)?;
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdaStridedBatched(
            ctx.as_raw(),
            jobz.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr().cast(),
            to_i64(s.stride, "stride_s")?,
            u_ptr.cast(),
            ldu,
            stride_u,
            v_ptr.cast(),
            ldv,
            stride_v,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            residual.map_or(ptr::null_mut(), |value| value as *mut f64),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}
