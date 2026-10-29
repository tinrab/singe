#![expect(
    clippy::many_single_char_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::ptr;

use singe_cuda::{
    data_type::DataTypeLike,
    view::{DeviceOutput, DeviceRegionProvider, DeviceRepr},
};

use crate::{
    context::Context,
    data_type::SolverElement,
    error::{Error, Result},
    layout::{ByteWorkspaceMut, MatrixMut, MatrixOutput, WorkspaceSizes},
    params::Params,
    svd::validation::{require_host_workspace, require_workspace_bytes, validate_x_matrix},
    sys, try_ffi,
    types::{EigenMode, SvdMode, TruncatedSvdMode},
    utility::{to_i64, to_usize},
};

/// Column-major leading dimensions for an allocation-free SVD workspace query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvdLayout {
    /// Element distance between input matrix column starts.
    pub input_leading_dimension: usize,
    /// Optional leading dimension in elements for the left singular-vector output.
    pub left_leading_dimension: Option<usize>,
    /// Optional leading dimension in elements for the right singular-vector output.
    pub right_leading_dimension: Option<usize>,
}

impl SvdLayout {
    /// Records the input leading dimension with no singular-vector output layouts.
    ///
    /// All distances are measured in elements. The solver workspace query validates dimensions and required outputs.
    #[must_use]
    pub const fn new(input_leading_dimension: usize) -> Self {
        Self {
            input_leading_dimension,
            left_leading_dimension: None,
            right_leading_dimension: None,
        }
    }

    /// Sets the left singular-vector output leading dimension in element units.
    ///
    /// Records metadata without validating output shape or storage capacity.
    #[must_use]
    pub const fn with_left_vectors(mut self, leading_dimension: usize) -> Self {
        self.left_leading_dimension = Some(leading_dimension);
        self
    }

    /// Sets the right singular-vector output leading dimension in element units.
    ///
    /// Records metadata without validating output shape or storage capacity.
    #[must_use]
    pub const fn with_right_vectors(mut self, leading_dimension: usize) -> Self {
        self.right_leading_dimension = Some(leading_dimension);
        self
    }
}

fn leading_dimension(value: usize, rows: usize) -> Result<i64> {
    if value < rows.max(1) {
        return Err(Error::InvalidMatrixShape);
    }
    to_i64(value, "leading dimension")
}

fn output_leading_dimension(value: Option<usize>, rows: usize, enabled: bool) -> Result<i64> {
    if enabled {
        leading_dimension(value.ok_or(Error::InvalidMatrixShape)?, rows)
    } else {
        Ok(1)
    }
}

fn layout_parts(
    layout: SvdLayout,
    m: usize,
    n: usize,
    left: bool,
    right: bool,
) -> Result<(i64, i64, i64)> {
    if m == 0 || n == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    Ok((
        leading_dimension(layout.input_leading_dimension, m)?,
        output_leading_dimension(layout.left_leading_dimension, m, left)?,
        output_leading_dimension(layout.right_leading_dimension, n, right)?,
    ))
}

fn classic_modes(m: usize, n: usize, left: SvdMode, right: SvdMode) -> Result<(bool, bool)> {
    if m < n {
        return Err(Error::InvalidMatrixShape);
    }
    if matches!(left, SvdMode::Overwrite) && matches!(right, SvdMode::Overwrite) {
        return Err(Error::InvalidSvdMode);
    }
    Ok((
        matches!(left, SvdMode::All | SvdMode::Some),
        matches!(right, SvdMode::All | SvdMode::Some),
    ))
}

fn output_parts<T: DataTypeLike>(
    matrix: Option<MatrixOutput<'_, T>>,
    rows: usize,
    columns: usize,
    enabled: bool,
) -> Result<(*mut T, i64)> {
    if !enabled {
        return Ok((ptr::null_mut(), 1));
    }
    let mut matrix = matrix.ok_or(Error::InvalidMatrixShape)?;
    validate_x_matrix(
        rows,
        columns,
        matrix.data.region().byte_len(),
        matrix.leading_dimension,
        T::data_type(),
    )?;
    Ok((
        matrix.data.as_mut_ptr(),
        to_i64(matrix.leading_dimension, "leading dimension")?,
    ))
}

fn output_layout<T: DeviceRepr>(
    a: &MatrixMut<'_, T>,
    u: Option<&MatrixOutput<'_, T>>,
    v: Option<&MatrixOutput<'_, T>>,
) -> SvdLayout {
    SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.map(|matrix| matrix.leading_dimension),
    }
}

fn validate_outputs<S: DeviceRepr>(
    singular_values: &impl DeviceOutput<S>,
    count: usize,
    info: &impl DeviceOutput<i32>,
) -> Result<()> {
    if singular_values.len() < count || info.is_empty() {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

/// Queries workspace sizes without allocating or initializing device matrices.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required singular-vector output layout is missing or its mode is unsupported, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgesvd_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    job_u: SvdMode,
    job_vt: SvdMode,
    m: usize,
    n: usize,
    layout: SvdLayout,
) -> Result<WorkspaceSizes> {
    let (left, right) = classic_modes(m, n, job_u, job_vt)?;
    let (lda, ldu, ldvt) = layout_parts(layout, m, n, left, right)?;
    context.bind()?;
    let mut device = 0;
    let mut host = 0;
    // SAFETY: workspace queries use dimensions and datatypes, not device contents.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvd_bufferSize(
            context.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_vt.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            lda,
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            ptr::null(),
            ldu,
            T::data_type().into(),
            ptr::null(),
            ldvt,
            T::data_type().into(),
            &raw mut device,
            &raw mut host,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device, "device workspace size")?,
        to_usize(host, "host workspace size")?,
    ))
}

/// Queues a column-major SVD, returning right vectors as V^H.
///
/// `All` produces square vector matrices; `Some` produces m-by-min(m,n) U and
/// min(m,n)-by-n V^H. `Overwrite` uses A for the requested vectors.
///
/// # Safety
///
/// Retain A, every output, both workspaces, and both handles through completion
/// on every path and throughout replay. A must be initialized and ordered before
/// this call. All ranges must belong to the handle's actual CUDA context and be
/// disjoint, with exclusive access to writes. A successful enqueue does not
/// establish completion. Check the first status element after completion before
/// using the solution. Only min(m,n) singular values, requested logical vector
/// elements, and the first status element are produced; padding and tails are not.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, a required singular-vector output is missing or its mode is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xgesvd<T: SolverElement>(
    context: &Context,
    params: &Params,
    job_u: SvdMode,
    job_vt: SvdMode,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, T>,
    s: &mut impl DeviceOutput<T::Real>,
    u: Option<MatrixOutput<'_, T>>,
    vt: Option<MatrixOutput<'_, T>>,
    mut workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut impl DeviceOutput<i32>,
) -> Result<()> {
    let sizes = xgesvd_buffer_size::<T>(
        context,
        params,
        job_u,
        job_vt,
        m,
        n,
        output_layout(&a, u.as_ref(), vt.as_ref()),
    )?;
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        T::data_type(),
    )?;
    validate_outputs(s, m.min(n), dev_info)?;
    require_workspace_bytes(workspace.device.len(), sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), sizes.host_bytes)?;
    let (left, right) = classic_modes(m, n, job_u, job_vt)?;
    let (u_ptr, ldu) = output_parts(
        u,
        m,
        if matches!(job_u, SvdMode::Some) {
            m.min(n)
        } else {
            m
        },
        left,
    )?;
    let (vt_ptr, ldvt) = output_parts(
        vt,
        if matches!(job_vt, SvdMode::Some) {
            m.min(n)
        } else {
            n
        },
        n,
        right,
    )?;
    // SAFETY: the caller retains and orders the checked input and write-only ranges.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvd(
            context.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_vt.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            s.as_mut_ptr().cast(),
            T::data_type().into(),
            u_ptr.cast(),
            ldu,
            T::data_type().into(),
            vt_ptr.cast(),
            ldvt,
            T::data_type().into(),
            workspace.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries polar-decomposition SVD workspace before allocating its buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required singular-vector output layout is missing or its mode is unsupported, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgesvdp_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    layout: SvdLayout,
) -> Result<WorkspaceSizes> {
    let vectors = matches!(jobz, EigenMode::Vector);
    let (lda, ldu, ldv) = layout_parts(layout, m, n, vectors, vectors)?;
    context.bind()?;
    let mut device = 0;
    let mut host = 0;
    // SAFETY: the query uses only dimensions, leading dimensions, and datatypes.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdp_bufferSize(
            context.as_raw(),
            params.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            lda,
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            ptr::null(),
            ldu,
            T::data_type().into(),
            ptr::null(),
            ldv,
            T::data_type().into(),
            &raw mut device,
            &raw mut host,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device, "device workspace size")?,
        to_usize(host, "host workspace size")?,
    ))
}

/// Queues polar-decomposition SVD, returning right vectors as V.
///
/// Vector allocations must accommodate square m-by-m U and n-by-n V as specified
/// by cuSOLVER. With `econ`, only the first min(m,n) columns are produced.
/// `err_sigma`, when present, receives the perturbation magnitude.
///
/// # Safety
///
/// Follow [`xgesvd`]'s context, ordering, aliasing, and retention contract. Retain
/// `err_sigma` through completion as well. Only requested logical vector columns,
/// min(m,n) singular values, and the first status element are produced. Workspace,
/// matrix padding, and unused output tails do not become initialized results.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, a required singular-vector output is missing or its mode is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xgesvdp<T: SolverElement>(
    context: &Context,
    params: &Params,
    jobz: EigenMode,
    econ: bool,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, T>,
    s: &mut impl DeviceOutput<T::Real>,
    u: Option<MatrixOutput<'_, T>>,
    v: Option<MatrixOutput<'_, T>>,
    mut workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut impl DeviceOutput<i32>,
    err_sigma: Option<&mut f64>,
) -> Result<()> {
    let sizes = xgesvdp_buffer_size::<T>(
        context,
        params,
        jobz,
        econ,
        m,
        n,
        output_layout(&a, u.as_ref(), v.as_ref()),
    )?;
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        T::data_type(),
    )?;
    validate_outputs(s, m.min(n), dev_info)?;
    require_workspace_bytes(workspace.device.len(), sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), sizes.host_bytes)?;
    let vectors = matches!(jobz, EigenMode::Vector);
    let (u_ptr, ldu) = output_parts(u, m, m, vectors)?;
    let (v_ptr, ldv) = output_parts(v, n, n, vectors)?;
    // SAFETY: the caller upholds the raw SVD lifetime and access contract.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdp(
            context.as_raw(),
            params.as_raw(),
            jobz.into(),
            i32::from(econ),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            s.as_mut_ptr().cast(),
            T::data_type().into(),
            u_ptr.cast(),
            ldu,
            T::data_type().into(),
            v_ptr.cast(),
            ldv,
            T::data_type().into(),
            workspace.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
            err_sigma.map_or(ptr::null_mut(), ptr::from_mut::<f64>),
        ))?;
    }
    Ok(())
}

/// Queries truncated SVD workspace without allocating its outputs.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, rank is zero or rank plus oversampling is not smaller than `min(m, n)`, a required singular-vector output layout is missing or its mode is unsupported, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgesvdr_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    job_u: TruncatedSvdMode,
    job_v: TruncatedSvdMode,
    m: usize,
    n: usize,
    k: usize,
    p: usize,
    niters: usize,
    layout: SvdLayout,
) -> Result<WorkspaceSizes> {
    let minimum = m.min(n);
    if k == 0 || k >= minimum || k.checked_add(p).is_none_or(|sum| sum >= minimum) {
        return Err(Error::InvalidMatrixShape);
    }
    let left = matches!(job_u, TruncatedSvdMode::Some);
    let right = matches!(job_v, TruncatedSvdMode::Some);
    let (lda, ldu, ldv) = layout_parts(layout, m, n, left, right)?;
    context.bind()?;
    let mut device = 0;
    let mut host = 0;
    // SAFETY: the workspace query does not access device matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdr_bufferSize(
            context.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_v.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            to_i64(p, "p")?,
            to_i64(niters, "niters")?,
            T::data_type().into(),
            ptr::null(),
            lda,
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            ptr::null(),
            ldu,
            T::data_type().into(),
            ptr::null(),
            ldv,
            T::data_type().into(),
            &raw mut device,
            &raw mut host,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device, "device workspace size")?,
        to_usize(host, "host workspace size")?,
    ))
}

/// Queues truncated SVD, producing the leading k singular vectors and values.
///
/// Storage must accommodate min(m,n) singular values and square m-by-m U and
/// n-by-n V allocations required by cuSOLVER, although only leading k results
/// are exposed by this operation's initialization contract.
///
/// # Safety
///
/// Follow [`xgesvd`]'s context, ordering, aliasing, and lifetime contract. After
/// completion and a zero first status element, only the leading k singular
/// values and requested vector columns may be treated as produced results.
/// Allocation padding, unused tails, and scratch remain outside that guarantee.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device or host workspace is smaller than the queried requirement, a required singular-vector output is missing or its mode is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xgesvdr<T: SolverElement>(
    context: &Context,
    params: &Params,
    job_u: TruncatedSvdMode,
    job_v: TruncatedSvdMode,
    m: usize,
    n: usize,
    k: usize,
    p: usize,
    niters: usize,
    mut a: MatrixMut<'_, T>,
    s: &mut impl DeviceOutput<T::Real>,
    u: Option<MatrixOutput<'_, T>>,
    v: Option<MatrixOutput<'_, T>>,
    mut workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut impl DeviceOutput<i32>,
) -> Result<()> {
    let sizes = xgesvdr_buffer_size::<T>(
        context,
        params,
        job_u,
        job_v,
        m,
        n,
        k,
        p,
        niters,
        output_layout(&a, u.as_ref(), v.as_ref()),
    )?;
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        T::data_type(),
    )?;
    validate_outputs(s, m.min(n), dev_info)?;
    require_workspace_bytes(workspace.device.len(), sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), sizes.host_bytes)?;
    let (u_ptr, ldu) = output_parts(u, m, m, matches!(job_u, TruncatedSvdMode::Some))?;
    let (v_ptr, ldv) = output_parts(v, n, n, matches!(job_v, TruncatedSvdMode::Some))?;
    // SAFETY: the caller retains the checked resources through completed execution.
    unsafe {
        try_ffi!(sys::cusolverDnXgesvdr(
            context.as_raw(),
            params.as_raw(),
            job_u.as_raw(),
            job_v.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            to_i64(p, "p")?,
            to_i64(niters, "niters")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            s.as_mut_ptr().cast(),
            T::data_type().into(),
            u_ptr.cast(),
            ldu,
            T::data_type().into(),
            v_ptr.cast(),
            ldv,
            T::data_type().into(),
            workspace.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
