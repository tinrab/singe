use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    eigen::{
        EigenSelection,
        info::SyevjInfo,
        validation::{
            matrix_mut_parts, matrix_mut_ref_option, matrix_mut_ref_parts, matrix_ref_parts,
            optional_xgeev_matrix_mut_ptr, optional_xgeev_matrix_ptr, require_host_workspace,
            require_info_buffer, require_info_buffer_len, require_workspace,
            require_workspace_bytes, selection_parts, validate_syev_buffers,
            validate_syevj_batched_buffers, validate_sygvd_buffers, validate_sygvj_buffers,
            validate_xgeev_inputs, validate_xsyev_batched_buffers, validate_xsyevd_buffers,
            validate_xsyevdx_range, validate_xsyevdx_value_type,
        },
    },
    error::Result,
    layout::{ByteWorkspaceMut, MatrixMut, MatrixRef, SelectionWorkspaceSizes, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{EigenMode, EigenRange, EigenType, FillMode},
    utility::{to_i32, to_i64, to_usize},
};

pub(crate) fn xsyevd_buffer_size<TA: DataTypeLike, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<TA>,
    lda: usize,
    w: &DeviceMemory<TW>,
) -> Result<WorkspaceSizes> {
    xsyevd_raw_buffer_size(
        ctx,
        params,
        mode,
        fill_mode,
        n,
        TA::data_type(),
        a,
        lda,
        TW::data_type(),
        w,
        TA::data_type(),
    )
}

pub(crate) fn xsyevd<TA: DataTypeLike, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<TA>,
    lda: usize,
    w: &mut DeviceMemory<TW>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    xsyevd_raw(
        ctx,
        params,
        mode,
        fill_mode,
        n,
        TA::data_type(),
        a,
        lda,
        TW::data_type(),
        w,
        TA::data_type(),
        workspace,
        dev_info,
    )
}

pub(crate) fn xsyevdx_buffer_size<TA: DataTypeLike, TR: Copy + Default, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<TR>,
    n: usize,
    a: &DeviceMemory<TA>,
    lda: usize,
    w: &DeviceMemory<TW>,
) -> Result<SelectionWorkspaceSizes> {
    let (range, value_range, index_range) = selection_parts(selection);
    xsyevdx_raw_buffer_size(
        ctx,
        params,
        mode,
        range,
        fill_mode,
        n,
        TA::data_type(),
        a,
        lda,
        value_range,
        index_range,
        TW::data_type(),
        w,
        TA::data_type(),
    )
}

pub(crate) fn xsyevdx<TA: DataTypeLike, TR: Copy + Default, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<TR>,
    n: usize,
    a: &mut DeviceMemory<TA>,
    lda: usize,
    w: &mut DeviceMemory<TW>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize> {
    let (range, value_range, index_range) = selection_parts(selection);
    xsyevdx_raw(
        ctx,
        params,
        mode,
        range,
        fill_mode,
        n,
        TA::data_type(),
        a,
        lda,
        value_range,
        index_range,
        TW::data_type(),
        w,
        TA::data_type(),
        workspace,
        dev_info,
    )
}

pub(crate) fn xsyev_batched_buffer_size<TA: DataTypeLike, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixRef<'_, TA>,
    w: &DeviceMemory<TW>,
    batch_count: usize,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    validate_xsyev_batched_buffers(
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
        w.byte_len(),
        TW::data_type(),
        batch_count,
    )?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevBatched_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            TW::data_type().into(),
            w.as_ptr().cast(),
            TA::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
            to_i64(batch_count, "batch_count")?,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Computes eigenvalues and eigenvectors of a sequence of symmetric
/// (Hermitian) $n \times n$ matrices.
///
/// where $\Lambda\_j$ is a real $n \times n$ diagonal matrix. $V\_j$ is an $n \times n$ unitary matrix.
/// The diagonal elements of $\Lambda\_j$ are the eigenvalues of $A\_j$ in ascending order.
///
/// `syevBatched` performs an eigendecomposition on each matrix.
/// It requires all matrices to have the same size `n` and be packed contiguously.
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// `w` contains the eigenvalues of each matrix contiguously.
///
/// The formula for random access of `W` is $W\_{k}\operatorname{(j)} = {W\lbrack\ j\ +\ n\cdot k\rbrack}$.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xsyev_batched_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] > 0` indicates that `syevBatched` did not converge on the `i`th matrix.
///
/// If `mode` is [`EigenMode::Vector`], $A\_{j}$ contains the orthonormal eigenvectors of the matrix $A\_{j}$.
///
/// The problem size is limited by
/// `n * lda * batch_count <= INT32_MAX` primarily due to the current implementation
/// constraints.
///
/// **Algorithms supported by [`xsyev_batched`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default; may switch between algorithms for best performance. |
/// | [`AlgorithmMode::Algorithm1`](crate::types::AlgorithmMode::Algorithm1) | Uses a single algorithm for consistent accuracy over all `n`. |
///
/// List of input arguments for [`xsyev_batched_buffer_size`] and [`xsyev_batched`]:
///
/// The generic operation has three data types: `data_type_a` is the data type
/// of matrix `A`, `data_type_w` is the data type of `W`, and `compute_type` is
/// the compute type of the operation.
/// [`xsyev_batched`] only supports the following four combinations:
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **data_type_w** | **compute_type** | **Meaning** |
/// | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SSYEVBATCHED` |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DSYEVBATCHED` |
/// | [`DataType::ComplexF32`] | [`DataType::F32`] | [`DataType::ComplexF32`] | `CSYEVBATCHED` |
/// | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::ComplexF64`] | `ZSYEVBATCHED` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, eigen mode, fill mode, or batch size
/// are invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn xsyev_batched<TA: DataTypeLike, TW: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixMut<'_, TA>,
    w: &mut DeviceMemory<TW>,
    batch_count: usize,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_xsyev_batched_buffers(
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
        w.byte_len(),
        TW::data_type(),
        batch_count,
    )?;
    require_info_buffer_len(dev_info, batch_count)?;
    let workspace_sizes =
        xsyev_batched_buffer_size(ctx, params, mode, fill_mode, n, a.as_ref(), w, batch_count)?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevBatched(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            TW::data_type().into(),
            w.as_mut_ptr().cast(),
            TA::data_type().into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
            to_i64(batch_count, "batch_count")?,
        ))?;
    }
    Ok(())
}

pub(crate) fn xgeev_buffer_size<TA: DataTypeLike, TW: DataTypeLike, TV: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    n: usize,
    a: MatrixRef<'_, TA>,
    eigenvalues: &DeviceMemory<TW>,
    right_vectors: Option<MatrixRef<'_, TV>>,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    validate_xgeev_inputs(
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
        eigenvalues.byte_len(),
        TW::data_type(),
        matrix_ref_parts(right_vectors),
        TV::data_type(),
    )?;
    let (vr_ptr, ldvr) = optional_xgeev_matrix_ptr(matrix_ref_parts(right_vectors))?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgeev_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            EigenMode::NoVector.into(),
            if right_vectors.is_some() {
                EigenMode::Vector
            } else {
                EigenMode::NoVector
            }
            .into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            TW::data_type().into(),
            eigenvalues.as_ptr().cast(),
            TA::data_type().into(),
            ptr::null(),
            1,
            TV::data_type().into(),
            vr_ptr.cast(),
            ldvr,
            TA::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Computes the eigenvalues and, optionally, the left and/or right eigenvectors
/// of an n-by-n real non-symmetric or complex non-Hermitian matrix `A`.
/// The right eigenvector `v(j)` of `A` satisfies
///
/// where `w(j)` is its eigenvalue.
/// The left eigenvalue `u(j)` of `A` satisfies
///
/// where $u(j)^{H}$ denotes the conjugate-transpose of `u(j)`.
///
/// The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real.
///
/// If `A` is real-valued, there are two options to return the eigenvalues in `W`.
/// The first option sets all data types to real-valued types.
/// Then `W` holds `2*n` entries.
/// The first n entries hold the real parts and the last n entries hold the imaginary parts.
/// The LAPACK interface with separate arrays for the real parts `WR` and the
/// imaginary parts `WI` can be recovered by setting pointers `WR = W` and
/// `WI = W+n`.
/// The second option uses a complex data type for `W`.
/// Then `W` is n entries long; each real eigenvalue is stored as a complex number and for each complex conjugate pair, both eigenvalues are returned.
/// The computation is still executed fully in real arithmetic.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgeev_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == 0`, the QR algorithm converged; `W` contains the computed eigenvalues of `A`, and any requested left or right eigenvectors have been computed.
/// If `info == i` with `i > 0`, the QR algorithm failed to compute all eigenvalues and no eigenvectors were computed.
/// The elements `i + 1:n` of `W` contain eigenvalues that converged.
///
/// - `geev` only supports the computation of right eigenvectors.
///   Therefore, `jobvl` must be [`EigenMode::NoVector`].
///
/// - `geev` uses balancing to improve the conditioning of the eigenvalues and eigenvectors.
///
/// - `geev` is a hybrid CPU-GPU algorithm.
///   Best performance is attained with pinned host memory.
///
/// Currently, [`xgeev`] supports only the default algorithm.
///
/// **Table of algorithms supported by [`xgeev`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// List of input arguments for [`xgeev_buffer_size`] and [`xgeev`]:
///
/// The generic operation has five data types: `data_type_a` is the data type
/// of matrix `A`, `data_type_w` is the data type of `W`, `data_type_vl` is the
/// data type of matrix `VL`, `data_type_vr` is the data type of matrix `VR`,
/// and `compute_type` is the compute type of the operation.
/// [`xgeev`] only supports the following four combinations:
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **data_type_w** | **data_type_vl** | **data_type_vr** | **compute_type** | **Meaning** |
/// | --- | --- | --- | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | `SGEEV` |
/// | [`DataType::F32`] | [`DataType::ComplexF32`] | [`DataType::F32`] | [`DataType::F32`] | [`DataType::F32`] | 32F mixed real-complex |
/// | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | `DGEEV` |
/// | [`DataType::F64`] | [`DataType::ComplexF64`] | [`DataType::F64`] | [`DataType::F64`] | [`DataType::F64`] | 64F mixed real-complex |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGEEV` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGEEV` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or requested eigenvector modes are
/// invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn xgeev<TA: DataTypeLike, TW: DataTypeLike, TV: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    n: usize,
    a: MatrixMut<'_, TA>,
    eigenvalues: &mut DeviceMemory<TW>,
    right_vectors: Option<MatrixMut<'_, TV>>,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_xgeev_inputs(
        n,
        a.data.byte_len(),
        a.leading_dimension,
        TA::data_type(),
        eigenvalues.byte_len(),
        TW::data_type(),
        matrix_mut_ref_parts(right_vectors.as_ref()),
        TV::data_type(),
    )?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xgeev_buffer_size(
        ctx,
        params,
        n,
        a.as_ref(),
        eigenvalues,
        matrix_mut_ref_option(right_vectors.as_ref()),
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    let (vr_ptr, ldvr) = optional_xgeev_matrix_mut_ptr(matrix_mut_parts(right_vectors))?;
    unsafe {
        try_ffi!(sys::cusolverDnXgeev(
            ctx.as_raw(),
            params.as_raw(),
            EigenMode::NoVector.into(),
            if vr_ptr.is_null() {
                EigenMode::NoVector
            } else {
                EigenMode::Vector
            }
            .into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            TW::data_type().into(),
            eigenvalues.as_mut_ptr().cast(),
            TA::data_type().into(),
            ptr::null_mut(),
            1,
            TV::data_type().into(),
            vr_ptr.cast(),
            ldvr,
            TA::data_type().into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

fn xsyevd_raw_buffer_size<TA, TW>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a_type: DataType,
    a: &DeviceMemory<TA>,
    lda: usize,
    w_type: DataType,
    w: &DeviceMemory<TW>,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    validate_xsyevd_buffers(n, a.byte_len(), lda, a_type, w.byte_len(), w_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevd_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.as_ptr().cast(),
            to_i64(lda, "lda")?,
            w_type.into(),
            w.as_ptr().cast(),
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

fn xsyevd_raw<TA, TW>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a_type: DataType,
    a: &mut DeviceMemory<TA>,
    lda: usize,
    w_type: DataType,
    w: &mut DeviceMemory<TW>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_xsyevd_buffers(n, a.byte_len(), lda, a_type, w.byte_len(), w_type)?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xsyevd_raw_buffer_size(
        ctx,
        params,
        mode,
        fill_mode,
        n,
        a_type,
        a,
        lda,
        w_type,
        w,
        compute_type,
    )?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevd(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.as_mut_ptr().cast(),
            to_i64(lda, "lda")?,
            w_type.into(),
            w.as_mut_ptr().cast(),
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

fn xsyevdx_raw_buffer_size<TA, TR, TW>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    range: EigenRange,
    fill_mode: FillMode,
    n: usize,
    a_type: DataType,
    a: &DeviceMemory<TA>,
    lda: usize,
    value_range: Option<(TR, TR)>,
    index_range: Option<(usize, usize)>,
    w_type: DataType,
    w: &DeviceMemory<TW>,
    compute_type: DataType,
) -> Result<SelectionWorkspaceSizes>
where
    TR: Copy + Default,
{
    ctx.bind()?;
    validate_xsyevd_buffers(n, a.byte_len(), lda, a_type, w.byte_len(), w_type)?;
    validate_xsyevdx_value_type::<TR>(w_type)?;
    let (mut vl, mut vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig = 0;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevdx_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.as_ptr().cast(),
            to_i64(lda, "lda")?,
            (&raw mut vl).cast(),
            (&raw mut vu).cast(),
            to_i64(il, "il")?,
            to_i64(iu, "iu")?,
            &raw mut meig,
            w_type.into(),
            w.as_ptr().cast(),
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(SelectionWorkspaceSizes::new(
        to_usize(meig, "meig")?,
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

fn xsyevdx_raw<TA, TR, TW>(
    ctx: &Context,
    params: &Params,
    mode: EigenMode,
    range: EigenRange,
    fill_mode: FillMode,
    n: usize,
    a_type: DataType,
    a: &mut DeviceMemory<TA>,
    lda: usize,
    value_range: Option<(TR, TR)>,
    index_range: Option<(usize, usize)>,
    w_type: DataType,
    w: &mut DeviceMemory<TW>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize>
where
    TR: Copy + Default,
{
    ctx.bind()?;
    validate_xsyevd_buffers(n, a.byte_len(), lda, a_type, w.byte_len(), w_type)?;
    validate_xsyevdx_value_type::<TR>(w_type)?;
    require_info_buffer(dev_info)?;
    let workspace_sizes = xsyevdx_raw_buffer_size(
        ctx,
        params,
        mode,
        range,
        fill_mode,
        n,
        a_type,
        a,
        lda,
        value_range,
        index_range,
        w_type,
        w,
        compute_type,
    )?;
    require_workspace_bytes(
        workspace.device.byte_len(),
        workspace_sizes.workspace.device_bytes,
    )?;
    require_host_workspace(workspace.host.len(), workspace_sizes.workspace.host_bytes)?;
    let (mut vl, mut vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig_raw = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXsyevdx(
            ctx.as_raw(),
            params.as_raw(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.as_mut_ptr().cast(),
            to_i64(lda, "lda")?,
            (&raw mut vl).cast(),
            (&raw mut vu).cast(),
            to_i64(il, "il")?,
            to_i64(iu, "iu")?,
            &raw mut meig_raw,
            w_type.into(),
            w.as_mut_ptr().cast(),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            workspace_sizes.workspace.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.workspace.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    debug_assert_eq!(workspace_sizes.selection_size, to_usize(meig_raw, "meig")?);
    Ok(workspace_sizes.selection_size)
}

pub(crate) fn ssyevj_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsyevj_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn dsyevj_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsyevj_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn cheevj_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCheevj_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn zheevj_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZheevj_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the workspace length before
/// allocating workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian)
/// $n \times n$ matrix `A`.
/// The standard symmetric eigenvalue problem is $A Q = Q \Lambda$, where `Λ` is a real $n \times n$ diagonal matrix.
/// `Q` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Λ` are the eigenvalues of `A` in ascending order.
///
/// `syevj` computes the same result as `syevd`, but uses the Jacobi method
/// instead of the QR algorithm.
/// The Jacobi method's parallelism gives GPUs better performance on small and
/// medium-size matrices.
/// `syevj` can also be configured to approximate results up to a chosen
/// accuracy.
///
/// `syevj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = Q(W + E)Q^{H}$, where `W` is diagonal and `E` is symmetric with a zero diagonal.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `W` is the set of eigenvalues.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
///
/// `syevj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`SyevjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the
/// number of Jacobi iterations.
/// The default value is 100, but [`SyevjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are typically enough to converge to machine
/// accuracy.
/// `syevj` stops when either the tolerance or the maximum number of sweeps is
/// reached.
///
/// The Jacobi method has quadratic convergence, so accuracy is not
/// proportional to the number of sweeps.
/// To target a specific accuracy, configure the tolerance.
///
/// After `syevj`, callers can query the residual with [`SyevjInfo::residual`] and the number of executed sweeps with [`SyevjInfo::executed_sweeps`].
/// However, the residual is the Frobenius norm of `E`, not the accuracy of each individual eigenvalue.
///
/// Provide workspace through `workspace`, just like the generic [`xsyevd`] operation.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == n + 1`, `syevj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `syevj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthonormal eigenvectors `V`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, eigen mode, or fill mode are invalid,
/// or if cuSOLVER reports an internal failure.
pub(crate) fn ssyevj(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = ssyevj_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsyevj(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
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
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian) $n \times n$ matrix `A`.
/// The standard symmetric eigenvalue problem is $A Q = Q \Lambda$, where `Λ` is a real $n \times n$ diagonal matrix.
/// `Q` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Λ` are the eigenvalues of `A` in ascending order.
///
/// `syevj` computes the same symmetric eigenvalue problem as `syevd`.
/// The difference is that `syevd` uses a QR algorithm and `syevj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `syevj` to target a chosen accuracy.
///
/// `syevj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = Q(W + E)Q^{H}$, where `W` is diagonal and `E` is symmetric with a zero diagonal.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `W` is the set of eigenvalues.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
///
/// `syevj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`SyevjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`SyevjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `syevj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// After `syevj`, callers can query the residual with [`SyevjInfo::residual`] and the number of executed sweeps with [`SyevjInfo::executed_sweeps`].
/// However, the residual is the Frobenius norm of `E`, not the accuracy of each individual eigenvalue.
///
/// Provide workspace through `workspace`, just like the generic [`xsyevd`] operation.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == n + 1`, `syevj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `syevj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthonormal eigenvectors `V`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, eigen mode, or fill mode are invalid,
/// or if cuSOLVER reports an internal failure.
pub(crate) fn dsyevj(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dsyevj_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsyevj(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn cheevj(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = cheevj_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCheevj(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn zheevj(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_syev_buffers(n, a.len(), lda, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zheevj_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZheevj(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn ssyevj_batched_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsyevjBatched_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn dsyevj_batched_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsyevjBatched_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn cheevj_batched_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCheevjBatched_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn zheevj_batched_buffer_size(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZheevjBatched_bufferSize(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
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
/// Computes eigenvalues and eigenvectors of a sequence of symmetric (Hermitian) $n \times n$ matrices
///
/// where $\Lambda\_{j}$ is a real $n \times n$ diagonal matrix. $Q\_j$ is an $n \times n$ unitary matrix.
/// The diagonal elements of $\Lambda\_j$ are the eigenvalues of $A\_j$ in either ascending order or non-sorting order.
///
/// `syevj_batched` performs `syevj` on each matrix.
/// It requires that all matrices are of the same size `n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `W` parameter also contains the eigenvalues of each matrix contiguously,
///
/// The formula for random access of `W` is $W\_{k}\operatorname{(j)} = {W\lbrack\ j\ +\ n\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `syevj_batched` can either sort the eigenvalues in ascending order (default) or choose as-is (without sorting) with [`SyevjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the spectra of those tiny matrices.
///
/// `syevj_batched` cannot report residual and executed sweeps through [`SyevjInfo::residual`] and [`SyevjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == n + 1` indicates that `syevj_batched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// If `mode` is [`EigenMode::Vector`], $A\_j$ contains the orthonormal eigenvectors $V\_j$.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, eigen mode, fill mode, or batch size
/// are invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn ssyevj_batched(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    require_info_buffer_len(dev_info, batch_count)?;
    let lwork =
        ssyevj_batched_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params, batch_count)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsyevjBatched(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
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
/// Computes eigenvalues and eigenvectors of a sequence of symmetric (Hermitian) $n \times n$ matrices
///
/// where $\Lambda\_{j}$ is a real $n \times n$ diagonal matrix. $Q\_j$ is an $n \times n$ unitary matrix.
/// The diagonal elements of $\Lambda\_j$ are the eigenvalues of $A\_j$ in either ascending order or non-sorting order.
///
/// `syevj_batched` performs `syevj` on each matrix.
/// It requires that all matrices are of the same size `n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `W` parameter also contains the eigenvalues of each matrix contiguously,
///
/// The formula for random access of `W` is $W\_{k}\operatorname{(j)} = {W\lbrack\ j\ +\ n\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `syevj_batched` can either sort the eigenvalues in ascending order (default) or choose as-is (without sorting) with [`SyevjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the spectra of those tiny matrices.
///
/// `syevj_batched` cannot report residual and executed sweeps through [`SyevjInfo::residual`] and [`SyevjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == n + 1` indicates that `syevj_batched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// If `mode` is [`EigenMode::Vector`], $A\_j$ contains the orthonormal eigenvectors $V\_j$.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, eigen mode, fill mode, or batch size
/// are invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn dsyevj_batched(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    require_info_buffer_len(dev_info, batch_count)?;
    let lwork =
        dsyevj_batched_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params, batch_count)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsyevjBatched(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    Ok(())
}

pub(crate) fn cheevj_batched(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    require_info_buffer_len(dev_info, batch_count)?;
    let lwork =
        cheevj_batched_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params, batch_count)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCheevjBatched(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    Ok(())
}

pub(crate) fn zheevj_batched(
    ctx: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    ctx.bind()?;
    validate_syevj_batched_buffers(n, a.len(), lda, w.len(), batch_count)?;
    require_info_buffer_len(dev_info, batch_count)?;
    let lwork =
        zheevj_batched_buffer_size(ctx, mode, fill_mode, n, a, lda, w, params, batch_count)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZheevjBatched(
            ctx.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?,
        ))?;
    }
    Ok(())
}

pub(crate) fn ssygvj_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    b: &DeviceMemory<f32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvj_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn dsygvj_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvj_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn chegvj_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    b: &DeviceMemory<Complex32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnChegvj_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn zhegvj_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
    params: &SyevjInfo,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvj_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
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
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem depends on
/// [`EigenType`].
///
/// where the matrix `B` is positive definite.
/// `Λ` is a real $n \times n$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times n$ orthogonal matrix.
/// The eigenvectors are normalized according to the selected generalized
/// eigenvalue problem.
///
/// This computes the same generalized symmetric eigenvalue problem as `sygvd`, but `sygvj` uses `syevj` where `sygvd` uses `syevd`.
/// Therefore, `sygvj` inherits properties of `syevj`; use [`SyevjInfo::set_tolerance`] and [`SyevjInfo::set_max_sweeps`] to configure tolerance and maximum sweeps.
///
/// However the meaning of residual is different from `syevj`.
/// `sygvj` first computes Cholesky factorization of matrix `B`,
///
/// transform the problem to standard eigenvalue problem, then calls `syevj`.
///
/// For example, the standard eigenvalue problem of type I is
///
/// where matrix `M` is symmetric
///
/// The residual is the result of `syevj` on matrix `M`, not `A`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == i` with `0 < i <= n`, `B` is not positive definite, the factorization of `B` could not be completed, and no eigenvalues or eigenvectors were computed.
/// If `info == n + 1`, `syevj` did not converge within the given tolerance and maximum sweep count.
/// In this case, the eigenvalues and eigenvectors are still computed because non-convergence comes from the configured tolerance or maximum sweep count.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors `V`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// mode, or fill mode are invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn ssygvj(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    b: &mut DeviceMemory<f32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = ssygvj_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvj(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
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
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem depends on
/// [`EigenType`].
///
/// where the matrix `B` is positive definite.
/// `Λ` is a real $n \times n$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times n$ orthogonal matrix.
/// The eigenvectors are normalized according to the selected generalized
/// eigenvalue problem.
///
/// This computes the same generalized symmetric eigenvalue problem as `sygvd`, but `sygvj` uses `syevj` where `sygvd` uses `syevd`.
/// Therefore, `sygvj` inherits properties of `syevj`; use [`SyevjInfo::set_tolerance`] and [`SyevjInfo::set_max_sweeps`] to configure tolerance and maximum sweeps.
///
/// However the meaning of residual is different from `syevj`.
/// `sygvj` first computes Cholesky factorization of matrix `B`,
///
/// transform the problem to standard eigenvalue problem, then calls `syevj`.
///
/// For example, the standard eigenvalue problem of type I is
///
/// where matrix `M` is symmetric
///
/// The residual is the result of `syevj` on matrix `M`, not `A`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == i` with `0 < i <= n`, `B` is not positive definite, the factorization of `B` could not be completed, and no eigenvalues or eigenvectors were computed.
/// If `info == n + 1`, `syevj` did not converge within the given tolerance and maximum sweep count.
/// In this case, the eigenvalues and eigenvectors are still computed because non-convergence comes from the configured tolerance or maximum sweep count.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors `V`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// mode, or fill mode are invalid, or if cuSOLVER reports an internal failure.
pub(crate) fn dsygvj(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    b: &mut DeviceMemory<f64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dsygvj_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvj(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn chegvj(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    b: &mut DeviceMemory<Complex32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = chegvj_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnChegvj(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn zhegvj(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    b: &mut DeviceMemory<Complex64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
    params: &SyevjInfo,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvj_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zhegvj_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w, params)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvj(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

pub(crate) fn ssygvd_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    b: &DeviceMemory<f32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvd_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn dsygvd_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvd_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn chegvd_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    b: &DeviceMemory<Complex32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnChegvd_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub(crate) fn zhegvd_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvd_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_ptr().cast(),
            &raw mut lwork,
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
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem depends on
/// [`EigenType`].
///
/// where the matrix `B` is positive definite.
/// `Λ` is a real $n \times n$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times n$ orthogonal matrix.
/// The eigenvectors are normalized according to the selected generalized
/// eigenvalue problem.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// If `dev_info = i` with `0 < i <= n` and `mode` is
/// [`EigenMode::NoVector`], `i` off-diagonal elements of an intermediate
/// tridiagonal form did not converge to zero.
/// If `dev_info = N + i` with `i > 0`, then the leading minor of order `i` of
/// `B` is not positive definite.
/// The factorization of `B` could not be completed and no eigenvalues or eigenvectors were computed.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors of the matrix `A`.
/// The eigenvectors are computed by divide and conquer algorithm.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// mode, or fill mode are invalid, if the current GPU architecture is
/// unsupported, or if cuSOLVER reports an internal failure.
pub(crate) fn ssygvd(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    b: &mut DeviceMemory<f32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = ssygvd_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvd(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
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
/// Computes eigenvalues and eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem depends on
/// [`EigenType`].
///
/// where the matrix `B` is positive definite.
/// `Λ` is a real $n \times n$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times n$ orthogonal matrix.
/// The eigenvectors are normalized according to the selected generalized
/// eigenvalue problem.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// If `dev_info = i` with `0 < i <= n` and `mode` is
/// [`EigenMode::NoVector`], `i` off-diagonal elements of an intermediate
/// tridiagonal form did not converge to zero.
/// If `dev_info = N + i` with `i > 0`, then the leading minor of order `i` of
/// `B` is not positive definite.
/// The factorization of `B` could not be completed and no eigenvalues or eigenvectors were computed.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors of the matrix `A`.
/// The eigenvectors are computed by divide and conquer algorithm.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// mode, or fill mode are invalid, if the current GPU architecture is
/// unsupported, or if cuSOLVER reports an internal failure.
pub(crate) fn dsygvd(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    b: &mut DeviceMemory<f64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dsygvd_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvd(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub(crate) fn chegvd(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    b: &mut DeviceMemory<Complex32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = chegvd_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnChegvd(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub(crate) fn zhegvd(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    b: &mut DeviceMemory<Complex64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zhegvd_buffer_size(ctx, eig_type, mode, fill_mode, n, a, lda, b, ldb, w)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvd(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub(crate) fn ssygvdx_selected_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f32>,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    b: &DeviceMemory<f32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
) -> Result<(usize, usize)> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig = 0;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvdx_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    Ok((to_usize(meig, "meig")?, to_usize(lwork, "lwork")?))
}

pub(crate) fn dsygvdx_selected_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f64>,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    b: &DeviceMemory<f64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
) -> Result<(usize, usize)> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig = 0;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvdx_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    Ok((to_usize(meig, "meig")?, to_usize(lwork, "lwork")?))
}

pub(crate) fn chegvdx_selected_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f32>,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    b: &DeviceMemory<Complex32>,
    ldb: usize,
    w: &DeviceMemory<f32>,
) -> Result<(usize, usize)> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig = 0;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnChegvdx_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    Ok((to_usize(meig, "meig")?, to_usize(lwork, "lwork")?))
}

pub(crate) fn zhegvdx_selected_buffer_size(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f64>,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    b: &DeviceMemory<Complex64>,
    ldb: usize,
    w: &DeviceMemory<f64>,
) -> Result<(usize, usize)> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig = 0;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvdx_bufferSize(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig,
            w.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    Ok((to_usize(meig, "meig")?, to_usize(lwork, "lwork")?))
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes all or selection of the eigenvalues and optionally eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem is selected by
/// `eig_type`, with positive-definite matrix `B`.
/// `Λ` is a real $n \times {h\_meig}$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times {h\_meig}$ orthogonal matrix.
/// `h_meig` is the number of eigenvalues/eigenvectors computed by the operation, `h_meig` is equal to `n` when the whole spectrum (for example, `range` = [`EigenRange::All`]) is requested.
/// For [`EigenType::Type1`] and [`EigenType::Type2`], eigenvectors are
/// normalized so that $V^{T} B V = I$. For [`EigenType::Type3`], they are
/// normalized so that $V^{T} B^{-1} V = I$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// If `dev_info = i` with `0 < i <= n` and `mode` is
/// [`EigenMode::NoVector`], `i` off-diagonal elements of an intermediate
/// tridiagonal form did not converge to zero.
/// If `dev_info = n + i` with `i > 0`, then the leading minor of order `i` of
/// `B` is not positive definite.
/// The factorization of `B` could not be completed and no eigenvalues or eigenvectors were computed.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors
/// of the matrix `A`.
/// The eigenvectors are computed by divide and conquer algorithm.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// selection, eigen mode, or fill mode are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub(crate) fn ssygvdx_selected(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f32>,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    b: &mut DeviceMemory<f32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (meig, lwork) = ssygvdx_selected_buffer_size(
        ctx, eig_type, mode, fill_mode, selection, n, a, lda, b, ldb, w,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig_raw = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsygvdx(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig_raw,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    debug_assert_eq!(meig, to_usize(meig_raw, "meig")?);
    Ok(meig)
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes all or selection of the eigenvalues and optionally eigenvectors of a symmetric (Hermitian) $n \times n$ matrix-pair (`A`,`B`).
/// The generalized symmetric-definite eigenvalue problem is selected by
/// `eig_type`, with positive-definite matrix `B`.
/// `Λ` is a real $n \times {h\_meig}$ diagonal matrix.
/// The diagonal elements of `Λ` are the eigenvalues of (`A`, `B`) in ascending order.
/// `V` is an $n \times {h\_meig}$ orthogonal matrix.
/// `h_meig` is the number of eigenvalues/eigenvectors computed by the operation, `h_meig` is equal to `n` when the whole spectrum (for example, `range` = [`EigenRange::All`]) is requested.
/// For [`EigenType::Type1`] and [`EigenType::Type2`], eigenvectors are
/// normalized so that $V^{T} B V = I$. For [`EigenType::Type3`], they are
/// normalized so that $V^{T} B^{-1} V = I$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// If `dev_info = i` with `0 < i <= n` and `mode` is
/// [`EigenMode::NoVector`], `i` off-diagonal elements of an intermediate
/// tridiagonal form did not converge to zero.
/// If `dev_info = n + i` with `i > 0`, then the leading minor of order `i` of
/// `B` is not positive definite.
/// The factorization of `B` could not be completed and no eigenvalues or eigenvectors were computed.
///
/// If `mode` is [`EigenMode::Vector`], `A` contains the orthogonal eigenvectors
/// of the matrix `A`.
/// The eigenvectors are computed by divide and conquer algorithm.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, generalized eigenproblem type, eigen
/// selection, eigen mode, or fill mode are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub(crate) fn dsygvdx_selected(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f64>,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    b: &mut DeviceMemory<f64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (meig, lwork) = dsygvdx_selected_buffer_size(
        ctx, eig_type, mode, fill_mode, selection, n, a, lda, b, ldb, w,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig_raw = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsygvdx(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig_raw,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    debug_assert_eq!(meig, to_usize(meig_raw, "meig")?);
    Ok(meig)
}

pub(crate) fn chegvdx_selected(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f32>,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    b: &mut DeviceMemory<Complex32>,
    ldb: usize,
    w: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (meig, lwork) = chegvdx_selected_buffer_size(
        ctx, eig_type, mode, fill_mode, selection, n, a, lda, b, ldb, w,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig_raw = 0;
    unsafe {
        try_ffi!(sys::cusolverDnChegvdx(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig_raw,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    debug_assert_eq!(meig, to_usize(meig_raw, "meig")?);
    Ok(meig)
}

pub(crate) fn zhegvdx_selected(
    ctx: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: EigenSelection<f64>,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    b: &mut DeviceMemory<Complex64>,
    ldb: usize,
    w: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sygvd_buffers(n, a.len(), lda, b.len(), ldb, w.len())?;
    require_info_buffer(dev_info)?;
    let (range, value_range, index_range) = selection_parts(selection);
    let (meig, lwork) = zhegvdx_selected_buffer_size(
        ctx, eig_type, mode, fill_mode, selection, n, a, lda, b, ldb, w,
    )?;
    require_workspace(workspace.len(), lwork)?;
    let (vl, vu, il, iu) = validate_xsyevdx_range(range, n, value_range, index_range)?;
    let mut meig_raw = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZhegvdx(
            ctx.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(ldb, "ldb")?,
            vl,
            vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            &raw mut meig_raw,
            w.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    debug_assert_eq!(meig, to_usize(meig_raw, "meig")?);
    Ok(meig)
}
