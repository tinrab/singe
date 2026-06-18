use std::ptr;

use singe_cuda::{data_type::DataType, memory::DeviceMemory};

use crate::{
    error::{Error, Result},
    layout::{MatrixMut, MatrixRef, StridedBatchedMatrixMut, StridedBatchedMatrixRef},
    types::{EigenMode, SvdMode, TruncatedSvdMode},
    utility::{to_i32, to_i64},
};

pub fn validate_gesvd_dims(m: usize, n: usize) -> Result<()> {
    if m == 0 || n == 0 || m < n {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn validate_xgesvdp_inputs<TU, TV>(
    m: usize,
    n: usize,
    a_bytes: usize,
    lda: usize,
    a_type: DataType,
    s_bytes: usize,
    s_type: DataType,
    jobz: EigenMode,
    econ: bool,
    u: Option<&(&DeviceMemory<TU>, usize)>,
    u_type: DataType,
    v: Option<&(&DeviceMemory<TV>, usize)>,
    v_type: DataType,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    validate_x_matrix(m, n, a_bytes, lda, a_type)?;
    validate_x_vector(m.min(n), s_bytes, s_type)?;
    match jobz {
        EigenMode::NoVector => Ok(()),
        EigenMode::Vector => {
            let Some((u, ldu)) = u else {
                return Err(Error::InvalidMatrixShape);
            };
            let Some((v, ldv)) = v else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_eig_output(m, n, u.byte_len(), *ldu, econ, u_type)?;
            validate_x_eig_output(n, n, v.byte_len(), *ldv, econ, v_type)
        }
    }
}

pub fn matrix_ref_parts<T>(matrix: Option<MatrixRef<'_, T>>) -> Option<(&DeviceMemory<T>, usize)> {
    matrix.map(|matrix| (matrix.data, matrix.leading_dimension))
}

pub fn matrix_mut_parts<T>(
    matrix: Option<MatrixMut<'_, T>>,
) -> Option<(&mut DeviceMemory<T>, usize)> {
    matrix.map(|matrix| (matrix.data, matrix.leading_dimension))
}

pub fn matrix_mut_ref_parts<'a, T>(
    matrix: Option<&'a MatrixMut<'a, T>>,
) -> Option<(&'a DeviceMemory<T>, usize)> {
    matrix.map(|matrix| (&*matrix.data, matrix.leading_dimension))
}

pub fn matrix_mut_ref_option<'a, T>(
    matrix: Option<&'a MatrixMut<'a, T>>,
) -> Option<MatrixRef<'a, T>> {
    matrix.map(MatrixMut::as_ref)
}

pub fn strided_batched_matrix_ref_parts<T>(
    matrix: Option<StridedBatchedMatrixRef<'_, T>>,
) -> Option<(&DeviceMemory<T>, usize, usize)> {
    matrix.map(|matrix| (matrix.data, matrix.leading_dimension, matrix.stride))
}

pub fn strided_batched_matrix_mut_parts<T>(
    matrix: Option<StridedBatchedMatrixMut<'_, T>>,
) -> Option<(&mut DeviceMemory<T>, usize, usize)> {
    matrix.map(|matrix| (matrix.data, matrix.leading_dimension, matrix.stride))
}

pub fn strided_batched_matrix_mut_ref_option<'a, T>(
    matrix: Option<&'a StridedBatchedMatrixMut<'a, T>>,
) -> Option<StridedBatchedMatrixRef<'a, T>> {
    matrix.map(StridedBatchedMatrixMut::as_ref)
}

pub fn validate_xgesvdr_inputs<TU, TV>(
    m: usize,
    n: usize,
    k: usize,
    p: usize,
    niters: usize,
    a_bytes: usize,
    lda: usize,
    a_type: DataType,
    s_bytes: usize,
    s_type: DataType,
    job_u: TruncatedSvdMode,
    u: Option<&(&DeviceMemory<TU>, usize)>,
    u_type: DataType,
    job_v: TruncatedSvdMode,
    v: Option<&(&DeviceMemory<TV>, usize)>,
    v_type: DataType,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 || k >= m.min(n) || p == 0 || k.checked_add(p).is_none() {
        return Err(Error::InvalidMatrixShape);
    }
    let kp = k.checked_add(p).ok_or(Error::InvalidMatrixShape)?;
    if kp >= m.min(n) || niters == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    validate_x_matrix(m, n, a_bytes, lda, a_type)?;
    validate_x_vector(k, s_bytes, s_type)?;
    if matches!(job_u, TruncatedSvdMode::Some) {
        let Some((u, ldu)) = u else {
            return Err(Error::InvalidMatrixShape);
        };
        validate_x_matrix(m, k, u.byte_len(), *ldu, u_type)?;
    }
    if matches!(job_v, TruncatedSvdMode::Some) {
        let Some((v, ldv)) = v else {
            return Err(Error::InvalidMatrixShape);
        };
        validate_x_matrix(n, k, v.byte_len(), *ldv, v_type)?;
    }
    Ok(())
}

pub fn validate_gesvdj_inputs<T>(
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    s_len: usize,
    jobz: EigenMode,
    econ: bool,
    u: Option<(&DeviceMemory<T>, usize)>,
    v: Option<(&DeviceMemory<T>, usize)>,
) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    validate_matrix(m, n, a_len, lda)?;
    if s_len < m.min(n) {
        return Err(Error::InvalidVectorShape);
    }
    validate_gesvdj_output(m, n, jobz, econ, u)?;
    validate_gesvdj_output(n, n, jobz, econ, v)?;
    Ok(())
}

pub fn validate_gesvda_strided_batched_inputs<T>(
    rank: usize,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    stride_a: usize,
    s_len: usize,
    stride_s: usize,
    jobz: EigenMode,
    u: Option<(&DeviceMemory<T>, usize, usize)>,
    v: Option<(&DeviceMemory<T>, usize, usize)>,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 || m == 0 || n == 0 || m < n || rank == 0 || rank > n {
        return Err(Error::InvalidMatrixShape);
    }

    validate_strided_matrix(m, n, a_len, lda, stride_a, batch_size)?;
    validate_strided_vector(s_len, n, stride_s, batch_size)?;

    match jobz {
        EigenMode::NoVector => {}
        EigenMode::Vector => {
            let Some((u, ldu, stride_u)) = u else {
                return Err(Error::InvalidMatrixShape);
            };
            let Some((v, ldv, stride_v)) = v else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_strided_matrix(m, rank, u.len(), ldu, stride_u, batch_size)?;
            validate_strided_matrix(n, rank, v.len(), ldv, stride_v, batch_size)?;
        }
    }
    Ok(())
}

pub fn validate_gesvdj_batched_inputs<T>(
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    s_len: usize,
    jobz: EigenMode,
    u: Option<(&DeviceMemory<T>, usize)>,
    v: Option<(&DeviceMemory<T>, usize)>,
    batch_size: usize,
) -> Result<()> {
    if batch_size == 0 || m == 0 || n == 0 || m > 32 || n > 32 {
        return Err(Error::InvalidMatrixShape);
    }

    let a_cols = n.checked_mul(batch_size).ok_or(Error::InvalidMatrixShape)?;
    validate_matrix(m, a_cols, a_len, lda)?;

    let s_required = m
        .min(n)
        .checked_mul(batch_size)
        .ok_or(Error::InvalidVectorShape)?;
    if s_len < s_required {
        return Err(Error::InvalidVectorShape);
    }

    validate_gesvdj_batched_output(m, n, jobz, u, batch_size)?;
    validate_gesvdj_batched_output(n, n, jobz, v, batch_size)?;
    Ok(())
}

pub fn validate_gesvdj_output<T>(
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    econ: bool,
    matrix: Option<(&DeviceMemory<T>, usize)>,
) -> Result<()> {
    match jobz {
        EigenMode::NoVector => Ok(()),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            let out_cols = if econ { rows.min(cols) } else { cols };
            validate_matrix(rows, out_cols, matrix.len(), ld)
        }
    }
}

pub fn validate_gesvdj_batched_output<T>(
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    matrix: Option<(&DeviceMemory<T>, usize)>,
    batch_size: usize,
) -> Result<()> {
    match jobz {
        EigenMode::NoVector => Ok(()),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            let out_cols = rows
                .min(cols)
                .checked_mul(batch_size)
                .ok_or(Error::InvalidMatrixShape)?;
            validate_matrix(rows, out_cols, matrix.len(), ld)
        }
    }
}

pub fn optional_gesvda_output_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
) -> Result<(*mut T, i32, i64)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1, 0)),
        EigenMode::Vector => {
            let Some((matrix, ld, stride)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_strided_matrix(rows, cols, matrix.len(), ld, stride, 1)?;
            Ok((
                matrix.as_ptr() as *mut T,
                to_i32(ld, "ld")?,
                to_i64(stride, "stride")?,
            ))
        }
    }
}

pub fn optional_gesvda_output_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
) -> Result<(*mut T, i32, i64)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1, 0)),
        EigenMode::Vector => {
            let Some((matrix, ld, stride)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_strided_matrix(rows, cols, matrix.len(), ld, stride, 1)?;
            Ok((
                matrix.as_mut_ptr().cast(),
                to_i32(ld, "ld")?,
                to_i64(stride, "stride")?,
            ))
        }
    }
}

pub fn validate_x_svd_output<T>(
    rows: usize,
    full_cols: usize,
    matrix: Option<(&DeviceMemory<T>, usize)>,
    mode: SvdMode,
    data_type: DataType,
) -> Result<()> {
    match mode {
        SvdMode::None | SvdMode::Overwrite => Ok(()),
        SvdMode::All => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, full_cols, matrix.byte_len(), ld, data_type)
        }
        SvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, full_cols.min(rows), matrix.byte_len(), ld, data_type)
        }
    }
}

pub fn validate_x_eig_output(
    rows: usize,
    cols: usize,
    bytes: usize,
    ld: usize,
    econ: bool,
    data_type: DataType,
) -> Result<()> {
    let out_cols = if econ { rows.min(cols) } else { cols };
    validate_x_matrix(rows, out_cols, bytes, ld, data_type)
}

pub fn optional_x_matrix_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: SvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        SvdMode::None | SvdMode::Overwrite => Ok((ptr::null_mut(), 1)),
        SvdMode::All => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_ptr() as *mut T, to_i64(ld, "ld")?))
        }
        SvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols.min(rows), matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_ptr() as *mut T, to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_matrix_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: SvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        SvdMode::None | SvdMode::Overwrite => Ok((ptr::null_mut(), 1)),
        SvdMode::All => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ld")?))
        }
        SvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols.min(rows), matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_eig_matrix_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    econ: bool,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1)),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_eig_output(rows, cols, matrix.byte_len(), ld, econ, data_type)?;
            Ok((matrix.as_ptr() as *mut T, to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_eig_matrix_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    econ: bool,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1)),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_eig_output(rows, cols, matrix.byte_len(), ld, econ, data_type)?;
            Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_truncated_u_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: TruncatedSvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        TruncatedSvdMode::None => Ok((ptr::null_mut(), 1)),
        TruncatedSvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_ptr() as *mut T, to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_truncated_u_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: TruncatedSvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        TruncatedSvdMode::None => Ok((ptr::null_mut(), 1)),
        TruncatedSvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_truncated_v_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: TruncatedSvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        TruncatedSvdMode::None => Ok((ptr::null_mut(), 1)),
        TruncatedSvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_ptr() as *mut T, to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_x_truncated_v_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    mode: TruncatedSvdMode,
    data_type: DataType,
) -> Result<(*mut T, i64)> {
    match mode {
        TruncatedSvdMode::None => Ok((ptr::null_mut(), 1)),
        TruncatedSvdMode::Some => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            validate_x_matrix(rows, cols, matrix.byte_len(), ld, data_type)?;
            Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ld")?))
        }
    }
}

pub fn optional_gesvdj_matrix_ptr<T>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    econ: bool,
) -> Result<(*mut T, i32)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1)),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            let out_cols = if econ { rows.min(cols) } else { cols };
            validate_matrix(rows, out_cols, matrix.len(), ld)?;
            Ok((matrix.as_ptr() as *mut T, to_i32(ld, "ld")?))
        }
    }
}

pub fn optional_gesvdj_matrix_mut_ptr<T>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
    rows: usize,
    cols: usize,
    jobz: EigenMode,
    econ: bool,
) -> Result<(*mut T, i32)> {
    match jobz {
        EigenMode::NoVector => Ok((ptr::null_mut(), 1)),
        EigenMode::Vector => {
            let Some((matrix, ld)) = matrix else {
                return Err(Error::InvalidMatrixShape);
            };
            let out_cols = if econ { rows.min(cols) } else { cols };
            validate_matrix(rows, out_cols, matrix.len(), ld)?;
            Ok((matrix.as_mut_ptr().cast(), to_i32(ld, "ld")?))
        }
    }
}

pub fn validate_matrix(rows: usize, cols: usize, len: usize, lda: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    if lda < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    let required = lda.checked_mul(cols).ok_or(Error::InvalidMatrixShape)?;
    if len < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn validate_x_matrix(
    rows: usize,
    cols: usize,
    bytes: usize,
    lda: usize,
    data_type: DataType,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    if lda < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    let elem_size = data_type.size_of();
    let required = lda
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(elem_size))
        .ok_or(Error::InvalidMatrixShape)?;
    if bytes < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn validate_x_vector(len: usize, bytes: usize, data_type: DataType) -> Result<()> {
    let required = len
        .checked_mul(data_type.size_of())
        .ok_or(Error::InvalidVectorShape)?;
    if bytes < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_strided_matrix(
    rows: usize,
    cols: usize,
    len: usize,
    lda: usize,
    stride: usize,
    batch_size: usize,
) -> Result<()> {
    validate_matrix(rows, cols, len, lda)?;
    if batch_size == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    let footprint = lda.checked_mul(cols).ok_or(Error::InvalidMatrixShape)?;
    if stride < footprint {
        return Err(Error::InvalidMatrixShape);
    }
    let required = if batch_size == 1 {
        footprint
    } else {
        stride
            .checked_mul(batch_size - 1)
            .and_then(|base| base.checked_add(footprint))
            .ok_or(Error::InvalidMatrixShape)?
    };
    if len < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn validate_strided_vector(
    len: usize,
    width: usize,
    stride: usize,
    batch_size: usize,
) -> Result<()> {
    if width == 0 || batch_size == 0 {
        return Err(Error::InvalidVectorShape);
    }
    if stride < width {
        return Err(Error::InvalidVectorShape);
    }
    let required = if batch_size == 1 {
        width
    } else {
        stride
            .checked_mul(batch_size - 1)
            .and_then(|base| base.checked_add(width))
            .ok_or(Error::InvalidVectorShape)?
    };
    if len < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_workspace(actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    Ok(())
}

pub fn require_workspace_bytes(actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    Ok(())
}

pub fn require_host_workspace(actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::InsufficientWorkspaceSize { required, actual });
    }
    Ok(())
}

pub fn require_info_buffer(dev_info: &DeviceMemory<i32>) -> Result<()> {
    if dev_info.is_empty() {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_info_buffer_len(dev_info: &DeviceMemory<i32>, required: usize) -> Result<()> {
    if dev_info.len() < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}
