use std::{mem::size_of, ptr};

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
};

use crate::{
    error::{Error, Result},
    layout::{MatrixMut, MatrixRef},
    types::EigenRange,
    utility::to_i64,
};

use super::EigenSelection;

pub fn validate_xsyevd_buffers(
    n: usize,
    a_bytes: usize,
    lda: usize,
    a_type: DataType,
    w_bytes: usize,
    w_type: DataType,
) -> Result<()> {
    validate_x_matrix(n, n, a_bytes, lda, a_type)?;
    validate_x_vector(n, w_bytes, w_type)?;
    Ok(())
}

pub fn validate_xsyev_batched_buffers(
    n: usize,
    a_bytes: usize,
    lda: usize,
    a_type: DataType,
    w_bytes: usize,
    w_type: DataType,
    batch_count: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    validate_x_matrix(
        n,
        n.checked_mul(batch_count)
            .ok_or(Error::InvalidMatrixShape)?,
        a_bytes,
        lda,
        a_type,
    )?;
    validate_x_vector(
        n.checked_mul(batch_count)
            .ok_or(Error::InvalidVectorShape)?,
        w_bytes,
        w_type,
    )?;
    let problem_size = n
        .checked_mul(lda)
        .and_then(|value| value.checked_mul(batch_count))
        .ok_or(Error::InvalidMatrixShape)?;
    if problem_size > i32::MAX as usize {
        return Err(Error::OutOfRange {
            name: "batched problem size".into(),
        });
    }
    Ok(())
}

pub fn validate_syev_buffers(n: usize, a_len: usize, lda: usize, w_len: usize) -> Result<()> {
    validate_square_matrix(n, a_len, lda)?;
    if w_len < n {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_square_matrix(n: usize, len: usize, lda: usize) -> Result<()> {
    validate_matrix(n, n, len, lda)
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

pub fn validate_syevj_batched_buffers(
    n: usize,
    a_len: usize,
    lda: usize,
    w_len: usize,
    batch_count: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    let matrix_cols = n
        .checked_mul(batch_count)
        .ok_or(Error::InvalidMatrixShape)?;
    validate_matrix(n, matrix_cols, a_len, lda)?;
    let eigenvalues = n
        .checked_mul(batch_count)
        .ok_or(Error::InvalidVectorShape)?;
    if w_len < eigenvalues {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_sygvj_buffers(
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    w_len: usize,
) -> Result<()> {
    validate_square_matrix(n, a_len, lda)?;
    validate_square_matrix(n, b_len, ldb)?;
    if w_len < n {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_sygvd_buffers(
    n: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    w_len: usize,
) -> Result<()> {
    validate_square_matrix(n, a_len, lda)?;
    validate_square_matrix(n, b_len, ldb)?;
    if w_len < n {
        return Err(Error::InvalidVectorShape);
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
    let required = lda
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(data_type.size_of()))
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

pub fn validate_xsyevdx_value_type<T>(w_type: DataType) -> Result<()> {
    if size_of::<T>() != w_type.size_of() {
        return Err(Error::InvalidEigenRange);
    }
    Ok(())
}

pub type EigenSelectionParts<T> = (EigenRange, Option<(T, T)>, Option<(usize, usize)>);

pub fn validate_xsyevdx_range<T>(
    range: EigenRange,
    n: usize,
    value_range: Option<(T, T)>,
    index_range: Option<(usize, usize)>,
) -> Result<(T, T, usize, usize)>
where
    T: Default,
{
    match range {
        EigenRange::All => {
            if value_range.is_some() || index_range.is_some() {
                return Err(Error::InvalidEigenRange);
            }
            Ok((T::default(), T::default(), 0, 0))
        }
        EigenRange::Value => {
            if index_range.is_some() {
                return Err(Error::InvalidEigenRange);
            }
            let Some((vl, vu)) = value_range else {
                return Err(Error::InvalidEigenRange);
            };
            Ok((vl, vu, 0, 0))
        }
        EigenRange::Index => {
            if value_range.is_some() {
                return Err(Error::InvalidEigenRange);
            }
            let Some((il, iu)) = index_range else {
                return Err(Error::InvalidEigenRange);
            };
            if il == 0 || iu == 0 || il > iu || iu > n {
                return Err(Error::InvalidEigenRange);
            }
            Ok((T::default(), T::default(), il, iu))
        }
    }
}

pub fn selection_parts<T>(selection: EigenSelection<T>) -> EigenSelectionParts<T> {
    match selection {
        EigenSelection::All => (EigenRange::All, None, None),
        EigenSelection::ByValue { lower, upper } => (EigenRange::Value, Some((lower, upper)), None),
        EigenSelection::ByIndex { start, end } => (EigenRange::Index, None, Some((start, end))),
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

pub fn validate_xgeev_inputs<TV>(
    n: usize,
    a_bytes: usize,
    lda: usize,
    a_type: DataType,
    w_bytes: usize,
    w_type: DataType,
    right_vectors: Option<(&DeviceMemory<TV>, usize)>,
    vr_type: DataType,
) -> Result<()> {
    validate_x_matrix(n, n, a_bytes, lda, a_type)?;
    validate_xgeev_eigenvalues(n, w_bytes, a_type, w_type)?;
    if let Some((vr, ldvr)) = right_vectors {
        validate_x_matrix(n, n, vr.byte_len(), ldvr, vr_type)?;
    }
    Ok(())
}

pub fn validate_xgeev_eigenvalues(
    n: usize,
    w_bytes: usize,
    a_type: DataType,
    w_type: DataType,
) -> Result<()> {
    let expected_len = match (a_type, w_type) {
        (DataType::F32, DataType::F32) | (DataType::F64, DataType::F64) => {
            n.checked_mul(2).ok_or(Error::InvalidVectorShape)?
        }
        (DataType::F32, DataType::ComplexF32)
        | (DataType::F64, DataType::ComplexF64)
        | (DataType::ComplexF32, DataType::ComplexF32)
        | (DataType::ComplexF64, DataType::ComplexF64) => n,
        _ => return Err(Error::InvalidVectorShape),
    };
    validate_x_vector(expected_len, w_bytes, w_type)
}

pub fn optional_xgeev_matrix_ptr<T: DataTypeLike>(
    matrix: Option<(&DeviceMemory<T>, usize)>,
) -> Result<(*const T, i64)> {
    match matrix {
        Some((matrix, ld)) => Ok((matrix.as_ptr().cast(), to_i64(ld, "ldvr")?)),
        None => Ok((ptr::null(), 1)),
    }
}

pub fn optional_xgeev_matrix_mut_ptr<T: DataTypeLike>(
    matrix: Option<(&mut DeviceMemory<T>, usize)>,
) -> Result<(*mut T, i64)> {
    match matrix {
        Some((matrix, ld)) => Ok((matrix.as_mut_ptr().cast(), to_i64(ld, "ldvr")?)),
        None => Ok((ptr::null_mut(), 1)),
    }
}

pub fn require_info_buffer_len(dev_info: &DeviceMemory<i32>, required: usize) -> Result<()> {
    if dev_info.len() < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}
