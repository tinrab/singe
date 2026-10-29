#![expect(
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use singe_cuda::{data_type::DataType, view::DeviceRepr};

use crate::{
    error::{Error, Result},
    layout::{BatchedMatrixRef, BatchedVectorRef},
    types::{SideMode, StorevMode},
};

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

pub fn require_info_buffer(len: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_info_entries(len: usize, required: usize) -> Result<()> {
    if len < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_pivot_buffer(len: usize, required: usize) -> Result<()> {
    if len < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_pivot64_buffer(len: usize, required: usize) -> Result<()> {
    if len < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn require_tau_buffer(len: usize, required: usize) -> Result<()> {
    if len < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn qr_rows(side: SideMode, m: usize, n: usize) -> usize {
    match side {
        SideMode::Left => m,
        SideMode::Right => n,
    }
}

pub fn tridiagonal_order(side: SideMode, m: usize, n: usize) -> usize {
    match side {
        SideMode::Left => m,
        SideMode::Right => n,
    }
}

pub fn validate_bidiagonal_dims(m: usize, n: usize) -> Result<()> {
    if m == 0 || n == 0 || m < n {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn validate_bidiagonal_buffers(
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    d_len: usize,
    e_len: usize,
    tauq_len: usize,
    taup_len: usize,
) -> Result<()> {
    validate_bidiagonal_dims(m, n)?;
    validate_matrix(m, n, a_len, lda)?;
    if d_len < n || e_len < n || tauq_len < n || taup_len < n {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_orgbr_inputs(
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    tau_len: usize,
) -> Result<()> {
    validate_orgbr_shape(side, m, n, k)?;
    validate_matrix(m, n, a_len, lda)?;
    let reflectors = match side {
        SideMode::Left => m.min(k),
        SideMode::Right => n.min(k),
    };
    if tau_len < reflectors {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_orgbr_shape(side: SideMode, m: usize, n: usize, k: usize) -> Result<()> {
    if m == 0 || n == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    match side {
        SideMode::Left if m < n || n < m.min(k) => Err(Error::InvalidMatrixShape),
        SideMode::Right if n < m || m < n.min(k) => Err(Error::InvalidMatrixShape),
        _ => Ok(()),
    }
}

pub fn validate_sytrd_inputs(
    n: usize,
    a_len: usize,
    lda: usize,
    d_len: usize,
    e_len: usize,
    tau_len: usize,
) -> Result<()> {
    validate_square_matrix(n, a_len, lda)?;
    let reflectors = n.saturating_sub(1);
    if d_len < n || e_len < reflectors || tau_len < reflectors {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_orgtr_inputs(n: usize, a_len: usize, lda: usize, tau_len: usize) -> Result<()> {
    validate_square_matrix(n, a_len, lda)?;
    if tau_len < n.saturating_sub(1) {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_ormtr_inputs(
    side: SideMode,
    m: usize,
    n: usize,
    a_len: usize,
    lda: usize,
    tau_len: usize,
    c_len: usize,
    ldc: usize,
) -> Result<()> {
    let nq = tridiagonal_order(side, m, n);
    validate_square_matrix(nq, a_len, lda)?;
    validate_matrix(m, n, c_len, ldc)?;
    if tau_len < nq.saturating_sub(1) {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}

pub fn validate_batched_square_matrix_pointers<T: DeviceRepr>(
    n: usize,
    matrices: BatchedMatrixRef<'_, T>,
) -> Result<()> {
    if n == 0 || matrices.is_empty() {
        return Err(Error::InvalidMatrixShape);
    }
    if matrices.leading_dimension < n {
        return Err(Error::InvalidLeadingDimension);
    }
    Ok(())
}

pub fn validate_batched_vector_pointers<T: DeviceRepr>(
    n: usize,
    vectors: BatchedVectorRef<'_, T>,
) -> Result<()> {
    if n == 0 || vectors.is_empty() {
        return Err(Error::InvalidVectorShape);
    }
    if vectors.leading_dimension < n {
        return Err(Error::InvalidLeadingDimension);
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

pub fn validate_xlarft_inputs(
    n: usize,
    k: usize,
    storev: StorevMode,
    v_bytes: usize,
    ldv: usize,
    v_type: DataType,
    tau_bytes: usize,
    tau_type: DataType,
    t_bytes: usize,
    ldt: usize,
    t_type: DataType,
) -> Result<()> {
    if n == 0 || k == 0 || k > n {
        return Err(Error::InvalidMatrixShape);
    }
    if storev != StorevMode::Columnwise {
        return Err(Error::InvalidMatrixShape);
    }
    validate_x_matrix(n, k, v_bytes, ldv, v_type)?;
    validate_x_vector(k, tau_bytes, tau_type)?;
    validate_x_matrix(k, k, t_bytes, ldt, t_type)?;
    Ok(())
}
