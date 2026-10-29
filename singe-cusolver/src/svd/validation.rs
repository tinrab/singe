use singe_cuda::data_type::DataType;

use crate::error::{Error, Result};

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
