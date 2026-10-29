use std::ptr;

use singe_cuda::view::{DeviceOutputView, DeviceRepr};

use crate::{
    error::{Error, Result},
    layout::{MatrixLayout, MatrixOutput, StridedBatchedMatrixOutput, StridedMatrixLayout},
    svd::SvdLayout,
    types::EigenMode,
    utility::{to_i32, to_i64},
};

/// Allocation-free layouts for approximate strided batched SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StridedSvdLayout {
    /// Column-major input layout and matrix stride, measured in elements.
    pub input: StridedMatrixLayout,
    /// Element distance between consecutive matrices' singular-value output prefixes.
    pub singular_value_stride: usize,
    /// Optional column-major left singular-vector layout and batch stride in elements.
    pub left_vectors: Option<StridedMatrixLayout>,
    /// Optional column-major right singular-vector layout and batch stride in elements.
    pub right_vectors: Option<StridedMatrixLayout>,
}

impl StridedSvdLayout {
    /// Records the input batch layout and singular-value stride with no singular-vector output layouts.
    ///
    /// All distances are measured in elements. The solver workspace query validates dimensions and required outputs.
    #[must_use]
    pub const fn new(input: StridedMatrixLayout, singular_value_stride: usize) -> Self {
        Self {
            input,
            singular_value_stride,
            left_vectors: None,
            right_vectors: None,
        }
    }

    /// Sets the left singular-vector output batch layout and stride in element units.
    ///
    /// Records metadata without validating output shape or storage capacity.
    #[must_use]
    pub const fn with_left_vectors(mut self, layout: StridedMatrixLayout) -> Self {
        self.left_vectors = Some(layout);
        self
    }

    /// Sets the right singular-vector output batch layout and stride in element units.
    ///
    /// Records metadata without validating output shape or storage capacity.
    #[must_use]
    pub const fn with_right_vectors(mut self, layout: StridedMatrixLayout) -> Self {
        self.right_vectors = Some(layout);
        self
    }
}

pub fn jacobi_columns(m: usize, n: usize, economy: bool) -> (usize, usize) {
    if economy {
        (m.min(n), m.min(n))
    } else {
        (m, n)
    }
}

pub fn jacobi_layout(
    m: usize,
    n: usize,
    mode: EigenMode,
    economy: bool,
    layout: SvdLayout,
    batch_size: usize,
    batched: bool,
) -> Result<(i32, i32, i32)> {
    if batch_size == 0 || batched && (m > 32 || n > 32) {
        return Err(Error::InvalidMatrixShape);
    }
    MatrixLayout::new(layout.input_leading_dimension).validate(m, n)?;
    m.min(n)
        .checked_mul(batch_size)
        .ok_or(Error::InvalidVectorShape)?;
    let (left, right) = jacobi_columns(m, n, economy);
    let dimensions = |leading: Option<usize>, rows: usize, columns: usize| -> Result<i32> {
        if mode == EigenMode::NoVector && leading.is_none() {
            return to_i32(rows, "leading dimension");
        }
        let leading = leading.ok_or(Error::InvalidMatrixShape)?;
        let columns = columns
            .checked_mul(batch_size)
            .ok_or(Error::InvalidMatrixShape)?;
        MatrixLayout::new(leading).validate(rows, columns)?;
        to_i32(leading, "leading dimension")
    };
    Ok((
        to_i32(layout.input_leading_dimension, "lda")?,
        dimensions(layout.left_leading_dimension, m, left)?,
        dimensions(layout.right_leading_dimension, n, right)?,
    ))
}

pub fn output_parts<T: DeviceRepr>(
    matrix: Option<MatrixOutput<'_, T>>,
    rows: usize,
    columns: usize,
    batch_size: usize,
    mode: EigenMode,
) -> Result<(*mut T, i32)> {
    if mode == EigenMode::NoVector && matrix.is_none() {
        return Ok((ptr::null_mut(), to_i32(rows, "leading dimension")?));
    }
    let mut matrix = matrix.ok_or(Error::InvalidMatrixShape)?;
    let columns = columns
        .checked_mul(batch_size)
        .ok_or(Error::InvalidMatrixShape)?;
    require_matrix(rows, columns, matrix.leading_dimension, matrix.data.len())?;
    Ok((
        matrix.data.as_mut_ptr(),
        to_i32(matrix.leading_dimension, "leading dimension")?,
    ))
}

pub fn require_matrix(rows: usize, columns: usize, leading: usize, len: usize) -> Result<()> {
    MatrixLayout::new(leading).validate(rows, columns)?;
    let required = leading
        .checked_mul(columns)
        .ok_or(Error::InvalidMatrixShape)?;
    if len < required {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

fn strided_capacity(stride: usize, batch_size: usize) -> Result<usize> {
    if batch_size == 0 {
        return Err(Error::InvalidMatrixShape);
    }
    let capacity = stride
        .checked_mul(batch_size)
        .ok_or(Error::InvalidMatrixShape)?;
    // The native implementation limits every complete batched storage extent.
    to_i32(capacity, "batched storage extent")?;
    Ok(capacity)
}

fn strided_layout(
    layout: StridedMatrixLayout,
    rows: usize,
    columns: usize,
    batch_size: usize,
) -> Result<()> {
    MatrixLayout::new(layout.leading_dimension).validate(rows, columns)?;
    let footprint = layout
        .leading_dimension
        .checked_mul(columns)
        .ok_or(Error::InvalidMatrixShape)?;
    if layout.stride < footprint {
        return Err(Error::InvalidMatrixShape);
    }
    strided_capacity(layout.stride, batch_size)?;
    Ok(())
}

pub fn approximate_layout(
    rank: usize,
    m: usize,
    n: usize,
    mode: EigenMode,
    layout: StridedSvdLayout,
    batch_size: usize,
) -> Result<(i32, i32, i64, i32, i64)> {
    if rank == 0 || rank > n || m < n {
        return Err(Error::InvalidMatrixShape);
    }
    strided_layout(layout.input, m, n, batch_size)?;
    if layout.singular_value_stride < rank {
        return Err(Error::InvalidVectorShape);
    }
    strided_capacity(layout.singular_value_stride, batch_size)?;
    let vectors = |layout: Option<StridedMatrixLayout>, rows: usize| -> Result<(i32, i64)> {
        if mode == EigenMode::NoVector {
            let stride = rows.checked_mul(rank).ok_or(Error::InvalidMatrixShape)?;
            strided_capacity(stride, batch_size)?;
            return Ok((
                to_i32(rows, "leading dimension")?,
                to_i64(stride, "stride")?,
            ));
        }
        let layout = layout.ok_or(Error::InvalidMatrixShape)?;
        strided_layout(layout, rows, rank, batch_size)?;
        Ok((
            to_i32(layout.leading_dimension, "leading dimension")?,
            to_i64(layout.stride, "stride")?,
        ))
    };
    let (ldu, su) = vectors(layout.left_vectors, m)?;
    let (ldv, sv) = vectors(layout.right_vectors, n)?;
    Ok((
        to_i32(layout.input.leading_dimension, "lda")?,
        ldu,
        su,
        ldv,
        sv,
    ))
}

pub fn require_strided(len: usize, stride: usize, batch_size: usize) -> Result<()> {
    if len < strided_capacity(stride, batch_size)? {
        return Err(Error::InvalidMatrixShape);
    }
    Ok(())
}

pub fn strided_output_parts<T: DeviceRepr>(
    matrix: Option<StridedBatchedMatrixOutput<'_, T>>,
    rows: usize,
    rank: usize,
    batch_size: usize,
    mode: EigenMode,
) -> Result<(*mut T, i32, i64)> {
    if mode == EigenMode::NoVector {
        let stride = rows.checked_mul(rank).ok_or(Error::InvalidMatrixShape)?;
        strided_capacity(stride, batch_size)?;
        return Ok((
            ptr::null_mut(),
            to_i32(rows, "leading dimension")?,
            to_i64(stride, "stride")?,
        ));
    }
    let mut matrix = matrix.ok_or(Error::InvalidMatrixShape)?;
    strided_layout(matrix.layout(), rows, rank, batch_size)?;
    require_strided(matrix.data.len(), matrix.stride, batch_size)?;
    Ok((
        matrix.data.as_mut_ptr(),
        to_i32(matrix.leading_dimension, "leading dimension")?,
        to_i64(matrix.stride, "stride")?,
    ))
}

pub fn validate_output<T: DeviceRepr>(
    output: &DeviceOutputView<'_, T>,
    required: usize,
) -> Result<()> {
    if output.len() < required {
        return Err(Error::InvalidVectorShape);
    }
    Ok(())
}
