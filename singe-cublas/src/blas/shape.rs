// Operand extents shared by the BLAS operations.
// Matrices are column-major, and a matrix span covers only the elements its columns address, so the last column needs no padding.

use crate::{
    error::{Error, Result},
    types::{Operation, SideMode},
    utility::{checked_pitched_extent, validate_min_len},
};

// Returns the number of elements `n` strided values span.
pub(crate) fn vector_span(n: usize, inc: usize) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }

    if inc == 0 {
        return Err(Error::InvalidIncrement);
    }

    checked_pitched_extent(n, inc, 1, "vector span")
}

// Checks that a buffer of `len` elements holds `n` values `inc` apart.
pub(crate) fn ensure_vector(name: &'static str, len: usize, n: usize, inc: usize) -> Result<()> {
    let required = vector_span(n, inc)?;
    validate_min_len(required, len, name)
}

// Returns the number of elements a `rows` x `cols` matrix with leading dimension `ld` spans.
pub(crate) fn matrix_span(rows: usize, cols: usize, ld: usize) -> Result<usize> {
    checked_pitched_extent(cols, ld, rows, "matrix span")
}

// Returns the `ld * cols` elements cuBLASXt's host-side checks require for a matrix of `cols` columns with leading dimension `ld`.
#[cfg(feature = "xt")]
pub(crate) fn required_matrix_len(ld: usize, cols: usize) -> Result<usize> {
    ld.checked_mul(cols)
        .ok_or_else(|| Error::out_of_range("matrix length"))
}

// Checks the leading dimension of a `rows` x `cols` matrix and that a buffer of `len` elements holds it.
pub(crate) fn ensure_matrix(
    name: &'static str,
    len: usize,
    rows: usize,
    cols: usize,
    ld: usize,
) -> Result<()> {
    if ld < rows.max(1) {
        return Err(Error::InvalidLeadingDimension);
    }

    let required = matrix_span(rows, cols, ld)?;
    validate_min_len(required, len, name)
}

// Returns the number of elements a packed triangle of order `n` holds.
pub(crate) fn packed_len(n: usize) -> Result<usize> {
    n.checked_add(1)
        .and_then(|next| next.checked_mul(n))
        .map(|product| product / 2)
        .ok_or_else(|| Error::out_of_range("packed matrix length"))
}

// Checks that a buffer of `len` elements holds a packed triangle of order `n`.
pub(crate) fn ensure_packed(name: &'static str, len: usize, n: usize) -> Result<()> {
    let required = packed_len(n)?;
    validate_min_len(required, len, name)
}

// Returns the stored rows and columns of a matrix whose operated form is `rows` x `cols`.
pub(crate) fn stored(operation: Operation, rows: usize, cols: usize) -> (usize, usize) {
    if operation == Operation::NonTranspose {
        (rows, cols)
    } else {
        (cols, rows)
    }
}

// Returns the order of the square matrix that multiplies an `m` x `n` matrix from `side`.
pub(crate) fn side_order(side: SideMode, m: usize, n: usize) -> usize {
    match side {
        SideMode::Left => m,
        SideMode::Right => n,
    }
}

// Checks a batch of `batch_count` matrices of `span` elements, `stride` apart, in a buffer of `len` elements.
// Input matrices may overlap, so a zero stride repeats one matrix; output matrices may not.
pub(crate) fn ensure_strided(
    name: &'static str,
    len: usize,
    span: usize,
    stride: i64,
    batch_count: usize,
    output: bool,
) -> Result<()> {
    let stride = usize::try_from(stride).map_err(|_| Error::out_of_range("batch stride"))?;
    if output && batch_count > 1 && span > 0 && stride < span {
        return Err(Error::InvalidMatrixShape);
    }

    let required = if batch_count == 0 || span == 0 {
        0
    } else {
        (batch_count - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(span))
            .ok_or_else(|| Error::out_of_range("strided batch span"))?
    };
    validate_min_len(required, len, name)
}
