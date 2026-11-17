#![allow(
    clippy::too_many_arguments,
    reason = "GEMM operations take each matrix and scalar separately"
)]

use singe_cuda::view::{DeviceInputView, DeviceOutput, DeviceOutputView, DeviceRange, DeviceRepr};

use crate::{
    blas::{Element, level3},
    context::Context,
    error::{Error, Result},
    scalar::Scalar,
    types::Operation,
    utility::to_i64,
};

/// The shape of `C = alpha * op(A) * op(B) + beta * C`, with `op(A)` `m` x `k`, `op(B)` `k` x `n`, and `C` `m` x `n`.
///
/// Matrices use cuBLAS column-major storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    m: usize,
    n: usize,
    k: usize,
    transpose_a: Operation,
    transpose_b: Operation,
}

impl GemmShape {
    /// Describes a product of untransposed inputs.
    #[must_use]
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            transpose_a: Operation::NonTranspose,
            transpose_b: Operation::NonTranspose,
        }
    }

    /// Sets the operation applied to `A` before multiplication.
    #[must_use]
    pub const fn with_transpose_a(mut self, transpose: Operation) -> Self {
        self.transpose_a = transpose;
        self
    }

    /// Sets the operation applied to `B` before multiplication.
    #[must_use]
    pub const fn with_transpose_b(mut self, transpose: Operation) -> Self {
        self.transpose_b = transpose;
        self
    }

    /// Returns the number of rows of `C`.
    #[must_use]
    pub const fn m(self) -> usize {
        self.m
    }

    /// Returns the number of columns of `C`.
    #[must_use]
    pub const fn n(self) -> usize {
        self.n
    }

    /// Returns the shared dimension of `op(A)` and `op(B)`.
    #[must_use]
    pub const fn k(self) -> usize {
        self.k
    }

    /// Returns the operation applied to `A`.
    #[must_use]
    pub const fn transpose_a(self) -> Operation {
        self.transpose_a
    }

    /// Returns the operation applied to `B`.
    #[must_use]
    pub const fn transpose_b(self) -> Operation {
        self.transpose_b
    }
}

/// A borrowed column-major input matrix.
#[derive(Debug, Clone, Copy)]
pub struct MatrixRef<'a, T: DeviceRepr> {
    /// Matrix storage.
    pub data: DeviceInputView<'a, T>,
    /// Distance between the starts of adjacent columns, in elements.
    pub leading_dimension: usize,
}

impl<'a, T: DeviceRepr> MatrixRef<'a, T> {
    /// Borrows `data` as a matrix with the given leading dimension.
    pub fn new(data: &'a (impl DeviceRange<T> + ?Sized), leading_dimension: usize) -> Self {
        Self {
            data: DeviceInputView::from_buffer(data),
            leading_dimension,
        }
    }
}

/// An exclusively borrowed column-major output matrix.
#[derive(Debug)]
pub struct MatrixMut<'a, T: DeviceRepr> {
    /// Matrix storage.
    pub data: DeviceOutputView<'a, T>,
    /// Distance between the starts of adjacent columns, in elements.
    pub leading_dimension: usize,
}

impl<'a, T: DeviceRepr> MatrixMut<'a, T> {
    /// Exclusively borrows `data` as a matrix with the given leading dimension.
    pub fn new(data: &'a mut (impl DeviceOutput<T> + ?Sized), leading_dimension: usize) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
            leading_dimension,
        }
    }
}

/// A borrowed batch of column-major input matrices `stride` elements apart.
///
/// A zero stride repeats one matrix for every product.
#[derive(Debug, Clone, Copy)]
pub struct StridedBatchedMatrixRef<'a, T: DeviceRepr> {
    /// Storage of the whole batch.
    pub data: DeviceInputView<'a, T>,
    /// Distance between the starts of adjacent columns, in elements.
    pub leading_dimension: usize,
    /// Distance between the starts of adjacent matrices, in elements.
    pub stride: usize,
}

impl<'a, T: DeviceRepr> StridedBatchedMatrixRef<'a, T> {
    /// Borrows `data` as a batch of matrices with the given layout.
    pub fn new(
        data: &'a (impl DeviceRange<T> + ?Sized),
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data: DeviceInputView::from_buffer(data),
            leading_dimension,
            stride,
        }
    }
}

/// An exclusively borrowed batch of column-major output matrices `stride` elements apart.
#[derive(Debug)]
pub struct StridedBatchedMatrixMut<'a, T: DeviceRepr> {
    /// Storage of the whole batch.
    pub data: DeviceOutputView<'a, T>,
    /// Distance between the starts of adjacent columns, in elements.
    pub leading_dimension: usize,
    /// Distance between the starts of adjacent matrices, in elements.
    pub stride: usize,
}

impl<'a, T: DeviceRepr> StridedBatchedMatrixMut<'a, T> {
    /// Exclusively borrows `data` as a batch of matrices with the given layout.
    pub fn new(
        data: &'a mut (impl DeviceOutput<T> + ?Sized),
        leading_dimension: usize,
        stride: usize,
    ) -> Self {
        Self {
            data: DeviceOutputView::from_output(data),
            leading_dimension,
            stride,
        }
    }
}

/// Enqueues `C = alpha * op(A) * op(B) + beta * C` on the handle's stream.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error for the reasons [`level3::gemm_async`] lists.
pub unsafe fn gemm_async<T: Element>(
    context: &Context,
    shape: GemmShape,
    alpha: Scalar<'_, T>,
    a: MatrixRef<'_, T>,
    b: MatrixRef<'_, T>,
    beta: Scalar<'_, T>,
    mut c: MatrixMut<'_, T>,
) -> Result<()> {
    // SAFETY: `level3::gemm_async` checks the shapes and contexts itself, and this caller takes on the raw execution contract it requires.
    unsafe {
        level3::gemm_async(
            context,
            shape.transpose_a(),
            shape.transpose_b(),
            shape.m(),
            shape.n(),
            shape.k(),
            alpha,
            &a.data,
            a.leading_dimension,
            &b.data,
            b.leading_dimension,
            beta,
            &mut c.data,
            c.leading_dimension,
        )
    }
}

/// Computes `C = alpha * op(A) * op(B) + beta * C` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_async`] lists.
pub fn gemm<T: Element>(
    context: &Context,
    shape: GemmShape,
    alpha: Scalar<'_, T>,
    a: MatrixRef<'_, T>,
    b: MatrixRef<'_, T>,
    beta: Scalar<'_, T>,
    c: MatrixMut<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed matrices are released.
    context.sync_scope(|| unsafe { gemm_async(context, shape, alpha, a, b, beta, c) })
}

/// Enqueues `batch_count` products of matrices taken `stride` elements apart from each batch.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a stride exceeds `i64`, or for the reasons [`level3::gemm_strided_batched_async`] lists.
pub unsafe fn gemm_strided_batched_async<T: Element>(
    context: &Context,
    shape: GemmShape,
    batch_count: usize,
    alpha: Scalar<'_, T>,
    a: StridedBatchedMatrixRef<'_, T>,
    b: StridedBatchedMatrixRef<'_, T>,
    beta: Scalar<'_, T>,
    mut c: StridedBatchedMatrixMut<'_, T>,
) -> Result<()> {
    let stride_a = to_i64(a.stride, "stride a")?;
    let stride_b = to_i64(b.stride, "stride b")?;
    let stride_c = to_i64(c.stride, "stride c")?;
    // SAFETY: `level3::gemm_strided_batched_async` checks the batch extents and contexts itself, and this caller takes on the raw execution contract it requires.
    unsafe {
        level3::gemm_strided_batched_async(
            context,
            shape.transpose_a(),
            shape.transpose_b(),
            shape.m(),
            shape.n(),
            shape.k(),
            alpha,
            &a.data,
            a.leading_dimension,
            stride_a,
            &b.data,
            b.leading_dimension,
            stride_b,
            beta,
            &mut c.data,
            c.leading_dimension,
            stride_c,
            batch_count,
        )
    }
}

/// Computes `batch_count` products of matrices taken `stride` elements apart from each batch and waits for them.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`gemm_strided_batched_async`] lists.
pub fn gemm_strided_batched<T: Element>(
    context: &Context,
    shape: GemmShape,
    batch_count: usize,
    alpha: Scalar<'_, T>,
    a: StridedBatchedMatrixRef<'_, T>,
    b: StridedBatchedMatrixRef<'_, T>,
    beta: Scalar<'_, T>,
    c: StridedBatchedMatrixMut<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed matrices are released.
    context.sync_scope(|| unsafe {
        gemm_strided_batched_async(context, shape, batch_count, alpha, a, b, beta, c)
    })
}

/// Computes the products of corresponding matrices of three batches and waits for them.
///
/// Each batch uses one leading dimension.
///
/// # Errors
///
/// Returns an error if a batch mixes leading dimensions, or for the reasons [`level3::gemm_batched`] lists.
pub fn gemm_batched<T: Element>(
    context: &Context,
    shape: GemmShape,
    alpha: Scalar<'_, T>,
    a: &[MatrixRef<'_, T>],
    b: &[MatrixRef<'_, T>],
    beta: Scalar<'_, T>,
    c: &mut [MatrixMut<'_, T>],
) -> Result<()> {
    let lda = uniform_leading_dimension(a.iter().map(|matrix| matrix.leading_dimension))?;
    let ldb = uniform_leading_dimension(b.iter().map(|matrix| matrix.leading_dimension))?;
    let ldc = uniform_leading_dimension(c.iter().map(|matrix| matrix.leading_dimension))?;
    let a = a.iter().map(|matrix| matrix.data).collect::<Vec<_>>();
    let b = b.iter().map(|matrix| matrix.data).collect::<Vec<_>>();
    let mut c = c
        .iter_mut()
        .map(|matrix| matrix.data.reborrow())
        .collect::<Vec<_>>();
    level3::gemm_batched(
        context,
        shape.transpose_a(),
        shape.transpose_b(),
        shape.m(),
        shape.n(),
        shape.k(),
        alpha,
        &a,
        lda,
        &b,
        ldb,
        beta,
        &mut c,
        ldc,
    )
}

// Returns the leading dimension every matrix of a batch shares, or one for an empty batch.
fn uniform_leading_dimension(mut values: impl Iterator<Item = usize>) -> Result<usize> {
    let Some(first) = values.next() else {
        return Ok(1);
    };
    if values.any(|value| value != first) {
        return Err(Error::InvalidLeadingDimension);
    }

    Ok(first)
}
