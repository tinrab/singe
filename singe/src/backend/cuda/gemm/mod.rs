//! Matrix multiplication through one cuBLAS strided batched GEMM, read directly from strided operands.
//!
//! Eager calls, the kernel selector, and compiled-graph lowering share [`GemmLayout`], so they agree on which operands cuBLAS reads without a copy.

mod scaled;

use singe_cublas::{
    blas::level3,
    context::Context as CublasContext,
    types::{ComputeType, GemmAlgorithm, Operation as CublasOperation},
};
use singe_cuda::{data_type::DataType as CublasDataType, view::DeviceRepr};

pub use self::scaled::{ScaledMatmulPointers, enqueue_scaled_matmul};
use crate::{
    backend::{
        cuda::{
            device_views::{tensor_input_storage_view, tensor_output_view},
            tensor_region::{CudaTensorRegion, CudaTensorSource},
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::{ComputePrecision, NumericalPolicy},
    layout::Layout,
    shape::Shape,
};

/// One operand of a [`GemmLayout`] as cuBLAS reads it: a column-major matrix, possibly transposed, with batch instances `batch_stride` elements apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmMatrix {
    pub operation: CublasOperation,
    pub leading_dimension: usize,
    pub batch_stride: i64,
}

/// `lhs @ rhs` as one cuBLAS strided batched GEMM that reads both operands in place and writes a contiguous output.
///
/// Row-major `C = A·B` is column-major `Cᵀ = Bᵀ·Aᵀ`, so cuBLAS multiplies `rhs` by `lhs`.
/// When `rhs` is the same matrix for every batch instance, as a weight is, the batch folds into the rows of a single GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmLayout {
    /// Output rows of each batch instance, cuBLAS's `n`.
    pub rows: usize,
    /// Output columns, cuBLAS's `m`.
    pub columns: usize,
    /// The contraction dimension, cuBLAS's `k`.
    pub inner: usize,
    pub batch_count: usize,
    pub lhs: GemmMatrix,
    pub rhs: GemmMatrix,
    pub output_shape: Shape,
}

impl GemmLayout {
    /// Describes `lhs @ rhs` as one strided batched GEMM, or returns `None` when the operands need a copy first or the product has no work.
    ///
    /// Each matrix must have a unit stride along its rows or its columns, and each operand's batch dimensions must collapse to a single stride, which may be zero for a broadcast operand.
    #[must_use]
    pub fn create(lhs: &Layout, rhs: &Layout) -> Option<Self> {
        let problem = spec::matmul_problem(lhs.shape(), rhs.shape(), "cuda gemm").ok()?;
        if problem.is_zero_work() {
            return None;
        }

        let rows = usize::try_from(problem.lhs_rows).ok()?;
        let inner = usize::try_from(problem.inner).ok()?;
        let columns = usize::try_from(problem.rhs_cols).ok()?;
        let batch = &problem.batch_dimensions;
        let (lhs_matrix, lhs_batch) = promoted_strides(lhs, true, batch)?;
        let (rhs_matrix, rhs_batch) = promoted_strides(rhs, false, batch)?;
        let (lhs_operation, lhs_leading) = matrix(rows, inner, lhs_matrix)?;
        let (rhs_operation, rhs_leading) = matrix(inner, columns, rhs_matrix)?;
        let rhs_broadcast = rhs_batch.iter().all(|&stride| stride == 0);
        if rhs_broadcast && lhs_operation == CublasOperation::NonTranspose {
            let mut dimensions = batch.clone();
            dimensions.push(problem.lhs_rows);
            let mut strides = lhs_batch.clone();
            strides.push(lhs_matrix[0]);
            if let Some((folded_rows, row_stride)) = collapse(&dimensions, &strides) {
                let leading_dimension = if folded_rows == 1 {
                    lhs_leading
                } else {
                    usize::try_from(row_stride).ok()?
                };
                if leading_dimension >= inner.max(1) {
                    return Some(Self {
                        rows: folded_rows,
                        columns,
                        inner,
                        batch_count: 1,
                        lhs: GemmMatrix {
                            operation: CublasOperation::NonTranspose,
                            leading_dimension,
                            batch_stride: 0,
                        },
                        rhs: GemmMatrix {
                            operation: rhs_operation,
                            leading_dimension: rhs_leading,
                            batch_stride: 0,
                        },
                        output_shape: problem.output_shape,
                    });
                }
            }
        }

        let (batch_count, lhs_stride) = collapse(batch, &lhs_batch)?;
        let (_, rhs_stride) = collapse(batch, &rhs_batch)?;
        Some(Self {
            rows,
            columns,
            inner,
            batch_count,
            lhs: GemmMatrix {
                operation: lhs_operation,
                leading_dimension: lhs_leading,
                batch_stride: lhs_stride,
            },
            rhs: GemmMatrix {
                operation: rhs_operation,
                leading_dimension: rhs_leading,
                batch_stride: rhs_stride,
            },
            output_shape: problem.output_shape,
        })
    }

    /// Enqueues the GEMM on `cublas`'s stream, accumulating in `F32`, and writes `output`'s contiguous elements.
    ///
    /// The caller orders the operands' storage on that stream until the work completes, as eager execution scopes and compiled graphs do.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not `F16`, `BF16`, or `F32`, the operands do not have the layouts this GEMM was planned for, or cuBLAS rejects the call.
    pub fn launch(
        &self,
        cublas: &CublasContext,
        lhs: &(impl CudaTensorSource + ?Sized),
        rhs: &(impl CudaTensorSource + ?Sized),
        output: CudaTensorRegion<'_>,
    ) -> Result<()> {
        let data_type = lhs.data_type();
        match data_type {
            DataType::F32 => {
                self.launch_typed::<f32>(cublas, CublasDataType::F32, lhs, rhs, output)
            }
            DataType::F16 => {
                self.launch_typed::<u16>(cublas, CublasDataType::F16, lhs, rhs, output)
            }
            DataType::Bf16 => {
                self.launch_typed::<u16>(cublas, CublasDataType::Bf16, lhs, rhs, output)
            }
            _ => Err(Error::unsupported_data_type("cuda gemm")),
        }
    }

    fn launch_typed<T: DeviceRepr>(
        &self,
        cublas: &CublasContext,
        cublas_type: CublasDataType,
        lhs: &(impl CudaTensorSource + ?Sized),
        rhs: &(impl CudaTensorSource + ?Sized),
        output: CudaTensorRegion<'_>,
    ) -> Result<()> {
        let data_type = lhs.data_type();
        if Self::create(lhs.layout(), rhs.layout()).as_ref() != Some(self) {
            return Err(Error::unsupported_operation(
                "cuda gemm operand layouts differ from the planned layouts",
            ));
        }

        let lhs_view = tensor_input_storage_view::<T>(lhs, data_type, "cuda gemm lhs")?;
        let rhs_view = tensor_input_storage_view::<T>(rhs, data_type, "cuda gemm rhs")?;
        let mut output_view = tensor_output_view::<T>(output, data_type, "cuda gemm output")?;
        let output_stride = self
            .rows
            .checked_mul(self.columns)
            .and_then(|stride| i64::try_from(stride).ok())
            .ok_or_else(|| Error::out_of_range("cuda gemm output batch stride"))?;
        // SAFETY: the views hold initialized values of `cublas_type`, `F32` compute reads `f32` scalars, and the caller orders the storage on the cuBLAS stream.
        unsafe {
            level3::gemm_strided_batched_ex_views(
                cublas,
                self.rhs.operation,
                self.lhs.operation,
                self.columns,
                self.rows,
                self.inner,
                &1.0_f32,
                &rhs_view,
                cublas_type,
                self.rhs.leading_dimension,
                self.rhs.batch_stride,
                &lhs_view,
                cublas_type,
                self.lhs.leading_dimension,
                self.lhs.batch_stride,
                &0.0_f32,
                &mut output_view,
                cublas_type,
                self.columns,
                output_stride,
                self.batch_count,
                ComputeType::F32,
                GemmAlgorithm::Default,
            )?;
        }
        Ok(())
    }
}

/// Returns the strides of `layout`'s last two dimensions after vector promotion, and its strides along `batch`, zero where it broadcasts.
fn promoted_strides(layout: &Layout, lhs: bool, batch: &[i64]) -> Option<([i64; 2], Vec<i64>)> {
    let dimensions = layout.dimensions();
    let strides = layout.strides();
    if strides.iter().any(|&stride| stride < 0) {
        return None;
    }

    let (matrix, batch_strides) = match (dimensions.len(), lhs) {
        (0, _) => return None,
        (1, true) => ([0, strides[0]], &[][..]),
        (1, false) => ([strides[0], 0], &[][..]),
        (rank, _) => ([strides[rank - 2], strides[rank - 1]], &strides[..rank - 2]),
    };
    let batch_dimensions = &dimensions[..dimensions.len().saturating_sub(2)];
    let leading = batch.len().checked_sub(batch_dimensions.len())?;
    let mut aligned = vec![0; batch.len()];
    for (index, (&dimension, &stride)) in batch_dimensions.iter().zip(batch_strides).enumerate() {
        if dimension != 1 {
            aligned[leading + index] = stride;
        }
    }
    Some((matrix, aligned))
}

/// Returns how cuBLAS reads a `rows × columns` row-major matrix with `strides`: untransposed with its row stride as the leading dimension, or transposed with its column stride.
fn matrix(rows: usize, columns: usize, strides: [i64; 2]) -> Option<(CublasOperation, usize)> {
    let row_stride = usize::try_from(strides[0]).ok()?;
    let column_stride = usize::try_from(strides[1]).ok()?;
    if columns == 1 || column_stride == 1 {
        let leading = if rows == 1 { columns } else { row_stride };
        if leading >= columns.max(1) {
            return Some((CublasOperation::NonTranspose, leading));
        }
    }

    if rows == 1 || row_stride == 1 {
        let leading = if columns == 1 { rows } else { column_stride };
        if leading >= rows.max(1) {
            return Some((CublasOperation::Transpose, leading));
        }
    }
    None
}

/// Collapses dimensions with `strides` into one count and stride, or returns `None` when they do not form a single arithmetic sequence.
///
/// Dimensions of size one are skipped, and all-zero strides collapse to a zero stride.
fn collapse(dimensions: &[i64], strides: &[i64]) -> Option<(usize, i64)> {
    let mut count = 1_usize;
    let mut inner: Option<(i64, i64)> = None;
    for (&dimension, &stride) in dimensions.iter().zip(strides).rev() {
        if dimension == 1 {
            continue;
        }

        if let Some((inner_dimension, inner_stride)) = inner
            && inner_stride.checked_mul(inner_dimension)? != stride
        {
            return None;
        }

        count = count.checked_mul(usize::try_from(dimension).ok()?)?;
        inner = Some((dimension, stride));
    }

    let stride = strides
        .iter()
        .zip(dimensions)
        .rev()
        .find(|&(_, &dimension)| dimension != 1)
        .map_or(0, |(&stride, _)| stride);
    Some((count, stride))
}

// Whether cuBLASLt can run a matmul whose leading dimensions and batch strides are `strides` elements of `element_bytes` each.
// Its zero-workspace algorithms found none for rows narrower than 16 bytes, such as `F32` widths of 2 or 3, so those stay with the next library or generated kernels.
#[must_use]
pub fn cublaslt_takes_strides(strides: &[i64], element_bytes: usize) -> bool {
    strides.iter().all(|stride| {
        usize::try_from(stride.unsigned_abs())
            .ok()
            .and_then(|stride| stride.checked_mul(element_bytes))
            .is_some_and(|bytes| bytes % 16 == 0)
    })
}

// Whether cuBLASLt can add a bias vector inside a matmul whose result has this type, under the current numerical policy.
// Its zero-workspace algorithms found none for a full-precision `F32` bias epilogue, such as GPT-2's `[1, 64]` by `[64, 64]`, while the same matmul with its addend read as `C` ran, so those add the bias another way.
#[must_use]
pub fn cublaslt_takes_bias(data_type: DataType) -> bool {
    match data_type {
        DataType::F16 | DataType::Bf16 => true,
        DataType::F32 => NumericalPolicy::current().precision() != ComputePrecision::Full,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
