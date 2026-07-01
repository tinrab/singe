//! GEMM, batched GEMM, ragged batched GEMM, and matvec variants.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len, ensure_len_at_least, ensure_rank3_reach},
};

trait FiniteScale {
    fn is_finite_scale(self) -> bool;
}

impl FiniteScale for f32 {
    fn is_finite_scale(self) -> bool {
        self.is_finite()
    }
}

impl FiniteScale for f64 {
    fn is_finite_scale(self) -> bool {
        self.is_finite()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub lhs_row_stride: usize,
    pub rhs_row_stride: usize,
    pub output_row_stride: usize,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BmmConfig {
    pub batch: usize,
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub lhs_batch_stride: usize,
    pub lhs_row_stride: usize,
    pub rhs_batch_stride: usize,
    pub rhs_row_stride: usize,
    pub output_batch_stride: usize,
    pub output_row_stride: usize,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RaggedBmmConfig {
    /// Number of independent right-hand-side batches. `row_indptr` must have
    /// `batch + 1` entries.
    pub batch: usize,
    /// Total logical lhs/output rows. `row_indptr[batch]` must equal this
    /// value.
    pub total_rows: usize,
    /// Maximum rows in any batch segment:
    /// `row_indptr[b + 1] - row_indptr[b] <= max_rows`.
    pub max_rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub lhs_row_stride: usize,
    pub rhs_batch_stride: usize,
    pub rhs_row_stride: usize,
    pub output_row_stride: usize,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupGemmConfig {
    /// Number of GEMM descriptors in each metadata array.
    pub group_count: usize,
    /// Upper bounds used for launch shape selection. Every `rows[g]`,
    /// `columns[g]`, and `reductions[g]` entry must be nonnegative and no
    /// larger than the matching maximum.
    pub max_rows: usize,
    pub max_columns: usize,
    pub max_reduction: usize,
    /// Whether each RHS matrix is physically `[columns, reduction]` instead of
    /// `[reduction, columns]`.
    pub transpose_rhs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedGemmConfig {
    /// Number of sorted rows processed and output rows produced.
    pub total_tokens: usize,
    /// Number of expert segments. `m_sizes` must have this many entries and
    /// sum to `total_tokens`.
    pub expert_count: usize,
    pub columns: usize,
    pub reduction: usize,
    /// Number of physical input rows. When `permute_input` is set,
    /// `gather_indices[row] / top_k` must be less than this value.
    pub input_rows: usize,
    pub input_row_stride: usize,
    pub weight_expert_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
    /// Read input rows through `gather_indices[row] / top_k`.
    pub permute_input: bool,
    /// Write output rows through `gather_indices[row]`.
    pub permute_output: bool,
    /// Divisor used to map gathered top-k rows back to source input rows.
    pub top_k: usize,
}

#[cfg(feature = "dtype-f8")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RaggedBlockScaledBmmConfig {
    /// Number of independent RHS batches. Uses the same `row_indptr` contract
    /// as [`RaggedBmmConfig`].
    pub batch: usize,
    /// Total logical lhs/output rows. `row_indptr[batch]` must equal this
    /// value.
    pub total_rows: usize,
    /// Maximum rows in any batch segment.
    pub max_rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub scale_block: usize,
    pub rhs_batch_stride: usize,
    pub rhs_row_stride: usize,
    pub lhs_scale_row_stride: usize,
    pub rhs_scale_batch_stride: usize,
    pub rhs_scale_row_stride: usize,
    pub output_row_stride: usize,
}

impl MatmulConfig {
    pub fn row_major(rows: usize, columns: usize, reduction: usize) -> Self {
        Self {
            rows,
            columns,
            reduction,
            lhs_row_stride: reduction,
            rhs_row_stride: columns,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: false,
        }
    }

    pub fn row_major_transposed_lhs(rows: usize, columns: usize, reduction: usize) -> Self {
        Self {
            rows,
            columns,
            reduction,
            lhs_row_stride: rows,
            rhs_row_stride: columns,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: false,
        }
    }

    pub fn row_major_transposed_rhs(rows: usize, columns: usize, reduction: usize) -> Self {
        Self {
            rows,
            columns,
            reduction,
            lhs_row_stride: reduction,
            rhs_row_stride: reduction,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: true,
        }
    }

    pub fn row_major_transposed_inputs(rows: usize, columns: usize, reduction: usize) -> Self {
        Self {
            rows,
            columns,
            reduction,
            lhs_row_stride: rows,
            rhs_row_stride: reduction,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: true,
        }
    }
}

impl BmmConfig {
    pub fn row_major(batch: usize, rows: usize, columns: usize, reduction: usize) -> Result<Self> {
        let lhs_batch_stride = checked_element_count(rows, reduction)?;
        let rhs_batch_stride = checked_element_count(reduction, columns)?;
        let output_batch_stride = checked_element_count(rows, columns)?;
        Ok(Self {
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            lhs_row_stride: reduction,
            rhs_batch_stride,
            rhs_row_stride: columns,
            output_batch_stride,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: false,
        })
    }

    pub fn row_major_transposed_lhs(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let lhs_batch_stride = checked_element_count(reduction, rows)?;
        let rhs_batch_stride = checked_element_count(reduction, columns)?;
        let output_batch_stride = checked_element_count(rows, columns)?;
        Ok(Self {
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            lhs_row_stride: rows,
            rhs_batch_stride,
            rhs_row_stride: columns,
            output_batch_stride,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: false,
        })
    }

    pub fn row_major_transposed_rhs(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let lhs_batch_stride = checked_element_count(rows, reduction)?;
        let rhs_batch_stride = checked_element_count(columns, reduction)?;
        let output_batch_stride = checked_element_count(rows, columns)?;
        Ok(Self {
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            lhs_row_stride: reduction,
            rhs_batch_stride,
            rhs_row_stride: reduction,
            output_batch_stride,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: true,
        })
    }

    pub fn row_major_transposed_inputs(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let lhs_batch_stride = checked_element_count(reduction, rows)?;
        let rhs_batch_stride = checked_element_count(columns, reduction)?;
        let output_batch_stride = checked_element_count(rows, columns)?;
        Ok(Self {
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            lhs_row_stride: rows,
            rhs_batch_stride,
            rhs_row_stride: reduction,
            output_batch_stride,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: true,
        })
    }
}

impl RaggedBmmConfig {
    pub fn row_major(
        batch: usize,
        total_rows: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let rhs_batch_stride = checked_element_count(reduction, columns)?;
        Ok(Self {
            batch,
            total_rows,
            max_rows,
            columns,
            reduction,
            lhs_row_stride: reduction,
            rhs_batch_stride,
            rhs_row_stride: columns,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: false,
        })
    }

    pub fn row_major_transposed_lhs(
        batch: usize,
        total_rows: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let rhs_batch_stride = checked_element_count(reduction, columns)?;
        Ok(Self {
            batch,
            total_rows,
            max_rows,
            columns,
            reduction,
            lhs_row_stride: total_rows,
            rhs_batch_stride,
            rhs_row_stride: columns,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: false,
        })
    }

    pub fn row_major_transposed_rhs(
        batch: usize,
        total_rows: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let rhs_batch_stride = checked_element_count(columns, reduction)?;
        Ok(Self {
            batch,
            total_rows,
            max_rows,
            columns,
            reduction,
            lhs_row_stride: reduction,
            rhs_batch_stride,
            rhs_row_stride: reduction,
            output_row_stride: columns,
            transpose_lhs: false,
            transpose_rhs: true,
        })
    }

    pub fn row_major_transposed_inputs(
        batch: usize,
        total_rows: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
    ) -> Result<Self> {
        let rhs_batch_stride = checked_element_count(columns, reduction)?;
        Ok(Self {
            batch,
            total_rows,
            max_rows,
            columns,
            reduction,
            lhs_row_stride: total_rows,
            rhs_batch_stride,
            rhs_row_stride: reduction,
            output_row_stride: columns,
            transpose_lhs: true,
            transpose_rhs: true,
        })
    }
}

impl GroupGemmConfig {
    pub fn create(
        group_count: usize,
        max_rows: usize,
        max_columns: usize,
        max_reduction: usize,
    ) -> Self {
        Self {
            group_count,
            max_rows,
            max_columns,
            max_reduction,
            transpose_rhs: false,
        }
    }

    pub fn transposed_rhs(
        group_count: usize,
        max_rows: usize,
        max_columns: usize,
        max_reduction: usize,
    ) -> Self {
        Self {
            group_count,
            max_rows,
            max_columns,
            max_reduction,
            transpose_rhs: true,
        }
    }
}

impl GroupedGemmConfig {
    pub fn contiguous(
        total_tokens: usize,
        expert_count: usize,
        columns: usize,
        reduction: usize,
    ) -> Self {
        Self {
            total_tokens,
            expert_count,
            columns,
            reduction,
            input_rows: total_tokens,
            input_row_stride: reduction,
            weight_expert_stride: columns.saturating_mul(reduction),
            weight_row_stride: reduction,
            output_row_stride: columns,
            permute_input: false,
            permute_output: false,
            top_k: 1,
        }
    }

    pub fn permuted_input(
        total_tokens: usize,
        input_rows: usize,
        expert_count: usize,
        columns: usize,
        reduction: usize,
        top_k: usize,
    ) -> Self {
        Self {
            input_rows,
            permute_input: true,
            top_k,
            ..Self::contiguous(total_tokens, expert_count, columns, reduction)
        }
    }

    pub fn permuted_output(
        total_tokens: usize,
        expert_count: usize,
        columns: usize,
        reduction: usize,
    ) -> Self {
        Self {
            permute_output: true,
            ..Self::contiguous(total_tokens, expert_count, columns, reduction)
        }
    }
}

#[cfg(feature = "dtype-f8")]
impl RaggedBlockScaledBmmConfig {
    pub fn row_major(
        batch: usize,
        total_rows: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
        scale_block: usize,
    ) -> Result<Self> {
        let k_scale_tiles = checked_div_ceil(reduction, scale_block)?;
        let n_scale_tiles = checked_div_ceil(columns, scale_block)?;
        let rhs_batch_stride = checked_element_count(columns, reduction)?;
        let rhs_scale_batch_stride = checked_element_count(n_scale_tiles, k_scale_tiles)?;
        Ok(Self {
            batch,
            total_rows,
            max_rows,
            columns,
            reduction,
            scale_block,
            rhs_batch_stride,
            rhs_row_stride: reduction,
            lhs_scale_row_stride: k_scale_tiles,
            rhs_scale_batch_stride,
            rhs_scale_row_stride: k_scale_tiles,
            output_row_stride: columns,
        })
    }
}

pub fn matmul_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn matmul_mma_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn matmul_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    lhs: &impl DeviceSlice<f64>,
    rhs: &impl DeviceSlice<f64>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_f64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matvec_transposed_rhs_f16_f32_1024(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_matvec_transposed_rhs_1024(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matvec_transposed_rhs_f16_f32_1024(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matvec_transposed_rhs_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_matvec_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matvec_transposed_rhs_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_8x32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_8x32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_16x32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_16x64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_2(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_2(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_4(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_4(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_transposed_rhs_f16_f32_tile_32x16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_transposed_rhs_f16_f32_tile_32x16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_mma_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn matmul_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn matmul_mma_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: MatmulConfig,
) -> Result<()> {
    validate_matmul(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_matmul(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_mma_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn matmul_alpha_beta_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: MatmulConfig,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    validate_matmul_alpha_beta(out.len(), lhs.len(), rhs.len(), config, alpha, beta)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_alpha_beta_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
        alpha,
        beta,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn matmul_alpha_beta_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: MatmulConfig,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    validate_matmul_alpha_beta(out.len(), lhs.len(), rhs.len(), config, alpha, beta)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_alpha_beta_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
        alpha,
        beta,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn matmul_alpha_beta_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: MatmulConfig,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    validate_matmul_alpha_beta(out.len(), lhs.len(), rhs.len(), config, alpha, beta)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_alpha_beta_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
        alpha,
        beta,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn matmul_alpha_beta_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    lhs: &impl DeviceSlice<f64>,
    rhs: &impl DeviceSlice<f64>,
    config: MatmulConfig,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    validate_matmul_alpha_beta(out.len(), lhs.len(), rhs.len(), config, alpha, beta)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::matmul_alpha_beta_f64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
        alpha,
        beta,
    )
}

pub fn bmm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_transposed_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_transposed_f32_tile_16x64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_f32_tile_16x64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_transposed_f32_tile_16x64_occupancy_2(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_f32_tile_16x64_occupancy_2(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_transposed_f32_tile_16x64_occupancy_4(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_f32_tile_16x64_occupancy_4(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn bmm_mma_transposed_f32_tile_32x16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_f32_tile_32x16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn bmm_mma_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn bmm_mma_transposed_rhs_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed_rhs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_rhs_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn bmm_mma_transposed_inputs_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed_inputs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_inputs_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn bmm_mma_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn bmm_mma_transposed_inputs_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    validate_compact_bmm_transposed_inputs(config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_mma_transposed_inputs_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn masked_bmm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    masked_rows: &impl DeviceSlice<i32>,
    config: BmmConfig,
) -> Result<()> {
    validate_masked_bmm(out.len(), lhs.len(), rhs.len(), masked_rows.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::masked_bmm_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(masked_rows),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn masked_bmm_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    masked_rows: &impl DeviceSlice<i32>,
    config: BmmConfig,
) -> Result<()> {
    validate_masked_bmm(out.len(), lhs.len(), rhs.len(), masked_rows.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::masked_bmm_f16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(masked_rows),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn masked_bmm_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    masked_rows: &impl DeviceSlice<i32>,
    config: BmmConfig,
) -> Result<()> {
    validate_masked_bmm(out.len(), lhs.len(), rhs.len(), masked_rows.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::masked_bmm_bf16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(masked_rows),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn ragged_bmm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBmmConfig,
) -> Result<()> {
    validate_ragged_bmm(out.len(), lhs.len(), rhs.len(), row_indptr.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_bmm_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn ragged_bmm_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBmmConfig,
) -> Result<()> {
    validate_ragged_bmm(out.len(), lhs.len(), rhs.len(), row_indptr.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_bmm_f16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn ragged_bmm_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBmmConfig,
) -> Result<()> {
    validate_ragged_bmm(out.len(), lhs.len(), rhs.len(), row_indptr.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_bmm_bf16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

pub fn group_gemm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f32>,
    rhs: &impl DeviceSlice<f32>,
    rows: &impl DeviceSlice<i32>,
    columns: &impl DeviceSlice<i32>,
    reductions: &impl DeviceSlice<i32>,
    lhs_offsets: &impl DeviceSlice<i32>,
    rhs_offsets: &impl DeviceSlice<i32>,
    output_offsets: &impl DeviceSlice<i32>,
    config: GroupGemmConfig,
) -> Result<()> {
    validate_group_gemm(
        rows.len(),
        columns.len(),
        reductions.len(),
        lhs_offsets.len(),
        rhs_offsets.len(),
        output_offsets.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::group_gemm_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(rows),
        input_pointer(columns),
        input_pointer(reductions),
        input_pointer(lhs_offsets),
        input_pointer(rhs_offsets),
        input_pointer(output_offsets),
        config.group_count,
        config.max_rows,
        config.max_columns,
        config.max_reduction,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn group_gemm_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    rows: &impl DeviceSlice<i32>,
    columns: &impl DeviceSlice<i32>,
    reductions: &impl DeviceSlice<i32>,
    lhs_offsets: &impl DeviceSlice<i32>,
    rhs_offsets: &impl DeviceSlice<i32>,
    output_offsets: &impl DeviceSlice<i32>,
    config: GroupGemmConfig,
) -> Result<()> {
    validate_group_gemm(
        rows.len(),
        columns.len(),
        reductions.len(),
        lhs_offsets.len(),
        rhs_offsets.len(),
        output_offsets.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::group_gemm_f16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(rows),
        input_pointer(columns),
        input_pointer(reductions),
        input_pointer(lhs_offsets),
        input_pointer(rhs_offsets),
        input_pointer(output_offsets),
        config.group_count,
        config.max_rows,
        config.max_columns,
        config.max_reduction,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]

pub fn group_gemm_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    rows: &impl DeviceSlice<i32>,
    columns: &impl DeviceSlice<i32>,
    reductions: &impl DeviceSlice<i32>,
    lhs_offsets: &impl DeviceSlice<i32>,
    rhs_offsets: &impl DeviceSlice<i32>,
    output_offsets: &impl DeviceSlice<i32>,
    config: GroupGemmConfig,
) -> Result<()> {
    validate_group_gemm(
        rows.len(),
        columns.len(),
        reductions.len(),
        lhs_offsets.len(),
        rhs_offsets.len(),
        output_offsets.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::group_gemm_bf16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(rows),
        input_pointer(columns),
        input_pointer(reductions),
        input_pointer(lhs_offsets),
        input_pointer(rhs_offsets),
        input_pointer(output_offsets),
        config.group_count,
        config.max_rows,
        config.max_columns,
        config.max_reduction,
        config.transpose_rhs,
    )
}

pub fn grouped_gemm_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weights: &impl DeviceSlice<f32>,
    m_sizes: &impl DeviceSlice<i32>,
    gather_indices: &impl DeviceSlice<i32>,
    config: GroupedGemmConfig,
) -> Result<()> {
    validate_grouped_gemm(
        out.len(),
        input.len(),
        weights.len(),
        m_sizes.len(),
        gather_indices.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::grouped_gemm_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weights),
        input_pointer(m_sizes),
        input_pointer(gather_indices),
        config.total_tokens,
        config.expert_count,
        config.columns,
        config.reduction,
        config.input_row_stride,
        config.weight_expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        config.permute_input,
        config.permute_output,
        config.top_k,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn grouped_gemm_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    weights: &impl DeviceSlice<f16>,
    m_sizes: &impl DeviceSlice<i32>,
    gather_indices: &impl DeviceSlice<i32>,
    config: GroupedGemmConfig,
) -> Result<()> {
    validate_grouped_gemm(
        out.len(),
        input.len(),
        weights.len(),
        m_sizes.len(),
        gather_indices.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::grouped_gemm_f16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weights),
        input_pointer(m_sizes),
        input_pointer(gather_indices),
        config.total_tokens,
        config.expert_count,
        config.columns,
        config.reduction,
        config.input_row_stride,
        config.weight_expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        config.permute_input,
        config.permute_output,
        config.top_k,
    )
}

#[cfg(feature = "dtype-bf16")]

pub fn grouped_gemm_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    input: &impl DeviceSlice<bf16>,
    weights: &impl DeviceSlice<bf16>,
    m_sizes: &impl DeviceSlice<i32>,
    gather_indices: &impl DeviceSlice<i32>,
    config: GroupedGemmConfig,
) -> Result<()> {
    validate_grouped_gemm(
        out.len(),
        input.len(),
        weights.len(),
        m_sizes.len(),
        gather_indices.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::grouped_gemm_bf16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weights),
        input_pointer(m_sizes),
        input_pointer(gather_indices),
        config.total_tokens,
        config.expert_count,
        config.columns,
        config.reduction,
        config.input_row_stride,
        config.weight_expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        config.permute_input,
        config.permute_output,
        config.top_k,
    )
}

#[cfg(feature = "dtype-f8")]

pub fn ragged_block_scaled_bmm_f8e4m3_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
    lhs_scale: &impl DeviceSlice<f32>,
    rhs_scale: &impl DeviceSlice<f32>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBlockScaledBmmConfig,
) -> Result<()> {
    validate_ragged_block_scaled_bmm(
        out.len(),
        lhs.len(),
        rhs.len(),
        lhs_scale.len(),
        rhs_scale.len(),
        row_indptr.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_block_scaled_bmm_f8e4m3_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(lhs_scale),
        input_pointer(rhs_scale),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.scale_block,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.lhs_scale_row_stride,
        config.rhs_scale_batch_stride,
        config.rhs_scale_row_stride,
        config.output_row_stride,
    )
}

#[cfg(all(feature = "dtype-f8", feature = "dtype-f16"))]

pub fn ragged_block_scaled_bmm_f8e4m3_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
    lhs_scale: &impl DeviceSlice<f32>,
    rhs_scale: &impl DeviceSlice<f32>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBlockScaledBmmConfig,
) -> Result<()> {
    validate_ragged_block_scaled_bmm(
        out.len(),
        lhs.len(),
        rhs.len(),
        lhs_scale.len(),
        rhs_scale.len(),
        row_indptr.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_block_scaled_bmm_f8e4m3_f16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(lhs_scale),
        input_pointer(rhs_scale),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.scale_block,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.lhs_scale_row_stride,
        config.rhs_scale_batch_stride,
        config.rhs_scale_row_stride,
        config.output_row_stride,
    )
}

#[cfg(all(feature = "dtype-f8", feature = "dtype-bf16"))]

pub fn ragged_block_scaled_bmm_f8e4m3_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
    lhs_scale: &impl DeviceSlice<f32>,
    rhs_scale: &impl DeviceSlice<f32>,
    row_indptr: &impl DeviceSlice<i32>,
    config: RaggedBlockScaledBmmConfig,
) -> Result<()> {
    validate_ragged_block_scaled_bmm(
        out.len(),
        lhs.len(),
        rhs.len(),
        lhs_scale.len(),
        rhs_scale.len(),
        row_indptr.len(),
        config,
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::ragged_block_scaled_bmm_f8e4m3_bf16(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        input_pointer(lhs_scale),
        input_pointer(rhs_scale),
        input_pointer(row_indptr),
        config.batch,
        config.max_rows,
        config.columns,
        config.reduction,
        config.scale_block,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.lhs_scale_row_stride,
        config.rhs_scale_batch_stride,
        config.rhs_scale_row_stride,
        config.output_row_stride,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn bmm_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    lhs: &impl DeviceSlice<f64>,
    rhs: &impl DeviceSlice<f64>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_f64(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn bmm_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<f16>,
    rhs: &impl DeviceSlice<f16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn bmm_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    lhs: &impl DeviceSlice<bf16>,
    rhs: &impl DeviceSlice<bf16>,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out.len(), lhs.len(), rhs.len(), config)?;
    let stream = borrowed_stream(stream)?;
    cutile::matmul::bmm_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        config.batch,
        config.rows,
        config.columns,
        config.reduction,
        config.lhs_batch_stride,
        config.lhs_row_stride,
        config.rhs_batch_stride,
        config.rhs_row_stride,
        config.output_batch_stride,
        config.output_row_stride,
        config.transpose_lhs,
        config.transpose_rhs,
    )
}

fn validate_matmul(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: MatmulConfig,
) -> Result<()> {
    let lhs_min_stride = if config.transpose_lhs {
        config.rows
    } else {
        config.reduction
    };
    let rhs_min_stride = if config.transpose_rhs {
        config.reduction
    } else {
        config.columns
    };
    if config.rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.lhs_row_stride < lhs_min_stride
        || config.rhs_row_stride < rhs_min_stride
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }
    let lhs_reach = if config.transpose_lhs {
        strided_matrix_reach(config.reduction, config.rows, config.lhs_row_stride)?
    } else {
        strided_matrix_reach(config.rows, config.reduction, config.lhs_row_stride)?
    };
    let rhs_reach = if config.transpose_rhs {
        strided_matrix_reach(config.columns, config.reduction, config.rhs_row_stride)?
    } else {
        strided_matrix_reach(config.reduction, config.columns, config.rhs_row_stride)?
    };
    ensure_len(lhs_len, lhs_reach)?;
    ensure_len(rhs_len, rhs_reach)?;
    ensure_len(
        out_len,
        strided_matrix_reach(config.rows, config.columns, config.output_row_stride)?,
    )?;
    Ok(())
}

fn validate_matmul_alpha_beta(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: MatmulConfig,
    alpha: impl FiniteScale,
    beta: impl FiniteScale,
) -> Result<()> {
    validate_matmul(out_len, lhs_len, rhs_len, config)?;
    if !alpha.is_finite_scale() || !beta.is_finite_scale() {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_compact_matmul(config: MatmulConfig) -> Result<()> {
    if config.transpose_lhs
        || config.transpose_rhs
        || config.lhs_row_stride != config.reduction
        || config.rhs_row_stride != config.columns
        || config.output_row_stride != config.columns
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_compact_matmul_transposed_rhs(config: MatmulConfig) -> Result<()> {
    if config.transpose_lhs
        || !config.transpose_rhs
        || config.lhs_row_stride != config.reduction
        || config.rhs_row_stride != config.reduction
        || config.output_row_stride != config.columns
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_bmm(out_len: usize, lhs_len: usize, rhs_len: usize, config: BmmConfig) -> Result<()> {
    let lhs_min_stride = if config.transpose_lhs {
        config.rows
    } else {
        config.reduction
    };
    let rhs_min_stride = if config.transpose_rhs {
        config.reduction
    } else {
        config.columns
    };
    if config.batch == 0
        || config.rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.lhs_row_stride < lhs_min_stride
        || config.rhs_row_stride < rhs_min_stride
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }
    let lhs_dims = if config.transpose_lhs {
        [config.batch, config.reduction, config.rows]
    } else {
        [config.batch, config.rows, config.reduction]
    };
    let rhs_dims = if config.transpose_rhs {
        [config.batch, config.columns, config.reduction]
    } else {
        [config.batch, config.reduction, config.columns]
    };
    ensure_rank3_reach(
        lhs_len,
        lhs_dims,
        [config.lhs_batch_stride, config.lhs_row_stride, 1usize],
    )?;
    ensure_rank3_reach(
        rhs_len,
        rhs_dims,
        [config.rhs_batch_stride, config.rhs_row_stride, 1usize],
    )?;
    ensure_rank3_reach(
        out_len,
        [config.batch, config.rows, config.columns],
        [config.output_batch_stride, config.output_row_stride, 1usize],
    )?;
    Ok(())
}

fn validate_matvec_transposed_rhs_1024(config: MatmulConfig) -> Result<()> {
    if config.rows != 1
        || config.reduction != 1024
        || config.lhs_row_stride != 1024
        || config.output_row_stride < config.columns
        || config.transpose_lhs
        || !config.transpose_rhs
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_matvec_transposed_rhs(config: MatmulConfig) -> Result<()> {
    if config.rows != 1
        || !matches!(config.reduction, 1024 | 2048 | 3072)
        || config.lhs_row_stride != config.reduction
        || config.output_row_stride < config.columns
        || config.transpose_lhs
        || !config.transpose_rhs
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_compact_bmm(config: BmmConfig) -> Result<()> {
    let lhs_matrix_len = checked_element_count(config.rows, config.reduction)?;
    let rhs_matrix_len = checked_element_count(config.reduction, config.columns)?;
    let output_matrix_len = checked_element_count(config.rows, config.columns)?;
    if config.transpose_lhs
        || config.transpose_rhs
        || config.lhs_batch_stride != lhs_matrix_len
        || config.lhs_row_stride != config.reduction
        || config.rhs_batch_stride != rhs_matrix_len
        || config.rhs_row_stride != config.columns
        || config.output_batch_stride != output_matrix_len
        || config.output_row_stride != config.columns
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_compact_bmm_transposed(config: BmmConfig) -> Result<()> {
    let lhs_rows = if config.transpose_lhs {
        config.reduction
    } else {
        config.rows
    };
    let lhs_columns = if config.transpose_lhs {
        config.rows
    } else {
        config.reduction
    };
    let rhs_rows = if config.transpose_rhs {
        config.columns
    } else {
        config.reduction
    };
    let rhs_columns = if config.transpose_rhs {
        config.reduction
    } else {
        config.columns
    };
    let lhs_matrix_len = checked_element_count(lhs_rows, lhs_columns)?;
    let rhs_matrix_len = checked_element_count(rhs_rows, rhs_columns)?;
    let output_matrix_len = checked_element_count(config.rows, config.columns)?;
    if config.lhs_batch_stride != lhs_matrix_len
        || config.lhs_row_stride != lhs_columns
        || config.rhs_batch_stride != rhs_matrix_len
        || config.rhs_row_stride != rhs_columns
        || config.output_batch_stride != output_matrix_len
        || config.output_row_stride != config.columns
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_compact_bmm_transposed_rhs(config: BmmConfig) -> Result<()> {
    if config.transpose_lhs || !config.transpose_rhs {
        return Err(Error::InvalidLength);
    }
    validate_compact_bmm_transposed(config)
}

fn validate_compact_bmm_transposed_inputs(config: BmmConfig) -> Result<()> {
    if !config.transpose_lhs || !config.transpose_rhs {
        return Err(Error::InvalidLength);
    }
    validate_compact_bmm_transposed(config)
}

fn validate_masked_bmm(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    masked_rows_len: usize,
    config: BmmConfig,
) -> Result<()> {
    validate_bmm(out_len, lhs_len, rhs_len, config)?;
    ensure_len(masked_rows_len, config.batch)?;
    Ok(())
}

fn validate_ragged_bmm(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    row_indptr_len: usize,
    config: RaggedBmmConfig,
) -> Result<()> {
    let lhs_min_stride = if config.transpose_lhs {
        config.total_rows
    } else {
        config.reduction
    };
    let rhs_min_stride = if config.transpose_rhs {
        config.reduction
    } else {
        config.columns
    };
    if config.batch == 0
        || config.total_rows == 0
        || config.max_rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.lhs_row_stride < lhs_min_stride
        || config.rhs_row_stride < rhs_min_stride
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }

    let lhs_reach = if config.transpose_lhs {
        strided_matrix_reach(config.reduction, config.total_rows, config.lhs_row_stride)?
    } else {
        strided_matrix_reach(config.total_rows, config.reduction, config.lhs_row_stride)?
    };
    let rhs_dims = if config.transpose_rhs {
        [config.batch, config.columns, config.reduction]
    } else {
        [config.batch, config.reduction, config.columns]
    };
    ensure_len(lhs_len, lhs_reach)?;
    ensure_rank3_reach(
        rhs_len,
        rhs_dims,
        [config.rhs_batch_stride, config.rhs_row_stride, 1usize],
    )?;
    ensure_len(
        out_len,
        strided_matrix_reach(config.total_rows, config.columns, config.output_row_stride)?,
    )?;
    ensure_len(row_indptr_len, config.batch + 1)?;
    Ok(())
}

/// Validates host-visible ragged row metadata before it is copied to the GPU.
///
/// The public CUDA launchers can validate only device-buffer lengths. Call this
/// helper when `row_indptr` is still host-visible, especially for debug checks
/// around metadata generated by custom routing code.
pub fn validate_ragged_row_indptr_host(
    row_indptr: &[i32],
    batch: usize,
    total_rows: usize,
    max_rows: usize,
) -> Result<()> {
    ensure_len(row_indptr.len(), batch + 1)?;
    if total_rows == 0 || max_rows == 0 || row_indptr[0] != 0 {
        return Err(Error::InvalidLength);
    }

    let mut previous = 0usize;
    for &entry in &row_indptr[1..] {
        let current = usize_from_metadata(entry)?;
        if current < previous || current > total_rows {
            return Err(Error::InvalidLength);
        }
        let rows = current - previous;
        if rows > max_rows {
            return Err(Error::InvalidLength);
        }
        previous = current;
    }
    if previous != total_rows {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

/// Validates ragged-BMM host metadata and the buffers it indexes.
///
/// `row_indptr` must be length `config.batch + 1`, start at zero, be
/// monotonically nondecreasing, end at `config.total_rows`, and every per-batch
/// segment must contain at most `config.max_rows` rows. `lhs` and `out` are
/// indexed by global ragged row, while `rhs` is indexed by dense batch.
pub fn validate_ragged_bmm_metadata_host(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    row_indptr: &[i32],
    config: RaggedBmmConfig,
) -> Result<()> {
    validate_ragged_bmm(out_len, lhs_len, rhs_len, row_indptr.len(), config)?;
    validate_ragged_row_indptr_host(row_indptr, config.batch, config.total_rows, config.max_rows)
}

fn validate_group_gemm(
    rows_len: usize,
    columns_len: usize,
    reductions_len: usize,
    lhs_offsets_len: usize,
    rhs_offsets_len: usize,
    output_offsets_len: usize,
    config: GroupGemmConfig,
) -> Result<()> {
    if config.group_count == 0
        || config.max_rows == 0
        || config.max_columns == 0
        || config.max_reduction == 0
    {
        return Err(Error::InvalidLength);
    }
    ensure_len(rows_len, config.group_count)?;
    ensure_len(columns_len, config.group_count)?;
    ensure_len(reductions_len, config.group_count)?;
    ensure_len(lhs_offsets_len, config.group_count)?;
    ensure_len(rhs_offsets_len, config.group_count)?;
    ensure_len(output_offsets_len, config.group_count)?;
    checked_element_count(config.max_rows, config.max_columns)?;
    checked_element_count(config.group_count, config.max_rows)?;
    checked_element_count(config.group_count, config.max_columns)?;
    Ok(())
}

/// Validates host-visible grouped-GEMM descriptors before launch.
///
/// Each metadata array must have `config.group_count` entries. Dimensions are
/// nonnegative and bounded by `config.max_*`. Offsets are nonnegative element
/// offsets into `lhs`, `rhs`, and `out`; this helper checks that every group
/// stays within the supplied buffer lengths.

pub fn validate_group_gemm_metadata_host(
    lhs_len: usize,
    rhs_len: usize,
    out_len: usize,
    rows: &[i32],
    columns: &[i32],
    reductions: &[i32],
    lhs_offsets: &[i32],
    rhs_offsets: &[i32],
    output_offsets: &[i32],
    config: GroupGemmConfig,
) -> Result<()> {
    validate_group_gemm(
        rows.len(),
        columns.len(),
        reductions.len(),
        lhs_offsets.len(),
        rhs_offsets.len(),
        output_offsets.len(),
        config,
    )?;
    for group in 0..config.group_count {
        let group_rows = bounded_metadata(rows[group], config.max_rows)?;
        let group_columns = bounded_metadata(columns[group], config.max_columns)?;
        let group_reduction = bounded_metadata(reductions[group], config.max_reduction)?;
        let lhs_base = usize_from_metadata(lhs_offsets[group])?;
        let rhs_base = usize_from_metadata(rhs_offsets[group])?;
        let output_base = usize_from_metadata(output_offsets[group])?;

        let lhs_reach =
            offset_matrix_reach(lhs_base, group_rows, group_reduction, group_reduction)?;
        let rhs_reach = if config.transpose_rhs {
            offset_matrix_reach(rhs_base, group_columns, group_reduction, group_reduction)?
        } else {
            offset_matrix_reach(rhs_base, group_reduction, group_columns, group_columns)?
        };
        let output_reach =
            offset_matrix_reach(output_base, group_rows, group_columns, group_columns)?;
        ensure_len_at_least(lhs_len, lhs_reach)?;
        ensure_len_at_least(rhs_len, rhs_reach)?;
        ensure_len_at_least(out_len, output_reach)?;
    }
    Ok(())
}

fn validate_grouped_gemm(
    out_len: usize,
    input_len: usize,
    weights_len: usize,
    m_sizes_len: usize,
    gather_indices_len: usize,
    config: GroupedGemmConfig,
) -> Result<()> {
    if config.total_tokens == 0
        || config.expert_count == 0
        || config.columns == 0
        || config.reduction == 0
        || config.input_rows == 0
        || config.input_row_stride < config.reduction
        || config.weight_row_stride < config.reduction
        || config.weight_expert_stride
            < strided_matrix_reach(config.columns, config.reduction, config.weight_row_stride)?
        || config.output_row_stride < config.columns
        || config.top_k == 0
        || (config.permute_input && config.permute_output)
    {
        return Err(Error::InvalidLength);
    }
    ensure_len(
        input_len,
        strided_matrix_reach(config.input_rows, config.reduction, config.input_row_stride)?,
    )?;
    let weight_matrix_reach =
        strided_matrix_reach(config.columns, config.reduction, config.weight_row_stride)?;
    let weights_reach = config
        .expert_count
        .checked_sub(1)
        .and_then(|prior_experts| prior_experts.checked_mul(config.weight_expert_stride))
        .and_then(|offset| offset.checked_add(weight_matrix_reach))
        .ok_or(Error::SizeOverflow)?;
    ensure_len(weights_len, weights_reach)?;
    ensure_len(
        out_len,
        strided_matrix_reach(
            config.total_tokens,
            config.columns,
            config.output_row_stride,
        )?,
    )?;
    ensure_len(m_sizes_len, config.expert_count)?;
    ensure_len(gather_indices_len, config.total_tokens)?;
    Ok(())
}

/// Validates host-visible grouped GEMM routing metadata before launch.
///
/// `m_sizes` must have `config.expert_count` nonnegative entries and sum exactly
/// to `config.total_tokens`. `gather_indices` must have `config.total_tokens`
/// entries. When input permutation is enabled, each gathered row divided by
/// `top_k` must be within `config.input_rows`; when output permutation is
/// enabled, each gathered row must be within `config.total_tokens`.
pub fn validate_grouped_gemm_metadata_host(
    out_len: usize,
    input_len: usize,
    weights_len: usize,
    m_sizes: &[i32],
    gather_indices: &[i32],
    config: GroupedGemmConfig,
) -> Result<()> {
    validate_grouped_gemm(
        out_len,
        input_len,
        weights_len,
        m_sizes.len(),
        gather_indices.len(),
        config,
    )?;

    let mut token_start = 0usize;
    for &m_size in m_sizes {
        let expert_tokens = usize_from_metadata(m_size)?;
        token_start = token_start
            .checked_add(expert_tokens)
            .ok_or(Error::SizeOverflow)?;
        if token_start > config.total_tokens {
            return Err(Error::InvalidLength);
        }
    }
    if token_start != config.total_tokens {
        return Err(Error::InvalidLength);
    }

    if config.permute_input {
        for &index in gather_indices {
            let input_row = usize_from_metadata(index)? / config.top_k;
            if input_row >= config.input_rows {
                return Err(Error::InvalidLength);
            }
        }
    } else if config.permute_output {
        for &index in gather_indices {
            let output_row = usize_from_metadata(index)?;
            if output_row >= config.total_tokens {
                return Err(Error::InvalidLength);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "dtype-f8")]
fn validate_ragged_block_scaled_bmm(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    lhs_scale_len: usize,
    rhs_scale_len: usize,
    row_indptr_len: usize,
    config: RaggedBlockScaledBmmConfig,
) -> Result<()> {
    if config.batch == 0
        || config.total_rows == 0
        || config.max_rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.scale_block == 0
        || config.rhs_row_stride < config.reduction
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }
    let k_scale_tiles = checked_div_ceil(config.reduction, config.scale_block)?;
    let n_scale_tiles = checked_div_ceil(config.columns, config.scale_block)?;
    if config.lhs_scale_row_stride < k_scale_tiles
        || config.rhs_scale_row_stride < k_scale_tiles
        || config.rhs_batch_stride < checked_element_count(config.columns, config.rhs_row_stride)?
        || config.rhs_scale_batch_stride
            < checked_element_count(n_scale_tiles, config.rhs_scale_row_stride)?
    {
        return Err(Error::InvalidLength);
    }

    ensure_len(
        lhs_len,
        strided_matrix_reach(config.total_rows, config.reduction, config.reduction)?,
    )?;
    ensure_rank3_reach(
        rhs_len,
        [config.batch, config.columns, config.reduction],
        [config.rhs_batch_stride, config.rhs_row_stride, 1usize],
    )?;
    ensure_len(
        lhs_scale_len,
        strided_matrix_reach(
            config.total_rows,
            k_scale_tiles,
            config.lhs_scale_row_stride,
        )?,
    )?;
    ensure_rank3_reach(
        rhs_scale_len,
        [config.batch, n_scale_tiles, k_scale_tiles],
        [
            config.rhs_scale_batch_stride,
            config.rhs_scale_row_stride,
            1usize,
        ],
    )?;
    ensure_len(
        out_len,
        strided_matrix_reach(config.total_rows, config.columns, config.output_row_stride)?,
    )?;
    ensure_len(row_indptr_len, config.batch + 1)?;
    Ok(())
}

/// Validates block-scaled ragged-BMM host metadata and the buffers it indexes.
#[cfg(feature = "dtype-f8")]
pub fn validate_ragged_block_scaled_bmm_metadata_host(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    lhs_scale_len: usize,
    rhs_scale_len: usize,
    row_indptr: &[i32],
    config: RaggedBlockScaledBmmConfig,
) -> Result<()> {
    validate_ragged_block_scaled_bmm(
        out_len,
        lhs_len,
        rhs_len,
        lhs_scale_len,
        rhs_scale_len,
        row_indptr.len(),
        config,
    )?;
    validate_ragged_row_indptr_host(row_indptr, config.batch, config.total_rows, config.max_rows)
}

#[cfg(feature = "dtype-f8")]
fn checked_div_ceil(value: usize, divisor: usize) -> Result<usize> {
    if divisor == 0 {
        return Err(Error::InvalidLength);
    }
    value
        .checked_add(divisor - 1)
        .map(|adjusted| adjusted / divisor)
        .ok_or(Error::SizeOverflow)
}

fn strided_matrix_reach(rows: usize, columns: usize, row_stride: usize) -> Result<usize> {
    let prior_rows = rows.checked_sub(1).ok_or(Error::InvalidLength)?;
    prior_rows
        .checked_mul(row_stride)
        .and_then(|offset| offset.checked_add(columns))
        .ok_or(Error::SizeOverflow)
}

fn usize_from_metadata(value: i32) -> Result<usize> {
    if value < 0 {
        return Err(Error::InvalidLength);
    }
    Ok(value as usize)
}

fn bounded_metadata(value: i32, maximum: usize) -> Result<usize> {
    let value = usize_from_metadata(value)?;
    if value > maximum {
        return Err(Error::InvalidLength);
    }
    Ok(value)
}

fn offset_matrix_reach(
    base: usize,
    rows: usize,
    columns: usize,
    row_stride: usize,
) -> Result<usize> {
    if rows == 0 || columns == 0 {
        return Ok(base);
    }
    let matrix_reach = strided_matrix_reach(rows, columns, row_stride)?;
    base.checked_add(matrix_reach).ok_or(Error::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_row_major_validation_accepts_exact_lengths() -> Result<()> {
        let config = MatmulConfig::row_major(3, 5, 7);
        validate_matmul(15, 21, 35, config)
    }

    #[test]
    fn matmul_transposed_lhs_validation_accepts_exact_lengths() -> Result<()> {
        let config = MatmulConfig::row_major_transposed_lhs(3, 5, 7);
        validate_matmul(15, 21, 35, config)
    }

    #[test]
    fn matmul_transposed_rhs_validation_accepts_exact_lengths() -> Result<()> {
        let config = MatmulConfig::row_major_transposed_rhs(3, 5, 7);
        validate_matmul(15, 21, 35, config)
    }

    #[test]
    fn matmul_transposed_inputs_validation_accepts_exact_lengths() -> Result<()> {
        let config = MatmulConfig::row_major_transposed_inputs(3, 5, 7);
        validate_matmul(15, 21, 35, config)
    }

    #[test]
    fn matmul_mma_validation_accepts_compact_row_major_only() -> Result<()> {
        let compact = MatmulConfig::row_major(3, 5, 7);
        validate_compact_matmul(compact)?;

        let transposed = MatmulConfig::row_major_transposed_rhs(3, 5, 7);
        assert!(matches!(
            validate_compact_matmul(transposed),
            Err(Error::InvalidLength)
        ));

        let mut padded = compact;
        padded.output_row_stride = 8;
        assert!(matches!(
            validate_compact_matmul(padded),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn bmm_mma_validation_accepts_compact_row_major_only() -> Result<()> {
        let compact = BmmConfig::row_major(2, 3, 5, 7)?;
        validate_compact_bmm(compact)?;

        let transposed = BmmConfig::row_major_transposed_lhs(2, 3, 5, 7)?;
        assert!(matches!(
            validate_compact_bmm(transposed),
            Err(Error::InvalidLength)
        ));

        let mut padded = compact;
        padded.output_batch_stride += 1;
        assert!(matches!(
            validate_compact_bmm(padded),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn bmm_mma_transposed_validation_accepts_compact_physical_layouts() -> Result<()> {
        validate_compact_bmm_transposed(BmmConfig::row_major(2, 3, 5, 7)?)?;
        validate_compact_bmm_transposed(BmmConfig::row_major_transposed_lhs(2, 3, 5, 7)?)?;
        validate_compact_bmm_transposed(BmmConfig::row_major_transposed_rhs(2, 3, 5, 7)?)?;
        validate_compact_bmm_transposed(BmmConfig::row_major_transposed_inputs(2, 3, 5, 7)?)?;

        let mut padded = BmmConfig::row_major_transposed_inputs(2, 3, 5, 7)?;
        padded.rhs_row_stride += 1;
        assert!(matches!(
            validate_compact_bmm_transposed(padded),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn matmul_validation_rejects_short_rhs() {
        let config = MatmulConfig::row_major(3, 5, 7);
        assert!(matches!(
            validate_matmul(15, 21, 34, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn matmul_transposed_rhs_validation_rejects_short_rhs() {
        let config = MatmulConfig::row_major_transposed_rhs(3, 5, 7);
        assert!(matches!(
            validate_matmul(15, 21, 34, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn matmul_alpha_beta_validation_rejects_non_finite_scale() {
        let config = MatmulConfig::row_major(3, 5, 7);
        assert!(matches!(
            validate_matmul_alpha_beta(15, 21, 35, config, f32::INFINITY, 0.0),
            Err(Error::InvalidLength)
        ));
        assert!(matches!(
            validate_matmul_alpha_beta(15, 21, 35, config, 1.0, f32::NAN),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(feature = "dtype-f64")]
    #[test]
    fn matmul_alpha_beta_f64_validation_rejects_non_finite_scale() {
        let config = MatmulConfig::row_major(3, 5, 7);
        assert!(matches!(
            validate_matmul_alpha_beta(15, 21, 35, config, f64::INFINITY, 0.0),
            Err(Error::InvalidLength)
        ));
        assert!(matches!(
            validate_matmul_alpha_beta(15, 21, 35, config, 1.0, f64::NAN),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn bmm_row_major_validation_accepts_exact_lengths() -> Result<()> {
        let config = BmmConfig::row_major(2, 3, 5, 7)?;
        validate_bmm(30, 42, 70, config)
    }

    #[test]
    fn bmm_transposed_lhs_validation_accepts_exact_lengths() -> Result<()> {
        let config = BmmConfig::row_major_transposed_lhs(2, 3, 5, 7)?;
        validate_bmm(30, 42, 70, config)
    }

    #[test]
    fn bmm_transposed_rhs_validation_accepts_exact_lengths() -> Result<()> {
        let config = BmmConfig::row_major_transposed_rhs(2, 3, 5, 7)?;
        validate_bmm(30, 42, 70, config)
    }

    #[test]
    fn bmm_transposed_inputs_validation_accepts_exact_lengths() -> Result<()> {
        let config = BmmConfig::row_major_transposed_inputs(2, 3, 5, 7)?;
        validate_bmm(30, 42, 70, config)
    }

    #[test]
    fn bmm_validation_accepts_padded_batch_strides() -> Result<()> {
        let config = BmmConfig {
            batch: 2,
            rows: 3,
            columns: 5,
            reduction: 7,
            lhs_batch_stride: 24,
            lhs_row_stride: 7,
            rhs_batch_stride: 40,
            rhs_row_stride: 6,
            output_batch_stride: 18,
            output_row_stride: 5,
            transpose_lhs: false,
            transpose_rhs: false,
        };
        validate_bmm(33, 45, 81, config)
    }

    #[test]
    fn bmm_transposed_rhs_validation_rejects_short_rhs() -> Result<()> {
        let config = BmmConfig::row_major_transposed_rhs(2, 3, 5, 7)?;
        assert!(matches!(
            validate_bmm(30, 42, 69, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn bmm_validation_rejects_short_output() -> Result<()> {
        let config = BmmConfig::row_major(2, 3, 5, 7)?;
        assert!(matches!(
            validate_bmm(29, 42, 70, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn masked_bmm_validation_accepts_exact_mask_len() -> Result<()> {
        let config = BmmConfig::row_major(2, 3, 5, 7)?;
        validate_masked_bmm(30, 42, 70, 2, config)
    }

    #[test]
    fn masked_bmm_validation_rejects_short_mask() -> Result<()> {
        let config = BmmConfig::row_major(2, 3, 5, 7)?;
        assert!(matches!(
            validate_masked_bmm(30, 42, 70, 1, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn ragged_bmm_validation_accepts_exact_lengths() -> Result<()> {
        let config = RaggedBmmConfig::row_major_transposed_rhs(3, 5, 4, 7, 11)?;
        validate_ragged_bmm(35, 55, 231, 4, config)
    }

    #[test]
    fn ragged_bmm_validation_rejects_short_indptr() -> Result<()> {
        let config = RaggedBmmConfig::row_major(3, 5, 4, 7, 11)?;
        assert!(matches!(
            validate_ragged_bmm(35, 55, 231, 3, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn ragged_bmm_transposed_lhs_validation_rejects_short_lhs() -> Result<()> {
        let config = RaggedBmmConfig::row_major_transposed_lhs(3, 5, 4, 7, 11)?;
        assert!(matches!(
            validate_ragged_bmm(35, 54, 231, 4, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn ragged_bmm_metadata_validation_accepts_valid_indptr() -> Result<()> {
        let config = RaggedBmmConfig::row_major_transposed_rhs(3, 5, 4, 7, 11)?;
        validate_ragged_bmm_metadata_host(35, 55, 231, &[0, 2, 2, 5], config)
    }

    #[test]
    fn ragged_bmm_metadata_validation_rejects_nonmonotonic_indptr() -> Result<()> {
        let config = RaggedBmmConfig::row_major(3, 5, 4, 7, 11)?;
        assert!(matches!(
            validate_ragged_bmm_metadata_host(35, 55, 231, &[0, 3, 2, 5], config),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn ragged_bmm_metadata_validation_rejects_large_segment() -> Result<()> {
        let config = RaggedBmmConfig::row_major(3, 5, 2, 7, 11)?;
        assert!(matches!(
            validate_ragged_bmm_metadata_host(35, 55, 231, &[0, 3, 4, 5], config),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn ragged_bmm_metadata_validation_rejects_wrong_final_row() -> Result<()> {
        let config = RaggedBmmConfig::row_major(3, 5, 4, 7, 11)?;
        assert!(matches!(
            validate_ragged_bmm_metadata_host(35, 55, 231, &[0, 2, 3, 4], config),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }

    #[test]
    fn group_gemm_validation_accepts_descriptor_lengths() -> Result<()> {
        let config = GroupGemmConfig::transposed_rhs(2, 4, 5, 7);
        validate_group_gemm(2, 2, 2, 2, 2, 2, config)
    }

    #[test]
    fn group_gemm_validation_rejects_short_descriptor() {
        let config = GroupGemmConfig::create(2, 4, 5, 7);
        assert!(matches!(
            validate_group_gemm(2, 2, 1, 2, 2, 2, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn group_gemm_validation_rejects_zero_max_shape() {
        let config = GroupGemmConfig::create(2, 4, 0, 7);
        assert!(matches!(
            validate_group_gemm(2, 2, 2, 2, 2, 2, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn group_gemm_metadata_validation_accepts_valid_descriptors() -> Result<()> {
        let config = GroupGemmConfig::create(2, 4, 5, 7);
        validate_group_gemm_metadata_host(
            13,
            22,
            8,
            &[2, 1],
            &[3, 2],
            &[4, 5],
            &[0, 8],
            &[0, 12],
            &[0, 6],
            config,
        )
    }

    #[test]
    fn group_gemm_metadata_validation_rejects_negative_dimension() {
        let config = GroupGemmConfig::create(1, 4, 5, 7);
        assert!(matches!(
            validate_group_gemm_metadata_host(
                8,
                12,
                6,
                &[-1],
                &[3],
                &[4],
                &[0],
                &[0],
                &[0],
                config,
            ),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn group_gemm_metadata_validation_rejects_dimension_above_max() {
        let config = GroupGemmConfig::create(1, 4, 5, 7);
        assert!(matches!(
            validate_group_gemm_metadata_host(
                20,
                30,
                20,
                &[5],
                &[3],
                &[4],
                &[0],
                &[0],
                &[0],
                config,
            ),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn group_gemm_metadata_validation_rejects_output_overreach() {
        let config = GroupGemmConfig::create(2, 4, 5, 7);
        assert!(matches!(
            validate_group_gemm_metadata_host(
                13,
                22,
                8,
                &[2, 1],
                &[3, 2],
                &[4, 5],
                &[0, 8],
                &[0, 12],
                &[0, 7],
                config,
            ),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn group_gemm_metadata_validation_rejects_negative_offset() {
        let config = GroupGemmConfig::transposed_rhs(1, 4, 5, 7);
        assert!(matches!(
            validate_group_gemm_metadata_host(
                8,
                12,
                6,
                &[2],
                &[3],
                &[4],
                &[0],
                &[-1],
                &[0],
                config,
            ),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn grouped_gemm_validation_accepts_exact_lengths() -> Result<()> {
        let config = GroupedGemmConfig::permuted_input(6, 3, 2, 5, 7, 2);
        validate_grouped_gemm(30, 21, 70, 2, 6, config)
    }

    #[test]
    fn grouped_gemm_validation_rejects_both_permutations() {
        let mut config = GroupedGemmConfig::contiguous(6, 2, 5, 7);
        config.permute_input = true;
        config.permute_output = true;
        assert!(matches!(
            validate_grouped_gemm(30, 42, 70, 2, 6, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn grouped_gemm_validation_rejects_short_weights() {
        let config = GroupedGemmConfig::contiguous(6, 2, 5, 7);
        assert!(matches!(
            validate_grouped_gemm(30, 42, 69, 2, 6, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn grouped_gemm_validation_rejects_zero_topk() {
        let mut config = GroupedGemmConfig::permuted_input(6, 3, 2, 5, 7, 2);
        config.top_k = 0;
        assert!(matches!(
            validate_grouped_gemm(30, 21, 70, 2, 6, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn grouped_gemm_metadata_validation_accepts_permuted_input() -> Result<()> {
        let config = GroupedGemmConfig::permuted_input(6, 3, 2, 5, 7, 2);
        validate_grouped_gemm_metadata_host(30, 21, 70, &[2, 4], &[0, 1, 2, 3, 4, 5], config)
    }

    #[test]
    fn grouped_gemm_metadata_validation_rejects_m_size_sum_mismatch() {
        let config = GroupedGemmConfig::contiguous(6, 2, 5, 7);
        assert!(matches!(
            validate_grouped_gemm_metadata_host(30, 42, 70, &[2, 3], &[0, 1, 2, 3, 4, 5], config,),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn grouped_gemm_metadata_validation_rejects_permuted_input_gather_range() {
        let config = GroupedGemmConfig::permuted_input(6, 3, 2, 5, 7, 2);
        assert!(matches!(
            validate_grouped_gemm_metadata_host(30, 21, 70, &[2, 4], &[0, 1, 2, 3, 4, 6], config,),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn grouped_gemm_metadata_validation_rejects_permuted_output_gather_range() {
        let config = GroupedGemmConfig::permuted_output(6, 2, 5, 7);
        assert!(matches!(
            validate_grouped_gemm_metadata_host(30, 42, 70, &[2, 4], &[0, 1, 2, 3, 4, 6], config,),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn ragged_block_scaled_bmm_validation_accepts_exact_lengths() -> Result<()> {
        let config = RaggedBlockScaledBmmConfig::row_major(2, 5, 3, 4, 5, 2)?;
        validate_ragged_block_scaled_bmm(20, 25, 40, 15, 12, 3, config)
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn ragged_block_scaled_bmm_validation_rejects_short_rhs_scale() -> Result<()> {
        let config = RaggedBlockScaledBmmConfig::row_major(2, 5, 3, 4, 5, 2)?;
        assert!(matches!(
            validate_ragged_block_scaled_bmm(20, 25, 40, 15, 11, 3, config),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn ragged_block_scaled_bmm_metadata_validation_rejects_bad_indptr() -> Result<()> {
        let config = RaggedBlockScaledBmmConfig::row_major(2, 5, 3, 4, 5, 2)?;
        assert!(matches!(
            validate_ragged_block_scaled_bmm_metadata_host(20, 25, 40, 15, 12, &[0, 4, 5], config,),
            Err(Error::InvalidLength)
        ));
        Ok(())
    }
}
