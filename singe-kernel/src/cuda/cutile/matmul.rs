use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};

use crate::{
    cuda::cutile::{
        DeviceOpExt, TileKernel,
        kernel::matmul as kernel_matmul,
        utility::{checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatmulLaunch {
    rows: i32,
    columns: i32,
    reduction: i32,
    lhs_row_stride: i32,
    rhs_row_stride: i32,
    output_row_stride: i32,
    transpose_lhs: i32,
    transpose_rhs: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatmulMmaLaunch {
    rows: i32,
    columns: i32,
    reduction: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BmmLaunch {
    batch: i32,
    rows: i32,
    columns: i32,
    reduction: i32,
    lhs_batch_stride: i32,
    lhs_row_stride: i32,
    rhs_batch_stride: i32,
    rhs_row_stride: i32,
    output_batch_stride: i32,
    output_row_stride: i32,
    transpose_lhs: i32,
    transpose_rhs: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BmmMmaLaunch {
    batch: i32,
    rows: i32,
    columns: i32,
    reduction: i32,
    transpose_lhs: bool,
    transpose_rhs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaggedBmmLaunch {
    batch: i32,
    max_rows: i32,
    columns: i32,
    reduction: i32,
    lhs_row_stride: i32,
    rhs_batch_stride: i32,
    rhs_row_stride: i32,
    output_row_stride: i32,
    transpose_lhs: i32,
    transpose_rhs: i32,
    virtual_output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GroupGemmLaunch {
    group_count: i32,
    max_rows: i32,
    max_columns: i32,
    max_reduction: i32,
    transpose_rhs: i32,
    virtual_output_len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GroupedGemmLaunch {
    columns: i32,
    reduction: i32,
    expert_count: i32,
    input_row_stride: i32,
    weight_expert_stride: i32,
    weight_row_stride: i32,
    output_row_stride: i32,
    permute_input: i32,
    permute_output: i32,
    top_k: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[cfg(feature = "dtype_f8")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaggedBlockScaledBmmLaunch {
    batch: i32,
    max_rows: i32,
    columns: i32,
    reduction: i32,
    scale_block: i32,
    rhs_batch_stride: i32,
    rhs_row_stride: i32,
    lhs_scale_row_stride: i32,
    rhs_scale_batch_stride: i32,
    rhs_scale_row_stride: i32,
    output_row_stride: i32,
    virtual_output_len: i32,
    grid: (u32, u32, u32),
}

impl MatmulLaunch {
    fn create(
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_row_stride: usize,
        rhs_row_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        let lhs_min_stride = if transpose_lhs { rows } else { reduction };
        let rhs_min_stride = if transpose_rhs { reduction } else { columns };
        if rows == 0
            || columns == 0
            || reduction == 0
            || lhs_row_stride < lhs_min_stride
            || rhs_row_stride < rhs_min_stride
            || output_row_stride < columns
        {
            return Err(Error::InvalidLength);
        }

        let output_len = checked_element_count(rows, columns)?;
        Ok(Self {
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            lhs_row_stride: checked_i32_value(lhs_row_stride)?,
            rhs_row_stride: checked_i32_value(rhs_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            transpose_lhs: i32::from(transpose_lhs),
            transpose_rhs: i32::from(transpose_rhs),
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl MatmulMmaLaunch {
    fn create(
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_row_stride: usize,
        rhs_row_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        if rows == 0
            || columns == 0
            || reduction == 0
            || lhs_row_stride != reduction
            || rhs_row_stride != columns
            || output_row_stride != columns
            || transpose_lhs
            || transpose_rhs
        {
            return Err(Error::InvalidLength);
        }

        Ok(Self {
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
        })
    }

    #[cfg(feature = "dtype_f16")]
    fn create_transposed_rhs(
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_row_stride: usize,
        rhs_row_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        if rows == 0
            || columns == 0
            || reduction == 0
            || lhs_row_stride != reduction
            || rhs_row_stride != reduction
            || output_row_stride != columns
            || transpose_lhs
            || !transpose_rhs
        {
            return Err(Error::InvalidLength);
        }

        Ok(Self {
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
        })
    }
}

impl BmmLaunch {
    fn create(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_batch_stride: usize,
        lhs_row_stride: usize,
        rhs_batch_stride: usize,
        rhs_row_stride: usize,
        output_batch_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        let lhs_min_stride = if transpose_lhs { rows } else { reduction };
        let rhs_min_stride = if transpose_rhs { reduction } else { columns };
        if batch == 0
            || rows == 0
            || columns == 0
            || reduction == 0
            || lhs_row_stride < lhs_min_stride
            || rhs_row_stride < rhs_min_stride
            || output_row_stride < columns
        {
            return Err(Error::InvalidLength);
        }

        let matrix_len = checked_element_count(rows, columns)?;
        let lhs_rows = if transpose_lhs { reduction } else { rows };
        let rhs_rows = if transpose_rhs { columns } else { reduction };
        if lhs_batch_stride < checked_element_count(lhs_rows, lhs_row_stride)?
            || rhs_batch_stride < checked_element_count(rhs_rows, rhs_row_stride)?
            || output_batch_stride < checked_element_count(rows, output_row_stride)?
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(batch, matrix_len)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            lhs_batch_stride: checked_i32_value(lhs_batch_stride)?,
            lhs_row_stride: checked_i32_value(lhs_row_stride)?,
            rhs_batch_stride: checked_i32_value(rhs_batch_stride)?,
            rhs_row_stride: checked_i32_value(rhs_row_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            transpose_lhs: i32::from(transpose_lhs),
            transpose_rhs: i32::from(transpose_rhs),
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

impl BmmMmaLaunch {
    fn create(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_batch_stride: usize,
        lhs_row_stride: usize,
        rhs_batch_stride: usize,
        rhs_row_stride: usize,
        output_batch_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        let lhs_matrix_len = checked_element_count(rows, reduction)?;
        let rhs_matrix_len = checked_element_count(reduction, columns)?;
        let output_matrix_len = checked_element_count(rows, columns)?;
        if batch == 0
            || rows == 0
            || columns == 0
            || reduction == 0
            || lhs_batch_stride != lhs_matrix_len
            || lhs_row_stride != reduction
            || rhs_batch_stride != rhs_matrix_len
            || rhs_row_stride != columns
            || output_batch_stride != output_matrix_len
            || output_row_stride != columns
            || transpose_lhs
            || transpose_rhs
        {
            return Err(Error::InvalidLength);
        }

        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            transpose_lhs,
            transpose_rhs,
        })
    }

    fn create_transposed(
        batch: usize,
        rows: usize,
        columns: usize,
        reduction: usize,
        lhs_batch_stride: usize,
        lhs_row_stride: usize,
        rhs_batch_stride: usize,
        rhs_row_stride: usize,
        output_batch_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        let lhs_rows = if transpose_lhs { reduction } else { rows };
        let lhs_columns = if transpose_lhs { rows } else { reduction };
        let rhs_rows = if transpose_rhs { columns } else { reduction };
        let rhs_columns = if transpose_rhs { reduction } else { columns };
        let lhs_matrix_len = checked_element_count(lhs_rows, lhs_columns)?;
        let rhs_matrix_len = checked_element_count(rhs_rows, rhs_columns)?;
        let output_matrix_len = checked_element_count(rows, columns)?;
        if batch == 0
            || rows == 0
            || columns == 0
            || reduction == 0
            || lhs_batch_stride != lhs_matrix_len
            || lhs_row_stride != lhs_columns
            || rhs_batch_stride != rhs_matrix_len
            || rhs_row_stride != rhs_columns
            || output_batch_stride != output_matrix_len
            || output_row_stride != columns
        {
            return Err(Error::InvalidLength);
        }

        Ok(Self {
            batch: checked_i32_value(batch)?,
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            transpose_lhs,
            transpose_rhs,
        })
    }
}

impl RaggedBmmLaunch {
    fn create(
        batch: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
        lhs_row_stride: usize,
        rhs_batch_stride: usize,
        rhs_row_stride: usize,
        output_row_stride: usize,
        transpose_lhs: bool,
        transpose_rhs: bool,
    ) -> Result<Self> {
        let lhs_min_stride = if transpose_lhs { max_rows } else { reduction };
        let rhs_min_stride = if transpose_rhs { reduction } else { columns };
        if batch == 0
            || max_rows == 0
            || columns == 0
            || reduction == 0
            || lhs_row_stride < lhs_min_stride
            || rhs_row_stride < rhs_min_stride
            || output_row_stride < columns
        {
            return Err(Error::InvalidLength);
        }

        let rhs_rows = if transpose_rhs { columns } else { reduction };
        if rhs_batch_stride < checked_element_count(rhs_rows, rhs_row_stride)? {
            return Err(Error::InvalidLength);
        }

        let virtual_matrix_len = checked_element_count(max_rows, columns)?;
        let virtual_output_len = checked_element_count(batch, virtual_matrix_len)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            max_rows: checked_i32_value(max_rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            lhs_row_stride: checked_i32_value(lhs_row_stride)?,
            rhs_batch_stride: checked_i32_value(rhs_batch_stride)?,
            rhs_row_stride: checked_i32_value(rhs_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            transpose_lhs: i32::from(transpose_lhs),
            transpose_rhs: i32::from(transpose_rhs),
            virtual_output_len: checked_i32_value(virtual_output_len)?,
            grid: raw_vector_grid(virtual_output_len)?,
        })
    }
}

impl GroupGemmLaunch {
    fn create(
        group_count: usize,
        max_rows: usize,
        max_columns: usize,
        max_reduction: usize,
        transpose_rhs: bool,
    ) -> Result<Self> {
        if group_count == 0 || max_rows == 0 || max_columns == 0 || max_reduction == 0 {
            return Err(Error::InvalidLength);
        }

        let virtual_matrix_len = checked_element_count(max_rows, max_columns)?;
        let virtual_output_len = checked_element_count(group_count, virtual_matrix_len)?;
        Ok(Self {
            group_count: checked_i32_value(group_count)?,
            max_rows: checked_i32_value(max_rows)?,
            max_columns: checked_i32_value(max_columns)?,
            max_reduction: checked_i32_value(max_reduction)?,
            transpose_rhs: i32::from(transpose_rhs),
            virtual_output_len: checked_i32_value(virtual_output_len)?,
            grid: raw_vector_grid(virtual_output_len)?,
        })
    }
}

impl GroupedGemmLaunch {
    fn create(
        total_tokens: usize,
        expert_count: usize,
        columns: usize,
        reduction: usize,
        input_row_stride: usize,
        weight_expert_stride: usize,
        weight_row_stride: usize,
        output_row_stride: usize,
        permute_input: bool,
        permute_output: bool,
        top_k: usize,
    ) -> Result<Self> {
        if total_tokens == 0
            || expert_count == 0
            || columns == 0
            || reduction == 0
            || input_row_stride < reduction
            || weight_row_stride < reduction
            || weight_expert_stride < checked_element_count(columns, weight_row_stride)?
            || output_row_stride < columns
            || top_k == 0
            || (permute_input && permute_output)
        {
            return Err(Error::InvalidLength);
        }

        let output_len = checked_element_count(total_tokens, columns)?;
        Ok(Self {
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            expert_count: checked_i32_value(expert_count)?,
            input_row_stride: checked_i32_value(input_row_stride)?,
            weight_expert_stride: checked_i32_value(weight_expert_stride)?,
            weight_row_stride: checked_i32_value(weight_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            permute_input: i32::from(permute_input),
            permute_output: i32::from(permute_output),
            top_k: checked_i32_value(top_k)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

#[cfg(feature = "dtype_f8")]
impl RaggedBlockScaledBmmLaunch {
    fn create(
        batch: usize,
        max_rows: usize,
        columns: usize,
        reduction: usize,
        scale_block: usize,
        rhs_batch_stride: usize,
        rhs_row_stride: usize,
        lhs_scale_row_stride: usize,
        rhs_scale_batch_stride: usize,
        rhs_scale_row_stride: usize,
        output_row_stride: usize,
    ) -> Result<Self> {
        if batch == 0
            || max_rows == 0
            || columns == 0
            || reduction == 0
            || scale_block == 0
            || rhs_row_stride < reduction
            || output_row_stride < columns
        {
            return Err(Error::InvalidLength);
        }

        let k_scale_tiles = reduction.div_ceil(scale_block);
        let n_scale_tiles = columns.div_ceil(scale_block);
        if lhs_scale_row_stride < k_scale_tiles
            || rhs_scale_row_stride < k_scale_tiles
            || rhs_batch_stride < checked_element_count(columns, rhs_row_stride)?
            || rhs_scale_batch_stride < checked_element_count(n_scale_tiles, rhs_scale_row_stride)?
        {
            return Err(Error::InvalidLength);
        }

        let virtual_matrix_len = checked_element_count(max_rows, columns)?;
        let virtual_output_len = checked_element_count(batch, virtual_matrix_len)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            max_rows: checked_i32_value(max_rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            scale_block: checked_i32_value(scale_block)?,
            rhs_batch_stride: checked_i32_value(rhs_batch_stride)?,
            rhs_row_stride: checked_i32_value(rhs_row_stride)?,
            lhs_scale_row_stride: checked_i32_value(lhs_scale_row_stride)?,
            rhs_scale_batch_stride: checked_i32_value(rhs_scale_batch_stride)?,
            rhs_scale_row_stride: checked_i32_value(rhs_scale_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            virtual_output_len: checked_i32_value(virtual_output_len)?,
            grid: raw_vector_grid(virtual_output_len)?,
        })
    }
}

#[cfg(feature = "dtype_f64")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_f64(
    stream: &Arc<Stream>,
    out: DevicePointer<f64>,
    lhs: DevicePointer<f64>,
    rhs: DevicePointer<f64>,
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    output_batch_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

    let params = BmmLaunch::create(
        batch,
        rows,
        columns,
        reduction,
        lhs_batch_stride,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_batch_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_matmul::bmm_f64(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_batch_stride,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_batch_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype_f16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    output_batch_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

    let params = BmmLaunch::create(
        batch,
        rows,
        columns,
        reduction,
        lhs_batch_stride,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_batch_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_matmul::bmm_f16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_batch_stride,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_batch_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype_bf16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    batch: usize,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_batch_stride: usize,
    lhs_row_stride: usize,
    rhs_batch_stride: usize,
    rhs_row_stride: usize,
    output_batch_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

    let params = BmmLaunch::create(
        batch,
        rows,
        columns,
        reduction,
        lhs_batch_stride,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_batch_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_matmul::bmm_bf16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_batch_stride,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_batch_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

mod batched;
mod dense;
mod grouped;

pub use batched::*;
pub use dense::*;
pub use grouped::*;

#[cfg(test)]
mod tests;
