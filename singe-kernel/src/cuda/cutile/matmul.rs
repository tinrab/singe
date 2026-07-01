use std::sync::Arc;

#[cfg(feature = "dtype-f8")]
use cutile::core::f8e4m3fn;
#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer,
    cuda_core::Stream,
    tile_kernel::{CompileOptions, TileKernel},
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
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

#[cfg(feature = "dtype-f8")]
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

#[cfg(feature = "dtype-f8")]
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

pub fn matmul_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
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

pub fn matmul_mma_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulMmaLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    let out = TensorAdapter::contiguous_2d(
        out,
        usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?,
        usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?,
    )?
    .partition([16, 16])?;
    let lhs = TensorAdapter::contiguous_2d(
        lhs,
        usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?,
        usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?,
    )?;
    let rhs = TensorAdapter::contiguous_2d(
        rhs,
        usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?,
        usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?,
    )?;
    kernel_matmul::matmul_mma_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f64")]

pub fn matmul_f64(
    stream: &Arc<Stream>,
    out: DevicePointer<f64>,
    lhs: DevicePointer<f64>,
    rhs: DevicePointer<f64>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_f64(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
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

#[cfg(feature = "dtype-f16")]

pub fn matmul_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_f16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
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

#[cfg(feature = "dtype-f16")]

pub fn matvec_transposed_rhs_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    if rows != 1
        || columns == 0
        || !matches!(reduction, 1024 | 2048 | 3072)
        || lhs_row_stride != reduction
        || rhs_row_stride < reduction
        || output_row_stride < columns
        || transpose_lhs
        || !transpose_rhs
    {
        return Err(Error::InvalidLength);
    }
    let columns = checked_i32_value(columns)?;
    let rhs_row_stride = checked_i32_value(rhs_row_stride)?;
    let grid = (
        u32::try_from(columns).map_err(|_| Error::SizeOverflow)?,
        1,
        1,
    );
    match reduction {
        1024 => unsafe {
            kernel_matmul::matvec_transposed_rhs_f16_f32_1024(
                out,
                lhs,
                rhs,
                columns,
                rhs_row_stride,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        2048 => unsafe {
            kernel_matmul::matvec_transposed_rhs_f16_f32_2048(
                out,
                lhs,
                rhs,
                columns,
                rhs_row_stride,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        3072 => unsafe {
            kernel_matmul::matvec_transposed_rhs_f16_f32_3072(
                out,
                lhs,
                rhs,
                columns,
                rhs_row_stride,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => return Err(Error::InvalidLength),
    };
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn matvec_transposed_rhs_f16_f32_1024(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matvec_transposed_rhs_f16_f32(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile::<16, 16>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

fn matmul_mma_transposed_rhs_f16_f32_tile<const BM: usize, const BN: usize>(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile_with_occupancy::<BM, BN>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
        None,
    )
}

#[cfg(feature = "dtype-f16")]

fn matmul_mma_transposed_rhs_f16_f32_tile_with_occupancy<const BM: usize, const BN: usize>(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    occupancy: Option<i32>,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulMmaLaunch::create_transposed_rhs(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_2d(out, rows, columns)?.partition([BM, BN])?;
    let lhs = TensorAdapter::contiguous_2d(lhs, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_2d(rhs, columns, reduction)?;
    let op = kernel_matmul::matmul_mma_transposed_rhs_f16_f32(out, lhs, rhs);
    if let Some(occupancy) = occupancy {
        op.compile_options(CompileOptions::default().occupancy(occupancy))
            .enqueue_on(stream)?;
    } else {
        op.enqueue_on(stream)?;
    }
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_8x32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile::<8, 32>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile::<16, 32>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_2(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile_with_occupancy::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
        Some(2),
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_4(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile_with_occupancy::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
        Some(4),
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_transposed_rhs_f16_f32_tile_32x16(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    matmul_mma_transposed_rhs_f16_f32_tile::<32, 16>(
        stream,
        out,
        lhs,
        rhs,
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_mma_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulMmaLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_2d(out, rows, columns)?.partition([16, 16])?;
    let lhs = TensorAdapter::contiguous_2d(lhs, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_2d(rhs, reduction, columns)?;
    kernel_matmul::matmul_mma_f16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn matmul_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_bf16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
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

#[cfg(feature = "dtype-bf16")]

pub fn matmul_mma_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulMmaLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_2d(out, rows, columns)?.partition([16, 16])?;
    let lhs = TensorAdapter::contiguous_2d(lhs, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_2d(rhs, reduction, columns)?;
    kernel_matmul::matmul_mma_bf16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

pub fn matmul_alpha_beta_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_alpha_beta_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            alpha,
            beta,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn matmul_alpha_beta_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_alpha_beta_f16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            alpha,
            beta,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn matmul_alpha_beta_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_alpha_beta_bf16_f32(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            alpha,
            beta,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f64")]

pub fn matmul_alpha_beta_f64(
    stream: &Arc<Stream>,
    out: DevicePointer<f64>,
    lhs: DevicePointer<f64>,
    rhs: DevicePointer<f64>,
    rows: usize,
    columns: usize,
    reduction: usize,
    lhs_row_stride: usize,
    rhs_row_stride: usize,
    output_row_stride: usize,
    transpose_lhs: bool,
    transpose_rhs: bool,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = MatmulLaunch::create(
        rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::matmul_alpha_beta_f64(
            out,
            lhs,
            rhs,
            params.rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            alpha,
            beta,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn bmm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

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
    unsafe {
        kernel_matmul::bmm_f32(
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

pub fn bmm_mma_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, reduction, columns)?;
    kernel_matmul::bmm_mma_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

pub fn bmm_mma_transposed_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile::<16, 16>(
        stream,
        out,
        lhs,
        rhs,
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
    )
}

fn bmm_mma_transposed_f32_tile<const BM: usize, const BN: usize>(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile_with_occupancy::<BM, BN>(
        stream,
        out,
        lhs,
        rhs,
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
        None,
    )
}

fn bmm_mma_transposed_f32_tile_with_occupancy<const BM: usize, const BN: usize>(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    occupancy: Option<i32>,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create_transposed(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, BM, BN])?;
    let lhs_rows = if params.transpose_lhs {
        reduction
    } else {
        rows
    };
    let lhs_columns = if params.transpose_lhs {
        rows
    } else {
        reduction
    };
    let rhs_rows = if params.transpose_rhs {
        columns
    } else {
        reduction
    };
    let rhs_columns = if params.transpose_rhs {
        reduction
    } else {
        columns
    };
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, lhs_rows, lhs_columns)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, rhs_rows, rhs_columns)?;
    match (params.transpose_lhs, params.transpose_rhs) {
        (false, false) => {
            let op = kernel_matmul::bmm_mma_f32(out, lhs, rhs);
            if let Some(occupancy) = occupancy {
                op.compile_options(CompileOptions::default().occupancy(occupancy))
                    .enqueue_on(stream)?;
            } else {
                op.enqueue_on(stream)?;
            }
        }
        (true, false) => {
            let op = kernel_matmul::bmm_mma_transposed_lhs_f32(out, lhs, rhs);
            if let Some(occupancy) = occupancy {
                op.compile_options(CompileOptions::default().occupancy(occupancy))
                    .enqueue_on(stream)?;
            } else {
                op.enqueue_on(stream)?;
            }
        }
        (false, true) => {
            let op = kernel_matmul::bmm_mma_transposed_rhs_f32(out, lhs, rhs);
            if let Some(occupancy) = occupancy {
                op.compile_options(CompileOptions::default().occupancy(occupancy))
                    .enqueue_on(stream)?;
            } else {
                op.enqueue_on(stream)?;
            }
        }
        (true, true) => {
            let op = kernel_matmul::bmm_mma_transposed_inputs_f32(out, lhs, rhs);
            if let Some(occupancy) = occupancy {
                op.compile_options(CompileOptions::default().occupancy(occupancy))
                    .enqueue_on(stream)?;
            } else {
                op.enqueue_on(stream)?;
            }
        }
    }
    Ok(())
}

pub fn bmm_mma_transposed_f32_tile_16x64(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
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
    )
}

pub fn bmm_mma_transposed_f32_tile_16x64_occupancy_2(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile_with_occupancy::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
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
        Some(2),
    )
}

pub fn bmm_mma_transposed_f32_tile_16x64_occupancy_4(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile_with_occupancy::<16, 64>(
        stream,
        out,
        lhs,
        rhs,
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
        Some(4),
    )
}

pub fn bmm_mma_transposed_f32_tile_32x16(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
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
    bmm_mma_transposed_f32_tile::<32, 16>(
        stream,
        out,
        lhs,
        rhs,
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
    )
}

#[cfg(feature = "dtype-f16")]

pub fn bmm_mma_f16_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, reduction, columns)?;
    kernel_matmul::bmm_mma_f16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn bmm_mma_transposed_rhs_f16_f32(
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
    if transpose_lhs || !transpose_rhs {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create_transposed(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, columns, reduction)?;
    kernel_matmul::bmm_mma_transposed_rhs_f16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn bmm_mma_transposed_inputs_f16_f32(
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
    if !transpose_lhs || !transpose_rhs {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create_transposed(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, reduction, rows)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, columns, reduction)?;
    kernel_matmul::bmm_mma_transposed_inputs_f16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn bmm_mma_bf16_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, rows, reduction)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, reduction, columns)?;
    kernel_matmul::bmm_mma_bf16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn bmm_mma_transposed_inputs_bf16_f32(
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
    if !transpose_lhs || !transpose_rhs {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

    let params = BmmMmaLaunch::create_transposed(
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
    let batch = usize::try_from(params.batch).map_err(|_| Error::SizeOverflow)?;
    let rows = usize::try_from(params.rows).map_err(|_| Error::SizeOverflow)?;
    let columns = usize::try_from(params.columns).map_err(|_| Error::SizeOverflow)?;
    let reduction = usize::try_from(params.reduction).map_err(|_| Error::SizeOverflow)?;
    let out = TensorAdapter::contiguous_3d(out, batch, rows, columns)?.partition([1, 16, 16])?;
    let lhs = TensorAdapter::contiguous_3d(lhs, batch, reduction, rows)?;
    let rhs = TensorAdapter::contiguous_3d(rhs, batch, columns, reduction)?;
    kernel_matmul::bmm_mma_transposed_inputs_bf16_f32(out, lhs, rhs).enqueue_on(stream)?;
    Ok(())
}

pub fn masked_bmm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    masked_rows: DevicePointer<i32>,
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(masked_rows)?;

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
    unsafe {
        kernel_matmul::masked_bmm_f32(
            out,
            lhs,
            rhs,
            masked_rows,
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

#[cfg(feature = "dtype-f16")]

pub fn masked_bmm_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    masked_rows: DevicePointer<i32>,
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(masked_rows)?;

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
    unsafe {
        kernel_matmul::masked_bmm_f16(
            out,
            lhs,
            rhs,
            masked_rows,
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

#[cfg(feature = "dtype-bf16")]

pub fn masked_bmm_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    masked_rows: DevicePointer<i32>,
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(masked_rows)?;

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
    unsafe {
        kernel_matmul::masked_bmm_bf16(
            out,
            lhs,
            rhs,
            masked_rows,
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

pub fn ragged_bmm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::ragged_bmm_f32(
            out,
            lhs,
            rhs,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn ragged_bmm_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::ragged_bmm_f16(
            out,
            lhs,
            rhs,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn ragged_bmm_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        lhs_row_stride,
        rhs_batch_stride,
        rhs_row_stride,
        output_row_stride,
        transpose_lhs,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::ragged_bmm_bf16(
            out,
            lhs,
            rhs,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.lhs_row_stride,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.output_row_stride,
            params.transpose_lhs,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn group_gemm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<f32>,
    rhs: DevicePointer<f32>,
    rows: DevicePointer<i32>,
    columns: DevicePointer<i32>,
    reductions: DevicePointer<i32>,
    lhs_offsets: DevicePointer<i32>,
    rhs_offsets: DevicePointer<i32>,
    output_offsets: DevicePointer<i32>,
    group_count: usize,
    max_rows: usize,
    max_columns: usize,
    max_reduction: usize,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(rows)?;
    checked_device_pointer(columns)?;
    checked_device_pointer(reductions)?;
    checked_device_pointer(lhs_offsets)?;
    checked_device_pointer(rhs_offsets)?;
    checked_device_pointer(output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::group_gemm_f32(
            out,
            lhs,
            rhs,
            rows,
            columns,
            reductions,
            lhs_offsets,
            rhs_offsets,
            output_offsets,
            params.max_rows,
            params.max_columns,
            params.max_reduction,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn group_gemm_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    lhs: DevicePointer<f16>,
    rhs: DevicePointer<f16>,
    rows: DevicePointer<i32>,
    columns: DevicePointer<i32>,
    reductions: DevicePointer<i32>,
    lhs_offsets: DevicePointer<i32>,
    rhs_offsets: DevicePointer<i32>,
    output_offsets: DevicePointer<i32>,
    group_count: usize,
    max_rows: usize,
    max_columns: usize,
    max_reduction: usize,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(rows)?;
    checked_device_pointer(columns)?;
    checked_device_pointer(reductions)?;
    checked_device_pointer(lhs_offsets)?;
    checked_device_pointer(rhs_offsets)?;
    checked_device_pointer(output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::group_gemm_f16(
            out,
            lhs,
            rhs,
            rows,
            columns,
            reductions,
            lhs_offsets,
            rhs_offsets,
            output_offsets,
            params.max_rows,
            params.max_columns,
            params.max_reduction,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn group_gemm_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    lhs: DevicePointer<bf16>,
    rhs: DevicePointer<bf16>,
    rows: DevicePointer<i32>,
    columns: DevicePointer<i32>,
    reductions: DevicePointer<i32>,
    lhs_offsets: DevicePointer<i32>,
    rhs_offsets: DevicePointer<i32>,
    output_offsets: DevicePointer<i32>,
    group_count: usize,
    max_rows: usize,
    max_columns: usize,
    max_reduction: usize,
    transpose_rhs: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(rows)?;
    checked_device_pointer(columns)?;
    checked_device_pointer(reductions)?;
    checked_device_pointer(lhs_offsets)?;
    checked_device_pointer(rhs_offsets)?;
    checked_device_pointer(output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    unsafe {
        kernel_matmul::group_gemm_bf16(
            out,
            lhs,
            rhs,
            rows,
            columns,
            reductions,
            lhs_offsets,
            rhs_offsets,
            output_offsets,
            params.max_rows,
            params.max_columns,
            params.max_reduction,
            params.transpose_rhs,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn grouped_gemm_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weights: DevicePointer<f32>,
    m_sizes: DevicePointer<i32>,
    gather_indices: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(m_sizes)?;
    checked_device_pointer(gather_indices)?;

    let params = GroupedGemmLaunch::create(
        total_tokens,
        expert_count,
        columns,
        reduction,
        input_row_stride,
        weight_expert_stride,
        weight_row_stride,
        output_row_stride,
        permute_input,
        permute_output,
        top_k,
    )?;
    unsafe {
        kernel_matmul::grouped_gemm_f32(
            out,
            input,
            weights,
            m_sizes,
            gather_indices,
            params.columns,
            params.reduction,
            params.expert_count,
            params.input_row_stride,
            params.weight_expert_stride,
            params.weight_row_stride,
            params.output_row_stride,
            params.permute_input,
            params.permute_output,
            params.top_k,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn grouped_gemm_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    input: DevicePointer<f16>,
    weights: DevicePointer<f16>,
    m_sizes: DevicePointer<i32>,
    gather_indices: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(m_sizes)?;
    checked_device_pointer(gather_indices)?;

    let params = GroupedGemmLaunch::create(
        total_tokens,
        expert_count,
        columns,
        reduction,
        input_row_stride,
        weight_expert_stride,
        weight_row_stride,
        output_row_stride,
        permute_input,
        permute_output,
        top_k,
    )?;
    unsafe {
        kernel_matmul::grouped_gemm_f16(
            out,
            input,
            weights,
            m_sizes,
            gather_indices,
            params.columns,
            params.reduction,
            params.expert_count,
            params.input_row_stride,
            params.weight_expert_stride,
            params.weight_row_stride,
            params.output_row_stride,
            params.permute_input,
            params.permute_output,
            params.top_k,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn grouped_gemm_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    input: DevicePointer<bf16>,
    weights: DevicePointer<bf16>,
    m_sizes: DevicePointer<i32>,
    gather_indices: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(m_sizes)?;
    checked_device_pointer(gather_indices)?;

    let params = GroupedGemmLaunch::create(
        total_tokens,
        expert_count,
        columns,
        reduction,
        input_row_stride,
        weight_expert_stride,
        weight_row_stride,
        output_row_stride,
        permute_input,
        permute_output,
        top_k,
    )?;
    unsafe {
        kernel_matmul::grouped_gemm_bf16(
            out,
            input,
            weights,
            m_sizes,
            gather_indices,
            params.columns,
            params.reduction,
            params.expert_count,
            params.input_row_stride,
            params.weight_expert_stride,
            params.weight_row_stride,
            params.output_row_stride,
            params.permute_input,
            params.permute_output,
            params.top_k,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f8")]

pub fn ragged_block_scaled_bmm_f8e4m3_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    lhs_scale: DevicePointer<f32>,
    rhs_scale: DevicePointer<f32>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(lhs_scale)?;
    checked_device_pointer(rhs_scale)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBlockScaledBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        scale_block,
        rhs_batch_stride,
        rhs_row_stride,
        lhs_scale_row_stride,
        rhs_scale_batch_stride,
        rhs_scale_row_stride,
        output_row_stride,
    )?;
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    unsafe {
        kernel_matmul::ragged_block_scaled_bmm_f8e4m3_f32(
            out,
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.scale_block,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.lhs_scale_row_stride,
            params.rhs_scale_batch_stride,
            params.rhs_scale_row_stride,
            params.output_row_stride,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f8", feature = "dtype-f16"))]

pub fn ragged_block_scaled_bmm_f8e4m3_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    lhs_scale: DevicePointer<f32>,
    rhs_scale: DevicePointer<f32>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(lhs_scale)?;
    checked_device_pointer(rhs_scale)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBlockScaledBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        scale_block,
        rhs_batch_stride,
        rhs_row_stride,
        lhs_scale_row_stride,
        rhs_scale_batch_stride,
        rhs_scale_row_stride,
        output_row_stride,
    )?;
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    unsafe {
        kernel_matmul::ragged_block_scaled_bmm_f8e4m3_f16(
            out,
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.scale_block,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.lhs_scale_row_stride,
            params.rhs_scale_batch_stride,
            params.rhs_scale_row_stride,
            params.output_row_stride,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f8", feature = "dtype-bf16"))]

pub fn ragged_block_scaled_bmm_f8e4m3_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    lhs: DevicePointer<u8>,
    rhs: DevicePointer<u8>,
    lhs_scale: DevicePointer<f32>,
    rhs_scale: DevicePointer<f32>,
    row_indptr: DevicePointer<i32>,
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
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;
    checked_device_pointer(lhs_scale)?;
    checked_device_pointer(rhs_scale)?;
    checked_device_pointer(row_indptr)?;

    let params = RaggedBlockScaledBmmLaunch::create(
        batch,
        max_rows,
        columns,
        reduction,
        scale_block,
        rhs_batch_stride,
        rhs_row_stride,
        lhs_scale_row_stride,
        rhs_scale_batch_stride,
        rhs_scale_row_stride,
        output_row_stride,
    )?;
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    unsafe {
        kernel_matmul::ragged_block_scaled_bmm_f8e4m3_bf16(
            out,
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            row_indptr,
            params.max_rows,
            params.columns,
            params.reduction,
            params.scale_block,
            params.rhs_batch_stride,
            params.rhs_row_stride,
            params.lhs_scale_row_stride,
            params.rhs_scale_batch_stride,
            params.rhs_scale_row_stride,
            params.output_row_stride,
            params.virtual_output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f64")]

pub fn bmm_f64(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

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

#[cfg(feature = "dtype-f16")]

pub fn bmm_f16_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

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

#[cfg(feature = "dtype-bf16")]

pub fn bmm_bf16_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lhs)?;
    checked_device_pointer(rhs)?;

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{cpu::matmul as cpu_matmul, error::Result};

    #[test]
    fn matmul_f32_square() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (4usize, 5usize, 3usize);
        let lhs_host = (0..rows * reduction)
            .map(|index| (index as f32 % 7.0) - 3.0)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * columns)
            .map(|index| (index as f32 % 5.0) - 2.0)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_host, &rhs_host, rows, columns, reduction, reduction, columns, false, false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_mma_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (17usize, 19usize, 10usize);
        let lhs_host = (0..rows * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * columns)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_host, &rhs_host, rows, columns, reduction, reduction, columns, false, false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_mma_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn matmul_f32_transposed_inputs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 5usize);
        let lhs_host = (0..reduction * rows)
            .map(|index| (index as f32 % 11.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let rhs_host = (0..columns * reduction)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_host, &rhs_host, rows, columns, reduction, rows, reduction, true, true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            rows,
            reduction,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_f32_wide_projection_shape() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (4usize, 768usize, 768usize);
        let lhs_host = (0..rows * reduction)
            .map(|index| ((index * 17 + 3) % 257) as f32 * 0.00390625 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..columns * reduction)
            .map(|index| ((index * 31 + 11) % 263) as f32 * 0.00390625 - 0.5)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_host, &rhs_host, rows, columns, reduction, reduction, reduction, false, true,
        );

        let expected_lhs = lhs_host.clone();
        let expected_rhs = rhs_host.clone();
        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-4);
        singe_core::assert_close!(&lhs.to_host_vec().sync_on(&stream)?, &expected_lhs, 0.0);
        singe_core::assert_close!(&rhs.to_host_vec().sync_on(&stream)?, &expected_rhs, 0.0);
        Ok(())
    }

    #[test]
    fn matmul_f32_handles_skinny_wide_padded_strides() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (2usize, 129usize, 7usize);
        let (lhs_row_stride, rhs_row_stride, output_row_stride) = (9usize, 133usize, 131usize);
        let lhs_host = (0..rows * lhs_row_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * rhs_row_stride)
            .map(|index| (index as f32 % 17.0) * 0.0625 - 0.5)
            .collect::<Vec<_>>();
        let out_host = vec![-99.0f32; rows * output_row_stride];
        let compact_expected = cpu_matmul::matmul_f32(
            &lhs_host,
            &rhs_host,
            rows,
            columns,
            reduction,
            lhs_row_stride,
            rhs_row_stride,
            false,
            false,
        );
        let mut expected = out_host.clone();
        for row in 0..rows {
            for column in 0..columns {
                expected[row * output_row_stride + column] =
                    compact_expected[row * columns + column];
            }
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        matmul_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            lhs_row_stride,
            rhs_row_stride,
            output_row_stride,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_alpha_beta_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 2usize);
        let alpha = 1.5f32;
        let beta = -0.25f32;
        let lhs_host = (0..rows * reduction)
            .map(|index| (index as f32 % 7.0) - 2.0)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * columns)
            .map(|index| (index as f32 % 5.0) - 1.0)
            .collect::<Vec<_>>();
        let out_host = (0..rows * columns)
            .map(|index| index as f32 * 0.5)
            .collect::<Vec<_>>();
        let mut expected = cpu_matmul::matmul_f32(
            &lhs_host, &rhs_host, rows, columns, reduction, reduction, columns, false, false,
        );
        for (value, prior) in expected.iter_mut().zip(&out_host) {
            *value = alpha * *value + beta * *prior;
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        matmul_alpha_beta_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
            alpha,
            beta,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matmul_alpha_beta_f16_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 2usize);
        let alpha = -0.75f32;
        let beta = 0.5f32;
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..reduction * columns)
            .map(|index| (index as f32 % 5.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let out_host = (0..rows * columns)
            .map(|index| index as f32 * 0.125)
            .collect::<Vec<_>>();
        let mut expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            reduction,
            columns,
            false,
            false,
        );
        for (value, prior) in expected.iter_mut().zip(&out_host) {
            *value = alpha * *value + beta * *prior;
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        matmul_alpha_beta_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
            alpha,
            beta,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn matmul_alpha_beta_bf16_f32_transposed_lhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 2usize);
        let alpha = 0.625f32;
        let beta = -0.375f32;
        let lhs_f32 = (0..reduction * rows)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..reduction * columns)
            .map(|index| (index as f32 % 5.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let out_host = (0..rows * columns)
            .map(|index| index as f32 * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let mut expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            rows,
            columns,
            true,
            false,
        );
        for (value, prior) in expected.iter_mut().zip(&out_host) {
            *value = alpha * *value + beta * *prior;
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        matmul_alpha_beta_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            rows,
            columns,
            columns,
            true,
            false,
            alpha,
            beta,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn bmm_f32_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 3usize, 4usize, 5usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 7.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 9.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn bmm_mma_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 19usize, 10usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = reduction * columns;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            output_batch_stride,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn bmm_mma_transposed_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 19usize, 10usize);
        let lhs_batch_stride = reduction * rows;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            true,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_transposed_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn bmm_mma_transposed_f32_16x64_occupancy_variants_match_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (1usize, 17usize, 70usize, 18usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 29.0) * 0.015625 - 0.125)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out_occ2 = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;
        let out_occ4 = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_transposed_f32_tile_16x64_occupancy_2(
            &stream,
            out_occ2.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            false,
            true,
        )?;
        bmm_mma_transposed_f32_tile_16x64_occupancy_4(
            &stream,
            out_occ4.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out_occ2.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        singe_core::assert_close!(&out_occ4.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn bmm_mma_f16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 18usize, 11usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = reduction * columns;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            output_batch_stride,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn bmm_mma_transposed_rhs_f16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 18usize, 11usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_transposed_rhs_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn bmm_mma_transposed_inputs_f16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 18usize, 11usize);
        let lhs_batch_stride = reduction * rows;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            true,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_transposed_inputs_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn bmm_mma_bf16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 18usize, 11usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = reduction * columns;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            output_batch_stride,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 5e-2);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn bmm_mma_transposed_inputs_bf16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 17usize, 18usize, 11usize);
        let lhs_batch_stride = reduction * rows;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            true,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_mma_transposed_inputs_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 5e-2);
        Ok(())
    }

    #[test]
    fn bmm_f32_handles_skinny_wide_padded_batches() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 1usize, 131usize, 6usize);
        let lhs_row_stride = 8usize;
        let rhs_row_stride = 137usize;
        let output_row_stride = 133usize;
        let lhs_batch_stride = rows * lhs_row_stride + 5;
        let rhs_batch_stride = reduction * rhs_row_stride + 7;
        let output_batch_stride = rows * output_row_stride + 3;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.375)
            .collect::<Vec<_>>();
        let out_host = vec![-88.0f32; batch * output_batch_stride];
        let compact_expected = cpu_matmul::bmm_f32(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            lhs_row_stride,
            rhs_batch_stride,
            rhs_row_stride,
            false,
            false,
        );
        let mut expected = out_host.clone();
        for batch_index in 0..batch {
            for row in 0..rows {
                for column in 0..columns {
                    expected
                        [batch_index * output_batch_stride + row * output_row_stride + column] =
                        compact_expected[batch_index * rows * columns + row * columns + column];
                }
            }
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        bmm_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
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
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn masked_bmm_f32_and_preserves_masked_rows() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (3usize, 4usize, 3usize, 5usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = reduction * columns;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let masked_rows_host = vec![4i32, 2i32, 0i32];
        let out_host = vec![-77.0f32; batch * output_batch_stride];
        let expected = cpu_matmul::masked_bmm_f32(
            &lhs_host,
            &rhs_host,
            &out_host,
            &masked_rows_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let masked_rows =
            api::copy_host_vec_to_device(&Arc::new(masked_rows_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        masked_bmm_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            masked_rows.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            output_batch_stride,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn masked_bmm_f16_and_preserves_masked_rows() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (3usize, 4usize, 3usize, 5usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = reduction * columns;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let masked_rows_host = vec![4i32, 2i32, 0i32];
        let out_host = vec![f16::from_f32(-77.0f32); batch * output_batch_stride];
        let expected = cpu_matmul::masked_bmm_f32(
            &lhs_expected,
            &rhs_expected,
            &vec![-77.0f32; batch * output_batch_stride],
            &masked_rows_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let masked_rows =
            api::copy_host_vec_to_device(&Arc::new(masked_rows_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        masked_bmm_f16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            masked_rows.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            columns,
            output_batch_stride,
            columns,
            false,
            false,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn ragged_bmm_f32_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction) =
            (3usize, 6usize, 3usize, 4usize, 5usize);
        let rhs_batch_stride = columns * reduction;
        let lhs_host = (0..total_rows * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 3i32, 5i32, 6i32];
        let expected = cpu_matmul::ragged_bmm_f32(
            &lhs_host,
            &rhs_host,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_bmm_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            reduction,
            rhs_batch_stride,
            reduction,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn ragged_bmm_bf16_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction) =
            (3usize, 6usize, 3usize, 4usize, 5usize);
        let rhs_batch_stride = columns * reduction;
        let lhs_f32 = (0..total_rows * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 3i32, 5i32, 6i32];
        let expected = cpu_matmul::ragged_bmm_f32(
            &lhs_expected,
            &rhs_expected,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_bmm_bf16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            reduction,
            rhs_batch_stride,
            reduction,
            columns,
            false,
            true,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn group_gemm_f32_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows_host = vec![2i32, 3i32];
        let columns_host = vec![3i32, 2i32];
        let reductions_host = vec![4i32, 5i32];
        let lhs_offsets_host = vec![0i32, 8i32];
        let rhs_offsets_host = vec![0i32, 12i32];
        let output_offsets_host = vec![0i32, 6i32];
        let lhs_len = 8 + 15;
        let rhs_len = 12 + 10;
        let output_len = 6 + 6;
        let lhs_host = (0..lhs_len)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..rhs_len)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::group_gemm_f32(
            &lhs_host,
            &rhs_host,
            &rows_host,
            &columns_host,
            &reductions_host,
            &lhs_offsets_host,
            &rhs_offsets_host,
            &output_offsets_host,
            output_len,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let rows = api::copy_host_vec_to_device(&Arc::new(rows_host)).sync_on(&stream)?;
        let columns = api::copy_host_vec_to_device(&Arc::new(columns_host)).sync_on(&stream)?;
        let reductions =
            api::copy_host_vec_to_device(&Arc::new(reductions_host)).sync_on(&stream)?;
        let lhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(lhs_offsets_host)).sync_on(&stream)?;
        let rhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(rhs_offsets_host)).sync_on(&stream)?;
        let output_offsets =
            api::copy_host_vec_to_device(&Arc::new(output_offsets_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[output_len]).sync_on(&stream)?;

        group_gemm_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows.device_pointer(),
            columns.device_pointer(),
            reductions.device_pointer(),
            lhs_offsets.device_pointer(),
            rhs_offsets.device_pointer(),
            output_offsets.device_pointer(),
            2,
            3,
            3,
            5,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn group_gemm_f16_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows_host = vec![2i32, 3i32];
        let columns_host = vec![3i32, 2i32];
        let reductions_host = vec![4i32, 5i32];
        let lhs_offsets_host = vec![0i32, 8i32];
        let rhs_offsets_host = vec![0i32, 12i32];
        let output_offsets_host = vec![0i32, 6i32];
        let lhs_len = 8 + 15;
        let rhs_len = 12 + 10;
        let output_len = 6 + 6;
        let lhs_f32 = (0..lhs_len)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..rhs_len)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::group_gemm_f32(
            &lhs_expected,
            &rhs_expected,
            &rows_host,
            &columns_host,
            &reductions_host,
            &lhs_offsets_host,
            &rhs_offsets_host,
            &output_offsets_host,
            output_len,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let rows = api::copy_host_vec_to_device(&Arc::new(rows_host)).sync_on(&stream)?;
        let columns = api::copy_host_vec_to_device(&Arc::new(columns_host)).sync_on(&stream)?;
        let reductions =
            api::copy_host_vec_to_device(&Arc::new(reductions_host)).sync_on(&stream)?;
        let lhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(lhs_offsets_host)).sync_on(&stream)?;
        let rhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(rhs_offsets_host)).sync_on(&stream)?;
        let output_offsets =
            api::copy_host_vec_to_device(&Arc::new(output_offsets_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[output_len]).sync_on(&stream)?;

        group_gemm_f16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows.device_pointer(),
            columns.device_pointer(),
            reductions.device_pointer(),
            lhs_offsets.device_pointer(),
            rhs_offsets.device_pointer(),
            output_offsets.device_pointer(),
            2,
            3,
            3,
            5,
            true,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn group_gemm_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows_host = vec![2i32, 3i32];
        let columns_host = vec![3i32, 2i32];
        let reductions_host = vec![4i32, 5i32];
        let lhs_offsets_host = vec![0i32, 8i32];
        let rhs_offsets_host = vec![0i32, 12i32];
        let output_offsets_host = vec![0i32, 6i32];
        let lhs_len = 8 + 15;
        let rhs_len = 12 + 10;
        let output_len = 6 + 6;
        let lhs_f32 = (0..lhs_len)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..rhs_len)
            .map(|index| (index as f32 % 13.0) * 0.25 - 1.0)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::group_gemm_f32(
            &lhs_expected,
            &rhs_expected,
            &rows_host,
            &columns_host,
            &reductions_host,
            &lhs_offsets_host,
            &rhs_offsets_host,
            &output_offsets_host,
            output_len,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let rows = api::copy_host_vec_to_device(&Arc::new(rows_host)).sync_on(&stream)?;
        let columns = api::copy_host_vec_to_device(&Arc::new(columns_host)).sync_on(&stream)?;
        let reductions =
            api::copy_host_vec_to_device(&Arc::new(reductions_host)).sync_on(&stream)?;
        let lhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(lhs_offsets_host)).sync_on(&stream)?;
        let rhs_offsets =
            api::copy_host_vec_to_device(&Arc::new(rhs_offsets_host)).sync_on(&stream)?;
        let output_offsets =
            api::copy_host_vec_to_device(&Arc::new(output_offsets_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[output_len]).sync_on(&stream)?;

        group_gemm_bf16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows.device_pointer(),
            columns.device_pointer(),
            reductions.device_pointer(),
            lhs_offsets.device_pointer(),
            rhs_offsets.device_pointer(),
            output_offsets.device_pointer(),
            2,
            3,
            3,
            5,
            false,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn grouped_gemm_f32_permuted_input() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (total_tokens, input_rows, expert_count, columns, reduction, top_k) =
            (5usize, 3usize, 2usize, 4usize, 3usize, 2usize);
        let input_host = (0..input_rows * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let weights_host = (0..expert_count * columns * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let m_sizes_host = vec![2i32, 3i32];
        let gather_indices_host = vec![0i32, 2i32, 1i32, 3i32, 4i32];
        let expected = cpu_matmul::grouped_gemm_f32(
            &input_host,
            &weights_host,
            &m_sizes_host,
            &gather_indices_host,
            total_tokens,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            true,
            false,
            top_k,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let m_sizes = api::copy_host_vec_to_device(&Arc::new(m_sizes_host)).sync_on(&stream)?;
        let gather_indices =
            api::copy_host_vec_to_device(&Arc::new(gather_indices_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[total_tokens * columns]).sync_on(&stream)?;

        grouped_gemm_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weights.device_pointer(),
            m_sizes.device_pointer(),
            gather_indices.device_pointer(),
            total_tokens,
            expert_count,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            true,
            false,
            top_k,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn grouped_gemm_f32_permuted_output() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (total_tokens, expert_count, columns, reduction) = (5usize, 2usize, 3usize, 4usize);
        let input_host = (0..total_tokens * reduction)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let weights_host = (0..expert_count * columns * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let m_sizes_host = vec![3i32, 2i32];
        let gather_indices_host = vec![3i32, 0i32, 4i32, 1i32, 2i32];
        let expected = cpu_matmul::grouped_gemm_f32(
            &input_host,
            &weights_host,
            &m_sizes_host,
            &gather_indices_host,
            total_tokens,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            false,
            true,
            1,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let m_sizes = api::copy_host_vec_to_device(&Arc::new(m_sizes_host)).sync_on(&stream)?;
        let gather_indices =
            api::copy_host_vec_to_device(&Arc::new(gather_indices_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[total_tokens * columns]).sync_on(&stream)?;

        grouped_gemm_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weights.device_pointer(),
            m_sizes.device_pointer(),
            gather_indices.device_pointer(),
            total_tokens,
            expert_count,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            false,
            true,
            1,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn grouped_gemm_f16_permuted_input() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (total_tokens, input_rows, expert_count, columns, reduction, top_k) =
            (5usize, 3usize, 2usize, 4usize, 3usize, 2usize);
        let input_f32 = (0..input_rows * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let weights_f32 = (0..expert_count * columns * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let input_host = input_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let weights_host = weights_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weights_expected = weights_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let m_sizes_host = vec![2i32, 3i32];
        let gather_indices_host = vec![0i32, 2i32, 1i32, 3i32, 4i32];
        let expected = cpu_matmul::grouped_gemm_f32(
            &input_expected,
            &weights_expected,
            &m_sizes_host,
            &gather_indices_host,
            total_tokens,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            true,
            false,
            top_k,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let m_sizes = api::copy_host_vec_to_device(&Arc::new(m_sizes_host)).sync_on(&stream)?;
        let gather_indices =
            api::copy_host_vec_to_device(&Arc::new(gather_indices_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[total_tokens * columns]).sync_on(&stream)?;

        grouped_gemm_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weights.device_pointer(),
            m_sizes.device_pointer(),
            gather_indices.device_pointer(),
            total_tokens,
            expert_count,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            true,
            false,
            top_k,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn grouped_gemm_bf16_permuted_output() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (total_tokens, expert_count, columns, reduction) = (5usize, 2usize, 3usize, 4usize);
        let input_f32 = (0..total_tokens * reduction)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let weights_f32 = (0..expert_count * columns * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let input_host = input_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let weights_host = weights_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weights_expected = weights_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let m_sizes_host = vec![3i32, 2i32];
        let gather_indices_host = vec![3i32, 0i32, 4i32, 1i32, 2i32];
        let expected = cpu_matmul::grouped_gemm_f32(
            &input_expected,
            &weights_expected,
            &m_sizes_host,
            &gather_indices_host,
            total_tokens,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            false,
            true,
            1,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let m_sizes = api::copy_host_vec_to_device(&Arc::new(m_sizes_host)).sync_on(&stream)?;
        let gather_indices =
            api::copy_host_vec_to_device(&Arc::new(gather_indices_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[total_tokens * columns]).sync_on(&stream)?;

        grouped_gemm_bf16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weights.device_pointer(),
            m_sizes.device_pointer(),
            gather_indices.device_pointer(),
            total_tokens,
            expert_count,
            columns,
            reduction,
            reduction,
            columns * reduction,
            reduction,
            columns,
            false,
            true,
            1,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .into_iter()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-2);
        Ok(())
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn ragged_block_scaled_bmm_f8e4m3_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction, scale_block) =
            (2usize, 5usize, 3usize, 4usize, 5usize, 2usize);
        let k_scale_tiles = reduction.div_ceil(scale_block);
        let n_scale_tiles = columns.div_ceil(scale_block);
        let rhs_batch_stride = columns * reduction;
        let rhs_scale_batch_stride = n_scale_tiles * k_scale_tiles;
        let lhs_host = vec![0x38u8; total_rows * reduction];
        let rhs_host = vec![0x38u8; batch * rhs_batch_stride];
        let lhs_scale_host = (0..total_rows * k_scale_tiles)
            .map(|index| 0.5f32 + (index as f32 % 3.0) * 0.25)
            .collect::<Vec<_>>();
        let rhs_scale_host = (0..batch * rhs_scale_batch_stride)
            .map(|index| 0.75f32 + (index as f32 % 5.0) * 0.125)
            .collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 3i32, 5i32];
        let expected = cpu_matmul::ragged_block_scaled_bmm_f32(
            &lhs_host,
            &rhs_host,
            &lhs_scale_host,
            &rhs_scale_host,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            scale_block,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let lhs_scale = api::copy_host_vec_to_device(&Arc::new(lhs_scale_host)).sync_on(&stream)?;
        let rhs_scale = api::copy_host_vec_to_device(&Arc::new(rhs_scale_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_block_scaled_bmm_f8e4m3_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            lhs_scale.device_pointer(),
            rhs_scale.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            scale_block,
            rhs_batch_stride,
            reduction,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
            columns,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn ragged_block_scaled_bmm_f8e4m3_f32_handles_tail_column_scale_tile() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction, scale_block) =
            (2usize, 4usize, 3usize, 5usize, 5usize, 2usize);
        let k_scale_tiles = reduction.div_ceil(scale_block);
        let n_scale_tiles = columns.div_ceil(scale_block);
        let rhs_batch_stride = columns * reduction;
        let rhs_scale_batch_stride = n_scale_tiles * k_scale_tiles;
        let lhs_host = vec![
            0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0xc0, 0x38, 0x40, 0x30, 0xb8, 0x30, 0xb8, 0x38, 0x40,
            0xc0, 0x40, 0x30, 0xb8, 0xc0, 0x38,
        ];
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| match index % 5 {
                0 => 0x38u8,
                1 => 0x40,
                2 => 0x30,
                3 => 0xb8,
                _ => 0xc0,
            })
            .collect::<Vec<_>>();
        let lhs_scale_host = (0..total_rows * k_scale_tiles)
            .map(|index| 0.5f32 + (index as f32 % 4.0) * 0.125)
            .collect::<Vec<_>>();
        let rhs_scale_host = (0..batch * rhs_scale_batch_stride)
            .map(|index| 0.75f32 + (index as f32 % 7.0) * 0.0625)
            .collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 1i32, 4i32];
        let expected = cpu_matmul::ragged_block_scaled_bmm_f32(
            &lhs_host,
            &rhs_host,
            &lhs_scale_host,
            &rhs_scale_host,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            scale_block,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let lhs_scale = api::copy_host_vec_to_device(&Arc::new(lhs_scale_host)).sync_on(&stream)?;
        let rhs_scale = api::copy_host_vec_to_device(&Arc::new(rhs_scale_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_block_scaled_bmm_f8e4m3_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            lhs_scale.device_pointer(),
            rhs_scale.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            scale_block,
            rhs_batch_stride,
            reduction,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
            columns,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f8", feature = "dtype-f16"))]
    #[test]
    fn ragged_block_scaled_bmm_f8e4m3_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction, scale_block) =
            (2usize, 5usize, 3usize, 4usize, 5usize, 2usize);
        let k_scale_tiles = reduction.div_ceil(scale_block);
        let n_scale_tiles = columns.div_ceil(scale_block);
        let rhs_batch_stride = columns * reduction;
        let rhs_scale_batch_stride = n_scale_tiles * k_scale_tiles;
        let lhs_host = vec![0x38u8; total_rows * reduction];
        let rhs_host = vec![0x38u8; batch * rhs_batch_stride];
        let lhs_scale_host = (0..total_rows * k_scale_tiles)
            .map(|index| 0.5f32 + (index as f32 % 3.0) * 0.25)
            .collect::<Vec<_>>();
        let rhs_scale_host = (0..batch * rhs_scale_batch_stride)
            .map(|index| 0.75f32 + (index as f32 % 5.0) * 0.125)
            .collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 3i32, 5i32];
        let expected = cpu_matmul::ragged_block_scaled_bmm_f32(
            &lhs_host,
            &rhs_host,
            &lhs_scale_host,
            &rhs_scale_host,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            scale_block,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let lhs_scale = api::copy_host_vec_to_device(&Arc::new(lhs_scale_host)).sync_on(&stream)?;
        let rhs_scale = api::copy_host_vec_to_device(&Arc::new(rhs_scale_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_block_scaled_bmm_f8e4m3_f16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            lhs_scale.device_pointer(),
            rhs_scale.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            scale_block,
            rhs_batch_stride,
            reduction,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
            columns,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f8", feature = "dtype-bf16"))]
    #[test]
    fn ragged_block_scaled_bmm_f8e4m3_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, total_rows, max_rows, columns, reduction, scale_block) =
            (2usize, 5usize, 3usize, 4usize, 5usize, 2usize);
        let k_scale_tiles = reduction.div_ceil(scale_block);
        let n_scale_tiles = columns.div_ceil(scale_block);
        let rhs_batch_stride = columns * reduction;
        let rhs_scale_batch_stride = n_scale_tiles * k_scale_tiles;
        let lhs_host = vec![0x38u8; total_rows * reduction];
        let rhs_host = vec![0x38u8; batch * rhs_batch_stride];
        let lhs_scale_host = (0..total_rows * k_scale_tiles)
            .map(|index| 0.5f32 + (index as f32 % 3.0) * 0.25)
            .collect::<Vec<_>>();
        let rhs_scale_host = (0..batch * rhs_scale_batch_stride)
            .map(|index| 0.75f32 + (index as f32 % 5.0) * 0.125)
            .collect::<Vec<_>>();
        let row_indptr_host = vec![0i32, 3i32, 5i32];
        let expected = cpu_matmul::ragged_block_scaled_bmm_f32(
            &lhs_host,
            &rhs_host,
            &lhs_scale_host,
            &rhs_scale_host,
            &row_indptr_host,
            batch,
            total_rows,
            columns,
            reduction,
            scale_block,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let lhs_scale = api::copy_host_vec_to_device(&Arc::new(lhs_scale_host)).sync_on(&stream)?;
        let rhs_scale = api::copy_host_vec_to_device(&Arc::new(rhs_scale_host)).sync_on(&stream)?;
        let row_indptr =
            api::copy_host_vec_to_device(&Arc::new(row_indptr_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[total_rows * columns]).sync_on(&stream)?;

        ragged_block_scaled_bmm_f8e4m3_bf16(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            lhs_scale.device_pointer(),
            rhs_scale.device_pointer(),
            row_indptr.device_pointer(),
            batch,
            max_rows,
            columns,
            reduction,
            scale_block,
            rhs_batch_stride,
            reduction,
            k_scale_tiles,
            rhs_scale_batch_stride,
            k_scale_tiles,
            columns,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 4e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-f64")]
    #[test]
    fn matmul_f64_transposed_lhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 5usize);
        let lhs_host = (0..reduction * rows)
            .map(|index| (index as f64 % 11.0) * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * columns)
            .map(|index| (index as f64 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f64(
            &lhs_host, &rhs_host, rows, columns, reduction, rows, columns, true, false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f64>(&[rows * columns]).sync_on(&stream)?;

        matmul_f64(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            rows,
            columns,
            columns,
            true,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-12);
        Ok(())
    }

    #[cfg(feature = "dtype-f64")]
    #[test]
    fn matmul_alpha_beta_f64() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 2usize);
        let alpha = -0.75f64;
        let beta = 0.5f64;
        let lhs_host = (0..rows * reduction)
            .map(|index| (index as f64 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..reduction * columns)
            .map(|index| (index as f64 % 5.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let out_host = (0..rows * columns)
            .map(|index| index as f64 * 0.125)
            .collect::<Vec<_>>();
        let mut expected = cpu_matmul::matmul_f64(
            &lhs_host, &rhs_host, rows, columns, reduction, reduction, columns, false, false,
        );
        for (value, prior) in expected.iter_mut().zip(&out_host) {
            *value = alpha * *value + beta * *prior;
        }

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::copy_host_vec_to_device(&Arc::new(out_host)).sync_on(&stream)?;

        matmul_alpha_beta_f64(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
            alpha,
            beta,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-12);
        Ok(())
    }

    #[cfg(feature = "dtype-f64")]
    #[test]
    fn bmm_f64_transposed_inputs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 3usize, 4usize, 5usize);
        let lhs_batch_stride = reduction * rows;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_host = (0..batch * lhs_batch_stride)
            .map(|index| (index as f64 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let rhs_host = (0..batch * rhs_batch_stride)
            .map(|index| (index as f64 % 9.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f64(
            &lhs_host,
            &rhs_host,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            true,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f64>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_f64(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-12);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matmul_f16_f32_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 5usize);
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..columns * reduction)
            .map(|index| (index as f32 % 9.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_f32, &rhs_f32, rows, columns, reduction, reduction, reduction, false, true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-4);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matvec_transposed_rhs_f16_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        for reduction in [1024usize, 2048usize, 3072usize] {
            let (rows, columns) = (1usize, 37usize);
            let lhs_f32 = (0..rows * reduction)
                .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
                .collect::<Vec<_>>();
            let rhs_f32 = (0..columns * reduction)
                .map(|index| (index as f32 % 29.0) * 0.015625 - 0.125)
                .collect::<Vec<_>>();
            let lhs_host = lhs_f32
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let rhs_host = rhs_f32
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
            let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
            let expected = cpu_matmul::matmul_f32(
                &lhs_expected,
                &rhs_expected,
                rows,
                columns,
                reduction,
                reduction,
                reduction,
                false,
                true,
            );

            let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
            let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
            let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

            matvec_transposed_rhs_f16_f32(
                &stream,
                out.device_pointer(),
                lhs.device_pointer(),
                rhs.device_pointer(),
                rows,
                columns,
                reduction,
                reduction,
                reduction,
                columns,
                false,
                true,
            )?;

            singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 5e-2);
        }
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matmul_mma_transposed_rhs_f16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (19usize, 35usize, 17usize);
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..columns * reduction)
            .map(|index| (index as f32 % 29.0) * 0.015625 - 0.125)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_mma_transposed_rhs_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 2e-2);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matmul_mma_transposed_rhs_f16_f32_16x64_occupancy_variants_match_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (19usize, 70usize, 17usize);
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..columns * reduction)
            .map(|index| (index as f32 % 29.0) * 0.015625 - 0.125)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out_occ2 = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;
        let out_occ4 = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_2(
            &stream,
            out_occ2.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            columns,
            false,
            true,
        )?;
        matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_4(
            &stream,
            out_occ4.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            reduction,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out_occ2.to_host_vec().sync_on(&stream)?, &expected, 2e-2);
        singe_core::assert_close!(&out_occ4.to_host_vec().sync_on(&stream)?, &expected, 2e-2);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn matmul_mma_f16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (17usize, 18usize, 11usize);
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..reduction * columns)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            reduction,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_mma_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn matmul_bf16_f32_transposed_lhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (3usize, 4usize, 5usize);
        let lhs_f32 = (0..reduction * rows)
            .map(|index| (index as f32 % 5.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..reduction * columns)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            rows,
            columns,
            true,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            rows,
            columns,
            columns,
            true,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn matmul_mma_bf16_f32_with_padding() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction) = (17usize, 18usize, 11usize);
        let lhs_f32 = (0..rows * reduction)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..reduction * columns)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::matmul_f32(
            &lhs_expected,
            &rhs_expected,
            rows,
            columns,
            reduction,
            reduction,
            columns,
            false,
            false,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_mma_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            rows,
            columns,
            reduction,
            reduction,
            columns,
            columns,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 5e-2);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn bmm_f16_f32_transposed_inputs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 3usize, 4usize, 5usize);
        let lhs_batch_stride = reduction * rows;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 9.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_f32,
            &rhs_f32,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            true,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_f16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            rows,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-4);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn bmm_bf16_f32_transposed_rhs() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, rows, columns, reduction) = (2usize, 3usize, 4usize, 5usize);
        let lhs_batch_stride = rows * reduction;
        let rhs_batch_stride = columns * reduction;
        let output_batch_stride = rows * columns;
        let lhs_f32 = (0..batch * lhs_batch_stride)
            .map(|index| (index as f32 % 5.0) * 0.5 - 1.0)
            .collect::<Vec<_>>();
        let rhs_f32 = (0..batch * rhs_batch_stride)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let lhs_host = lhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let rhs_host = rhs_f32
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let lhs_expected = lhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let rhs_expected = rhs_host.iter().copied().map(f32::from).collect::<Vec<_>>();
        let expected = cpu_matmul::bmm_f32(
            &lhs_expected,
            &rhs_expected,
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            false,
            true,
        );

        let lhs = api::copy_host_vec_to_device(&Arc::new(lhs_host)).sync_on(&stream)?;
        let rhs = api::copy_host_vec_to_device(&Arc::new(rhs_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * output_batch_stride]).sync_on(&stream)?;

        bmm_bf16_f32(
            &stream,
            out.device_pointer(),
            lhs.device_pointer(),
            rhs.device_pointer(),
            batch,
            rows,
            columns,
            reduction,
            lhs_batch_stride,
            reduction,
            rhs_batch_stride,
            reduction,
            output_batch_stride,
            columns,
            false,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-3);
        Ok(())
    }
}
