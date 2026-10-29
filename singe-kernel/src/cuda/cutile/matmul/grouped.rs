use std::sync::Arc;

#[cfg(any(
    all(feature = "dtype_bf16", feature = "dtype_f8"),
    all(feature = "dtype_f16", feature = "dtype_f8"),
    feature = "dtype_f8"
))]
use ::cutile::core::f8e4m3fn;
#[cfg(any(
    feature = "dtype_bf16",
    all(feature = "dtype_bf16", feature = "dtype_f8")
))]
use ::cutile::half::bf16;
#[cfg(any(
    feature = "dtype_f16",
    all(feature = "dtype_f16", feature = "dtype_f8")
))]
use ::cutile::half::f16;
use ::cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};

#[cfg(any(
    all(feature = "dtype_bf16", feature = "dtype_f8"),
    all(feature = "dtype_f16", feature = "dtype_f8"),
    feature = "dtype_f8"
))]
use crate::cuda::cutile::matmul::RaggedBlockScaledBmmLaunch;
use crate::{
    cuda::cutile::{
        DeviceOpExt, TileKernel,
        kernel::matmul as kernel_matmul,
        matmul::{BmmLaunch, GroupGemmLaunch, GroupedGemmLaunch, RaggedBmmLaunch},
        utility::checked_device_pointer,
    },
    error::Result,
};

/// Multiplies the active rows of each batch.
/// Rows at or beyond each `masked_rows` count remain untouched.
/// Uninitialized output outside those active rows remains uninitialized.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn masked_bmm_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&masked_rows)?;

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

#[cfg(feature = "dtype_f16")]
/// Multiplies the active rows of each batch.
/// Rows at or beyond each `masked_rows` count remain untouched.
/// Uninitialized output outside those active rows remains uninitialized.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn masked_bmm_f16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&masked_rows)?;

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

#[cfg(feature = "dtype_bf16")]
/// Multiplies the active rows of each batch.
/// Rows at or beyond each `masked_rows` count remain untouched.
/// Uninitialized output outside those active rows remains uninitialized.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn masked_bmm_bf16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&masked_rows)?;

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

/// Multiplies batches whose row segments are described by `row_indptr`.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_bmm_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies batches whose row segments are described by `row_indptr`.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_bmm_f16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_bf16")]
/// Multiplies batches whose row segments are described by `row_indptr`.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_bmm_bf16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

/// Multiplies matrices described by device dimension and offset arrays.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn group_gemm_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&rows)?;
    checked_device_pointer(&columns)?;
    checked_device_pointer(&reductions)?;
    checked_device_pointer(&lhs_offsets)?;
    checked_device_pointer(&rhs_offsets)?;
    checked_device_pointer(&output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies matrices described by device dimension and offset arrays.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn group_gemm_f16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&rows)?;
    checked_device_pointer(&columns)?;
    checked_device_pointer(&reductions)?;
    checked_device_pointer(&lhs_offsets)?;
    checked_device_pointer(&rhs_offsets)?;
    checked_device_pointer(&output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_bf16")]
/// Multiplies matrices described by device dimension and offset arrays.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn group_gemm_bf16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&rows)?;
    checked_device_pointer(&columns)?;
    checked_device_pointer(&reductions)?;
    checked_device_pointer(&lhs_offsets)?;
    checked_device_pointer(&rhs_offsets)?;
    checked_device_pointer(&output_offsets)?;

    let params = GroupGemmLaunch::create(
        group_count,
        max_rows,
        max_columns,
        max_reduction,
        transpose_rhs,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
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

/// Multiplies expert row segments with optional input gathering and output scattering.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn grouped_gemm_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&input)?;
    checked_device_pointer(&weights)?;
    checked_device_pointer(&m_sizes)?;
    checked_device_pointer(&gather_indices)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies expert row segments with optional input gathering and output scattering.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn grouped_gemm_f16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&input)?;
    checked_device_pointer(&weights)?;
    checked_device_pointer(&m_sizes)?;
    checked_device_pointer(&gather_indices)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_bf16")]
/// Multiplies expert row segments with optional input gathering and output scattering.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn grouped_gemm_bf16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&input)?;
    checked_device_pointer(&weights)?;
    checked_device_pointer(&m_sizes)?;
    checked_device_pointer(&gather_indices)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f8")]
/// Multiplies ragged FP8 batches with block scales.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_block_scaled_bmm_f8e4m3_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&lhs_scale)?;
    checked_device_pointer(&rhs_scale)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(all(feature = "dtype_f8", feature = "dtype_f16"))]
/// Multiplies ragged FP8 batches with block scales.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_block_scaled_bmm_f8e4m3_f16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&lhs_scale)?;
    checked_device_pointer(&rhs_scale)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(all(feature = "dtype_f8", feature = "dtype_bf16"))]
/// Multiplies ragged FP8 batches with block scales.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
/// Device metadata must describe valid input ranges and in-bounds, nonoverlapping output writes.
pub unsafe fn ragged_block_scaled_bmm_f8e4m3_bf16(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
    checked_device_pointer(&lhs_scale)?;
    checked_device_pointer(&rhs_scale)?;
    checked_device_pointer(&row_indptr)?;

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
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let lhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(lhs.cu_deviceptr()) };
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let rhs = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(rhs.cu_deviceptr()) };
    // SAFETY: the caller upholds this function's safety contract.
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
