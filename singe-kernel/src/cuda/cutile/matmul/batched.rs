use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::CompileOptions,
};

use crate::{
    cuda::{
        autotune::KernelCompileConfig,
        cutile::{
            DeviceOpExt, TileKernel,
            adapter::TensorAdapter,
            kernel::matmul as kernel_matmul,
            matmul::{BmmLaunch, BmmMmaLaunch},
            utility::checked_device_pointer,
        },
    },
    error::{Error, Result},
};

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_f32(
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

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32(
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
    bmm_mma_transposed_f32_tile_with_compile_config::<BM, BN>(
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
        KernelCompileConfig {
            occupancy,
            ..KernelCompileConfig::default()
        },
    )
}

fn bmm_mma_transposed_f32_tile_with_compile_config<const BM: usize, const BN: usize>(
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
    compile: KernelCompileConfig,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
            if compile.is_default() {
                op.enqueue_on(stream)?;
            } else {
                let mut options = CompileOptions::default();
                if let Some(occupancy) = compile.occupancy {
                    options = options.occupancy(occupancy);
                }
                if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
                    options = options.num_cta_in_cga(num_cta_in_cga);
                }
                if let Some(max_divisibility) = compile.max_divisibility {
                    options = options.max_divisibility(max_divisibility);
                }
                op.compile_options(options).enqueue_on(stream)?;
            }
        }
        (true, false) => {
            let op = kernel_matmul::bmm_mma_transposed_lhs_f32(out, lhs, rhs);
            if compile.is_default() {
                op.enqueue_on(stream)?;
            } else {
                let mut options = CompileOptions::default();
                if let Some(occupancy) = compile.occupancy {
                    options = options.occupancy(occupancy);
                }
                if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
                    options = options.num_cta_in_cga(num_cta_in_cga);
                }
                if let Some(max_divisibility) = compile.max_divisibility {
                    options = options.max_divisibility(max_divisibility);
                }
                op.compile_options(options).enqueue_on(stream)?;
            }
        }
        (false, true) => {
            let op = kernel_matmul::bmm_mma_transposed_rhs_f32(out, lhs, rhs);
            if compile.is_default() {
                op.enqueue_on(stream)?;
            } else {
                let mut options = CompileOptions::default();
                if let Some(occupancy) = compile.occupancy {
                    options = options.occupancy(occupancy);
                }
                if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
                    options = options.num_cta_in_cga(num_cta_in_cga);
                }
                if let Some(max_divisibility) = compile.max_divisibility {
                    options = options.max_divisibility(max_divisibility);
                }
                op.compile_options(options).enqueue_on(stream)?;
            }
        }
        (true, true) => {
            let op = kernel_matmul::bmm_mma_transposed_inputs_f32(out, lhs, rhs);
            if compile.is_default() {
                op.enqueue_on(stream)?;
            } else {
                let mut options = CompileOptions::default();
                if let Some(occupancy) = compile.occupancy {
                    options = options.occupancy(occupancy);
                }
                if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
                    options = options.num_cta_in_cga(num_cta_in_cga);
                }
                if let Some(max_divisibility) = compile.max_divisibility {
                    options = options.max_divisibility(max_divisibility);
                }
                op.compile_options(options).enqueue_on(stream)?;
            }
        }
    }
    Ok(())
}

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32_tuned(
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
    tile_rows: usize,
    tile_columns: usize,
    compile: KernelCompileConfig,
) -> Result<()> {
    match (tile_rows, tile_columns) {
        (16, 16) => bmm_mma_transposed_f32_tile_with_compile_config::<16, 16>(
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
            compile,
        ),
        (16, 64) => bmm_mma_transposed_f32_tile_with_compile_config::<16, 64>(
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
            compile,
        ),
        (32, 16) => bmm_mma_transposed_f32_tile_with_compile_config::<32, 16>(
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
            compile,
        ),
        _ => Err(Error::UnsupportedConfiguration {
            op: "bmm_mma_transposed_f32".into(),
            reason: format!("unsupported mma tile {tile_rows}x{tile_columns}"),
        }),
    }
}

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32_tile_16x64(
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

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32_tile_16x64_occupancy_2(
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

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32_tile_16x64_occupancy_4(
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

/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_f32_tile_32x16(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_f16_f32(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_rhs_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

#[cfg(feature = "dtype_f16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_inputs_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

#[cfg(feature = "dtype_bf16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_bf16_f32(
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

#[cfg(feature = "dtype_bf16")]
/// Multiplies matrix batches using the supplied row and batch layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn bmm_mma_transposed_inputs_bf16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
