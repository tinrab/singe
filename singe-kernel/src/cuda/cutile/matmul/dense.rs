use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use ::cutile::half::bf16;
use ::cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};
#[cfg(feature = "dtype_f16")]
use ::cutile::{half::f16, tile_kernel::CompileOptions};

#[cfg(feature = "dtype_f16")]
use crate::{cuda::autotune::KernelCompileConfig, utility::checked_i32_value};
use crate::{
    cuda::cutile::{
        DeviceOpExt, TileKernel,
        adapter::TensorAdapter,
        kernel::matmul as kernel_matmul,
        matmul::{MatmulLaunch, MatmulMmaLaunch},
        utility::checked_device_pointer,
    },
    error::{Error, Result},
};

/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

#[cfg(feature = "dtype_f64")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_f64(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies a vector by the transposed right-hand matrix.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matvec_transposed_rhs_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;
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
        // SAFETY: the caller upholds this function's safety contract.
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
        // SAFETY: the caller upholds this function's safety contract.
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
        // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Multiplies a vector by the transposed right-hand matrix.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matvec_transposed_rhs_f16_f32_1024(
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
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
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
}

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32(
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

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
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
    matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<BM, BN>(
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
        KernelCompileConfig {
            occupancy,
            ..KernelCompileConfig::default()
        },
    )
}

#[cfg(feature = "dtype_f16")]
fn matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config<const BM: usize, const BN: usize>(
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
    compile: KernelCompileConfig,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    Ok(())
}

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tuned(
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
    tile_rows: usize,
    tile_columns: usize,
    compile: KernelCompileConfig,
) -> Result<()> {
    match (tile_rows, tile_columns) {
        (8, 32) => matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<8, 32>(
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
            compile,
        ),
        (16, 16) => matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<16, 16>(
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
            compile,
        ),
        (16, 32) => matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<16, 32>(
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
            compile,
        ),
        (16, 64) => matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<16, 64>(
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
            compile,
        ),
        (32, 16) => matmul_mma_transposed_rhs_f16_f32_tile_with_compile_config::<32, 16>(
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
            compile,
        ),
        _ => Err(Error::UnsupportedConfiguration {
            op: "matmul_mma_transposed_rhs_f16_f32".into(),
            reason: format!("unsupported mma tile {tile_rows}x{tile_columns}"),
        }),
    }
}

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_8x32(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_16x32(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_16x64(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_2(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_16x64_occupancy_4(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_transposed_rhs_f16_f32_tile_32x16(
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

#[cfg(feature = "dtype_f16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

#[cfg(feature = "dtype_bf16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_bf16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_bf16")]
/// Multiplies the logical matrices using the supplied dimensions and layout.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Output may be uninitialized; only the configured logical elements are written.
pub unsafe fn matmul_mma_bf16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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

/// Computes `alpha * lhs * rhs + beta * out` using the prior output values.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Prior logical output elements must be initialized even when `beta` is zero.
pub unsafe fn matmul_alpha_beta_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f16")]
/// Computes `alpha * lhs * rhs + beta * out` using the prior output values.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Prior logical output elements must be initialized even when `beta` is zero.
pub unsafe fn matmul_alpha_beta_f16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_bf16")]
/// Computes `alpha * lhs * rhs + beta * out` using the prior output values.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Prior logical output elements must be initialized even when `beta` is zero.
pub unsafe fn matmul_alpha_beta_bf16_f32(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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

#[cfg(feature = "dtype_f64")]
/// Computes `alpha * lhs * rhs + beta * out` using the prior output values.
///
/// # Safety
///
/// Pointers must identify appropriately sized, aligned ranges in the stream's context.
/// Order input initialization before device reads and keep all allocations live,
/// with exclusive writable access, until completion on every path and throughout replay.
/// Prior logical output elements must be initialized even when `beta` is zero.
pub unsafe fn matmul_alpha_beta_f64(
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
    checked_device_pointer(&out)?;
    checked_device_pointer(&lhs)?;
    checked_device_pointer(&rhs)?;

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
    // SAFETY: the caller upholds this function's safety contract.
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
