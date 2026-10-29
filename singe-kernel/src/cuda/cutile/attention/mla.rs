use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};

use crate::{
    cuda::cutile::{
        DeviceOpExt, kernel::attention as kernel_attention, runtime::tile_kernel::TileKernel,
        utility::checked_device_pointer,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn mla_prefill_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    query_pe: DevicePointer<f32>,
    key: DevicePointer<f32>,
    key_pe: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pe_dim: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&query)?;
    checked_device_pointer(&query_pe)?;
    checked_device_pointer(&key)?;
    checked_device_pointer(&key_pe)?;
    checked_device_pointer(&value)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        launch_mla(
            stream, out, None, query, query_pe, key, key_pe, value, None, None, None, batch,
            query_len, key_len, heads, kv_heads, head_dim, pe_dim, 0, 1, 0, 0, 1, 0, scale, 1.0, 0,
            "f32",
        )
    }?;
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! mla_prefill_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            query_pe: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            key_pe: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            batch: usize,
            query_len: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            pe_dim: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&query)?;
            checked_device_pointer(&query_pe)?;
            checked_device_pointer(&key)?;
            checked_device_pointer(&key_pe)?;
            checked_device_pointer(&value)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                launch_mla(
                    stream,
                    out,
                    None,
                    query,
                    query_pe,
                    key,
                    key_pe,
                    value,
                    None,
                    None,
                    None,
                    batch,
                    query_len,
                    key_len,
                    heads,
                    kv_heads,
                    head_dim,
                    pe_dim,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    scale,
                    1.0,
                    0,
                    stringify!($ty),
                )
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
mla_prefill_half_fn!(mla_prefill_f16, f16, mla_prefill_f16);

#[cfg(feature = "dtype_bf16")]
mla_prefill_half_fn!(mla_prefill_bf16, bf16, mla_prefill_bf16);

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn sparse_mla_prefill_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    query_pe: DevicePointer<f32>,
    key: DevicePointer<f32>,
    key_pe: DevicePointer<f32>,
    value: DevicePointer<f32>,
    indices: DevicePointer<i32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pe_dim: usize,
    topk: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&query)?;
    checked_device_pointer(&query_pe)?;
    checked_device_pointer(&key)?;
    checked_device_pointer(&key_pe)?;
    checked_device_pointer(&value)?;
    checked_device_pointer(&indices)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        launch_mla(
            stream,
            out,
            None,
            query,
            query_pe,
            key,
            key_pe,
            value,
            Some(indices),
            None,
            None,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            head_dim,
            pe_dim,
            topk,
            1,
            0,
            0,
            1,
            0,
            scale,
            1.0,
            2,
            "f32",
        )
    }?;
    Ok(())
}

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn mla_decode_lse_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    query_pe: DevicePointer<f32>,
    key_value: DevicePointer<f32>,
    key_pe: DevicePointer<f32>,
    batch: usize,
    key_len: usize,
    heads: usize,
    head_dim: usize,
    pe_dim: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lse)?;
    checked_device_pointer(&query)?;
    checked_device_pointer(&query_pe)?;
    checked_device_pointer(&key_value)?;
    checked_device_pointer(&key_pe)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        launch_mla(
            stream,
            out,
            Some(lse),
            query,
            query_pe,
            key_value,
            key_pe,
            key_value,
            None,
            None,
            None,
            batch,
            1,
            key_len,
            heads,
            1,
            head_dim,
            pe_dim,
            0,
            1,
            0,
            0,
            1,
            0,
            scale,
            1.0,
            1,
            "f32",
        )
    }?;
    Ok(())
}

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn paged_mla_decode_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    query_pe: DevicePointer<f32>,
    key_value_cache: DevicePointer<f32>,
    key_pe_cache: DevicePointer<f32>,
    actual_seq_lens: DevicePointer<i32>,
    block_table: DevicePointer<u32>,
    batch: usize,
    heads: usize,
    head_dim: usize,
    pe_dim: usize,
    block_size: usize,
    physical_blocks: usize,
    block_table_batch_stride: usize,
    scale: f32,
    output_scale: f32,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lse)?;
    checked_device_pointer(&query)?;
    checked_device_pointer(&query_pe)?;
    checked_device_pointer(&key_value_cache)?;
    checked_device_pointer(&key_pe_cache)?;
    checked_device_pointer(&actual_seq_lens)?;
    checked_device_pointer(&block_table)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        launch_mla(
            stream,
            out,
            Some(lse),
            query,
            query_pe,
            key_value_cache,
            key_pe_cache,
            key_value_cache,
            None,
            Some(actual_seq_lens),
            Some(block_table),
            batch,
            1,
            0,
            heads,
            1,
            head_dim,
            pe_dim,
            0,
            block_size,
            physical_blocks,
            block_table_batch_stride,
            1,
            0,
            scale,
            output_scale,
            3,
            "f32",
        )
    }?;
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! paged_mla_decode_attention_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            query_pe: DevicePointer<$ty>,
            key_value_cache: DevicePointer<$ty>,
            key_pe_cache: DevicePointer<$ty>,
            actual_seq_lens: DevicePointer<i32>,
            block_table: DevicePointer<u32>,
            batch: usize,
            heads: usize,
            head_dim: usize,
            pe_dim: usize,
            block_size: usize,
            physical_blocks: usize,
            block_table_batch_stride: usize,
            scale: f32,
            output_scale: f32,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&lse)?;
            checked_device_pointer(&query)?;
            checked_device_pointer(&query_pe)?;
            checked_device_pointer(&key_value_cache)?;
            checked_device_pointer(&key_pe_cache)?;
            checked_device_pointer(&actual_seq_lens)?;
            checked_device_pointer(&block_table)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                launch_mla(
                    stream,
                    out,
                    Some(lse),
                    query,
                    query_pe,
                    key_value_cache,
                    key_pe_cache,
                    key_value_cache,
                    None,
                    Some(actual_seq_lens),
                    Some(block_table),
                    batch,
                    1,
                    0,
                    heads,
                    1,
                    head_dim,
                    pe_dim,
                    0,
                    block_size,
                    physical_blocks,
                    block_table_batch_stride,
                    1,
                    0,
                    scale,
                    output_scale,
                    3,
                    stringify!($ty),
                )
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
paged_mla_decode_attention_half_fn!(
    paged_mla_decode_attention_f16,
    f16,
    paged_mla_decode_attention_f16
);

#[cfg(feature = "dtype_bf16")]
paged_mla_decode_attention_half_fn!(
    paged_mla_decode_attention_bf16,
    bf16,
    paged_mla_decode_attention_bf16
);

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! mla_decode_lse_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            query_pe: DevicePointer<$ty>,
            key_value: DevicePointer<$ty>,
            key_pe: DevicePointer<$ty>,
            batch: usize,
            key_len: usize,
            heads: usize,
            head_dim: usize,
            pe_dim: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&lse)?;
            checked_device_pointer(&query)?;
            checked_device_pointer(&query_pe)?;
            checked_device_pointer(&key_value)?;
            checked_device_pointer(&key_pe)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                launch_mla(
                    stream,
                    out,
                    Some(lse),
                    query,
                    query_pe,
                    key_value,
                    key_pe,
                    key_value,
                    None,
                    None,
                    None,
                    batch,
                    1,
                    key_len,
                    heads,
                    1,
                    head_dim,
                    pe_dim,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    scale,
                    1.0,
                    1,
                    stringify!($ty),
                )
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
mla_decode_lse_half_fn!(mla_decode_lse_f16, f16, mla_decode_lse_f16);

#[cfg(feature = "dtype_bf16")]
mla_decode_lse_half_fn!(mla_decode_lse_bf16, bf16, mla_decode_lse_bf16);

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn mla_decode_splitk_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    query_pe: DevicePointer<f32>,
    key_value: DevicePointer<f32>,
    key_pe: DevicePointer<f32>,
    batch: usize,
    key_len: usize,
    heads: usize,
    splits: usize,
    kv_len_per_split: usize,
    head_dim: usize,
    pe_dim: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&lse)?;
    checked_device_pointer(&query)?;
    checked_device_pointer(&query_pe)?;
    checked_device_pointer(&key_value)?;
    checked_device_pointer(&key_pe)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        launch_mla(
            stream,
            out,
            Some(lse),
            query,
            query_pe,
            key_value,
            key_pe,
            key_value,
            None,
            None,
            None,
            batch,
            1,
            key_len,
            heads,
            1,
            head_dim,
            pe_dim,
            0,
            1,
            0,
            0,
            splits,
            kv_len_per_split,
            scale,
            1.0,
            4,
            "f32",
        )
    }?;
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! mla_decode_splitk_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<f32>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            query_pe: DevicePointer<$ty>,
            key_value: DevicePointer<$ty>,
            key_pe: DevicePointer<$ty>,
            batch: usize,
            key_len: usize,
            heads: usize,
            splits: usize,
            kv_len_per_split: usize,
            head_dim: usize,
            pe_dim: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&lse)?;
            checked_device_pointer(&query)?;
            checked_device_pointer(&query_pe)?;
            checked_device_pointer(&key_value)?;
            checked_device_pointer(&key_pe)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                launch_mla(
                    stream,
                    out,
                    Some(lse),
                    query,
                    query_pe,
                    key_value,
                    key_pe,
                    key_value,
                    None,
                    None,
                    None,
                    batch,
                    1,
                    key_len,
                    heads,
                    1,
                    head_dim,
                    pe_dim,
                    0,
                    1,
                    0,
                    0,
                    splits,
                    kv_len_per_split,
                    scale,
                    1.0,
                    4,
                    stringify!($ty),
                )
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
mla_decode_splitk_half_fn!(mla_decode_splitk_f16, f16, mla_decode_splitk_f16);

#[cfg(feature = "dtype_bf16")]
mla_decode_splitk_half_fn!(mla_decode_splitk_bf16, bf16, mla_decode_splitk_bf16);

/// Every optional pointer is replaced by an unused alias for specialization modes
/// that do not read it; all actual reads remain covered by the raw caller's contract.
unsafe fn launch_mla<
    T: ::cutile::core::ElementType + ::cutile::cuda_core::DType + Send,
    O: ::cutile::core::ElementType + ::cutile::cuda_core::DType + Send,
>(
    stream: &Arc<Stream>,
    out: DevicePointer<O>,
    lse: Option<DevicePointer<f32>>,
    query: DevicePointer<T>,
    query_pe: DevicePointer<T>,
    key: DevicePointer<T>,
    key_pe: DevicePointer<T>,
    value: DevicePointer<T>,
    indices: Option<DevicePointer<i32>>,
    lengths: Option<DevicePointer<i32>>,
    table: Option<DevicePointer<u32>>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pe_dim: usize,
    topk: usize,
    block_size: usize,
    physical_blocks: usize,
    table_stride: usize,
    splits: usize,
    per_split: usize,
    scale: f32,
    output_scale: f32,
    mode: i32,
    dtype: &str,
) -> Result<()> {
    if batch == 0
        || query_len == 0
        || heads == 0
        || kv_heads == 0
        || head_dim == 0
        || pe_dim == 0
        || !heads.is_multiple_of(kv_heads)
        || !scale.is_finite()
        || !output_scale.is_finite()
        || (mode != 3 && key_len == 0)
        || (mode == 2 && topk == 0)
        || (mode == 3 && (block_size == 0 || physical_blocks == 0 || table_stride == 0))
        || splits == 0
        || (mode == 4 && per_split == 0)
    {
        return Err(Error::InvalidLength);
    }
    let query_rows = checked_element_count(checked_element_count(batch, query_len)?, heads)?;
    let kv_rows = if mode == 3 {
        checked_element_count(physical_blocks, block_size)?
    } else {
        checked_element_count(checked_element_count(batch, key_len)?, kv_heads)?
    };
    for width in [head_dim, pe_dim] {
        checked_i32_value(checked_element_count(query_rows, width)?)?;
        checked_i32_value(checked_element_count(kv_rows, width)?)?;
    }
    checked_i32_value(checked_element_count(
        checked_element_count(checked_element_count(batch, query_len)?, kv_heads)?,
        topk,
    )?)?;
    checked_i32_value(checked_element_count(
        checked_element_count(checked_element_count(batch, heads)?, splits)?,
        head_dim,
    )?)?;
    checked_i32_value(checked_element_count(batch, table_stride)?)?;
    checked_i32_value(checked_element_count(table_stride, block_size)?)?;
    let bm = if mode == 0 { 16 } else { 1 };
    let d = head_dim
        .max(32)
        .checked_next_power_of_two()
        .ok_or(Error::SizeOverflow)?
        .min(512);
    let p = pe_dim
        .max(32)
        .checked_next_power_of_two()
        .ok_or(Error::SizeOverflow)?
        .min(128);
    let grid_x = checked_element_count(batch, heads)?;
    let grid_z = checked_element_count(splits, head_dim.div_ceil(d))?;
    // SAFETY: these aliases are never dereferenced by modes whose arguments are absent.
    let unused_f32 = unsafe { DevicePointer::from_cu_deviceptr(query.cu_deviceptr()) };
    // SAFETY: same unused-alias contract as above.
    let unused_i32 = unsafe { DevicePointer::from_cu_deviceptr(query.cu_deviceptr()) };
    // SAFETY: same unused-alias contract as above.
    let unused_u32 = unsafe { DevicePointer::from_cu_deviceptr(query.cu_deviceptr()) };
    // SAFETY: the wrappers validated the layout, and the caller retains all disjoint buffers.
    unsafe {
        kernel_attention::mla_tiled::mla(
            out,
            lse.unwrap_or(unused_f32),
            query,
            query_pe,
            key,
            key_pe,
            value,
            indices.unwrap_or(unused_i32),
            lengths.unwrap_or(unused_i32),
            table.unwrap_or(unused_u32),
            checked_i32_value(query_len)?,
            checked_i32_value(key_len)?,
            checked_i32_value(heads)?,
            checked_i32_value(kv_heads)?,
            checked_i32_value(head_dim)?,
            checked_i32_value(pe_dim)?,
            checked_i32_value(topk)?,
            checked_i32_value(block_size)?,
            checked_i32_value(physical_blocks)?,
            checked_i32_value(table_stride)?,
            checked_i32_value(splits)?,
            checked_i32_value(per_split)?,
            scale,
            output_scale,
        )
    }
    .generics(vec![
        dtype.into(),
        if mode == 4 {
            "f32".into()
        } else {
            dtype.into()
        },
        bm.to_string(),
        d.to_string(),
        p.to_string(),
        mode.to_string(),
    ])
    .grid((
        u32::try_from(grid_x).map_err(|_| Error::SizeOverflow)?,
        u32::try_from(query_len.div_ceil(bm)).map_err(|_| Error::SizeOverflow)?,
        u32::try_from(grid_z).map_err(|_| Error::SizeOverflow)?,
    ))
    .enqueue_on(stream)?;
    Ok(())
}
