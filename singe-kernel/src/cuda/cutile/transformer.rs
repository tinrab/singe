//! Raw fused transformer launch helpers.

use std::sync::Arc;

use cutile::{cuda_core::Stream, half::f16, tile_kernel::DevicePointer};

use crate::{
    cuda::{
        cutile::{
            DeviceOpExt, TileKernel, adapter::TensorAdapter, kernel::transformer as kernel,
            utility::checked_device_pointer,
        },
        fused::QkNormRopeConfig,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

/// Enqueues Q/K normalization, half-split RoPE, and optional KV-cache updates.
///
/// # Safety
/// Pointers must address correctly sized, nonaliasing buffers on the stream's device.
/// Keep the buffers alive and order all accesses until completion and throughout graph replay.
/// Device positions outside the table or cache capacity suppress writes.
pub unsafe fn qk_norm_rope_f16(
    stream: &Arc<Stream>,
    q_out: DevicePointer<f16>,
    k_out: DevicePointer<f16>,
    query: DevicePointer<f16>,
    key: DevicePointer<f16>,
    value: DevicePointer<f16>,
    q_weight: DevicePointer<f16>,
    k_weight: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    table_rows: usize,
    position: Option<DevicePointer<u32>>,
    cache: Option<(DevicePointer<f16>, DevicePointer<f16>, usize)>,
    config: QkNormRopeConfig,
) -> Result<()> {
    let c = config;
    if c.seq_len == 0
        || c.heads == 0
        || c.kv_heads == 0
        || c.head_dim == 0
        || !c.head_dim.is_multiple_of(2)
        || c.head_dim > 2048
        || !c.eps.is_finite()
        || c.eps < 0.0
    {
        return Err(Error::InvalidLength);
    }
    let dynamic = position.is_some();
    let cache_enabled = cache.is_some();
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let null = unsafe { DevicePointer::from_cu_deviceptr(0) };
    for pointer in [q_out, k_out, query, key, q_weight, k_weight] {
        checked_device_pointer(&pointer)?;
    }
    checked_device_pointer(&cos)?;
    checked_device_pointer(&sin)?;
    if let Some(position) = position {
        checked_device_pointer(&position)?;
    }
    if !dynamic
        && c.table_start
            .checked_add(c.seq_len)
            .ok_or(Error::SizeOverflow)?
            > table_rows
    {
        return Err(Error::InvalidLength);
    }
    let (k_cache, v_cache, max_seq) = cache.unwrap_or((null, null, 0));
    if cache_enabled {
        checked_device_pointer(&k_cache)?;
        checked_device_pointer(&v_cache)?;
        checked_device_pointer(&value)?;
        checked_i32_value(checked_element_count(
            checked_element_count(c.kv_heads, max_seq)?,
            c.head_dim,
        )?)?;
        if !dynamic
            && c.position_start
                .checked_add(c.seq_len)
                .ok_or(Error::SizeOverflow)?
                > max_seq
        {
            return Err(Error::InvalidLength);
        }
    }
    checked_i32_value(checked_element_count(
        checked_element_count(c.seq_len, c.heads.max(c.kv_heads))?,
        c.head_dim,
    )?)?;
    checked_i32_value(checked_element_count(table_rows, c.head_dim / 2)?)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel::qk_norm_rope_f16(
            query,
            key,
            value,
            q_weight,
            k_weight,
            q_out,
            k_out,
            k_cache,
            v_cache,
            cos,
            sin,
            position.unwrap_or_else(|| DevicePointer::from_cu_deviceptr(0)),
            checked_i32_value(c.heads)?,
            checked_i32_value(c.kv_heads)?,
            checked_i32_value(c.head_dim)?,
            checked_i32_value(table_rows)?,
            checked_i32_value(c.position_start)?,
            checked_i32_value(c.table_start)?,
            checked_i32_value(max_seq)?,
            c.eps,
        )
    }
    .generics(vec![
        c.head_dim.next_power_of_two().to_string(),
        i32::from(cache_enabled).to_string(),
        i32::from(dynamic).to_string(),
    ])
    .grid((
        u32::try_from(c.heads.checked_add(c.kv_heads).ok_or(Error::SizeOverflow)?)
            .map_err(|_| Error::SizeOverflow)?,
        u32::try_from(c.seq_len).map_err(|_| Error::SizeOverflow)?,
        1,
    ))
    .enqueue_on(stream)?;
    Ok(())
}

/// Enqueues LM-head matvec and writes one maximum and vocabulary index per 64 rows.
///
/// # Safety
/// Buffers must have sufficient capacity on the stream's device and remain alive until completion.
/// Order writes and reads across streams and retain every allocation through graph replay.
pub unsafe fn lm_head_argmax_f16(
    stream: &Arc<Stream>,
    block_max: DevicePointer<f32>,
    block_idx: DevicePointer<i32>,
    weights: DevicePointer<f16>,
    hidden: DevicePointer<f16>,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<()> {
    if vocab_size == 0 || hidden_size == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(&block_max)?;
    checked_device_pointer(&block_idx)?;
    checked_device_pointer(&weights)?;
    checked_device_pointer(&hidden)?;
    checked_element_count(vocab_size, hidden_size)?;
    let weights = TensorAdapter::contiguous_2d(weights, vocab_size, hidden_size)?;
    let hidden = TensorAdapter::contiguous_2d(hidden, 1, hidden_size)?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel::lm_head_argmax_f16(
            weights,
            hidden,
            block_max,
            block_idx,
            checked_i32_value(vocab_size)?,
            checked_i32_value(hidden_size)?,
        )
    }
    .grid((
        u32::try_from(vocab_size.div_ceil(64)).map_err(|_| Error::SizeOverflow)?,
        1,
        1,
    ))
    .enqueue_on(stream)?;
    Ok(())
}
