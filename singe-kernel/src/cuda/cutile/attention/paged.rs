use super::*;

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! sliding_window_attention_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            batch: usize,
            query_len: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            query_start: usize,
            key_start: usize,
            window: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_sequence_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_sequence_stride: usize,
            query_head_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
            output_head_stride: usize,
            scale: f32,
            output_scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = SlidingWindowAttention::create(
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
                query_start,
                key_start,
                window,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                scale,
                output_scale,
            )?;
            let SlidingWindowAttention {
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                query_start,
                key_start,
                window,
                output_len: output_len_i32,
                output_values_per_batch,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                grid,
            } = params;

            match head_dim {
                4 => unsafe {
                    kernel_attention::$hd4(
                        out,
                        query,
                        key,
                        value,
                        batch,
                        query_len,
                        key_len,
                        heads,
                        kv_heads,
                        query_start,
                        key_start,
                        window,
                        query_batch_stride,
                        key_batch_stride,
                        value_batch_stride,
                        output_batch_stride,
                        query_sequence_stride,
                        key_sequence_stride,
                        value_sequence_stride,
                        output_sequence_stride,
                        query_head_stride,
                        key_head_stride,
                        value_head_stride,
                        output_head_stride,
                        scale,
                        output_scale,
                        output_len_i32,
                        output_values_per_batch,
                    )
                }
                .grid(grid)
                .enqueue_on(stream)?,
                8 => unsafe {
                    kernel_attention::$hd8(
                        out,
                        query,
                        key,
                        value,
                        batch,
                        query_len,
                        key_len,
                        heads,
                        kv_heads,
                        query_start,
                        key_start,
                        window,
                        query_batch_stride,
                        key_batch_stride,
                        value_batch_stride,
                        output_batch_stride,
                        query_sequence_stride,
                        key_sequence_stride,
                        value_sequence_stride,
                        output_sequence_stride,
                        query_head_stride,
                        key_head_stride,
                        value_head_stride,
                        output_head_stride,
                        scale,
                        output_scale,
                        output_len_i32,
                        output_values_per_batch,
                    )
                }
                .grid(grid)
                .enqueue_on(stream)?,
                16 => unsafe {
                    kernel_attention::$hd16(
                        out,
                        query,
                        key,
                        value,
                        batch,
                        query_len,
                        key_len,
                        heads,
                        kv_heads,
                        query_start,
                        key_start,
                        window,
                        query_batch_stride,
                        key_batch_stride,
                        value_batch_stride,
                        output_batch_stride,
                        query_sequence_stride,
                        key_sequence_stride,
                        value_sequence_stride,
                        output_sequence_stride,
                        query_head_stride,
                        key_head_stride,
                        value_head_stride,
                        output_head_stride,
                        scale,
                        output_scale,
                        output_len_i32,
                        output_values_per_batch,
                    )
                }
                .grid(grid)
                .enqueue_on(stream)?,
                32 => unsafe {
                    kernel_attention::$hd32(
                        out,
                        query,
                        key,
                        value,
                        batch,
                        query_len,
                        key_len,
                        heads,
                        kv_heads,
                        query_start,
                        key_start,
                        window,
                        query_batch_stride,
                        key_batch_stride,
                        value_batch_stride,
                        output_batch_stride,
                        query_sequence_stride,
                        key_sequence_stride,
                        value_sequence_stride,
                        output_sequence_stride,
                        query_head_stride,
                        key_head_stride,
                        value_head_stride,
                        output_head_stride,
                        scale,
                        output_scale,
                        output_len_i32,
                        output_values_per_batch,
                    )
                }
                .grid(grid)
                .enqueue_on(stream)?,
                64 => unsafe {
                    kernel_attention::$hd64(
                        out,
                        query,
                        key,
                        value,
                        batch,
                        query_len,
                        key_len,
                        heads,
                        kv_heads,
                        query_start,
                        key_start,
                        window,
                        query_batch_stride,
                        key_batch_stride,
                        value_batch_stride,
                        output_batch_stride,
                        query_sequence_stride,
                        key_sequence_stride,
                        value_sequence_stride,
                        output_sequence_stride,
                        query_head_stride,
                        key_head_stride,
                        value_head_stride,
                        output_head_stride,
                        scale,
                        output_scale,
                        output_len_i32,
                        output_values_per_batch,
                    )
                }
                .grid(grid)
                .enqueue_on(stream)?,
                _ => {
                    return Err(Error::UnsupportedWidth {
                        op: stringify!($name).into(),
                        width: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
sliding_window_attention_half_fn!(
    sliding_window_attention_f16,
    f16,
    sliding_window_attention_f16_hd4,
    sliding_window_attention_f16_hd8,
    sliding_window_attention_f16_hd16,
    sliding_window_attention_f16_hd32,
    sliding_window_attention_f16_hd64
);
#[cfg(feature = "dtype-bf16")]
sliding_window_attention_half_fn!(
    sliding_window_attention_bf16,
    bf16,
    sliding_window_attention_bf16_hd4,
    sliding_window_attention_bf16_hd8,
    sliding_window_attention_bf16_hd16,
    sliding_window_attention_bf16_hd32,
    sliding_window_attention_bf16_hd64
);

#[cfg(feature = "dtype-f16")]
pub fn swa_attention_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    query: DevicePointer<f16>,
    key: DevicePointer<f16>,
    value: DevicePointer<f16>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    head_dim: usize,
    window_size: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if !causal {
        return Err(Error::InvalidLength);
    }
    let features = checked_element_count(heads, head_dim)?;
    let window = if window_size == 0 {
        key_len
    } else {
        window_size
    };
    sliding_window_attention_f16(
        stream,
        out,
        query,
        key,
        value,
        batch,
        query_len,
        key_len,
        heads,
        heads,
        head_dim,
        0,
        0,
        window,
        checked_element_count(query_len, features)?,
        checked_element_count(key_len, features)?,
        checked_element_count(key_len, features)?,
        checked_element_count(query_len, features)?,
        features,
        features,
        features,
        features,
        head_dim,
        head_dim,
        head_dim,
        head_dim,
        scale,
        1.0,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn swa_attention_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    query: DevicePointer<bf16>,
    key: DevicePointer<bf16>,
    value: DevicePointer<bf16>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    head_dim: usize,
    window_size: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if !causal {
        return Err(Error::InvalidLength);
    }
    let features = checked_element_count(heads, head_dim)?;
    let window = if window_size == 0 {
        key_len
    } else {
        window_size
    };
    sliding_window_attention_bf16(
        stream,
        out,
        query,
        key,
        value,
        batch,
        query_len,
        key_len,
        heads,
        heads,
        head_dim,
        0,
        0,
        window,
        checked_element_count(query_len, features)?,
        checked_element_count(key_len, features)?,
        checked_element_count(key_len, features)?,
        checked_element_count(query_len, features)?,
        features,
        features,
        features,
        features,
        head_dim,
        head_dim,
        head_dim,
        head_dim,
        scale,
        1.0,
    )
}

#[cfg(feature = "dtype-f32")]
/// Causal left-window attention over a paged KV cache.
///
/// The dense and paged variants use the same left-window mask semantics.
pub fn sliding_window_attention_paged_kv_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key_cache: DevicePointer<f32>,
    value_cache: DevicePointer<f32>,
    block_table: DevicePointer<u32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_start: usize,
    key_start: usize,
    window: usize,
    query_batch_stride: usize,
    output_batch_stride: usize,
    query_sequence_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_sequence_stride: usize,
    query_head_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    output_head_stride: usize,
    block_size: usize,
    physical_blocks: usize,
    block_table_batch_stride: usize,
    key_cache_block_stride: usize,
    value_cache_block_stride: usize,
    scale: f32,
    output_scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key_cache)?;
    checked_device_pointer(value_cache)?;
    checked_device_pointer(block_table)?;

    let params = PagedSlidingWindowAttention::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
        query_start,
        key_start,
        window,
        query_batch_stride,
        output_batch_stride,
        query_sequence_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_sequence_stride,
        query_head_stride,
        key_head_stride,
        value_head_stride,
        output_head_stride,
        block_size,
        physical_blocks,
        block_table_batch_stride,
        key_cache_block_stride,
        value_cache_block_stride,
        scale,
        output_scale,
    )?;
    let PagedSlidingWindowAttention {
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        query_start,
        key_start,
        window,
        output_len: output_len_i32,
        output_values_per_batch,
        query_batch_stride,
        output_batch_stride,
        query_sequence_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_sequence_stride,
        query_head_stride,
        key_head_stride,
        value_head_stride,
        output_head_stride,
        block_size,
        physical_blocks,
        block_table_batch_stride,
        key_cache_block_stride,
        value_cache_block_stride,
        grid,
    } = params;

    match head_dim {
        8 => unsafe {
            kernel_attention::sliding_window_attention_paged_kv_f32_hd8(
                out,
                query,
                key_cache,
                value_cache,
                block_table,
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                query_start,
                key_start,
                window,
                query_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                key_cache_block_stride,
                value_cache_block_stride,
                scale,
                output_scale,
                output_len_i32,
                output_values_per_batch,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::sliding_window_attention_paged_kv_f32_hd16(
                out,
                query,
                key_cache,
                value_cache,
                block_table,
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                query_start,
                key_start,
                window,
                query_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                key_cache_block_stride,
                value_cache_block_stride,
                scale,
                output_scale,
                output_len_i32,
                output_values_per_batch,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::sliding_window_attention_paged_kv_f32_hd32(
                out,
                query,
                key_cache,
                value_cache,
                block_table,
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                query_start,
                key_start,
                window,
                query_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                key_cache_block_stride,
                value_cache_block_stride,
                scale,
                output_scale,
                output_len_i32,
                output_values_per_batch,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::sliding_window_attention_paged_kv_f32_hd64(
                out,
                query,
                key_cache,
                value_cache,
                block_table,
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                query_start,
                key_start,
                window,
                query_batch_stride,
                output_batch_stride,
                query_sequence_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_sequence_stride,
                query_head_stride,
                key_head_stride,
                value_head_stride,
                output_head_stride,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                key_cache_block_stride,
                value_cache_block_stride,
                scale,
                output_scale,
                output_len_i32,
                output_values_per_batch,
            )
        }
        .grid(grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedWidth {
                op: "sliding_window_attention_paged_kv_f32 head_dim".into(),
                width: head_dim,
            });
        }
    };
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn paged_kv_decode_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key_cache: DevicePointer<f32>,
    value_cache: DevicePointer<f32>,
    actual_seq_lens: DevicePointer<i32>,
    block_table: DevicePointer<u32>,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    physical_blocks: usize,
    block_table_batch_stride: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key_cache)?;
    checked_device_pointer(value_cache)?;
    checked_device_pointer(actual_seq_lens)?;
    checked_device_pointer(block_table)?;

    let params = PagedKvDecodeAttention::create(
        batch,
        heads,
        kv_heads,
        head_dim,
        block_size,
        physical_blocks,
        block_table_batch_stride,
        scale,
    )?;
    macro_rules! launch {
        ($kernel:ident) => {
            unsafe {
                kernel_attention::$kernel(
                    out,
                    query,
                    key_cache,
                    value_cache,
                    actual_seq_lens,
                    block_table,
                    params.batch,
                    params.heads,
                    params.kv_heads,
                    params.block_size,
                    params.physical_blocks,
                    params.block_table_batch_stride,
                    scale,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?
        };
    }
    match head_dim {
        4 => launch!(paged_kv_decode_attention_f32_hd4),
        8 => launch!(paged_kv_decode_attention_f32_hd8),
        16 => launch!(paged_kv_decode_attention_f32_hd16),
        32 => launch!(paged_kv_decode_attention_f32_hd32),
        64 => launch!(paged_kv_decode_attention_f32_hd64),
        _ => {
            return Err(Error::UnsupportedWidth {
                op: "paged_kv_decode_attention_f32 head_dim".into(),
                width: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! paged_kv_decode_attention_half_fn {
    ($name:ident, $dtype:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            query: DevicePointer<$dtype>,
            key_cache: DevicePointer<$dtype>,
            value_cache: DevicePointer<$dtype>,
            actual_seq_lens: DevicePointer<i32>,
            block_table: DevicePointer<u32>,
            batch: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            block_size: usize,
            physical_blocks: usize,
            block_table_batch_stride: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key_cache)?;
            checked_device_pointer(value_cache)?;
            checked_device_pointer(actual_seq_lens)?;
            checked_device_pointer(block_table)?;

            let params = PagedKvDecodeAttention::create(
                batch,
                heads,
                kv_heads,
                head_dim,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                scale,
            )?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            query,
                            key_cache,
                            value_cache,
                            actual_seq_lens,
                            block_table,
                            params.batch,
                            params.heads,
                            params.kv_heads,
                            params.block_size,
                            params.physical_blocks,
                            params.block_table_batch_stride,
                            scale,
                            params.output_len,
                        )
                    }
                    .grid(params.grid)
                    .enqueue_on(stream)?
                };
            }
            match head_dim {
                4 => launch!($hd4),
                8 => launch!($hd8),
                16 => launch!($hd16),
                32 => launch!($hd32),
                64 => launch!($hd64),
                _ => {
                    return Err(Error::UnsupportedWidth {
                        op: $op.into(),
                        width: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
paged_kv_decode_attention_half_fn!(
    paged_kv_decode_attention_f16,
    f16,
    "paged_kv_decode_attention_f16 head_dim",
    paged_kv_decode_attention_f16_hd4,
    paged_kv_decode_attention_f16_hd8,
    paged_kv_decode_attention_f16_hd16,
    paged_kv_decode_attention_f16_hd32,
    paged_kv_decode_attention_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
paged_kv_decode_attention_half_fn!(
    paged_kv_decode_attention_bf16,
    bf16,
    "paged_kv_decode_attention_bf16 head_dim",
    paged_kv_decode_attention_bf16_hd4,
    paged_kv_decode_attention_bf16_hd8,
    paged_kv_decode_attention_bf16_hd16,
    paged_kv_decode_attention_bf16_hd32,
    paged_kv_decode_attention_bf16_hd64
);

#[cfg(feature = "dtype-f32")]
pub fn paged_kv_prefill_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key_cache: DevicePointer<f32>,
    value_cache: DevicePointer<f32>,
    actual_seq_lens_q: DevicePointer<i32>,
    actual_seq_lens_kv: DevicePointer<i32>,
    actual_seq_offsets: DevicePointer<i32>,
    block_table: DevicePointer<u32>,
    batch: usize,
    total_query_tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    physical_blocks: usize,
    block_table_batch_stride: usize,
    causal: bool,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key_cache)?;
    checked_device_pointer(value_cache)?;
    checked_device_pointer(actual_seq_lens_q)?;
    checked_device_pointer(actual_seq_lens_kv)?;
    checked_device_pointer(actual_seq_offsets)?;
    checked_device_pointer(block_table)?;

    let params = PagedKvPrefillAttention::create(
        batch,
        total_query_tokens,
        heads,
        kv_heads,
        head_dim,
        block_size,
        physical_blocks,
        block_table_batch_stride,
        causal,
        scale,
    )?;
    macro_rules! launch {
        ($kernel:ident) => {
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    key_cache,
                    value_cache,
                    actual_seq_lens_q,
                    actual_seq_lens_kv,
                    actual_seq_offsets,
                    block_table,
                    params.batch,
                    params.total_query_tokens,
                    params.heads,
                    params.kv_heads,
                    params.block_size,
                    params.physical_blocks,
                    params.block_table_batch_stride,
                    params.causal,
                    scale,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?
        };
    }
    match head_dim {
        4 => launch!(paged_kv_prefill_attention_f32_hd4),
        8 => launch!(paged_kv_prefill_attention_f32_hd8),
        16 => launch!(paged_kv_prefill_attention_f32_hd16),
        32 => launch!(paged_kv_prefill_attention_f32_hd32),
        64 => launch!(paged_kv_prefill_attention_f32_hd64),
        _ => {
            return Err(Error::UnsupportedWidth {
                op: "paged_kv_prefill_attention_f32 head_dim".into(),
                width: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! paged_kv_prefill_attention_half_fn {
    ($name:ident, $ty:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key_cache: DevicePointer<$ty>,
            value_cache: DevicePointer<$ty>,
            actual_seq_lens_q: DevicePointer<i32>,
            actual_seq_lens_kv: DevicePointer<i32>,
            actual_seq_offsets: DevicePointer<i32>,
            block_table: DevicePointer<u32>,
            batch: usize,
            total_query_tokens: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            block_size: usize,
            physical_blocks: usize,
            block_table_batch_stride: usize,
            causal: bool,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key_cache)?;
            checked_device_pointer(value_cache)?;
            checked_device_pointer(actual_seq_lens_q)?;
            checked_device_pointer(actual_seq_lens_kv)?;
            checked_device_pointer(actual_seq_offsets)?;
            checked_device_pointer(block_table)?;

            let params = PagedKvPrefillAttention::create(
                batch,
                total_query_tokens,
                heads,
                kv_heads,
                head_dim,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                causal,
                scale,
            )?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            lse,
                            query,
                            key_cache,
                            value_cache,
                            actual_seq_lens_q,
                            actual_seq_lens_kv,
                            actual_seq_offsets,
                            block_table,
                            params.batch,
                            params.total_query_tokens,
                            params.heads,
                            params.kv_heads,
                            params.block_size,
                            params.physical_blocks,
                            params.block_table_batch_stride,
                            params.causal,
                            scale,
                            params.output_len,
                        )
                    }
                    .grid(params.grid)
                    .enqueue_on(stream)?
                };
            }
            match head_dim {
                4 => launch!($hd4),
                8 => launch!($hd8),
                16 => launch!($hd16),
                32 => launch!($hd32),
                64 => launch!($hd64),
                _ => {
                    return Err(Error::UnsupportedWidth {
                        op: $op.into(),
                        width: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
paged_kv_prefill_attention_half_fn!(
    paged_kv_prefill_attention_f16,
    f16,
    "paged_kv_prefill_attention_f16 head_dim",
    paged_kv_prefill_attention_f16_hd4,
    paged_kv_prefill_attention_f16_hd8,
    paged_kv_prefill_attention_f16_hd16,
    paged_kv_prefill_attention_f16_hd32,
    paged_kv_prefill_attention_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
paged_kv_prefill_attention_half_fn!(
    paged_kv_prefill_attention_bf16,
    bf16,
    "paged_kv_prefill_attention_bf16 head_dim",
    paged_kv_prefill_attention_bf16_hd4,
    paged_kv_prefill_attention_bf16_hd8,
    paged_kv_prefill_attention_bf16_hd16,
    paged_kv_prefill_attention_bf16_hd32,
    paged_kv_prefill_attention_bf16_hd64
);

#[cfg(feature = "dtype-f32")]
pub fn ragged_kv_prefill_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    actual_seq_lens_q: DevicePointer<i32>,
    actual_seq_lens_kv: DevicePointer<i32>,
    actual_seq_offsets: DevicePointer<i32>,
    batch: usize,
    total_query_tokens: usize,
    total_kv_tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;
    checked_device_pointer(actual_seq_lens_q)?;
    checked_device_pointer(actual_seq_lens_kv)?;
    checked_device_pointer(actual_seq_offsets)?;

    let params = RaggedKvPrefillAttention::create(
        batch,
        total_query_tokens,
        total_kv_tokens,
        heads,
        kv_heads,
        head_dim,
        causal,
        scale,
    )?;
    macro_rules! launch {
        ($kernel:ident) => {
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    key,
                    value,
                    actual_seq_lens_q,
                    actual_seq_lens_kv,
                    actual_seq_offsets,
                    params.batch,
                    params.total_query_tokens,
                    params.total_kv_tokens,
                    params.heads,
                    params.kv_heads,
                    params.causal,
                    scale,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?
        };
    }
    match head_dim {
        4 => launch!(ragged_kv_prefill_attention_f32_hd4),
        8 => launch!(ragged_kv_prefill_attention_f32_hd8),
        16 => launch!(ragged_kv_prefill_attention_f32_hd16),
        32 => launch!(ragged_kv_prefill_attention_f32_hd32),
        64 => launch!(ragged_kv_prefill_attention_f32_hd64),
        _ => {
            return Err(Error::UnsupportedWidth {
                op: "ragged_kv_prefill_attention_f32 head_dim".into(),
                width: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! ragged_kv_prefill_attention_half_fn {
    ($name:ident, $ty:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            actual_seq_lens_q: DevicePointer<i32>,
            actual_seq_lens_kv: DevicePointer<i32>,
            actual_seq_offsets: DevicePointer<i32>,
            batch: usize,
            total_query_tokens: usize,
            total_kv_tokens: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            causal: bool,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;
            checked_device_pointer(actual_seq_lens_q)?;
            checked_device_pointer(actual_seq_lens_kv)?;
            checked_device_pointer(actual_seq_offsets)?;

            let params = RaggedKvPrefillAttention::create(
                batch,
                total_query_tokens,
                total_kv_tokens,
                heads,
                kv_heads,
                head_dim,
                causal,
                scale,
            )?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            lse,
                            query,
                            key,
                            value,
                            actual_seq_lens_q,
                            actual_seq_lens_kv,
                            actual_seq_offsets,
                            params.batch,
                            params.total_query_tokens,
                            params.total_kv_tokens,
                            params.heads,
                            params.kv_heads,
                            params.causal,
                            scale,
                            params.output_len,
                        )
                    }
                    .grid(params.grid)
                    .enqueue_on(stream)?
                };
            }
            match head_dim {
                4 => launch!($hd4),
                8 => launch!($hd8),
                16 => launch!($hd16),
                32 => launch!($hd32),
                64 => launch!($hd64),
                _ => {
                    return Err(Error::UnsupportedWidth {
                        op: $op.into(),
                        width: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
ragged_kv_prefill_attention_half_fn!(
    ragged_kv_prefill_attention_f16,
    f16,
    "ragged_kv_prefill_attention_f16 head_dim",
    ragged_kv_prefill_attention_f16_hd4,
    ragged_kv_prefill_attention_f16_hd8,
    ragged_kv_prefill_attention_f16_hd16,
    ragged_kv_prefill_attention_f16_hd32,
    ragged_kv_prefill_attention_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
ragged_kv_prefill_attention_half_fn!(
    ragged_kv_prefill_attention_bf16,
    bf16,
    "ragged_kv_prefill_attention_bf16 head_dim",
    ragged_kv_prefill_attention_bf16_hd4,
    ragged_kv_prefill_attention_bf16_hd8,
    ragged_kv_prefill_attention_bf16_hd16,
    ragged_kv_prefill_attention_bf16_hd32,
    ragged_kv_prefill_attention_bf16_hd64
);
