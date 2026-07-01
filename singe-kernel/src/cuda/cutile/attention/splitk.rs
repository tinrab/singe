use super::*;

pub fn splitk_reduce_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    attn_splitk: DevicePointer<f32>,
    lse_splitk: DevicePointer<f32>,
    batch: usize,
    heads: usize,
    splits: usize,
    head_dim: usize,
    attn_batch_stride: usize,
    attn_head_stride: usize,
    attn_split_stride: usize,
    lse_batch_stride: usize,
    lse_head_stride: usize,
    output_batch_stride: usize,
    output_head_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(attn_splitk)?;
    checked_device_pointer(lse_splitk)?;

    let params = SplitKReduce::create(
        batch,
        heads,
        splits,
        head_dim,
        attn_batch_stride,
        attn_head_stride,
        attn_split_stride,
        lse_batch_stride,
        lse_head_stride,
        output_batch_stride,
        output_head_stride,
    )?;
    unsafe {
        kernel_attention::splitk_reduce_f32(
            out,
            attn_splitk,
            lse_splitk,
            params.batch,
            params.heads,
            params.splits,
            params.head_dim,
            params.attn_batch_stride,
            params.attn_head_stride,
            params.attn_split_stride,
            params.lse_batch_stride,
            params.lse_head_stride,
            params.output_batch_stride,
            params.output_head_stride,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn fmha_decode_splitk_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    splits: usize,
    kv_len_per_split: usize,
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    output_split_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    lse_batch_stride: usize,
    lse_head_stride: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    let params = FmhaDecodeSplitK::create(
        batch,
        key_len,
        heads,
        kv_heads,
        splits,
        kv_len_per_split,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        output_split_stride,
        key_head_stride,
        value_head_stride,
        lse_batch_stride,
        lse_head_stride,
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
                    params.batch,
                    params.key_len,
                    params.heads,
                    params.kv_heads,
                    params.splits,
                    params.kv_len_per_split,
                    params.query_batch_stride,
                    params.key_batch_stride,
                    params.value_batch_stride,
                    params.output_batch_stride,
                    params.query_head_stride,
                    params.key_sequence_stride,
                    params.value_sequence_stride,
                    params.output_head_stride,
                    params.output_split_stride,
                    params.key_head_stride,
                    params.value_head_stride,
                    params.lse_batch_stride,
                    params.lse_head_stride,
                    scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?
        };
    }
    match head_dim {
        4 => launch!(fmha_decode_splitk_f32_hd4),
        8 => launch!(fmha_decode_splitk_f32_hd8),
        16 => launch!(fmha_decode_splitk_f32_hd16),
        32 => launch!(fmha_decode_splitk_f32_hd32),
        64 => launch!(fmha_decode_splitk_f32_hd64),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_decode_splitk_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

pub fn softcapped_window_decode_splitk_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    splits: usize,
    kv_len_per_split: usize,
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    output_split_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    lse_batch_stride: usize,
    lse_head_stride: usize,
    scale: f32,
    window_size: usize,
    soft_cap: Option<f32>,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    if let Some(soft_cap) = soft_cap
        && (!soft_cap.is_finite() || soft_cap <= 0.0)
    {
        return Err(Error::InvalidLength);
    }

    let params = FmhaDecodeSplitK::create(
        batch,
        key_len,
        heads,
        kv_heads,
        splits,
        kv_len_per_split,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        output_split_stride,
        key_head_stride,
        value_head_stride,
        lse_batch_stride,
        lse_head_stride,
        scale,
    )?;
    let soft_cap_value = soft_cap.unwrap_or(0.0);
    let has_soft_cap = i32::from(soft_cap.is_some());
    let window_size = checked_i32_value(window_size)?;
    match head_dim {
        4 => unsafe {
            kernel_attention::softcapped_window_decode_splitk_f32_hd4(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.splits,
                params.kv_len_per_split,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.output_split_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.lse_batch_stride,
                params.lse_head_stride,
                scale,
                window_size,
                soft_cap_value,
                has_soft_cap,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        8 => unsafe {
            kernel_attention::softcapped_window_decode_splitk_f32_hd8(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.splits,
                params.kv_len_per_split,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.output_split_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.lse_batch_stride,
                params.lse_head_stride,
                scale,
                window_size,
                soft_cap_value,
                has_soft_cap,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::softcapped_window_decode_splitk_f32_hd16(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.splits,
                params.kv_len_per_split,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.output_split_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.lse_batch_stride,
                params.lse_head_stride,
                scale,
                window_size,
                soft_cap_value,
                has_soft_cap,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::softcapped_window_decode_splitk_f32_hd32(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.splits,
                params.kv_len_per_split,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.output_split_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.lse_batch_stride,
                params.lse_head_stride,
                scale,
                window_size,
                soft_cap_value,
                has_soft_cap,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::softcapped_window_decode_splitk_f32_hd64(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.splits,
                params.kv_len_per_split,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.output_split_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.lse_batch_stride,
                params.lse_head_stride,
                scale,
                window_size,
                soft_cap_value,
                has_soft_cap,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "softcapped_window_decode_splitk_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! softcapped_window_decode_splitk_half_fn {
    ($name:ident, $dtype:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<f32>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$dtype>,
            key: DevicePointer<$dtype>,
            value: DevicePointer<$dtype>,
            batch: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            splits: usize,
            kv_len_per_split: usize,
            head_dim: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_head_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_head_stride: usize,
            output_split_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
            lse_batch_stride: usize,
            lse_head_stride: usize,
            scale: f32,
            window_size: usize,
            soft_cap: Option<f32>,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            if let Some(soft_cap) = soft_cap
                && (!soft_cap.is_finite() || soft_cap <= 0.0)
            {
                return Err(Error::InvalidLength);
            }

            let params = FmhaDecodeSplitK::create(
                batch,
                key_len,
                heads,
                kv_heads,
                splits,
                kv_len_per_split,
                head_dim,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_head_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_head_stride,
                output_split_stride,
                key_head_stride,
                value_head_stride,
                lse_batch_stride,
                lse_head_stride,
                scale,
            )?;
            let soft_cap_value = soft_cap.unwrap_or(0.0);
            let has_soft_cap = i32::from(soft_cap.is_some());
            let window_size = checked_i32_value(window_size)?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            lse,
                            query,
                            key,
                            value,
                            params.batch,
                            params.key_len,
                            params.heads,
                            params.kv_heads,
                            params.splits,
                            params.kv_len_per_split,
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_head_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_head_stride,
                            params.output_split_stride,
                            params.key_head_stride,
                            params.value_head_stride,
                            params.lse_batch_stride,
                            params.lse_head_stride,
                            scale,
                            window_size,
                            soft_cap_value,
                            has_soft_cap,
                            params.output_len,
                            params.output_values_per_batch,
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
                    return Err(Error::UnsupportedParameter {
                        op: $op.into(),
                        parameter: "head_dim".into(),
                        value: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
softcapped_window_decode_splitk_half_fn!(
    softcapped_window_decode_splitk_f16,
    f16,
    "softcapped_window_decode_splitk_f16",
    softcapped_window_decode_splitk_f16_hd4,
    softcapped_window_decode_splitk_f16_hd8,
    softcapped_window_decode_splitk_f16_hd16,
    softcapped_window_decode_splitk_f16_hd32,
    softcapped_window_decode_splitk_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
softcapped_window_decode_splitk_half_fn!(
    softcapped_window_decode_splitk_bf16,
    bf16,
    "softcapped_window_decode_splitk_bf16",
    softcapped_window_decode_splitk_bf16_hd4,
    softcapped_window_decode_splitk_bf16_hd8,
    softcapped_window_decode_splitk_bf16_hd16,
    softcapped_window_decode_splitk_bf16_hd32,
    softcapped_window_decode_splitk_bf16_hd64
);

pub fn attention_sink_decode_splitk_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    sinks: DevicePointer<f32>,
    batch: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    splits: usize,
    kv_len_per_split: usize,
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    output_split_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    lse_batch_stride: usize,
    lse_head_stride: usize,
    start_q: usize,
    window: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;
    checked_device_pointer(sinks)?;

    let params = FmhaDecodeSplitK::create(
        batch,
        key_len,
        heads,
        kv_heads,
        splits,
        kv_len_per_split,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        output_split_stride,
        key_head_stride,
        value_head_stride,
        lse_batch_stride,
        lse_head_stride,
        scale,
    )?;
    let start_q = checked_i32_value(start_q)?;
    let window = checked_i32_value(window)?;
    macro_rules! launch {
        ($kernel:ident) => {
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    key,
                    value,
                    sinks,
                    params.batch,
                    params.key_len,
                    params.heads,
                    params.kv_heads,
                    params.splits,
                    params.kv_len_per_split,
                    params.query_batch_stride,
                    params.key_batch_stride,
                    params.value_batch_stride,
                    params.output_batch_stride,
                    params.query_head_stride,
                    params.key_sequence_stride,
                    params.value_sequence_stride,
                    params.output_head_stride,
                    params.output_split_stride,
                    params.key_head_stride,
                    params.value_head_stride,
                    params.lse_batch_stride,
                    params.lse_head_stride,
                    start_q,
                    window,
                    scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?
        };
    }
    match head_dim {
        4 => launch!(attention_sink_decode_splitk_f32_hd4),
        8 => launch!(attention_sink_decode_splitk_f32_hd8),
        16 => launch!(attention_sink_decode_splitk_f32_hd16),
        32 => launch!(attention_sink_decode_splitk_f32_hd32),
        64 => launch!(attention_sink_decode_splitk_f32_hd64),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "attention_sink_decode_splitk_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! fmha_decode_splitk_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<f32>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            batch: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            splits: usize,
            kv_len_per_split: usize,
            head_dim: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_head_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_head_stride: usize,
            output_split_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
            lse_batch_stride: usize,
            lse_head_stride: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = FmhaDecodeSplitK::create(
                batch,
                key_len,
                heads,
                kv_heads,
                splits,
                kv_len_per_split,
                head_dim,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_head_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_head_stride,
                output_split_stride,
                key_head_stride,
                value_head_stride,
                lse_batch_stride,
                lse_head_stride,
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
                            params.batch,
                            params.key_len,
                            params.heads,
                            params.kv_heads,
                            params.splits,
                            params.kv_len_per_split,
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_head_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_head_stride,
                            params.output_split_stride,
                            params.key_head_stride,
                            params.value_head_stride,
                            params.lse_batch_stride,
                            params.lse_head_stride,
                            scale,
                            params.output_len,
                            params.output_values_per_batch,
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
                    return Err(Error::UnsupportedParameter {
                        op: stringify!($name).into(),
                        parameter: "head_dim".into(),
                        value: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
fmha_decode_splitk_half_fn!(
    fmha_decode_splitk_f16,
    f16,
    fmha_decode_splitk_f16_hd4,
    fmha_decode_splitk_f16_hd8,
    fmha_decode_splitk_f16_hd16,
    fmha_decode_splitk_f16_hd32,
    fmha_decode_splitk_f16_hd64
);
#[cfg(feature = "dtype-bf16")]
fmha_decode_splitk_half_fn!(
    fmha_decode_splitk_bf16,
    bf16,
    fmha_decode_splitk_bf16_hd4,
    fmha_decode_splitk_bf16_hd8,
    fmha_decode_splitk_bf16_hd16,
    fmha_decode_splitk_bf16_hd32,
    fmha_decode_splitk_bf16_hd64
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! attention_sink_decode_splitk_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<f32>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            sinks: DevicePointer<f32>,
            batch: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            splits: usize,
            kv_len_per_split: usize,
            head_dim: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_head_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_head_stride: usize,
            output_split_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
            lse_batch_stride: usize,
            lse_head_stride: usize,
            start_q: usize,
            window: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;
            checked_device_pointer(sinks)?;

            let params = FmhaDecodeSplitK::create(
                batch,
                key_len,
                heads,
                kv_heads,
                splits,
                kv_len_per_split,
                head_dim,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_head_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_head_stride,
                output_split_stride,
                key_head_stride,
                value_head_stride,
                lse_batch_stride,
                lse_head_stride,
                scale,
            )?;
            let start_q = checked_i32_value(start_q)?;
            let window = checked_i32_value(window)?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            lse,
                            query,
                            key,
                            value,
                            sinks,
                            params.batch,
                            params.key_len,
                            params.heads,
                            params.kv_heads,
                            params.splits,
                            params.kv_len_per_split,
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_head_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_head_stride,
                            params.output_split_stride,
                            params.key_head_stride,
                            params.value_head_stride,
                            params.lse_batch_stride,
                            params.lse_head_stride,
                            start_q,
                            window,
                            scale,
                            params.output_len,
                            params.output_values_per_batch,
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
                    return Err(Error::UnsupportedParameter {
                        op: stringify!($name).into(),
                        parameter: "head_dim".into(),
                        value: head_dim,
                    });
                }
            };
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
attention_sink_decode_splitk_half_fn!(
    attention_sink_decode_splitk_f16,
    f16,
    attention_sink_decode_splitk_f16_hd4,
    attention_sink_decode_splitk_f16_hd8,
    attention_sink_decode_splitk_f16_hd16,
    attention_sink_decode_splitk_f16_hd32,
    attention_sink_decode_splitk_f16_hd64
);
#[cfg(feature = "dtype-bf16")]
attention_sink_decode_splitk_half_fn!(
    attention_sink_decode_splitk_bf16,
    bf16,
    attention_sink_decode_splitk_bf16_hd4,
    attention_sink_decode_splitk_bf16_hd8,
    attention_sink_decode_splitk_bf16_hd16,
    attention_sink_decode_splitk_bf16_hd32,
    attention_sink_decode_splitk_bf16_hd64
);
