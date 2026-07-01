use super::*;

pub fn fmha_prefill_lse_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
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
    causal: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    let params = FmhaPrefill::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
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
        causal,
    )?;
    match head_dim {
        4 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd4(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        8 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd8(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd16(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd32(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd64(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        128 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd128(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        256 => unsafe {
            kernel_attention::fmha_prefill_lse_f32_hd256(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_prefill_lse_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

pub fn fmha_prefill_lse_mma_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
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
    causal: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    FmhaPrefill::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
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
        causal,
    )?;

    let q_features = checked_element_count(heads, head_dim)?;
    let kv_features = checked_element_count(kv_heads, head_dim)?;
    if query_batch_stride != checked_element_count(query_len, q_features)?
        || key_batch_stride != checked_element_count(key_len, kv_features)?
        || value_batch_stride != checked_element_count(key_len, kv_features)?
        || output_batch_stride != checked_element_count(query_len, q_features)?
        || query_sequence_stride != q_features
        || key_sequence_stride != kv_features
        || value_sequence_stride != kv_features
        || output_sequence_stride != q_features
        || query_head_stride != head_dim
        || key_head_stride != head_dim
        || value_head_stride != head_dim
        || output_head_stride != head_dim
    {
        return Err(Error::InvalidLength);
    }

    let batch_heads = checked_element_count(batch, heads)?;
    let query_group_size = checked_i32_value(heads / kv_heads)?;
    let output_batch_stride = checked_i32_value(output_batch_stride)?;
    let output_sequence_stride = checked_i32_value(output_sequence_stride)?;
    let output_head_stride = checked_i32_value(output_head_stride)?;
    let lse = TensorAdapter::contiguous_2d(lse, batch_heads, query_len)?.partition([1, 16])?;
    let query = TensorAdapter::contiguous_4d(query, batch, query_len, heads, head_dim)?;
    let key = TensorAdapter::contiguous_4d(key, batch, key_len, kv_heads, head_dim)?;
    let value = TensorAdapter::contiguous_4d(value, batch, key_len, kv_heads, head_dim)?;
    let head_dim_generic = match head_dim {
        4 | 8 | 16 | 32 | 64 => head_dim.to_string(),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_prefill_lse_mma_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    unsafe {
        kernel_attention::fmha_prefill_lse_mma_f32(
            out,
            lse,
            query,
            key,
            value,
            output_batch_stride,
            output_sequence_stride,
            output_head_stride,
            scale,
            i32::from(causal),
            query_group_size,
        )
    }
    .generics(vec!["32".into(), head_dim_generic])
    .enqueue_on(stream)?;
    Ok(())
}

pub fn fmha_prefill_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
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
    causal: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    let params = FmhaPrefill::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
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
        causal,
    )?;
    match head_dim {
        4 => unsafe {
            kernel_attention::fmha_prefill_f32_hd4(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        8 => unsafe {
            kernel_attention::fmha_prefill_f32_hd8(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::fmha_prefill_f32_hd16(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::fmha_prefill_f32_hd32(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::fmha_prefill_f32_hd64(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        128 => unsafe {
            kernel_attention::fmha_prefill_f32_hd128(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        256 => unsafe {
            kernel_attention::fmha_prefill_f32_hd256(
                out,
                query,
                key,
                value,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_prefill_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

pub fn attention_sink_prefill_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    sinks: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
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
    start_q: usize,
    window: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;
    checked_device_pointer(sinks)?;

    let params = FmhaPrefill::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
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
        causal,
    )?;
    let start_q = checked_i32_value(start_q)?;
    let window = checked_i32_value(window)?;
    match head_dim {
        4 => unsafe {
            kernel_attention::attention_sink_prefill_f32_hd4(
                out,
                query,
                key,
                value,
                sinks,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                start_q,
                window,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        8 => unsafe {
            kernel_attention::attention_sink_prefill_f32_hd8(
                out,
                query,
                key,
                value,
                sinks,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                start_q,
                window,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::attention_sink_prefill_f32_hd16(
                out,
                query,
                key,
                value,
                sinks,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                start_q,
                window,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::attention_sink_prefill_f32_hd32(
                out,
                query,
                key,
                value,
                sinks,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                start_q,
                window,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::attention_sink_prefill_f32_hd64(
                out,
                query,
                key,
                value,
                sinks,
                params.batch,
                params.query_len,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_sequence_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_sequence_stride,
                params.query_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                params.output_head_stride,
                start_q,
                window,
                scale,
                params.causal,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "attention_sink_prefill_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

pub fn softcapped_window_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    value: DevicePointer<f32>,
    batch: usize,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
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
    causal: bool,
    window_size: usize,
    soft_cap: Option<f32>,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    if let Some(soft_cap) = soft_cap
        && (!soft_cap.is_finite() || soft_cap <= 0.0)
    {
        return Err(Error::InvalidLength);
    }

    let params = FmhaPrefill::create(
        batch,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
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
        causal,
    )?;
    let soft_cap_value = soft_cap.unwrap_or(0.0);
    let has_soft_cap = i32::from(soft_cap.is_some());
    let window_size = checked_i32_value(window_size)?;
    macro_rules! launch {
        ($kernel:ident) => {
            unsafe {
                kernel_attention::$kernel(
                    out,
                    query,
                    key,
                    value,
                    params.batch,
                    params.query_len,
                    params.key_len,
                    params.heads,
                    params.kv_heads,
                    params.query_batch_stride,
                    params.key_batch_stride,
                    params.value_batch_stride,
                    params.output_batch_stride,
                    params.query_sequence_stride,
                    params.key_sequence_stride,
                    params.value_sequence_stride,
                    params.output_sequence_stride,
                    params.query_head_stride,
                    params.key_head_stride,
                    params.value_head_stride,
                    params.output_head_stride,
                    scale,
                    params.causal,
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
        4 => launch!(softcapped_window_attention_f32_hd4),
        8 => launch!(softcapped_window_attention_f32_hd8),
        16 => launch!(softcapped_window_attention_f32_hd16),
        32 => launch!(softcapped_window_attention_f32_hd32),
        64 => launch!(softcapped_window_attention_f32_hd64),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "softcapped_window_attention_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! softcapped_window_attention_half_fn {
    ($name:ident, $dtype:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            query: DevicePointer<$dtype>,
            key: DevicePointer<$dtype>,
            value: DevicePointer<$dtype>,
            batch: usize,
            query_len: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
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
            causal: bool,
            window_size: usize,
            soft_cap: Option<f32>,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            if let Some(soft_cap) = soft_cap
                && (!soft_cap.is_finite() || soft_cap <= 0.0)
            {
                return Err(Error::InvalidLength);
            }

            let params = FmhaPrefill::create(
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
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
                causal,
            )?;
            let soft_cap_value = soft_cap.unwrap_or(0.0);
            let has_soft_cap = i32::from(soft_cap.is_some());
            let window_size = checked_i32_value(window_size)?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            query,
                            key,
                            value,
                            params.batch,
                            params.query_len,
                            params.key_len,
                            params.heads,
                            params.kv_heads,
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_sequence_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_sequence_stride,
                            params.query_head_stride,
                            params.key_head_stride,
                            params.value_head_stride,
                            params.output_head_stride,
                            scale,
                            params.causal,
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
softcapped_window_attention_half_fn!(
    softcapped_window_attention_f16,
    f16,
    "softcapped_window_attention_f16",
    softcapped_window_attention_f16_hd4,
    softcapped_window_attention_f16_hd8,
    softcapped_window_attention_f16_hd16,
    softcapped_window_attention_f16_hd32,
    softcapped_window_attention_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
softcapped_window_attention_half_fn!(
    softcapped_window_attention_bf16,
    bf16,
    "softcapped_window_attention_bf16",
    softcapped_window_attention_bf16_hd4,
    softcapped_window_attention_bf16_hd8,
    softcapped_window_attention_bf16_hd16,
    softcapped_window_attention_bf16_hd32,
    softcapped_window_attention_bf16_hd64
);

pub fn fmha_decode_lse_f32(
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
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;

    let params = FmhaDecode::create(
        batch,
        key_len,
        heads,
        kv_heads,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        key_head_stride,
        value_head_stride,
        scale,
    )?;
    match head_dim {
        4 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd4(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        8 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd8(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        16 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd16(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        32 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd32(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        64 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd64(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        128 => unsafe {
            kernel_attention::fmha_decode_lse_f32_hd128(
                out,
                lse,
                query,
                key,
                value,
                params.batch,
                params.key_len,
                params.heads,
                params.kv_heads,
                params.query_batch_stride,
                params.key_batch_stride,
                params.value_batch_stride,
                params.output_batch_stride,
                params.query_head_stride,
                params.key_sequence_stride,
                params.value_sequence_stride,
                params.output_head_stride,
                params.key_head_stride,
                params.value_head_stride,
                scale,
                params.output_len,
                params.output_values_per_batch,
            )
        }
        .grid(params.grid)
        .enqueue_on(stream)?,
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_decode_lse_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

pub fn softcapped_window_decode_lse_f32(
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
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
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

    let params = FmhaDecode::create(
        batch,
        key_len,
        heads,
        kv_heads,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        key_head_stride,
        value_head_stride,
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
                    params.query_batch_stride,
                    params.key_batch_stride,
                    params.value_batch_stride,
                    params.output_batch_stride,
                    params.query_head_stride,
                    params.key_sequence_stride,
                    params.value_sequence_stride,
                    params.output_head_stride,
                    params.key_head_stride,
                    params.value_head_stride,
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
        4 => launch!(softcapped_window_decode_lse_f32_hd4),
        8 => launch!(softcapped_window_decode_lse_f32_hd8),
        16 => launch!(softcapped_window_decode_lse_f32_hd16),
        32 => launch!(softcapped_window_decode_lse_f32_hd32),
        64 => launch!(softcapped_window_decode_lse_f32_hd64),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "softcapped_window_decode_lse_f32".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}

macro_rules! softcapped_window_decode_lse_half_fn {
    ($name:ident, $dtype:ty, $op:literal, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$dtype>,
            key: DevicePointer<$dtype>,
            value: DevicePointer<$dtype>,
            batch: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_head_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_head_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
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

            let params = FmhaDecode::create(
                batch,
                key_len,
                heads,
                kv_heads,
                head_dim,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_head_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_head_stride,
                key_head_stride,
                value_head_stride,
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
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_head_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_head_stride,
                            params.key_head_stride,
                            params.value_head_stride,
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
softcapped_window_decode_lse_half_fn!(
    softcapped_window_decode_lse_f16,
    f16,
    "softcapped_window_decode_lse_f16",
    softcapped_window_decode_lse_f16_hd4,
    softcapped_window_decode_lse_f16_hd8,
    softcapped_window_decode_lse_f16_hd16,
    softcapped_window_decode_lse_f16_hd32,
    softcapped_window_decode_lse_f16_hd64
);

#[cfg(feature = "dtype-bf16")]
softcapped_window_decode_lse_half_fn!(
    softcapped_window_decode_lse_bf16,
    bf16,
    "softcapped_window_decode_lse_bf16",
    softcapped_window_decode_lse_bf16_hd4,
    softcapped_window_decode_lse_bf16_hd8,
    softcapped_window_decode_lse_bf16_hd16,
    softcapped_window_decode_lse_bf16_hd32,
    softcapped_window_decode_lse_bf16_hd64
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! fmha_prefill_lse_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident, $hd128:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            batch: usize,
            query_len: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
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
            causal: bool,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = FmhaPrefill::create(
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
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
                causal,
            )?;
            match head_dim {
                4 => unsafe {
                    kernel_attention::$hd4(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                8 => unsafe {
                    kernel_attention::$hd8(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                16 => unsafe {
                    kernel_attention::$hd16(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                32 => unsafe {
                    kernel_attention::$hd32(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                64 => unsafe {
                    kernel_attention::$hd64(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                128 => unsafe {
                    kernel_attention::$hd128(
                        out,
                        lse,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
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
fmha_prefill_lse_half_fn!(
    fmha_prefill_lse_f16,
    f16,
    fmha_prefill_lse_f16_hd4,
    fmha_prefill_lse_f16_hd8,
    fmha_prefill_lse_f16_hd16,
    fmha_prefill_lse_f16_hd32,
    fmha_prefill_lse_f16_hd64,
    fmha_prefill_lse_f16_hd128
);
#[cfg(feature = "dtype-bf16")]
fmha_prefill_lse_half_fn!(
    fmha_prefill_lse_bf16,
    bf16,
    fmha_prefill_lse_bf16_hd4,
    fmha_prefill_lse_bf16_hd8,
    fmha_prefill_lse_bf16_hd16,
    fmha_prefill_lse_bf16_hd32,
    fmha_prefill_lse_bf16_hd64,
    fmha_prefill_lse_bf16_hd128
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! fmha_prefill_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident, $hd128:ident) => {
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
            causal: bool,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = FmhaPrefill::create(
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
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
                causal,
            )?;
            match head_dim {
                4 => unsafe {
                    kernel_attention::$hd4(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                8 => unsafe {
                    kernel_attention::$hd8(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                16 => unsafe {
                    kernel_attention::$hd16(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                32 => unsafe {
                    kernel_attention::$hd32(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                64 => unsafe {
                    kernel_attention::$hd64(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
                128 => unsafe {
                    kernel_attention::$hd128(
                        out,
                        query,
                        key,
                        value,
                        params.batch,
                        params.query_len,
                        params.key_len,
                        params.heads,
                        params.kv_heads,
                        params.query_batch_stride,
                        params.key_batch_stride,
                        params.value_batch_stride,
                        params.output_batch_stride,
                        params.query_sequence_stride,
                        params.key_sequence_stride,
                        params.value_sequence_stride,
                        params.output_sequence_stride,
                        params.query_head_stride,
                        params.key_head_stride,
                        params.value_head_stride,
                        params.output_head_stride,
                        scale,
                        params.causal,
                        params.output_len,
                        params.output_values_per_batch,
                    )
                }
                .grid(params.grid)
                .enqueue_on(stream)?,
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
fmha_prefill_half_fn!(
    fmha_prefill_f16,
    f16,
    fmha_prefill_f16_hd4,
    fmha_prefill_f16_hd8,
    fmha_prefill_f16_hd16,
    fmha_prefill_f16_hd32,
    fmha_prefill_f16_hd64,
    fmha_prefill_f16_hd128
);
#[cfg(feature = "dtype-bf16")]
fmha_prefill_half_fn!(
    fmha_prefill_bf16,
    bf16,
    fmha_prefill_bf16_hd4,
    fmha_prefill_bf16_hd8,
    fmha_prefill_bf16_hd16,
    fmha_prefill_bf16_hd32,
    fmha_prefill_bf16_hd64,
    fmha_prefill_bf16_hd128
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! attention_sink_prefill_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            sinks: DevicePointer<f32>,
            batch: usize,
            query_len: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
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
            start_q: usize,
            window: usize,
            scale: f32,
            causal: bool,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;
            checked_device_pointer(sinks)?;

            let params = FmhaPrefill::create(
                batch,
                query_len,
                key_len,
                heads,
                kv_heads,
                head_dim,
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
                causal,
            )?;
            let start_q = checked_i32_value(start_q)?;
            let window = checked_i32_value(window)?;
            macro_rules! launch {
                ($kernel:ident) => {
                    unsafe {
                        kernel_attention::$kernel(
                            out,
                            query,
                            key,
                            value,
                            sinks,
                            params.batch,
                            params.query_len,
                            params.key_len,
                            params.heads,
                            params.kv_heads,
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_sequence_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_sequence_stride,
                            params.query_head_stride,
                            params.key_head_stride,
                            params.value_head_stride,
                            params.output_head_stride,
                            start_q,
                            window,
                            scale,
                            params.causal,
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
attention_sink_prefill_half_fn!(
    attention_sink_prefill_f16,
    f16,
    attention_sink_prefill_f16_hd4,
    attention_sink_prefill_f16_hd8,
    attention_sink_prefill_f16_hd16,
    attention_sink_prefill_f16_hd32,
    attention_sink_prefill_f16_hd64
);
#[cfg(feature = "dtype-bf16")]
attention_sink_prefill_half_fn!(
    attention_sink_prefill_bf16,
    bf16,
    attention_sink_prefill_bf16_hd4,
    attention_sink_prefill_bf16_hd8,
    attention_sink_prefill_bf16_hd16,
    attention_sink_prefill_bf16_hd32,
    attention_sink_prefill_bf16_hd64
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! fmha_decode_lse_half_fn {
    ($name:ident, $ty:ty, $hd4:ident, $hd8:ident, $hd16:ident, $hd32:ident, $hd64:ident, $hd128:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lse: DevicePointer<f32>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            batch: usize,
            key_len: usize,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            query_batch_stride: usize,
            key_batch_stride: usize,
            value_batch_stride: usize,
            output_batch_stride: usize,
            query_head_stride: usize,
            key_sequence_stride: usize,
            value_sequence_stride: usize,
            output_head_stride: usize,
            key_head_stride: usize,
            value_head_stride: usize,
            scale: f32,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;

            let params = FmhaDecode::create(
                batch,
                key_len,
                heads,
                kv_heads,
                head_dim,
                query_batch_stride,
                key_batch_stride,
                value_batch_stride,
                output_batch_stride,
                query_head_stride,
                key_sequence_stride,
                value_sequence_stride,
                output_head_stride,
                key_head_stride,
                value_head_stride,
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
                            params.query_batch_stride,
                            params.key_batch_stride,
                            params.value_batch_stride,
                            params.output_batch_stride,
                            params.query_head_stride,
                            params.key_sequence_stride,
                            params.value_sequence_stride,
                            params.output_head_stride,
                            params.key_head_stride,
                            params.value_head_stride,
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
                128 => launch!($hd128),
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
fmha_decode_lse_half_fn!(
    fmha_decode_lse_f16,
    f16,
    fmha_decode_lse_f16_hd4,
    fmha_decode_lse_f16_hd8,
    fmha_decode_lse_f16_hd16,
    fmha_decode_lse_f16_hd32,
    fmha_decode_lse_f16_hd64,
    fmha_decode_lse_f16_hd128
);
#[cfg(feature = "dtype-bf16")]
fmha_decode_lse_half_fn!(
    fmha_decode_lse_bf16,
    bf16,
    fmha_decode_lse_bf16_hd4,
    fmha_decode_lse_bf16_hd8,
    fmha_decode_lse_bf16_hd16,
    fmha_decode_lse_bf16_hd32,
    fmha_decode_lse_bf16_hd64,
    fmha_decode_lse_bf16_hd128
);

#[cfg(feature = "dtype-f16")]

pub fn fmha_decode_lse_dynpos_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    lse: DevicePointer<f32>,
    query: DevicePointer<f16>,
    key: DevicePointer<f16>,
    value: DevicePointer<f16>,
    position_start: DevicePointer<u32>,
    batch: usize,
    max_key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_batch_stride: usize,
    key_batch_stride: usize,
    value_batch_stride: usize,
    output_batch_stride: usize,
    query_head_stride: usize,
    key_sequence_stride: usize,
    value_sequence_stride: usize,
    output_head_stride: usize,
    key_head_stride: usize,
    value_head_stride: usize,
    scale: f32,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(value)?;
    checked_device_pointer(position_start)?;

    let params = FmhaDecode::create(
        batch,
        max_key_len,
        heads,
        kv_heads,
        head_dim,
        query_batch_stride,
        key_batch_stride,
        value_batch_stride,
        output_batch_stride,
        query_head_stride,
        key_sequence_stride,
        value_sequence_stride,
        output_head_stride,
        key_head_stride,
        value_head_stride,
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
                    position_start,
                    params.batch,
                    params.heads,
                    params.kv_heads,
                    params.query_batch_stride,
                    params.key_batch_stride,
                    params.value_batch_stride,
                    params.output_batch_stride,
                    params.query_head_stride,
                    params.key_sequence_stride,
                    params.value_sequence_stride,
                    params.output_head_stride,
                    params.key_head_stride,
                    params.value_head_stride,
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
        4 => launch!(fmha_decode_lse_f16_dynpos_hd4),
        8 => launch!(fmha_decode_lse_f16_dynpos_hd8),
        16 => launch!(fmha_decode_lse_f16_dynpos_hd16),
        32 => launch!(fmha_decode_lse_f16_dynpos_hd32),
        64 => launch!(fmha_decode_lse_f16_dynpos_hd64),
        128 => launch!(fmha_decode_lse_f16_dynpos_hd128),
        _ => {
            return Err(Error::UnsupportedParameter {
                op: "fmha_decode_lse_dynpos_f16".into(),
                parameter: "head_dim".into(),
                value: head_dim,
            });
        }
    };
    Ok(())
}
