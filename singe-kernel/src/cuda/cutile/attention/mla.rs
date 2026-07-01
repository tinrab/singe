use super::*;

pub fn mla_prefill_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(query_pe)?;
    checked_device_pointer(key)?;
    checked_device_pointer(key_pe)?;
    checked_device_pointer(value)?;

    let params = MlaPrefill::create(
        batch, query_len, key_len, heads, kv_heads, head_dim, pe_dim, scale,
    )?;
    unsafe {
        kernel_attention::mla_prefill_f32(
            out,
            query,
            query_pe,
            key,
            key_pe,
            value,
            params.batch,
            params.query_len,
            params.key_len,
            params.heads,
            params.kv_heads,
            params.head_dim,
            params.pe_dim,
            scale,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! mla_prefill_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
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
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(query_pe)?;
            checked_device_pointer(key)?;
            checked_device_pointer(key_pe)?;
            checked_device_pointer(value)?;

            let params = MlaPrefill::create(
                batch, query_len, key_len, heads, kv_heads, head_dim, pe_dim, scale,
            )?;
            unsafe {
                kernel_attention::$kernel(
                    out,
                    query,
                    query_pe,
                    key,
                    key_pe,
                    value,
                    params.batch,
                    params.query_len,
                    params.key_len,
                    params.heads,
                    params.kv_heads,
                    params.head_dim,
                    params.pe_dim,
                    scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
mla_prefill_half_fn!(mla_prefill_f16, f16, mla_prefill_f16);
#[cfg(feature = "dtype-bf16")]
mla_prefill_half_fn!(mla_prefill_bf16, bf16, mla_prefill_bf16);

pub fn sparse_mla_prefill_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(query)?;
    checked_device_pointer(query_pe)?;
    checked_device_pointer(key)?;
    checked_device_pointer(key_pe)?;
    checked_device_pointer(value)?;
    checked_device_pointer(indices)?;

    let params = SparseMlaPrefill::create(
        batch, query_len, key_len, heads, kv_heads, head_dim, pe_dim, topk, scale,
    )?;
    unsafe {
        kernel_attention::sparse_mla_prefill_f32(
            out,
            query,
            query_pe,
            key,
            key_pe,
            value,
            indices,
            params.batch,
            params.query_len,
            params.key_len,
            params.heads,
            params.kv_heads,
            params.head_dim,
            params.pe_dim,
            params.topk,
            scale,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn mla_decode_lse_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(query_pe)?;
    checked_device_pointer(key_value)?;
    checked_device_pointer(key_pe)?;

    let params = MlaDecode::create(batch, key_len, heads, head_dim, pe_dim, scale)?;
    unsafe {
        kernel_attention::mla_decode_lse_f32(
            out,
            lse,
            query,
            query_pe,
            key_value,
            key_pe,
            params.batch,
            params.key_len,
            params.heads,
            params.head_dim,
            params.pe_dim,
            scale,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn paged_mla_decode_attention_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(query_pe)?;
    checked_device_pointer(key_value_cache)?;
    checked_device_pointer(key_pe_cache)?;
    checked_device_pointer(actual_seq_lens)?;
    checked_device_pointer(block_table)?;

    let params = PagedMlaDecodeAttention::create(
        batch,
        heads,
        head_dim,
        pe_dim,
        block_size,
        physical_blocks,
        block_table_batch_stride,
        scale,
        output_scale,
    )?;
    unsafe {
        kernel_attention::paged_mla_decode_attention_f32(
            out,
            lse,
            query,
            query_pe,
            key_value_cache,
            key_pe_cache,
            actual_seq_lens,
            block_table,
            params.batch,
            params.heads,
            params.head_dim,
            params.pe_dim,
            params.block_size,
            params.physical_blocks,
            params.block_table_batch_stride,
            scale,
            output_scale,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}
#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! paged_mla_decode_attention_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
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
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(query_pe)?;
            checked_device_pointer(key_value_cache)?;
            checked_device_pointer(key_pe_cache)?;
            checked_device_pointer(actual_seq_lens)?;
            checked_device_pointer(block_table)?;

            let params = PagedMlaDecodeAttention::create(
                batch,
                heads,
                head_dim,
                pe_dim,
                block_size,
                physical_blocks,
                block_table_batch_stride,
                scale,
                output_scale,
            )?;
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    query_pe,
                    key_value_cache,
                    key_pe_cache,
                    actual_seq_lens,
                    block_table,
                    params.batch,
                    params.heads,
                    params.head_dim,
                    params.pe_dim,
                    params.block_size,
                    params.physical_blocks,
                    params.block_table_batch_stride,
                    scale,
                    output_scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
paged_mla_decode_attention_half_fn!(
    paged_mla_decode_attention_f16,
    f16,
    paged_mla_decode_attention_f16
);

#[cfg(feature = "dtype-bf16")]
paged_mla_decode_attention_half_fn!(
    paged_mla_decode_attention_bf16,
    bf16,
    paged_mla_decode_attention_bf16
);

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! mla_decode_lse_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
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
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(query_pe)?;
            checked_device_pointer(key_value)?;
            checked_device_pointer(key_pe)?;

            let params = MlaDecode::create(batch, key_len, heads, head_dim, pe_dim, scale)?;
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    query_pe,
                    key_value,
                    key_pe,
                    params.batch,
                    params.key_len,
                    params.heads,
                    params.head_dim,
                    params.pe_dim,
                    scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
mla_decode_lse_half_fn!(mla_decode_lse_f16, f16, mla_decode_lse_f16);
#[cfg(feature = "dtype-bf16")]
mla_decode_lse_half_fn!(mla_decode_lse_bf16, bf16, mla_decode_lse_bf16);

pub fn mla_decode_splitk_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(lse)?;
    checked_device_pointer(query)?;
    checked_device_pointer(query_pe)?;
    checked_device_pointer(key_value)?;
    checked_device_pointer(key_pe)?;

    let params = MlaDecodeSplitK::create(
        batch,
        key_len,
        heads,
        splits,
        kv_len_per_split,
        head_dim,
        pe_dim,
        scale,
    )?;
    unsafe {
        kernel_attention::mla_decode_splitk_f32(
            out,
            lse,
            query,
            query_pe,
            key_value,
            key_pe,
            params.batch,
            params.key_len,
            params.heads,
            params.splits,
            params.kv_len_per_split,
            params.head_dim,
            params.pe_dim,
            scale,
            params.output_len,
            params.output_values_per_batch,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! mla_decode_splitk_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
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
            checked_device_pointer(out)?;
            checked_device_pointer(lse)?;
            checked_device_pointer(query)?;
            checked_device_pointer(query_pe)?;
            checked_device_pointer(key_value)?;
            checked_device_pointer(key_pe)?;

            let params = MlaDecodeSplitK::create(
                batch,
                key_len,
                heads,
                splits,
                kv_len_per_split,
                head_dim,
                pe_dim,
                scale,
            )?;
            unsafe {
                kernel_attention::$kernel(
                    out,
                    lse,
                    query,
                    query_pe,
                    key_value,
                    key_pe,
                    params.batch,
                    params.key_len,
                    params.heads,
                    params.splits,
                    params.kv_len_per_split,
                    params.head_dim,
                    params.pe_dim,
                    scale,
                    params.output_len,
                    params.output_values_per_batch,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
mla_decode_splitk_half_fn!(mla_decode_splitk_f16, f16, mla_decode_splitk_f16);
#[cfg(feature = "dtype-bf16")]
mla_decode_splitk_half_fn!(mla_decode_splitk_bf16, bf16, mla_decode_splitk_bf16);
