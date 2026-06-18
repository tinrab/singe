mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaInputs, SdpaOutputs},
        graph::Graph,
        operation::{AttentionConfig, AttentionPagedCache, HeuristicMode},
    },
    version,
};

fn build_case(is_ragged_prefill: bool) -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let cudnn_version = version()?;

    if ctx.get_compute_capability() == 120 {
        println!("skipping paged decode/prefill: not supported on SM version 120 yet");
        return Ok(());
    }

    if is_ragged_prefill && cudnn_version < 90800 {
        println!("skipping ragged prefill: requires cuDNN >= 9.8.0");
        return Ok(());
    }
    if cudnn_version < 90500 {
        println!("skipping paged decode/prefill: requires cuDNN >= 9.5.0");
        return Ok(());
    }

    let b = 8_i64;
    let h_q = 8_i64;
    let h_k = 1_i64;
    let h_v = 1_i64;
    let s_kv = 32 * 1024_i64;
    let s_q = if is_ragged_prefill { 512_i64 } else { 1_i64 };
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let block_size = 1_i64;
    let table_size = (s_kv + block_size - 1) / block_size;
    let num_blocks_k = table_size * b;
    let num_blocks_v = table_size * b;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let q = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides(vec![
            h_q * s_q * d_qk,
            s_q * d_qk,
            d_qk,
            1,
        ])?,
    ));
    if is_ragged_prefill {
        let q_offsets = graph.tensor(TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b + 1, 1, 1, 1])?,
        ));
        graph.set_ragged_offset(q, q_offsets)?;
    }

    let k = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([num_blocks_k, h_k, block_size, d_qk])?.with_strides(vec![
            h_k * block_size * d_qk,
            block_size * d_qk,
            d_qk,
            1,
        ])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([num_blocks_v, h_v, block_size, d_v])?.with_strides(vec![
            h_v * block_size * d_v,
            block_size * d_v,
            d_v,
            1,
        ])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let page_table_k = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, table_size, 1])?.with_strides([table_size, table_size, 1, 1])?,
    ));
    let page_table_v = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, table_size, 1])?.with_strides([table_size, table_size, 1, 1])?,
    ));

    let SdpaOutputs { output: o, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_causal_mask()
            .with_padding_mask(sequence_length_query, sequence_length_key_value)
            .with_paged_k(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_k,
            ))
            .with_paged_v(AttentionPagedCache::new(
                sequence_length_key_value,
                page_table_v,
            ))
            .with_max_sequence_length_key_value(s_kv),
    )?;

    let o_tensor = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
            h_q * d_v,
            d_v,
            b * h_q * d_v,
            1,
        ])?,
    )
    .output_tensor();
    graph.replace_tensor(o, o_tensor)?;
    if let Some(stats) = stats {
        graph.mark_as_output(stats)?;
    }

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;
    let mut q_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_qk) as usize)?;
    let mut k_dev =
        DeviceMemory::<bf16>::zeroes((num_blocks_k * h_k * block_size * d_qk) as usize)?;
    let mut v_dev = DeviceMemory::<bf16>::zeroes((num_blocks_v * h_v * block_size * d_v) as usize)?;
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(
        &(0..b)
            .map(|index| {
                if is_ragged_prefill {
                    if index & 1 == 0 { 32_i32 } else { 16_i32 }
                } else {
                    1_i32
                }
            })
            .collect::<Vec<_>>(),
    )?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(
        &(0..b)
            .map(|index| (s_kv - (index * 17)).max(1) as i32)
            .collect::<Vec<_>>(),
    )?;
    let mut page_table_k_dev = DeviceMemory::from_slice(
        &(0..(b * table_size))
            .map(|index| (index % num_blocks_k) as i32)
            .collect::<Vec<_>>(),
    )?;
    let mut page_table_v_dev = DeviceMemory::from_slice(
        &(0..(b * table_size))
            .map(|index| ((index * 13) % num_blocks_v) as i32)
            .collect::<Vec<_>>(),
    )?;
    let mut stats_dev = if stats.is_some() {
        Some(DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?)
    } else {
        None
    };
    let mut q_offsets_dev = if is_ragged_prefill {
        let mut offsets = Vec::with_capacity((b + 1) as usize);
        offsets.push(0_i32);
        let mut cumulative = 0_i32;
        for index in 0..b {
            let sequence = if index & 1 == 0 { 32_i32 } else { 16_i32 };
            cumulative += sequence * h_q as i32 * d_qk as i32;
            offsets.push(cumulative);
        }
        Some(DeviceMemory::from_slice(&offsets)?)
    } else {
        None
    };
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(page_table_k, &mut page_table_k_dev)?;
    bindings.set(page_table_v, &mut page_table_v_dev)?;
    if let (Some(stats), Some(stats_dev)) = (stats, stats_dev.as_mut()) {
        bindings.set(stats, stats_dev)?;
    }
    if let Some(q_offsets) = q_offsets_dev.as_mut() {
        let q_offsets_tensor = graph
            .tensor_config(q)?
            .ragged_offset
            .expect("q ragged offset");
        bindings.set(q_offsets_tensor, q_offsets)?;
    }
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn run() -> Result<()> {
    build_case(true)?;
    build_case(false)
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_paged_decode_and_prefill", run())
}
