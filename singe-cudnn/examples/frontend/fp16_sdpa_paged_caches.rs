mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionConfig, AttentionMaskMode, AttentionPagedCache, AttentionPagedKvCache,
            AttentionScoreConfig, HeuristicMode, SdpaAuxOutputRequest,
        },
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 2_i64;
    let h_q = 8_i64;
    let h_k = 1_i64;
    let h_v = 1_i64;
    let s_q = 1_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let block_size = 1_i64;
    let table_size = (s_kv + block_size - 1) / block_size;
    let num_blocks_k = table_size * b;
    let num_blocks_v = table_size * b;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides(vec![
                h_q * s_q * d_qk,
                s_q * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks_k, h_k, block_size, d_qk])?.with_strides([
                h_k * block_size * d_qk,
                block_size * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([num_blocks_v, h_v, block_size, d_v])?.with_strides([
                h_v * block_size * d_v,
                block_size * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1003),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let sequence_length_query = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1007),
    );
    let sequence_length_key_value = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_id(1008),
    );
    let page_table_k = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, table_size, 1])?
                .with_strides([table_size, table_size, 1, 1])?,
        )
        .with_id(1009),
    );
    let page_table_v = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, table_size, 1])?
                .with_strides([table_size, table_size, 1, 1])?,
        )
        .with_id(1010),
    );

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_aux_outputs(SdpaAuxOutputRequest::stats_only())
            .with_score_config(AttentionScoreConfig::new().with_mask_mode(
                AttentionMaskMode::Padding {
                    sequence_length_query: sequence_length_query,
                    sequence_length_key_value: sequence_length_key_value,
                },
            ))
            .with_paged_cache(
                AttentionPagedKvCache::new()
                    .with_k(AttentionPagedCache::new(
                        sequence_length_key_value,
                        page_table_k,
                    ))
                    .with_v(AttentionPagedCache::new(
                        sequence_length_key_value,
                        page_table_v,
                    ))
                    .with_max_sequence_length_key_value(s_kv),
            ),
    )?;
    let o = outputs.output();
    let stats = outputs.stats();

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * s_q * d_v,
                s_q * d_v,
                d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(1004),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(1005);
    graph.replace_tensor(stats, stats_tensor)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev = DeviceMemory::from_slice(&init_bf16_image(
        (num_blocks_k * h_k * block_size * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev = DeviceMemory::from_slice(&init_bf16_image(
        (num_blocks_v * h_v * block_size * d_v) as usize,
        &mut seed,
    ))?;
    let mut page_table_k_host = vec![0_i32; (b * table_size) as usize];
    let mut page_table_v_host = vec![0_i32; (b * table_size) as usize];
    let blocks_per_batch = table_size as usize;
    for batch in 0..(b as usize) {
        for slot in 0..blocks_per_batch {
            let value = (batch * blocks_per_batch + slot) as i32;
            page_table_k_host[batch * blocks_per_batch + slot] = value;
            page_table_v_host[batch * blocks_per_batch + slot] = value;
        }
    }
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![1_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![s_kv as i32; b as usize])?;
    let mut page_table_k_dev = DeviceMemory::from_slice(&page_table_k_host)?;
    let mut page_table_v_dev = DeviceMemory::from_slice(&page_table_v_host)?;
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(page_table_k, &mut page_table_k_dev)?;
    bindings.set(page_table_v, &mut page_table_v_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_paged_caches", run())
}
