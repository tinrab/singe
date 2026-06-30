mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cuda::types::f16;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionConfig, AttentionMaskMode, AttentionScoreConfig, HeuristicMode,
            SdpaAuxOutputRequest,
        },
    },
    version,
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let float_value = (*seed as f32) * 2.328_306_4e-10_f32;
        values.push(bf16::from_bits(f16::from_f32(float_value).to_bits()));
    }
    values
}

const Q_ID: TensorId = TensorId::new(1101);
const K_ID: TensorId = TensorId::new(1102);
const V_ID: TensorId = TensorId::new(1103);
const O_ID: TensorId = TensorId::new(1104);
const STATS_ID: TensorId = TensorId::new(1105);
const SEQ_LEN_Q_ID: TensorId = TensorId::new(1107);
const SEQ_LEN_KV_ID: TensorId = TensorId::new(1108);

fn create_sdpa_forward_graph(
    b: i64,
    h_q: i64,
    h_k: i64,
    h_v: i64,
    s_q: i64,
    s_kv: i64,
    d_qk: i64,
    d_v: i64,
    attention_scale: f32,
    generate_stats: bool,
    causal_mask: bool,
    padding_mask: bool,
) -> Result<(
    Graph,
    TensorId,
    TensorId,
    TensorId,
    TensorId,
    TensorId,
    TensorId,
    TensorId,
)> {
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
            Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides([
                h_q * s_q * d_qk,
                s_q * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_name("Q")
        .with_id(Q_ID),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_k, s_kv, d_qk])?.with_strides([
                h_k * s_kv * d_qk,
                s_kv * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_name("K")
        .with_id(K_ID),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_v, s_kv, d_v])?.with_strides([
                h_v * s_kv * d_v,
                s_kv * d_v,
                d_v,
                1,
            ])?,
        )
        .with_name("V")
        .with_id(V_ID),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(attention_scale)?);
    let sequence_length_query = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_name("seq_q")
        .with_id(SEQ_LEN_Q_ID),
    );
    let sequence_length_key_value = graph.tensor(
        TensorSpec::new(
            DataType::I32,
            Shape::contiguous([b, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
        )
        .with_name("seq_kv")
        .with_id(SEQ_LEN_KV_ID),
    );

    let mut sdpa_options = AttentionConfig::new(DataType::F32);
    let mut score_config = AttentionScoreConfig::new();
    if generate_stats {
        sdpa_options = sdpa_options.with_aux_outputs(SdpaAuxOutputRequest::stats_only());
    }
    if causal_mask {
        score_config = score_config.with_mask_mode(AttentionMaskMode::CausalTopLeft);
    }
    if padding_mask {
        score_config = score_config.with_mask_mode(AttentionMaskMode::Padding {
            sequence_length_query: sequence_length_query,
            sequence_length_key_value: sequence_length_key_value,
        });
    }
    sdpa_options = sdpa_options.with_score_config(score_config);

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        sdpa_options,
    )?;
    let o = outputs.output();
    let stats = outputs.stats();

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides([
                h_q * d_v,
                d_v,
                b * h_q * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(O_ID),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(STATS_ID);
    graph.replace_tensor(stats, stats_tensor)?;

    Ok((
        graph,
        q,
        k,
        v,
        sequence_length_query,
        sequence_length_key_value,
        o,
        stats,
    ))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let generate_stats = true;
    let attention_scale = 0.123_f32;
    let causal_mask = true;

    if version()? < 8903 {
        println!("frontend_fp16_sdpa: skipped, requires cuDNN 8.9.3+");
        return Ok(());
    }

    let (graph, q, k, v, sequence_length_query, sequence_length_key_value, o, stats) =
        create_sdpa_forward_graph(
            b,
            h_q,
            h_k,
            h_v,
            s_q,
            s_kv,
            d_qk,
            d_v,
            attention_scale,
            generate_stats,
            causal_mask,
            true, // padding_mask
        )?;

    let compiled_graph = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_tensor =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_tensor = DeviceMemory::from_slice(&init_bf16_image(
        (b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_tensor =
        DeviceMemory::from_slice(&init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;

    let mut o_tensor = DeviceMemory::<bf16>::zeroes((b * s_q * h_q * d_v) as usize)?;

    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled_graph.workspace_size()?)?;

    let mut bindings = compiled_graph.bindings();
    bindings
        .set(q, &mut q_tensor)?
        .set(k, &mut k_tensor)?
        .set(v, &mut v_tensor)?
        .set(sequence_length_query, &mut seq_len_q_dev)?
        .set(sequence_length_key_value, &mut seq_len_kv_dev)?
        .set(o, &mut o_tensor)?
        .set(stats, &mut stats_dev)?;
    compiled_graph.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    ctx.cudnn.cuda_context().synchronize()?;

    let o_host = o_tensor.copy_to_host_vec()?;
    let stats_host = stats_dev.copy_to_host_vec()?;
    let q_host = q_tensor.copy_to_host_vec()?;

    println!(
        "o_bits: first={}, last={}, checksum={}, len={}",
        o_host[0].to_bits(),
        o_host[o_host.len() - 1].to_bits(),
        o_host
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .sum::<u64>(),
        o_host.len()
    );
    println!(
        "stats: first={}, last={}, checksum={}, nonfinite={}, len={}",
        stats_host[0],
        stats_host[stats_host.len() - 1],
        stats_host.iter().copied().sum::<f32>(),
        common::count_nonfinite_f32(&stats_host),
        stats_host.len()
    );
    println!(
        "q_bits: first={}, second={}, third={}, fourth={}",
        q_host[0].to_bits(),
        q_host[1].to_bits(),
        q_host[2].to_bits(),
        q_host[3].to_bits()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa", run())
}
