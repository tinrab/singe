mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionScoreConfig, AttentionSequenceInputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let float_value = (*seed as f32) * 2.328_306_4e-10_f32;
        values.push(bf16::from_f32(float_value));
    }
    values
}

const Q_ID: TensorId = TensorId::new(1101);
const K_ID: TensorId = TensorId::new(1102);
const V_ID: TensorId = TensorId::new(1103);
const SEQ_LEN_Q_ID: TensorId = TensorId::new(1107);
const SEQ_LEN_KV_ID: TensorId = TensorId::new(1108);

struct SdpaForwardGraph {
    graph: Graph,
    inputs: SdpaInputs,
    query_lengths: TensorId,
    key_value_lengths: TensorId,
    outputs: SdpaForwardOutputTensors,
}

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
    output_requests: SdpaForwardOutputTensors<TensorOutputConfig>,
    causal_mask: bool,
    padding_mask: bool,
) -> Result<SdpaForwardGraph> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.insert_tensor(
        Q_ID,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_q, s_q, d_qk], [h_q * s_q * d_qk, s_q * d_qk, d_qk, 1])?,
        )
        .with_name("Q"),
    )?;
    let k = graph.insert_tensor(
        K_ID,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides(
                [b, h_k, s_kv, d_qk],
                [h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1],
            )?,
        )
        .with_name("K"),
    )?;
    let v = graph.insert_tensor(
        V_ID,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_v, s_kv, d_v], [h_v * s_kv * d_v, s_kv * d_v, d_v, 1])?,
        )
        .with_name("V"),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(attention_scale));
    let sequence_length_query = graph.insert_tensor(
        SEQ_LEN_Q_ID,
        TensorSpec::new(
            DataType::I32,
            Shape::with_strides([b, 1, 1, 1], [1, 1, 1, 1])?,
        )
        .with_name("seq_q"),
    )?;
    let sequence_length_key_value = graph.insert_tensor(
        SEQ_LEN_KV_ID,
        TensorSpec::new(
            DataType::I32,
            Shape::with_strides([b, 1, 1, 1], [1, 1, 1, 1])?,
        )
        .with_name("seq_kv"),
    )?;

    let mut sdpa_options = AttentionConfig::new();
    let mut score_config = AttentionScoreConfig::new();
    if causal_mask {
        score_config = score_config.with_mask_mode(AttentionMaskMode::DiagonalBand(
            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft)
                .with_padding(padding_mask),
        ));
    }
    if padding_mask {
        score_config = score_config.with_sequence_inputs(AttentionSequenceInputs::per_batch(
            sequence_length_query,
            sequence_length_key_value,
        ));
        if !causal_mask {
            score_config = score_config.with_mask_mode(AttentionMaskMode::Padding);
        }
    }
    sdpa_options = sdpa_options.with_score_config(score_config);

    let inputs = SdpaInputs {
        query: q,
        key: k,
        value: v,
        scale,
    };
    let outputs = graph.sdpa_auto(inputs, sdpa_options, output_requests)?;
    let o = outputs.output;

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_q, s_q, d_v], [h_q * d_v, d_v, b * h_q * d_v, 1])?,
        )
        .output_tensor(),
    )?;
    Ok(SdpaForwardGraph {
        graph,
        inputs,
        query_lengths: sequence_length_query,
        key_value_lengths: sequence_length_key_value,
        outputs,
    })
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;

    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let output_requests = SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
        .with_stats(TensorOutputConfig::new().output_tensor())
        .with_logit_max(TensorOutputConfig::new().output_tensor())
        .with_score_sum_exp(
            TensorOutputConfig::new()
                .with_data_type(DataType::F32)
                .output_tensor(),
        );
    let attention_scale = 0.123_f32;
    let causal_mask = true;

    let SdpaForwardGraph {
        graph,
        inputs,
        query_lengths,
        key_value_lengths,
        outputs,
    } = create_sdpa_forward_graph(
        b,
        h_q,
        h_k,
        h_v,
        s_q,
        s_kv,
        d_qk,
        d_v,
        attention_scale,
        output_requests,
        causal_mask,
        true, // padding_mask
    )?;

    let compiled_graph = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let mut q_tensor = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed),
    )?;
    let mut k_tensor = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_k * s_kv * d_qk) as usize, &mut seed),
    )?;
    let mut v_tensor = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed),
    )?;

    let mut o_tensor =
        DeviceMemory::<bf16>::zeroed(context.cudnn.cuda_context(), (b * s_q * h_q * d_v) as usize)?;

    let mut seq_len_q_dev =
        DeviceMemory::from_slice(context.cudnn.cuda_context(), &vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev =
        DeviceMemory::from_slice(context.cudnn.cuda_context(), &vec![20_i32; b as usize])?;
    let mut auxiliary_buffers = Vec::new();
    for id in [outputs.stats, outputs.logit_max, outputs.score_sum_exp]
        .into_iter()
        .flatten()
    {
        let bytes = graph.tensor_spec(id)?.storage_byte_len(None)?;
        auxiliary_buffers.push((
            id,
            DeviceMemory::<u8>::zeroed(context.cudnn.cuda_context(), bytes)?,
        ));
    }
    let mut workspace = common::workspace(
        context.cudnn.cuda_context(),
        compiled_graph.workspace_size(),
    )?;

    let mut bindings = compiled_graph.bindings()?;
    bindings
        .set(inputs.query, &mut q_tensor)?
        .set(inputs.key, &mut k_tensor)?
        .set(inputs.value, &mut v_tensor)?
        .set(query_lengths, &mut seq_len_q_dev)?
        .set(key_value_lengths, &mut seq_len_kv_dev)?
        .set(outputs.output, &mut o_tensor)?;
    for (id, memory) in &mut auxiliary_buffers {
        bindings.set_encoded_bytes(*id, memory)?;
    }
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    unsafe {
        compiled_graph.execute(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    }?;

    context.cudnn.cuda_context().synchronize()?;

    let o_host = o_tensor.copy_to_host_vec()?;
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
    for (id, memory) in &auxiliary_buffers {
        let spec = graph.tensor_spec(*id)?;
        let bytes = memory.copy_to_host_vec()?;
        println!(
            "auxiliary {id}: encoding={}, bytes={}, prefix={:02x?}",
            spec.data_type,
            bytes.len(),
            &bytes[..bytes.len().min(16)],
        );
    }
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
