mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::operation::attention::AttentionSequenceInputs,
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaBackwardInputs, SdpaGradientOutputConfig},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionBackwardConfig, AttentionBackwardWorkspaceLimit, AttentionDiagonalAlignment,
            AttentionDiagonalBand, AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    tensor::{Shape, TensorId, TensorSpec},
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

#[expect(
    clippy::many_single_char_names,
    reason = "keeps the upstream sample's single-letter dimension names"
)]
fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_value = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::Bf16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h_q, s_q, d_qk])?),
    )?;
    let k = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h_k, s_kv, d_qk])?),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h_v, s_kv, d_value])?),
    )?;
    let o = graph.insert_tensor(
        TensorId::new(1004),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h_q, s_q, d_value])?),
    )?;
    let o_grad = graph.insert_tensor(
        TensorId::new(1101),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h_q, s_q, d_value])?),
    )?;
    let stats = graph.insert_tensor(
        TensorId::new(1005),
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, 1])?),
    )?;
    let sequence_length_query = graph.insert_tensor(
        TensorId::new(1008),
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?),
    )?;
    let sequence_length_key_value = graph.insert_tensor(
        TensorId::new(1009),
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let outputs = graph.sdpa_backward(
        SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: o_grad,
            stats,
            scale,
        },
        AttentionBackwardConfig::new()
            .with_name("attention_backward")
            .with_compute_type(DataType::F32)
            .with_deterministic_algorithm(true)
            // On Hopper, zero dP workspace opts into ordered FP32 dQ accumulation.
            .with_probability_gradient_workspace_limit(
                if args().any(|argument| argument == "--inherit-workspace-limit") {
                    None
                } else {
                    Some(AttentionBackwardWorkspaceLimit::Bytes(0))
                },
            )
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                        sequence_length_query,
                        sequence_length_key_value,
                    ))
                    .with_mask_mode(AttentionMaskMode::DiagonalBand(
                        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft)
                            .with_padding(true),
                    )),
            ),
        SdpaGradientOutputConfig::new(TensorOutputConfig::new().output_tensor()),
    )?;
    let q_grad = outputs.query_gradient;
    let k_grad = outputs.key_gradient;
    let v_grad = outputs.value_gradient;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 987_654_321_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(graph.shape(q)?.storage_span()?, &mut seed),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(graph.shape(k)?.storage_span()?, &mut seed),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(graph.shape(v)?.storage_span()?, &mut seed),
    )?;
    let o_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(graph.shape(o)?.storage_span()?, &mut seed),
    )?;
    let o_grad_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(graph.shape(o_grad)?.storage_span()?, &mut seed),
    )?;
    let seq_len_q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![20_i32; graph.shape(sequence_length_query)?.storage_span()?],
    )?;
    let seq_len_kv_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![20_i32; graph.shape(sequence_length_key_value)?.storage_span()?],
    )?;
    let stats_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f32_image(graph.shape(stats)?.storage_span()?, &mut seed),
    )?;
    let mut q_grad_dev = DeviceMemory::<bf16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(q_grad)?.storage_span()?,
    )?;
    let mut k_grad_dev = DeviceMemory::<bf16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(k_grad)?.storage_span()?,
    )?;
    let mut v_grad_dev = DeviceMemory::<bf16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(v_grad)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev)?;
    bindings.set_input(k, &k_dev)?;
    bindings.set_input(v, &v_dev)?;
    bindings.set_input(o, &o_dev)?;
    bindings.set_input(o_grad, &o_grad_dev)?;
    bindings.set_input(sequence_length_query, &seq_len_q_dev)?;
    bindings.set_input(sequence_length_key_value, &seq_len_kv_dev)?;
    bindings.set_input(stats, &stats_dev)?;
    bindings.set(q_grad, &mut q_grad_dev)?;
    bindings.set(k_grad, &mut k_grad_dev)?;
    bindings.set(v_grad, &mut v_grad_dev)?;
    // SAFETY: input buffers, writable gradients, plan and workspace remain live
    // through stream completion, including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute_async(
            context,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
        )
    })?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    common::finish("frontend_fp16_sdpa_backward", run())
}
