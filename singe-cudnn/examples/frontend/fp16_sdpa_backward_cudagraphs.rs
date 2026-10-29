mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, bf16},
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::{SdpaBackwardInputs, SdpaGradientOutputConfig},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionBackwardConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionMaskMode, AttentionScoreConfig, AttentionSequenceInputs,
        },
        plan::{CompileConfig, EngineConfigFilter, ExecuteArgs, HeuristicMode},
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
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

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides(
                [b, h_q, s_q, d_qk],
                vec![h_q * s_q * d_qk, s_q * d_qk, d_qk, 1],
            )?,
        ),
    )?;
    let k = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_k, s_kv, d_qk])?),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h_v, s_kv, d_v])?),
    )?;
    let o = graph.insert_tensor(
        TensorId::new(1004),
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_q, s_q, d_v], vec![h_q * d_v, d_v, b * h_q * d_v, 1])?,
        ),
    )?;
    let stats = graph.insert_tensor(
        TensorId::new(1005),
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides([b, h_q, s_q, 1], [h_q * s_q, s_q, 1, 1])?,
        ),
    )?;
    let d_o = graph.insert_tensor(
        TensorId::new(1101),
        TensorSpec::new(
            DataType::BF16,
            Shape::with_strides([b, h_q, s_q, d_v], vec![h_q * d_v, d_v, b * h_q * d_v, 1])?,
        ),
    )?;
    let sequence_length_query = graph.insert_tensor(
        TensorId::new(1008),
        TensorSpec::new(
            DataType::I32,
            Shape::with_strides([b, 1, 1, 1], [1, 1, 1, 1])?,
        ),
    )?;
    let sequence_length_key_value = graph.insert_tensor(
        TensorId::new(1009),
        TensorSpec::new(
            DataType::I32,
            Shape::with_strides([b, 1, 1, 1], [1, 1, 1, 1])?,
        ),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(1.0));

    let outputs = graph.sdpa_backward(
        SdpaBackwardInputs {
            query: q,
            key: k,
            value: v,
            output: o,
            output_gradient: d_o,
            stats,
            scale,
        },
        AttentionBackwardConfig::new()
            .with_compute_type(DataType::F32)
            .with_attention_scale(0.123)
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
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v_out = outputs.value_gradient;

    let compiled = graph
        .plan_candidates(
            Some(&context.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?
        .filter(
            &EngineConfigFilter::new()
                .with_included_behavior_notes(&[BackendBehaviorNote::SupportsCudaGraphNativeApi]),
        )
        .compile(Some(&context.cudnn), &CompileConfig::new())?;

    let mut seed = 987_654_321_u32;
    let mut q_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed),
    )?;
    let mut k_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_k * s_kv * d_qk) as usize, &mut seed),
    )?;
    let mut v_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed),
    )?;
    let mut o_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_q * s_q * d_v) as usize, &mut seed),
    )?;
    let mut d_o_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image((b * h_q * s_q * d_v) as usize, &mut seed),
    )?;
    let mut seq_len_q_dev =
        DeviceMemory::from_slice(context.cudnn.cuda_context(), &vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev =
        DeviceMemory::from_slice(context.cudnn.cuda_context(), &vec![20_i32; b as usize])?;
    let mut stats_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &common::init_f32_image((b * h_q * s_q) as usize, &mut seed),
    )?;
    let mut d_q_dev = DeviceMemory::<bf16>::zeroed(
        context.cudnn.cuda_context(),
        (b * h_q * s_q * d_qk) as usize,
    )?;
    let mut d_k_dev = DeviceMemory::<bf16>::zeroed(
        context.cudnn.cuda_context(),
        (b * h_k * s_kv * d_qk) as usize,
    )?;
    let mut d_v_dev = DeviceMemory::<bf16>::zeroed(
        context.cudnn.cuda_context(),
        (b * h_v * s_kv * d_v) as usize,
    )?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(d_o, &mut d_o_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    bindings.set(d_q, &mut d_q_dev)?;
    bindings.set(d_k, &mut d_k_dev)?;
    bindings.set(d_v_out, &mut d_v_dev)?;

    let mut cudnn_graph = context.cudnn.cuda_context().create_graph()?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    match unsafe {
        compiled.populate_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            &mut cudnn_graph,
        )
    } {
        Ok(()) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }

    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    match unsafe {
        compiled.update_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            &cudnn_graph,
        )
    } {
        Ok(()) => Ok(()),
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_backward_cudagraphs", run())
}
