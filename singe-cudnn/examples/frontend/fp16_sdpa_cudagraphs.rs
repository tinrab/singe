mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::operation::attention::AttentionSequenceInputs,
    data_type::{DataType, bf16},
    error::{Error, Status},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionScoreConfig,
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
fn run() -> anyhow::Result<()> {
    let context = common::ExampleContext::create()?;

    let b = 3_i64;
    let h = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

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
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h, s_q, d_qk])?),
    )?;
    let k = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h, s_kv, d_qk])?),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(DataType::Bf16, Shape::contiguous([b, h, s_kv, d_v])?),
    )?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let sequence_length_query = graph.insert_tensor(
        TensorId::new(1007),
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?),
    )?;
    let sequence_length_key_value = graph.insert_tensor(
        TensorId::new(1008),
        TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?),
    )?;

    let outputs = graph.sdpa(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        AttentionConfig::new()
            .with_compute_type(DataType::F32)
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
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = outputs.output;
    let stats = outputs.stats;

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::Bf16,
            Shape::with_strides([b, h, s_q, d_v], vec![h * d_v, d_v, b * h * d_v, 1])?,
        )
        .output_tensor(),
    )?;
    let stats = stats.expect("stats");
    let stats_tensor = graph.tensor_spec(stats)?.clone().output_tensor();
    graph.replace_tensor(stats, stats_tensor)?;

    let compiled = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let mut q_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image(usize::try_from(b * h * s_q * d_qk)?, &mut seed),
    )?;
    let mut k_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image(usize::try_from(b * h * s_kv * d_qk)?, &mut seed),
    )?;
    let mut v_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_bf16_image(usize::try_from(b * h * s_kv * d_v)?, &mut seed),
    )?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![20_i32; usize::try_from(b)?],
    )?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![20_i32; usize::try_from(b)?],
    )?;
    let mut o_dev = DeviceMemory::<bf16>::zeroed(
        context.cudnn.cuda_context(),
        usize::try_from(b * h * s_q * d_v)?,
    )?;
    let mut stats_dev =
        DeviceMemory::<f32>::zeroed(context.cudnn.cuda_context(), usize::try_from(b * h * s_q)?)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;

    let mut cuda_graph = context.cudnn.cuda_context().create_graph()?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    match unsafe {
        compiled.populate_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
            &mut cuda_graph,
        )
    } {
        Ok(()) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    }

    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    match unsafe {
        compiled.update_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
            &mut cuda_graph,
        )
    } {
        Ok(()) => Ok(()),
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn main() -> anyhow::Result<()> {
    common::finish("frontend_fp16_sdpa_cudagraphs", run())
}
