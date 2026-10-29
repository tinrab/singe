mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaFp8Inputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let generate_stats = args().any(|argument| argument == "--stats");
    let implementation = if args().any(|argument| argument == "--composite") {
        AttentionImplementation::Composite
    } else {
        AttentionImplementation::Auto
    };
    let example = common::ExampleContext::create()?;
    if example.compute_capability()? < 90 {
        println!("frontend_fp8_sdpa: skipped, requires SM90 or newer");
        return Ok(());
    }
    let [b, h, s, d] = [2_i64, 2, 512, 128];
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    // Physical [B, S, 3, H, D] storage supplies three logical [B, H, S, D] views.
    let qkv_storage_shape = Shape::contiguous([b, s, 3, h, d])?;
    let strides = qkv_storage_shape.strides();
    let qkv_shape = Shape::with_strides(
        [b, h, s, d],
        [strides[0], strides[3], strides[1], strides[4]],
    )?;
    let offset_error = || Error::OutOfRange {
        name: "interleaved qkv byte offset".into(),
    };
    // E4M3 descriptor elements occupy one byte.
    let key_offset = u64::try_from(strides[2]).map_err(|_| offset_error())?;
    let value_offset = key_offset.checked_mul(2).ok_or_else(offset_error)?;
    let qkv_storage =
        graph.tensor(TensorSpec::new(DataType::F8E4M3, qkv_storage_shape).with_name("QKV"));
    let q = graph.storage_view(
        qkv_storage,
        TensorSpec::new(DataType::F8E4M3, qkv_shape.clone()).with_name("Q"),
        0,
    )?;
    let k = graph.storage_view(
        qkv_storage,
        TensorSpec::new(DataType::F8E4M3, qkv_shape.clone()).with_name("K"),
        key_offset,
    )?;
    let v = graph.storage_view(
        qkv_storage,
        TensorSpec::new(DataType::F8E4M3, qkv_shape).with_name("V"),
        value_offset,
    )?;
    let descale_q = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1; 4])?).with_name("Descale_Q"));
    let descale_k = graph.tensor_like(descale_q, Some("Descale_K"))?;
    let descale_v = graph.tensor_like(descale_q, Some("Descale_V"))?;
    let descale_s = graph.tensor_like(descale_q, Some("Descale_S"))?;
    let scale_s = graph.tensor_like(descale_q, Some("Scale_S"))?;
    let scale_o = graph.tensor_like(descale_q, Some("Scale_O"))?;
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.123));
    let config = AttentionConfig::new()
        .with_implementation(implementation)
        .with_compute_type(DataType::F32)
        .with_name("sdpa_fp8")
        .with_score_config(AttentionScoreConfig::new().with_mask_mode(
            AttentionMaskMode::DiagonalBand(AttentionDiagonalBand::causal(
                AttentionDiagonalAlignment::TopLeft,
            )),
        ));
    let outputs = graph.sdpa_fp8(
        SdpaFp8Inputs::new(
            q,
            k,
            v,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_s,
            scale_s,
            scale_o,
        ),
        &config,
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(generate_stats.then_some(TensorOutputConfig::new().output_tensor()))
            .with_absolute_max_probabilities(TensorOutputConfig::new().output_tensor())
            .with_absolute_max_output(TensorOutputConfig::new().output_tensor()),
    )?;
    let output = outputs.output;
    let stats = outputs.stats;
    let absolute_max_s = outputs
        .absolute_max_probabilities
        .ok_or_else(|| FrontendError::Compile("requested score amax is missing".into()))?;
    let absolute_max_o = outputs
        .absolute_max_output
        .ok_or_else(|| FrontendError::Compile("requested output amax is missing".into()))?;
    let output_storage = Shape::contiguous([b, s, h, d])?;
    let strides = output_storage.strides();
    let output_shape = Shape::with_strides(
        [b, h, s, d],
        [strides[0], strides[2], strides[1], strides[3]],
    )?;
    graph.modify_tensor(output, |tensor| tensor.with_shape(output_shape))?;
    for id in [absolute_max_s, absolute_max_o] {
        let shape = Shape::contiguous([1; 4])?;
        graph.modify_tensor(id, |tensor| tensor.with_shape(shape).output_tensor())?;
    }
    if let Some(stats) = stats {
        graph.mark_as_output(stats)?;
    }
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let qkv_values =
        common::init_f8e4m3_image(graph.shape(qkv_storage)?.storage_span()?, &mut seed);
    let qkv_device = DeviceMemory::from_slice(cuda, &qkv_values)?;
    let scales = DeviceMemory::from_slice(cuda, &[1.0_f32])?;
    let mut output_device =
        DeviceMemory::<f8e4m3>::zeroed(cuda, graph.shape(output)?.storage_span()?)?;
    let mut score_amax =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(absolute_max_s)?.storage_span()?)?;
    let mut output_amax =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(absolute_max_o)?.storage_span()?)?;
    let mut statistics = if let Some(id) = stats {
        Some((
            id,
            DeviceMemory::<f32>::zeroed(cuda, graph.shape(id)?.storage_span()?)?,
        ))
    } else {
        None
    };
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(qkv_storage, &qkv_device)?;
    for id in [descale_q, descale_k, descale_v, descale_s, scale_s, scale_o] {
        bindings.set_input(id, &scales)?;
    }
    bindings.set(output, &mut output_device)?;
    bindings.set(absolute_max_s, &mut score_amax)?;
    bindings.set(absolute_max_o, &mut output_amax)?;
    if let Some((id, memory)) = &mut statistics {
        bindings.set(*id, memory)?;
    }
    // SAFETY: the interleaved QKV owner, initialized scales, writable outputs and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed fp8 attention with finite interleaved QKV, stats: {generate_stats}");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa", run())
}
