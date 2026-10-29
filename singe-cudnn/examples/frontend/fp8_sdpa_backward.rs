mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3},
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaForwardOutputTensors, SdpaFp8BackwardInputs, SdpaFp8BackwardOutputConfig,
            SdpaFp8Inputs, SdpaGradientOutputConfig,
        },
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionBackwardConfig, AttentionConfig, AttentionDiagonalAlignment,
            AttentionDiagonalBand, AttentionMaskMode, AttentionScoreConfig, SoftmaxStatsBase,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn interleaved_shape([b, h, s, d]: [i64; 4]) -> Result<Shape> {
    let storage = Shape::contiguous([b, s, 3, h, d])?;
    let strides = storage.strides();
    Shape::with_strides(
        [b, h, s, d],
        [strides[0], strides[3], strides[1], strides[4]],
    )
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 2_i64;
    let h_kv = 2_i64;
    let h = if args().any(|argument| argument == "--gqa") {
        4_i64
    } else {
        h_kv
    };
    let s = 512_i64;
    let d = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph
        .tensor(TensorSpec::new(DataType::F8E4M3, interleaved_shape([b, h, s, d])?).with_name("Q"));
    let k = graph.tensor(
        TensorSpec::new(DataType::F8E4M3, interleaved_shape([b, h_kv, s, d])?).with_name("K"),
    );
    let v = graph.tensor_like(k, Some("V"))?;
    let output_storage = Shape::contiguous([b, s, h, d])?;
    let strides = output_storage.strides();
    let o = graph.tensor(
        TensorSpec::new(
            DataType::F8E4M3,
            Shape::with_strides(
                [b, h, s, d],
                [strides[0], strides[2], strides[1], strides[3]],
            )?,
        )
        .with_name("O"),
    );
    let d_o = graph.tensor_like(o, Some("dO"))?;
    let stats = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h, s, 1])?).with_name("Stats"),
    );

    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0));
    let descale_k = graph.tensor_like(descale_q, Some("descale_k"))?;
    let descale_v = graph.tensor_like(descale_q, Some("descale_v"))?;
    let descale_o = graph.tensor_like(descale_q, Some("descale_o"))?;
    let descale_d_o = graph.tensor_like(descale_q, Some("descale_d_o"))?;
    let descale_s = graph.tensor_like(descale_q, Some("descale_s"))?;
    let descale_d_p = graph.tensor_like(descale_q, Some("descale_d_p"))?;
    let scale_s = graph.tensor_like(descale_q, Some("scale_s"))?;
    let scale_o = graph.tensor_like(descale_q, Some("scale_o"))?;
    let scale_d_q = graph.tensor_like(descale_q, Some("scale_d_q"))?;
    let scale_d_k = graph.tensor_like(descale_q, Some("scale_d_k"))?;
    let scale_d_v = graph.tensor_like(descale_q, Some("scale_d_v"))?;
    let scale_d_p = graph.tensor_like(descale_q, Some("scale_d_p"))?;
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.123));

    let score_config = AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::DiagonalBand(
        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
    ));
    let mut forward_graph = graph.clone();
    let forward_outputs = forward_graph.sdpa_fp8(
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
        &AttentionConfig::new()
            .with_name("fp8_attention_forward")
            .with_score_config(score_config.clone())
            .with_stats_base(SoftmaxStatsBase::NaturalLog),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
            .with_stats(TensorOutputConfig::new().output_tensor())
            .with_absolute_max_probabilities(TensorOutputConfig::new().output_tensor())
            .with_absolute_max_output(TensorOutputConfig::new().output_tensor()),
    )?;
    let forward_o = forward_outputs.output;
    let forward_stats = forward_outputs.stats.ok_or_else(|| {
        FrontendError::Compile("requested forward attention statistics are missing".into())
    })?;
    let output_shape = graph.shape(o)?.clone();
    forward_graph.modify_tensor(forward_o, |tensor| tensor.with_shape(output_shape))?;
    let stats_shape = graph.shape(stats)?.clone();
    forward_graph.modify_tensor(forward_stats, |tensor| {
        tensor.with_shape(stats_shape).output_tensor()
    })?;
    let forward_amax_ids = [
        forward_outputs.absolute_max_probabilities.ok_or_else(|| {
            FrontendError::Compile("requested forward score amax is missing".into())
        })?,
        forward_outputs.absolute_max_output.ok_or_else(|| {
            FrontendError::Compile("requested forward output amax is missing".into())
        })?,
    ];

    let outputs = graph.sdpa_fp8_backward(
        SdpaFp8BackwardInputs::new(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            attention_scale,
            descale_q,
            descale_k,
            descale_v,
            descale_o,
            descale_d_o,
            descale_s,
            descale_d_p,
            scale_s,
            scale_d_q,
            scale_d_k,
            scale_d_v,
            scale_d_p,
        ),
        AttentionBackwardConfig::new()
            .with_name("fp8_attention_backward")
            .with_compute_type(DataType::F32)
            .with_score_config(score_config),
        SdpaFp8BackwardOutputConfig::new(SdpaGradientOutputConfig::new(
            TensorOutputConfig::new().output_tensor(),
        )),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

    let output_ids = [
        d_q,
        d_k,
        d_v,
        absolute_max_d_q,
        absolute_max_d_k,
        absolute_max_d_v,
        absolute_max_d_p,
    ];
    for output in output_ids {
        graph.mark_as_output(output)?;
    }

    let forward_compiled = forward_graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789;
    let inputs = [q, k, v]
        .into_iter()
        .map(|id| {
            let elements = graph.shape(id)?.storage_span()?;
            let values = common::init_f8e4m3_image(elements, &mut seed);
            Ok((id, DeviceMemory::from_slice(cuda, &values)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let gradient_values = common::init_f8e4m3_image(graph.shape(d_o)?.storage_span()?, &mut seed);
    let d_o_dev = DeviceMemory::from_slice(cuda, &gradient_values)?;
    let mut o_dev = DeviceMemory::<f8e4m3>::zeroed(cuda, graph.shape(o)?.storage_span()?)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroed(cuda, graph.shape(stats)?.storage_span()?)?;
    {
        let mut amax_buffers = forward_amax_ids
            .into_iter()
            .map(|id| {
                Ok((
                    id,
                    DeviceMemory::<f32>::zeroed(cuda, forward_graph.shape(id)?.storage_span()?)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut workspace = common::workspace(cuda, forward_compiled.workspace_size())?;
        let mut bindings = forward_compiled.bindings()?;
        for (id, memory) in &inputs {
            bindings.set_input(*id, memory)?;
        }
        bindings.set(forward_o, &mut o_dev)?;
        bindings.set(forward_stats, &mut stats_dev)?;
        for (id, memory) in &mut amax_buffers {
            bindings.set(*id, memory)?;
        }
        // SAFETY: inputs, writable saved state and forward scratch remain live through stream completion.
        example.execute_and_wait(|context| unsafe {
            forward_compiled.execute(
                context,
                ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            )
        })?;
    }
    let mut output_buffers = output_ids
        .into_iter()
        .map(|id| {
            let bytes = graph.tensor_spec(id)?.storage_byte_len(None)?;
            Ok((id, DeviceMemory::<u8>::zeroed(cuda, bytes)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    for (id, memory) in &inputs {
        bindings.set_input(*id, memory)?;
    }
    bindings.set_input(o, &o_dev)?;
    bindings.set_input(d_o, &d_o_dev)?;
    bindings.set_input(stats, &stats_dev)?;
    for (id, memory) in &mut output_buffers {
        bindings.set_encoded_bytes(*id, memory)?;
    }
    // SAFETY: FP8 inputs, actual saved forward state, writable gradients and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed fp8 forward and backward attention with saved statistics, query heads: {h}");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_backward", run())
}
