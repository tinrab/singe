mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e4m3},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaFp8Inputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let output_type = if args().any(|argument| argument == "--fp16-output") {
        DataType::F16
    } else {
        DataType::BF16
    };
    let implementation = if args().any(|argument| argument == "--composite") {
        AttentionImplementation::Composite
    } else if args().any(|argument| argument == "--unified") {
        AttentionImplementation::Unified
    } else {
        AttentionImplementation::Auto
    };
    let b = 2_i64;
    let h = 2_i64;
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

    let qkvo_dims = [b, h, s, d];
    let qkv_strides = [s * 3 * h * d, d, 3 * h * d, 1];
    let o_strides = [s * h * d, d, h * d, 1];

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::with_strides(qkvo_dims, qkv_strides)?,
    ));
    let k = graph.tensor_like(q, Some("K"))?;
    let v = graph.tensor_like(q, Some("V"))?;

    let descale_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([1, 1, 1, 1], [1, 1, 1, 1])?,
    ));
    let descale_k = graph.tensor_like(descale_q, Some("descale_k"))?;
    let descale_v = graph.tensor_like(descale_q, Some("descale_v"))?;
    let descale_s = graph.tensor_like(descale_q, Some("descale_s"))?;
    let attention_scale = graph.tensor_like(descale_q, Some("attention_scale"))?;
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0));
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0));

    let inputs = SdpaFp8Inputs::new(
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
    );
    let config = AttentionConfig::new()
        .with_name("sdpa_fp8_current_scaling")
        .with_implementation(implementation)
        .with_score_config(AttentionScoreConfig::new().with_mask_mode(
            AttentionMaskMode::DiagonalBand(AttentionDiagonalBand::causal(
                AttentionDiagonalAlignment::TopLeft,
            )),
        ));
    let output_shape = Shape::with_strides(qkvo_dims, o_strides)?;
    let summaries = args().any(|argument| argument == "--softmax-summaries");
    let (o, stats, absolute_max_s, absolute_max_o, logit_max, score_sum_exp) =
        if args().any(|argument| argument == "--explicit-output") {
            let output = graph.tensor(TensorSpec::new(output_type, output_shape).with_name("O"));
            let stats = graph.tensor(
                TensorSpec::new(DataType::F32, Shape::contiguous([b, h, s, 1])?)
                    .with_name("saved_stats"),
            );
            let amax = TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?);
            let amax_s = graph.tensor(amax.clone().with_name("probability_amax"));
            let amax_o = graph.tensor(amax.with_name("output_amax"));
            let (logit_max, score_sum_exp) = if summaries {
                let summary = TensorSpec::new(DataType::F32, Shape::contiguous([b, h, s, 1])?);
                (
                    Some(graph.tensor(summary.clone().with_name("logit_max_fp32"))),
                    Some(graph.tensor(summary.with_name("sum_exp_fp32"))),
                )
            } else {
                (None, None)
            };
            graph.sdpa_fp8_into(
                inputs,
                SdpaForwardOutputTensors {
                    stats: Some(stats),
                    absolute_max_probabilities: Some(amax_s),
                    absolute_max_output: Some(amax_o),
                    logit_max,
                    score_sum_exp,
                    ..SdpaForwardOutputTensors::new(output)
                },
                &config,
            )?;
            (
                output,
                Some(stats),
                Some(amax_s),
                Some(amax_o),
                logit_max,
                score_sum_exp,
            )
        } else {
            let outputs = graph.sdpa_fp8(
                inputs,
                &config,
                SdpaForwardOutputTensors::new(
                    TensorOutputConfig::new()
                        .with_data_type(output_type)
                        .output_tensor(),
                )
                .with_stats(TensorOutputConfig::new().output_tensor())
                .with_absolute_max_probabilities(TensorOutputConfig::new().output_tensor())
                .with_absolute_max_output(TensorOutputConfig::new().output_tensor())
                .with_logit_max(
                    summaries.then_some(
                        TensorOutputConfig::new()
                            .with_data_type(DataType::F32)
                            .output_tensor(),
                    ),
                )
                .with_score_sum_exp(
                    summaries.then_some(
                        TensorOutputConfig::new()
                            .with_data_type(DataType::F32)
                            .output_tensor(),
                    ),
                ),
            )?;
            graph.modify_tensor(outputs.output, |tensor| tensor.with_shape(output_shape))?;
            (
                outputs.output,
                outputs.stats,
                outputs.absolute_max_probabilities,
                outputs.absolute_max_output,
                outputs.logit_max,
                outputs.score_sum_exp,
            )
        };

    for output in [
        stats,
        absolute_max_s,
        absolute_max_o,
        logit_max,
        score_sum_exp,
    ]
    .into_iter()
    .flatten()
    {
        graph.mark_as_output(output)?;
    }

    let head_elements = Shape::contiguous([h, d])?.element_count()?;
    let value_offset = head_elements.checked_mul(2).ok_or(Error::OutOfRange {
        name: "interleaved value offset".into(),
    })?;
    let qkv_elements = graph
        .shape(q)?
        .storage_span()?
        .checked_add(value_offset)
        .ok_or(Error::OutOfRange {
            name: "interleaved qkv storage".into(),
        })?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let qkv_dev = DeviceMemory::from_slice(cuda, &vec![f8e4m3::from_bits(0x30); qkv_elements])?;
    let descale_dev = DeviceMemory::from_slice(cuda, &[1.0_f32])?;
    let descale_s_dev = DeviceMemory::from_slice(cuda, &[1.0_f32 / 448.0])?;
    let attention_scale_dev = DeviceMemory::from_slice(cuda, &[0.123_f32])?;
    let mut output_buffers = [
        Some(o),
        stats,
        absolute_max_s,
        absolute_max_o,
        logit_max,
        score_sum_exp,
    ]
    .into_iter()
    .flatten()
    .map(|id| {
        let bytes = graph.tensor_spec(id)?.storage_byte_len(None)?;
        Ok((id, DeviceMemory::<u8>::zeroed(cuda, bytes)?))
    })
    .collect::<Result<Vec<_>>>()?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &qkv_dev.view())?;
    bindings.set_input(k, &qkv_dev.view().slice(head_elements..)?)?;
    bindings.set_input(v, &qkv_dev.view().slice(value_offset..)?)?;
    for descale in [descale_q, descale_k, descale_v] {
        bindings.set_input(descale, &descale_dev.view())?;
    }
    bindings.set_input(descale_s, &descale_s_dev.view())?;
    bindings.set_input(attention_scale, &attention_scale_dev.view())?;
    for (id, memory) in &mut output_buffers {
        bindings.set_encoded_bytes(*id, memory)?;
    }
    // SAFETY: initialized inputs, encoded output allocations, host scales and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_current_scaling", run())
}
