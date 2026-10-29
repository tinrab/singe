mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionImplementation, AttentionScoreConfig, PointwiseOperation,
            SdpaScoreSubgraph,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 16_i64;
    let h_q = 32_i64;
    let h_k = 32_i64;
    let h_v = 32_i64;
    let s_q = 2048_i64;
    let s_kv = 2048_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_override_shape(true)
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let q = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides(
                [b, h_q, s_q, d_qk],
                vec![h_q * s_q * d_qk, s_q * d_qk, d_qk, 1],
            )?,
        ),
    )?;
    let k = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides(
                [b, h_k, s_kv, d_qk],
                vec![h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1],
            )?,
        ),
    )?;
    let v = graph.insert_tensor(
        TensorId::new(1003),
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides(
                [b, h_v, s_kv, d_v],
                vec![h_v * s_kv * d_v, s_kv * d_v, d_v, 1],
            )?,
        ),
    )?;
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides([1, 1, 1, 1], [1, 1, 1, 1])?,
    ));
    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h_q, s_q, s_kv])?,
    ));
    let score_with_shift = if args().any(|argument| argument == "--parent-score-shift") {
        let shift = graph.tensor(
            TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?).virtual_tensor(),
        );
        graph.pointwise_into(PointwiseOperation::Binary {
            name: None,
            mode: PointwiseMode::Mul,
            lhs: scale,
            rhs: scale,
            output: shift,
            compute_type: Some(DataType::F32),
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        })?;
        score_subgraph.insert_tensor(shift, graph.tensor_spec(shift)?.clone())?;
        let shifted = score_subgraph.tensor(
            TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?)
                .virtual_tensor(),
        );
        score_subgraph.pointwise_into(PointwiseOperation::Binary {
            name: None,
            mode: PointwiseMode::Add,
            lhs: score_input,
            rhs: shift,
            output: shifted,
            compute_type: Some(DataType::F32),
            nan_propagation: NanPropagation::Propagate,
            alpha1: 1.0,
            alpha2: 1.0,
        })?;
        shifted
    } else {
        score_input
    };
    let bias = score_subgraph.insert_tensor(
        TensorId::new(1006),
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides([b, 1, s_q, s_kv], vec![s_q * s_kv, s_q * s_kv, s_kv, 1])?,
        ),
    )?;
    let biased = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Add,
        lhs: score_with_shift,
        rhs: bias,
        output: biased,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;
    let soft_cap = if args().any(|argument| argument == "--captured-scalar-alignment") {
        let source = TensorSpec::scalar_f32(0.8);
        let soft_cap = graph.tensor(source.clone());
        score_subgraph.insert_tensor(soft_cap, source.with_byte_alignment(128)?)?;
        soft_cap
    } else {
        score_subgraph.tensor(TensorSpec::scalar_f32(0.8))
    };
    let divided = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Div,
        lhs: biased,
        rhs: soft_cap,
        output: divided,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;
    let activated = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::TanhFwd,
        input: divided,
        output: activated,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    })?;
    let score_output = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Mul,
        lhs: activated,
        rhs: soft_cap,
        output: score_output,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

    let mut score_config = AttentionScoreConfig::new();
    if args().any(|argument| argument == "--built-in-bias") {
        // Reuse the captured bias after the nonlinear custom score transform.
        // This is a separate addition from the bias already inside the soft cap.
        graph.insert_tensor(bias, score_subgraph.tensor_spec(bias)?.clone())?;
        score_config = score_config.with_bias(bias);
    }
    let score_config = score_config.with_score_subgraph(SdpaScoreSubgraph::new(
        score_subgraph,
        score_input,
        score_output,
    ));
    let implementation = if args().any(|argument| argument == "--composite") {
        AttentionImplementation::Composite
    } else if args().any(|argument| argument == "--unified") {
        AttentionImplementation::Unified
    } else {
        AttentionImplementation::Auto
    };
    let inputs = SdpaInputs {
        query: q,
        key: k,
        value: v,
        scale,
    };
    let config = AttentionConfig::new()
        .with_compute_type(DataType::F32)
        .with_implementation(implementation)
        .with_score_config(score_config);
    let output_shape = Shape::with_strides([b, h_q, s_q, d_v], [h_q * d_v, d_v, b * h_q * d_v, 1])?;
    let summaries = args().any(|argument| argument == "--softmax-summaries");
    let (o, stats, logit_max, score_sum_exp) =
        if summaries || args().any(|argument| argument == "--explicit-output") {
            let o = graph.tensor(
                TensorSpec::new(DataType::F16, output_shape)
                    .with_name("attention_result")
                    .output_tensor(),
            );
            let stats = graph.tensor(
                TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, 1])?)
                    .with_name("saved_stats")
                    .output_tensor(),
            );
            let (logit_max, score_sum_exp) = if summaries {
                let summary = TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, 1])?)
                    .output_tensor();
                (
                    Some(graph.tensor(summary.clone().with_name("logit_max_fp32"))),
                    Some(graph.tensor(summary.with_name("sum_exp_fp32"))),
                )
            } else {
                (None, None)
            };
            graph.sdpa_auto_into(
                inputs,
                SdpaForwardOutputTensors {
                    stats: Some(stats),
                    logit_max,
                    score_sum_exp,
                    ..SdpaForwardOutputTensors::new(o)
                },
                &config,
            )?;
            (o, stats, logit_max, score_sum_exp)
        } else {
            let outputs = graph.sdpa_auto(
                inputs,
                config,
                SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor())
                    .with_stats(TensorOutputConfig::new().output_tensor()),
            )?;
            let stats = outputs
                .stats
                .ok_or_else(|| FrontendError::SdpaOptionalOutputMismatch {
                    output: "attention statistics",
                })?;
            graph.modify_tensor(outputs.output, |tensor| {
                tensor.with_shape(output_shape).output_tensor()
            })?;
            graph.modify_tensor(stats, TensorSpec::output_tensor)?;
            (outputs.output, stats, None, None)
        };
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let bias_row_stride = s_kv + 16;
    let bias_layout = Shape::with_strides(
        [b, 1, s_q, s_kv],
        [
            s_q * bias_row_stride,
            s_q * bias_row_stride,
            bias_row_stride,
            1,
        ],
    )?;
    let bias_elements = bias_layout.storage_span()?;
    let mut overrides = compiled.runtime_overrides();
    overrides.set(bias, bias_layout)?;
    let overrides = overrides.build();

    let mut seed = 123_456_789_u32;
    let q_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(q)?.storage_span()?, &mut seed),
    )?;
    let k_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(k)?.storage_span()?, &mut seed),
    )?;
    let v_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(v)?.storage_span()?, &mut seed),
    )?;
    let scale_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &[0.123_f32])?;
    let bias_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f32_image(bias_elements, &mut seed),
    )?;
    let mut output_buffers = [Some(o), Some(stats), logit_max, score_sum_exp]
        .into_iter()
        .flatten()
        .map(|id| {
            let bytes = graph.tensor_spec(id)?.storage_byte_len(None)?;
            Ok((
                id,
                DeviceMemory::<u8>::zeroed(example.cudnn.cuda_context(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut workspace = common::workspace(
        example.cudnn.cuda_context(),
        compiled.workspace_size_with_runtime_overrides(&example.cudnn, &overrides)?,
    )?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(q, &q_dev.view())?;
    bindings.set_input(k, &k_dev.view())?;
    bindings.set_input(v, &v_dev.view())?;
    bindings.set_input(scale, &scale_dev.view())?;
    bindings.set_input(bias, &bias_dev.view())?;
    for (id, buffer) in &mut output_buffers {
        bindings.set_encoded_bytes(*id, buffer)?;
    }
    // SAFETY: initialized inputs, writable outputs, overrides and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings)
                .with_workspace(&mut workspace)
                .with_overrides(&overrides),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_flexible_graphs", run())
}
