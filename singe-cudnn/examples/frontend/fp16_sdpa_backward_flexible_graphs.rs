mod common;

use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{SdpaBackwardInputs, SdpaGradientOutputConfig},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionBackwardConfig, AttentionScoreConfig, PointwiseOperation, SdpaScoreSubgraph,
        },
        plan::{CompileConfig, HeuristicMode},
    },
    math::NanPropagation,
};

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let properties = context.cudnn.cuda_context().device().properties()?;
    let sm = properties.major * 10 + properties.minor;

    if sm < 90 {
        println!("skipping: requires Hopper or above");
        return Ok(());
    }
    if sm == 120 {
        println!(
            "skipping: backward pass with flexible graphs is not supported on SM version 120 yet"
        );
        return Ok(());
    }

    let b = 2_i64;
    let h = 4_i64;
    let s_q = 128_i64;
    let s_kv = 128_i64;
    let d_qk = 64_i64;
    let d_v = 64_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, h, s_q, d_qk])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, h, s_kv, d_qk])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, h, s_kv, d_v])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, h, s_q, d_v])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, h, s_q, d_v])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h, s_q, 1])?,
    ));
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123));

    let mut score_subgraph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(DataTypePolicy::new().with_compute(DataType::F32)),
    );
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h, s_q, s_kv])?,
    ));
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8));
    let divided = score_subgraph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Div,
            score_input,
            soft_cap,
            TensorOutputConfig::new().virtual_tensor(),
        )
        .with_compute_type(None),
    )?;
    let activated = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::TanhFwd,
        input: divided,
        output: activated,
        compute_type: None,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    })?;
    let score_output = score_subgraph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            activated,
            soft_cap,
            TensorOutputConfig::new().virtual_tensor(),
        )
        .with_compute_type(None),
    )?;

    let mut score_subgraph_bprop = Graph::new();
    let bprop_input = score_subgraph_bprop.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h, s_q, s_kv])?,
    ));
    let bprop_scale = score_subgraph_bprop.tensor(TensorSpec::scalar_f32(0.5));
    let bprop_output = score_subgraph_bprop.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            bprop_input,
            bprop_scale,
            TensorOutputConfig::new().virtual_tensor(),
        )
        .with_compute_type(None),
    )?;

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
            .with_score_config(AttentionScoreConfig::new().with_score_subgraph(
                SdpaScoreSubgraph::new(score_subgraph, score_input, score_output),
            ))
            .with_score_subgraph_bprop(SdpaScoreSubgraph::new(
                score_subgraph_bprop,
                bprop_input,
                bprop_output,
            )),
        SdpaGradientOutputConfig::new(TensorOutputConfig::new().output_tensor()),
    )?;
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;

    for output in [d_q, d_k, d_v] {
        graph.mark_as_output(output)?;
    }

    let _ = graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_backward_flexible_graphs", run())
}
