mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs},
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionBackwardConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionMaskMode, AttentionScoreConfig,
        },
        plan::{CompileConfig, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;

    if !(100..110).contains(&context.compute_capability()?) {
        println!("frontend_mxfp8_sdpa_backward: skipped, requires Blackwell Computing GPU");
        return Ok(());
    }
    // The attention request below overrides the graph's FP16 compute default with FP32.
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F16),
        ),
    );

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([2, 2, 512, 128])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 2, 512, 1])?,
    ));
    let scale_q = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 512, 4])?)
            .with_f8_128x4_reordering(),
    );
    let scale_q_t = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 16, 128])?)
            .with_f8_128x4_reordering(),
    );
    let scale_k = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 512, 4])?)
            .with_f8_128x4_reordering(),
    );
    let scale_k_t = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 16, 128])?)
            .with_f8_128x4_reordering(),
    );
    let scale_v = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 512, 4])?)
            .with_f8_128x4_reordering(),
    );
    let scale_d_o = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 512, 4])?)
            .with_f8_128x4_reordering(),
    );
    let scale_d_o_t = graph.tensor(
        TensorSpec::new(DataType::F8UE8M0, Shape::contiguous([2, 2, 16, 128])?)
            .with_f8_128x4_reordering(),
    );
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0));
    let sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));
    let d_sink_token = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 1, 1])?,
    ));

    let gradient = TensorSpec::new(DataType::BF16, Shape::contiguous([2, 2, 512, 128])?);
    let d_q = graph.tensor(gradient.clone());
    let d_k = graph.tensor(gradient.clone());
    let d_v = graph.tensor(gradient);
    let amax_d_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    graph.sdpa_mxfp8_backward_into(
        SdpaMxfp8BackwardInputs {
            query: q,
            sequence_scaled_query: q_t,
            key: k,
            sequence_scaled_key: k_t,
            value: v,
            output: o,
            output_gradient: d_o,
            quantized_output_gradient: d_o_quantized,
            sequence_scaled_output_gradient: d_o_t,
            stats,
            scale_query: scale_q,
            sequence_scale_query: scale_q_t,
            scale_key: scale_k,
            sequence_scale_key: scale_k_t,
            scale_value: scale_v,
            scale_output_gradient: scale_d_o,
            sequence_scale_output_gradient: scale_d_o_t,
            attention_scale,
        },
        SdpaMxfp8BackwardGradientTensors::new(d_q, d_k, d_v)
            .with_sink_token_gradient(d_sink_token)
            .with_absolute_max_query_gradient(amax_d_q),
        AttentionBackwardConfig::new()
            .with_compute_type(DataType::F32)
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_mask_mode(AttentionMaskMode::DiagonalBand(
                        AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
                    ))
                    .with_sink_token(sink_token),
            ),
    )?;

    for output in [d_q, d_k, d_v, d_sink_token, amax_d_q] {
        graph.mark_as_output(output)?;
    }

    graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_mxfp8_sdpa_backward", run())
}
