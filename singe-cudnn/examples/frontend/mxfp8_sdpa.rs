mod common;

use std::env::args;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaMxfp8Inputs},
        graph::{
            BlockScaleQuantizeOutputs, DataTypePolicy, Graph, GraphConfig, TensorOutputConfig,
        },
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionImplementation, AttentionMaskMode, AttentionPagedKvCache,
            AttentionScoreConfig, BlockScaleQuantizeConfig, MatmulConfig, SoftmaxStatsBase,
        },
        plan::{CompileConfig, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;

    if !(100..110).contains(&context.compute_capability()?) {
        println!("frontend_mxfp8_sdpa: skipped, requires Blackwell Computing GPU");
        return Ok(());
    }

    let generated_scales = args().any(|argument| argument == "--generated-scales");
    let project_output = args().any(|argument| argument == "--project-output");
    let output_type = if args().any(|argument| argument == "--fp16-output") {
        DataType::F16
    } else {
        DataType::BF16
    };
    let b = 2_i64;
    let h = 2_i64;
    let s = 512_i64;
    let d = 128_i64;
    let block_size = 32_i64;
    let d_scale = (d + block_size - 1) / block_size;
    let s_scale = (s + block_size - 1) / block_size;
    let s_padded = ((s + 127) / 128) * 128;
    let d_scale_padded = ((d_scale + 3) / 4) * 4;
    let s_scale_padded = ((s_scale + 3) / 4) * 4;
    let d_padded = ((d + 127) / 128) * 128;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let (q, k, v, scale_q, scale_k, scale_v) = if generated_scales {
        let input_spec = TensorSpec::new(DataType::F16, Shape::contiguous([b, h, s, d])?);
        let input_q = graph.tensor(input_spec.clone().with_name("q_f16"));
        let input_k = graph.tensor(input_spec.with_name("k_f16"));
        // V's sequence axis is contiguous because its quantization blocks run along sequence.
        let input_v = graph.tensor(
            TensorSpec::new(
                DataType::F16,
                Shape::with_strides([b, h, s, d], [h * s * d, s * d, 1, s])?,
            )
            .with_name("v_f16"),
        );
        let quantized_outputs = BlockScaleQuantizeOutputs::new(
            TensorOutputConfig::new().with_data_type(DataType::F8E4M3),
            TensorOutputConfig::new().with_data_type(DataType::F8UE8M0),
        );
        let query = graph.block_scale_quantize(
            input_q,
            quantized_outputs,
            BlockScaleQuantizeConfig::new(block_size).with_axis(3),
        )?;
        let key = graph.block_scale_quantize(
            input_k,
            quantized_outputs,
            BlockScaleQuantizeConfig::new(block_size).with_axis(3),
        )?;
        let value = graph.block_scale_quantize(
            input_v,
            quantized_outputs,
            BlockScaleQuantizeConfig::new(block_size).with_axis(2),
        )?;
        // Quantized values and scales stay virtual and are consumed by the composite graph.
        (
            query.output,
            key.output,
            value.output,
            query.scale,
            key.scale,
            value.scale,
        )
    } else {
        let qkv_dims = vec![b, h, s, d];
        let qkv_strides = vec![s * 3 * h * d, d, 3 * h * d, 1];
        let q = graph.tensor(TensorSpec::new(
            DataType::F8E4M3,
            Shape::with_strides(qkv_dims.clone(), qkv_strides.clone())?,
        ));
        let k = graph.tensor(TensorSpec::new(
            DataType::F8E4M3,
            Shape::with_strides(qkv_dims.clone(), qkv_strides.clone())?,
        ));
        let v = graph.tensor(TensorSpec::new(
            DataType::F8E4M3,
            Shape::with_strides(qkv_dims, qkv_strides)?,
        ));
        let scale_q = graph.tensor(
            TensorSpec::new(
                DataType::F8UE8M0,
                Shape::contiguous([b, h, s_padded, d_scale_padded])?,
            )
            .with_f8_128x4_reordering(),
        );
        let scale_k = graph.tensor(
            TensorSpec::new(
                DataType::F8UE8M0,
                Shape::contiguous([b, h, s_padded, d_scale_padded])?,
            )
            .with_f8_128x4_reordering(),
        );
        let scale_v = graph.tensor(
            TensorSpec::new(
                DataType::F8UE8M0,
                Shape::contiguous([b, h, s_scale_padded, d_padded])?,
            )
            .with_f8_128x4_reordering(),
        );
        (q, k, v, scale_q, scale_k, scale_v)
    };
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0));

    let length_hint = args()
        .any(|argument| argument == "--dense-length-hint")
        .then_some(s);
    let output = TensorOutputConfig::new().with_data_type(output_type);
    let stats_base = SoftmaxStatsBase::Log2;
    let outputs = graph.sdpa_mxfp8(
        SdpaMxfp8Inputs::new(q, k, v, scale_q, scale_k, scale_v, attention_scale),
        &AttentionConfig::new()
            .with_implementation(if generated_scales {
                AttentionImplementation::Composite
            } else {
                AttentionImplementation::Auto
            })
            .with_name("sdpa_mxfp8")
            .with_paged_cache(
                AttentionPagedKvCache::new().with_max_sequence_length_key_value(length_hint),
            )
            .with_score_config(AttentionScoreConfig::new().with_mask_mode(
                AttentionMaskMode::DiagonalBand(AttentionDiagonalBand::causal(
                    AttentionDiagonalAlignment::TopLeft,
                )),
            ))
            .with_stats_base(stats_base),
        SdpaForwardOutputTensors::new(if project_output {
            output
        } else {
            output.output_tensor()
        })
        .with_stats(TensorOutputConfig::new().output_tensor())
        .with_absolute_max_output(TensorOutputConfig::new().output_tensor()),
    )?;
    let o = if project_output {
        let weights = graph.tensor(
            TensorSpec::new(output_type, Shape::contiguous([1, 1, d, d])?)
                .with_name("projection_weights"),
        );
        graph.matmul(
            outputs.output,
            weights,
            MatmulConfig::new().with_compute_type(DataType::F32),
            TensorOutputConfig::new()
                .with_data_type(output_type)
                .output_tensor(),
        )?
    } else {
        outputs.output
    };
    let stats = outputs.stats;
    let absolute_max_o = outputs.absolute_max_output;
    println!("MXFP8 statistics encoding: {:?}", stats_base);

    for output in [Some(o), stats, absolute_max_o].into_iter().flatten() {
        graph.mark_as_output(output)?;
    }

    graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_mxfp8_sdpa", run())
}
