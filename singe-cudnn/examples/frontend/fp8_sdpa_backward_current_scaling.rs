mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::SdpaFp8BackwardInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionBackwardConfig, AttentionMaskMode, AttentionScoreConfig, HeuristicMode,
        },
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12000 {
        println!(
            "frontend_fp8_sdpa_backward_current_scaling: skipped, requires CUDA toolkit 12.0+"
        );
        return Ok(());
    }

    let properties = ctx.cudnn.cuda_context().device().properties()?;
    let sm = properties.major * 10 + properties.minor;
    if sm < 100 || sm >= 110 {
        println!(
            "frontend_fp8_sdpa_backward_current_scaling: skipped, current scaling requires SM 100.x"
        );
        return Ok(());
    }
    if version()? < 91300 {
        println!("frontend_fp8_sdpa_backward_current_scaling: skipped, requires cuDNN 9.13+");
        return Ok(());
    }

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

    let qkvo_dims = vec![b, h, s, d];
    let qkv_strides = vec![s * 3 * h * d, d, 3 * h * d, 1];
    let o_strides = vec![s * h * d, d, h * d, 1];

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkvo_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkvo_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkvo_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkvo_dims.clone())?.with_strides(o_strides.clone())?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkvo_dims)?.with_strides(o_strides.clone())?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h, s, 1])?.with_strides([h * s, s, 1, 1])?,
    ));

    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor_like_named(descale_q, "descale_k")?;
    let descale_v = graph.tensor_like_named(descale_q, "descale_v")?;
    let descale_o = graph.tensor_like_named(descale_q, "descale_o")?;
    let descale_d_o = graph.tensor_like_named(descale_q, "descale_d_o")?;
    let descale_s = graph.tensor_like_named(descale_q, "descale_s")?;
    let descale_d_p = graph.tensor_like_named(descale_q, "descale_d_p")?;
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0)?);
    let scale_d_q = graph.tensor_like_named(descale_q, "scale_d_q")?;
    let scale_d_k = graph.tensor_like_named(descale_q, "scale_d_k")?;
    let scale_d_v = graph.tensor_like_named(descale_q, "scale_d_v")?;
    let scale_d_p = graph.tensor_like_named(descale_q, "scale_d_p")?;

    let outputs = graph.sdpa_fp8_backward_infer_with_aux_and_output_types(
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
        DataType::BF16,
        DataType::BF16,
        DataType::BF16,
        AttentionBackwardConfig::new(DataType::F32).with_score_config(
            AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::CausalTopLeft),
        ),
    )?;
    let d_q = outputs.query_gradient();
    let d_k = outputs.key_gradient();
    let d_v = outputs.value_gradient();
    let absolute_max_d_q = outputs.absolute_max_query_gradient();
    let absolute_max_d_k = outputs.absolute_max_key_gradient();
    let absolute_max_d_v = outputs.absolute_max_value_gradient();
    let absolute_max_d_p = outputs.absolute_max_probability_gradient();

    for output in [
        d_q,
        d_k,
        d_v,
        absolute_max_d_q,
        absolute_max_d_k,
        absolute_max_d_v,
        absolute_max_d_p,
    ] {
        graph.mark_as_output(output)?;
    }

    let _ = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_backward_current_scaling", run())
}
