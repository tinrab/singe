mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::SdpaFp8Inputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, SdpaAuxOutputRequest, SdpaConfig, SdpaFusedMaskMode},
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12000 {
        println!("frontend_fp8_sdpa_current_scaling: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }

    let properties = ctx.cudnn.cuda_context().device().properties()?;
    let sm = properties.major * 10 + properties.minor;
    if sm < 100 || sm >= 110 {
        println!("frontend_fp8_sdpa_current_scaling: skipped, current scaling requires SM 100.x");
        return Ok(());
    }
    if version()? < 91300 {
        println!("frontend_fp8_sdpa_current_scaling: skipped, requires cuDNN 9.13+");
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
        Shape::contiguous(qkvo_dims.clone())?.with_strides(qkv_strides)?,
    ));

    let descale_q = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let descale_k = graph.tensor_like_named(descale_q, "descale_k")?;
    let descale_v = graph.tensor_like_named(descale_q, "descale_v")?;
    let descale_s = graph.tensor_like_named(descale_q, "descale_s")?;
    let scale_s = graph.tensor(TensorSpec::scalar_f32(448.0)?);
    let scale_o = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = graph.sdpa_fp8_infer_with_output_type(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        DataType::BF16,
        &SdpaConfig::new()
            .with_name("sdpa_fp8_current_scaling")
            .with_attention_scale(0.123)
            .with_aux_outputs(SdpaAuxOutputRequest::stats_only())
            .with_mask(SdpaFusedMaskMode::CausalTopLeft)
            .with_aux_outputs(SdpaAuxOutputRequest::fp8_amax()),
    )?;
    let o = outputs.output();
    let stats = outputs.stats();
    let absolute_max_s = outputs.logit_max();
    let absolute_max_o = outputs.score_sum_exp();

    let mut o_tensor = graph.tensor_config(o)?.clone();
    o_tensor = o_tensor
        .output_tensor()
        .with_shape(Shape::contiguous(qkvo_dims)?.with_strides(o_strides)?);
    graph.replace_tensor(o, o_tensor)?;

    for output in [stats, absolute_max_s, absolute_max_o]
        .into_iter()
        .flatten()
    {
        graph.mark_as_output(output)?;
    }

    let _ = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_current_scaling", run())
}
