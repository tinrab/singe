mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaFp8BackwardInputs,
        graph::Graph,
        operation::{AttentionBackwardConfig, HeuristicMode},
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!(
            "frontend_fp8_sdpa_backward_bottom_right_causal: skipped, requires CUDA toolkit 12.0+"
        );
        return Ok(());
    }

    let properties = ctx.cudnn.cuda_context().device().properties()?;
    let sm = properties.major * 10 + properties.minor;
    if !(100..=120).contains(&sm) {
        println!("frontend_fp8_sdpa_backward_bottom_right_causal: skipped, requires SM 100.x");
        return Ok(());
    }
    if version()? < 90_700 {
        println!(
            "frontend_fp8_sdpa_backward_bottom_right_causal: skipped, graph is not supported on this cuDNN/GPU combination"
        );
        return Ok(());
    }

    let b = 1_i64;
    let h = 2_i64;
    let s_q = 64_i64;
    let s_kv = 128_i64;
    let d = 128_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F8E4M3)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_q, d])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_kv, d])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_kv, d])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_q, d])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_q, d])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h, s_q, 1])?,
    ));

    let attention_scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor_like_named(descale_q, "descale_k")?;
    let descale_v = graph.tensor_like_named(descale_q, "descale_v")?;
    let descale_o = graph.tensor_like_named(descale_q, "descale_o")?;
    let descale_d_o = graph.tensor_like_named(descale_q, "descale_d_o")?;
    let descale_s = graph.tensor_like_named(descale_q, "descale_s")?;
    let descale_d_p = graph.tensor_like_named(descale_q, "descale_d_p")?;
    let scale_s = graph.tensor_like_named(descale_q, "scale_s")?;
    let scale_d_q = graph.tensor_like_named(descale_q, "scale_d_q")?;
    let scale_d_k = graph.tensor_like_named(descale_q, "scale_d_k")?;
    let scale_d_v = graph.tensor_like_named(descale_q, "scale_d_v")?;
    let scale_d_p = graph.tensor_like_named(descale_q, "scale_d_p")?;
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([1, 1, 1, 1])?,
    ));

    let outputs = match graph.sdpa_fp8_backward_infer(
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
        AttentionBackwardConfig::new(DataType::F32)
            .with_causal_bottom_right(sequence_length_query, sequence_length_key_value),
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            if matches!(
                error,
                Error::Cudnn {
                    code,
                    ..
                } if code == Status::NotSupported
                    || code == Status::NotSupportedGraphPattern
                    || code == Status::NotSupportedShape
                    || code == Status::NotSupportedDataType
                    || code == Status::NotSupportedLayout
            ) {
                println!("frontend_fp8_sdpa_backward_bottom_right_causal: skipped, not supported");
                return Ok(());
            }
            return Err(error);
        }
    };
    let d_q = outputs.query_gradient;
    let d_k = outputs.key_gradient;
    let d_v = outputs.value_gradient;
    let absolute_max_d_q = outputs.absolute_max_query_gradient;
    let absolute_max_d_k = outputs.absolute_max_key_gradient;
    let absolute_max_d_v = outputs.absolute_max_value_gradient;
    let absolute_max_d_p = outputs.absolute_max_probability_gradient;

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

    if let Err(error) = graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        if matches!(
            error,
            Error::Cudnn {
                code,
                ..
            } if code == Status::NotSupported
                || code == Status::NotSupportedGraphPattern
                || code == Status::NotSupportedShape
                || code == Status::NotSupportedDataType
                || code == Status::NotSupportedLayout
        ) {
            println!(
                "frontend_fp8_sdpa_backward_bottom_right_causal: skipped, not supported on this configuration"
            );
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_backward_bottom_right_causal", run())
}
