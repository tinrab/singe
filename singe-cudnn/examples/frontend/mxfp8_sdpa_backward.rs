mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaMxfp8BackwardInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionBackwardConfig, AttentionMaskMode, AttentionScoreConfig, HeuristicMode,
        },
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!("frontend_mxfp8_sdpa_backward: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }
    if !ctx.is_blackwell_device() {
        println!("frontend_mxfp8_sdpa_backward: skipped, requires Blackwell Computing GPU");
        return Ok(());
    }
    if version()? < 90_700 {
        println!("frontend_mxfp8_sdpa_backward: skipped, requires cuDNN 9.7.0+");
        return Ok(());
    }

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F8E4M3)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let q_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 32])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let k_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 32])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 6, 16])?,
    ));
    let o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o = graph.tensor(TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_quantized = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let d_o_t = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([1, 2, 4, 16])?,
    ));
    let stats = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_q_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_k_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 32])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 6, 1])?,
    ));
    let scale_d_o = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 4, 1])?,
    ));
    let scale_d_o_t = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([1, 2, 1, 16])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = match graph.sdpa_mxfp8_backward_infer(
        SdpaMxfp8BackwardInputs::new(
            q,
            q_t,
            k,
            k_t,
            v,
            o,
            d_o,
            d_o_quantized,
            d_o_t,
            stats,
            scale_q,
            scale_q_t,
            scale_k,
            scale_k_t,
            scale_v,
            scale_d_o,
            scale_d_o_t,
            attention_scale,
        ),
        AttentionBackwardConfig::new(DataType::F32).with_score_config(
            AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::CausalTopLeft),
        ),
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            if matches!(
                error,
                Error::Cudnn { code, .. }
                    if code == Status::NotSupported
                        || code == Status::NotSupportedGraphPattern
                        || code == Status::NotSupportedShape
                        || code == Status::NotSupportedDataType
                        || code == Status::NotSupportedLayout
            ) {
                println!(
                    "frontend_mxfp8_sdpa_backward: skipped, not supported on current configuration"
                );
                return Ok(());
            }
            return Err(error);
        }
    };

    for output in [
        outputs.query_gradient(),
        outputs.key_gradient(),
        outputs.value_gradient(),
        outputs.absolute_max_query_gradient(),
        outputs.absolute_max_key_gradient(),
        outputs.absolute_max_value_gradient(),
    ] {
        graph.mark_as_output(output)?;
    }

    if let Err(error) = graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        if matches!(
            error,
            Error::Cudnn { code, .. }
                if code == Status::NotSupported
                    || code == Status::NotSupportedGraphPattern
                    || code == Status::NotSupportedShape
                    || code == Status::NotSupportedDataType
                    || code == Status::NotSupportedLayout
        ) {
            println!(
                "frontend_mxfp8_sdpa_backward: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        return Err(error);
    };
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_mxfp8_sdpa_backward", run())
}
