mod common;

use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaMxfp8Inputs,
        graph::Graph,
        operation::{HeuristicMode, SdpaConfig},
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!("frontend_mxfp8_sdpa: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }
    if !ctx.is_blackwell_device() {
        println!("frontend_mxfp8_sdpa: skipped, requires Blackwell Computing GPU");
        return Ok(());
    }
    if version()? < 90_700 {
        println!("frontend_mxfp8_sdpa: skipped, requires cuDNN 9.7.0+");
        return Ok(());
    }

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

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F8E4M3)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let qkv_dims = vec![b, h, s, d];
    let qkv_strides = vec![s * 3 * h * d, d, 3 * h * d, 1];
    let q = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims.clone())?.with_strides(qkv_strides.clone())?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous(qkv_dims)?.with_strides(qkv_strides)?,
    ));
    let scale_q = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));
    let scale_k = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_padded, d_scale_padded])?.with_strides(vec![
            h * s_padded * d_scale_padded,
            s_padded * d_scale_padded,
            d_scale_padded,
            1,
        ])?,
    ));
    let scale_v = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E8M0,
        Shape::contiguous([b, h, s_scale_padded, d_padded])?.with_strides(vec![
            h * s_scale_padded * d_padded,
            s_scale_padded * d_padded,
            1,
            s_scale_padded,
        ])?,
    ));
    let attention_scale = graph.tensor(TensorSpec::scalar_f32(1.0)?);

    let outputs = match graph.sdpa_mxfp8_infer(
        SdpaMxfp8Inputs::new(q, k, v, scale_q, scale_k, scale_v, attention_scale),
        &SdpaConfig::new()
            .with_name("sdpa_mxfp8")
            .with_causal_mask()
            .with_stats()
            .with_absolute_max_o(),
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
                println!("frontend_mxfp8_sdpa: skipped, not supported on current configuration");
                return Ok(());
            }
            return Err(error);
        }
    };
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_o = outputs.absolute_max_output;

    for output in [Some(o), stats, absolute_max_o].into_iter().flatten() {
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
            println!("frontend_mxfp8_sdpa: skipped, not supported on current configuration");
            return Ok(());
        }
        return Err(error);
    };
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_mxfp8_sdpa", run())
}
