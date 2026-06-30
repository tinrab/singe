mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaFp8Inputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, SdpaAuxOutputRequest, SdpaConfig, SdpaFusedMaskMode},
    },
    version,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!("frontend_fp8_sdpa_bottom_right_causal: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }

    if !ctx.is_hopper_device() && !ctx.is_blackwell_device() {
        println!(
            "frontend_fp8_sdpa_bottom_right_causal: skipped, requires Hopper or Blackwell Computing GPU"
        );
        return Ok(());
    }
    if version()? < 90_700 || ctx.get_compute_capability() < 100 {
        println!(
            "frontend_fp8_sdpa_bottom_right_causal: skipped, graph is not supported on this cuDNN/GPU combination"
        );
        return Ok(());
    }

    let b = 2_i64;
    let h = 2_i64;
    let s_q = 512_i64;
    let s_kv = 1024_i64;
    let d = 128_i64;

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
        Shape::contiguous([b, h, s_q, d])?.with_strides([s_q * h * d, d, h * d, 1])?,
    ));
    let k = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_kv, d])?.with_strides([s_kv * h * d, d, h * d, 1])?,
    ));
    let v = graph.tensor(TensorSpec::new(
        DataType::F8E4M3,
        Shape::contiguous([b, h, s_kv, d])?.with_strides([s_kv * h * d, d, h * d, 1])?,
    ));
    let descale_q = graph.tensor(TensorSpec::scalar_f32(1.0)?);
    let descale_k = graph.tensor_like_named(descale_q, "descale_k")?;
    let descale_v = graph.tensor_like_named(descale_q, "descale_v")?;
    let descale_s = graph.tensor_like_named(descale_q, "descale_s")?;
    let scale_s = graph.tensor_like_named(descale_q, "scale_s")?;
    let scale_o = graph.tensor_like_named(descale_q, "scale_o")?;
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));

    let outputs = match graph.sdpa_fp8_infer(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        &SdpaConfig::new()
            .with_name("sdpa_fp8_bottom_right_causal")
            .with_attention_scale(0.123)
            .with_mask(SdpaFusedMaskMode::CausalBottomRight {
                sequence_length_query: sequence_length_query,
                sequence_length_key_value: sequence_length_key_value,
            })
            .with_aux_outputs(SdpaAuxOutputRequest::fp8_amax()),
    ) {
        Ok(result) => result,
        Err(error) => {
            if matches!(
                error,
                Error::Cudnn { code, .. }
                    if code == Status::NotSupported
                        || code == Status::NotSupportedGraphPattern
                        || code == Status::NotSupportedShape
                        || code == Status::NotSupportedDataType
                        || code == Status::NotSupportedLayout
                        || code == Status::InternalErrorCompilationFailed
            ) {
                println!(
                    "frontend_fp8_sdpa_bottom_right_causal: skipped, not supported on current configuration"
                );
                return Ok(());
            }
            return Err(error);
        }
    };

    let o = outputs.output();
    let stats = outputs.stats();
    let absolute_max_s = outputs.logit_max();
    let absolute_max_o = outputs.score_sum_exp();
    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::F8E4M3,
            Shape::contiguous([b, h, s_q, d])?.with_strides([s_q * h * d, d, h * d, 1])?,
        )
        .output_tensor(),
    )?;
    assert!(stats.is_none(), "stats should not be generated");
    let absolute_max_s = absolute_max_s.expect("absolute_max_s requested");
    let absolute_max_o = absolute_max_o.expect("absolute_max_o requested");
    for output in [absolute_max_s, absolute_max_o] {
        graph.mark_as_output(output)?;
    }

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        Ok(compiled) => compiled,
        Err(error) => {
            if matches!(
                error,
                Error::Cudnn { code, .. }
                    if code == Status::NotSupported
                        || code == Status::NotSupportedGraphPattern
                        || code == Status::NotSupportedShape
                        || code == Status::NotSupportedDataType
                        || code == Status::NotSupportedLayout
                        || code == Status::InternalErrorCompilationFailed
            ) {
                println!(
                    "frontend_fp8_sdpa_bottom_right_causal: skipped, not supported on current configuration"
                );
                return Ok(());
            }
            return Err(error);
        }
    };

    let mut q_dev = DeviceMemory::<u8>::zeroes((b * h * s_q * d) as usize)?;
    let mut k_dev = DeviceMemory::<u8>::zeroes((b * h * s_kv * d) as usize)?;
    let mut v_dev = DeviceMemory::<u8>::zeroes((b * h * s_kv * d) as usize)?;
    let mut o_dev = DeviceMemory::<u8>::zeroes((b * h * s_q * d) as usize)?;
    let mut descale_q_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_k_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_v_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_s_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut scale_s_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut scale_o_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![s_q as i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![s_kv as i32; b as usize])?;
    let mut absolute_max_s_dev = DeviceMemory::from_slice(&[0.0_f32])?;
    let mut absolute_max_o_dev = DeviceMemory::from_slice(&[0.0_f32])?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(descale_q, &mut descale_q_dev)?;
    bindings.set(descale_k, &mut descale_k_dev)?;
    bindings.set(descale_v, &mut descale_v_dev)?;
    bindings.set(descale_s, &mut descale_s_dev)?;
    bindings.set(scale_s, &mut scale_s_dev)?;
    bindings.set(scale_o, &mut scale_o_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(absolute_max_s, &mut absolute_max_s_dev)?;
    bindings.set(absolute_max_o, &mut absolute_max_o_dev)?;

    if let Err(error) = compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace)) {
        if matches!(
            error,
            Error::Cudnn { code, .. }
                if code == Status::NotSupported
                    || code == Status::NotSupportedGraphPattern
                    || code == Status::NotSupportedShape
                    || code == Status::NotSupportedDataType
                    || code == Status::NotSupportedLayout
                    || code == Status::InternalErrorCompilationFailed
        ) {
            println!(
                "frontend_fp8_sdpa_bottom_right_causal: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa_bottom_right_causal", run())
}
