mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::{DataType, f16},
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionConfig, AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
            HeuristicMode,
        },
    },
    version,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    if cudart_version()? < 12_000 {
        println!("frontend_fp16_sdpa_bottom_right_causal: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }

    if version()? < 90_600 {
        println!("frontend_fp16_sdpa_bottom_right_causal: skipped, requires cuDNN 9.6.0+");
        return Ok(());
    }

    let ctx = common::ExampleContext::create()?;
    if !ctx.is_hopper_device() && !ctx.is_blackwell_device() {
        println!(
            "frontend_fp16_sdpa_bottom_right_causal: skipped, requires Hopper or Blackwell Computing GPU"
        );
        return Ok(());
    }

    let b = 2_i64;
    let h = 2_i64;
    let s_q = 64_i64;
    let s_kv = 64_i64;
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
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125)?);
    let sequence_length_query = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));
    let sequence_length_key_value = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([b, 1, 1, 1])?,
    ));

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_implementation(AttentionImplementation::Composite)
            .with_score_config(AttentionScoreConfig::new().with_mask_mode(
                AttentionMaskMode::CausalBottomRight {
                    sequence_length_query,
                    sequence_length_key_value,
                },
            )),
    )?;
    let o = outputs.output();
    let stats = outputs.stats();

    for output in [Some(o), stats].into_iter().flatten() {
        graph.mark_as_output(output)?;
    }

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        Ok(compiled) => compiled,
        Err(Error::Cudnn { code, .. }) if code == Status::NotSupported => {
            println!("frontend_fp16_sdpa_bottom_right_causal: skipped, not supported by backend");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h * s_kv * d_qk) as usize, &mut seed))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h * s_kv * d_v) as usize, &mut seed))?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![48_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![64_i32; b as usize])?;
    let mut o_dev = DeviceMemory::<f16>::zeroes((b * h * s_q * d_v) as usize)?;
    let stats = stats.expect("stats");
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings
        .set(q, &mut q_dev)?
        .set(k, &mut k_dev)?
        .set(v, &mut v_dev)?
        .set(sequence_length_query, &mut seq_len_q_dev)?
        .set(sequence_length_key_value, &mut seq_len_kv_dev)?
        .set(o, &mut o_dev)?
        .set(stats, &mut stats_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_bottom_right_causal", run())
}
