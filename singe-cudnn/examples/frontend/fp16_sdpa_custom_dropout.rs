mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        composite::sdpa::{SdpaInputs, SdpaOutputs},
        graph::Graph,
        operation::{AttentionConfig, HeuristicMode},
    },
    version,
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if version()? < 8903 {
        println!("frontend_fp16_sdpa_custom_dropout: skipped, requires cuDNN 8.9.3+");
        return Ok(());
    }

    let device = ctx.cudnn.cuda_context().device();
    let properties = device.properties()?;
    if properties.major >= 12 {
        println!(
            "frontend_fp16_sdpa_custom_dropout: skipped, custom dropout is not supported on blackwell"
        );
        return Ok(());
    }

    let b = 3_i64;
    let h_q = 4_i64;
    let h_k = 4_i64;
    let h_v = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;
    let has_attn_bias = version()? >= 8903;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let q = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides(vec![
                h_q * s_q * d_qk,
                s_q * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_k, s_kv, d_qk])?.with_strides(vec![
                h_k * s_kv * d_qk,
                s_kv * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_v, s_kv, d_v])?.with_strides(vec![
                h_v * s_kv * d_v,
                s_kv * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1003),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let dropout_mask = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, s_kv])?.with_strides(vec![
                h_q * s_q * s_kv,
                s_q * s_kv,
                s_kv,
                1,
            ])?,
        )
        .with_id(1008),
    );
    let dropout_scale = graph.tensor(TensorSpec::scalar_f32(0.1)?.with_id(1007));
    let bias = if has_attn_bias {
        Some(
            graph.tensor(
                TensorSpec::new(
                    DataType::F32,
                    Shape::contiguous([b, 1, s_q, s_kv])?
                        .with_strides([s_q * s_kv, s_q * s_kv, s_kv, 1])
                        .expect("bias shape"),
                )
                .with_id(1006),
            ),
        )
    } else {
        None
    };

    let mut config = AttentionConfig::new(DataType::F32)
        .with_stats()
        .with_causal_mask()
        .with_dropout(dropout_mask, dropout_scale);
    if let Some(bias) = bias {
        config = config.with_bias(bias);
    }

    let SdpaOutputs { output: o, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        config,
    )?;

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * d_v,
                d_v,
                b * h_q * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(1004),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(1005);
    graph.replace_tensor(stats, stats_tensor)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev = DeviceMemory::from_slice(&init_bf16_image(
        (b * h_k * s_kv * d_qk) as usize,
        &mut seed,
    ))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;
    let mut dropout_mask_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h_q * s_q * s_kv) as usize, &mut seed))?;
    let mut dropout_scale_dev = DeviceMemory::from_slice(&[0.1_f32])?;
    let mut bias_dev = if has_attn_bias {
        Some(DeviceMemory::from_slice(&common::init_f32_image(
            (b * s_q * s_kv) as usize,
            &mut seed,
        ))?)
    } else {
        None
    };
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(dropout_mask, &mut dropout_mask_dev)?;
    bindings.set(dropout_scale, &mut dropout_scale_dev)?;
    if let (Some(bias), Some(bias_dev)) = (bias, bias_dev.as_mut()) {
        bindings.set(bias, bias_dev)?;
    }
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_custom_dropout", run())
}
