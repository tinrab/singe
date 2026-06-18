mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{AdaptiveLayerNormalizationBackwardConfig, HeuristicMode},
    },
};

fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        values.push((*seed as f32) * 2.328_306_4e-10_f32);
    }
    values
}

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn count_nonfinite_f16(values: &[f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 4;
    let s = 1024;
    let d = 128;

    let mut graph = Graph::new()
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, s, d])?).with_name("X"));
    let dy =
        graph.tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, s, d])?).with_name("DY"));
    let scale = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, 1, d])?).with_name("scale"));
    let mean = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, s, 1])?).with_name("mean"));
    let inv_variance = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, s, 1])?).with_name("inv_variance"),
    );
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([b, s, d])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, 1, d])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, 1, d])?,
    ));
    graph.adaptive_layer_normalization_backward(
        x,
        mean,
        inv_variance,
        dy,
        scale,
        dscale,
        Some(bias_gradient),
        dx,
        AdaptiveLayerNormalizationBackwardConfig::new().with_bias_gradient(),
    )?;

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((b * s * d) as usize, &mut half_seed))?;
    let mut dy_dev =
        DeviceMemory::from_slice(&init_f16_image((b * s * d) as usize, &mut half_seed))?;
    let mut mean_dev =
        DeviceMemory::from_slice(&init_f32_image((b * s) as usize, &mut float_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&init_f32_image((b * s) as usize, &mut float_seed))?;
    let mut scale_dev =
        DeviceMemory::from_slice(&init_f32_image((b * d) as usize, &mut float_seed))?;
    let mut dscale_dev =
        DeviceMemory::from_slice(&init_f32_image((b * d) as usize, &mut float_seed))?;
    let mut bias_gradient_dev =
        DeviceMemory::from_slice(&init_f32_image((b * d) as usize, &mut float_seed))?;
    let mut dx_dev =
        DeviceMemory::from_slice(&init_f16_image((b * s * d) as usize, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let dx_host = dx_dev.copy_to_host_vec()?;
    let dscale_host = dscale_dev.copy_to_host_vec()?;
    let bias_gradient_host = bias_gradient_dev.copy_to_host_vec()?;

    let dx_first = dx_host[0].to_f32();
    let dx_last = dx_host[dx_host.len() - 1].to_f32();
    let dx_checksum = dx_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dx_nonfinite = count_nonfinite_f16(&dx_host);
    let dscale_first = dscale_host[0];
    let dscale_last = dscale_host[dscale_host.len() - 1];
    let dscale_checksum = dscale_host.iter().copied().sum::<f32>();
    let dscale_nonfinite = count_nonfinite_f32(&dscale_host);
    let bias_gradient_first = bias_gradient_host[0];
    let bias_gradient_last = bias_gradient_host[bias_gradient_host.len() - 1];
    let bias_gradient_checksum = bias_gradient_host.iter().copied().sum::<f32>();
    let bias_gradient_nonfinite = count_nonfinite_f32(&bias_gradient_host);

    println!(
        "dx: first={dx_first}, last={dx_last}, checksum={dx_checksum}, nonfinite={dx_nonfinite}, len={}",
        dx_host.len()
    );
    println!(
        "dscale: first={dscale_first}, last={dscale_last}, checksum={dscale_checksum}, nonfinite={dscale_nonfinite}, len={}",
        dscale_host.len()
    );
    println!(
        "bias_gradient: first={bias_gradient_first}, last={bias_gradient_last}, checksum={bias_gradient_checksum}, nonfinite={bias_gradient_nonfinite}, len={}",
        bias_gradient_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_adaptive_layer_norm_backward", run())
}
