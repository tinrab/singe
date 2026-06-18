mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{CompileConfig, HeuristicMode, LayerNormalizationBackwardConfig},
        plan::BuildPlanPolicy,
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

    let batch_size = 4;
    let seq_length = 1024;
    let hidden_size = 128;

    let mut graph = Graph::new();
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size * seq_length, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size * seq_length, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let dscale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let bias_gradient = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    graph.layer_normalization_backward(
        x,
        mean,
        inv_variance,
        dy,
        scale,
        dscale,
        bias_gradient,
        dx,
        LayerNormalizationBackwardConfig::new(epsilon),
    )?;

    let compiled = match graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_f16_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut half_seed,
    ))?;
    let mut dy_dev = DeviceMemory::from_slice(&init_f16_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut half_seed,
    ))?;
    let mut mean_dev = DeviceMemory::from_slice(&init_f32_image(
        (batch_size * seq_length) as usize,
        &mut float_seed,
    ))?;
    let mut inv_variance_dev = DeviceMemory::from_slice(&init_f32_image(
        (batch_size * seq_length) as usize,
        &mut float_seed,
    ))?;
    let mut scale_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut dscale_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut bias_gradient_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut dx_dev = DeviceMemory::from_slice(&init_f16_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut half_seed,
    ))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
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

    assert_eq!(dx_nonfinite, 0, "dx.nonfinite mismatch");
    assert_eq!(dscale_nonfinite, 0, "dscale.nonfinite mismatch");
    assert_eq!(
        bias_gradient_nonfinite, 0,
        "bias_gradient.nonfinite mismatch"
    );
    assert_close!(dx_first, 0.0748901, 1.0e-5, "dx.first");
    assert_close!(dx_last, -0.0279999, 1.0e-5, "dx.last");
    assert_close!(dx_checksum, -2824.66, 0.1, "dx.checksum");
    assert_close!(dscale_first, -3.24039, 1.0e-4, "dscale.first");
    assert_close!(dscale_last, -5.93121, 1.0e-4, "dscale.last");
    assert_close!(dscale_checksum, -300.269, 0.05, "dscale.checksum");
    assert_close!(bias_gradient_first, 2060.29, 1.0e-2, "bias_gradient.first");
    assert_close!(bias_gradient_last, 2033.52, 1.0e-2, "bias_gradient.last");
    assert_close!(
        bias_gradient_checksum,
        262112.0,
        0.5,
        "bias_gradient.checksum"
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm_backward", run())
}
