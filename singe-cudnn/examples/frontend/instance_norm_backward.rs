mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    data_type::f16 as half_f16,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, InstanceNormalizationBackwardConfig},
    },
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<half_f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(half_f16::from_f32)
        .collect()
}

fn count_nonfinite_f16(values: &[half_f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let mut half_seed = 123_456_789_u32;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?,
    ));
    let stats = Shape::contiguous([4, 32, 1, 1])?;
    let scale_shape = Shape::contiguous([1, 32, 1, 1])?;
    let mean = graph.tensor(TensorSpec::new(DataType::F32, stats.clone()));
    let inv_variance = graph.tensor(TensorSpec::new(DataType::F32, stats.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F16, scale_shape));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let outputs = graph.instance_normalization_backward_infer(
        input,
        mean,
        inv_variance,
        dy,
        scale,
        InstanceNormalizationBackwardConfig::new(epsilon),
    )?;
    let dx = outputs.dx;
    let dscale = outputs.dscale;
    let bias_gradient = outputs.bias_gradient;

    let compiled = match graph.compile(
        &ctx.cudnn,
        &[HeuristicMode::Instant, HeuristicMode::Fallback],
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut input_dev =
        DeviceMemory::from_slice(&init_f16_image(4 * 32 * 16 * 16, &mut half_seed))?;
    let mut dy_dev = DeviceMemory::from_slice(&init_f16_image(4 * 32 * 16 * 16, &mut half_seed))?;
    let mut mean_dev = DeviceMemory::from_slice(&common::init_f32_image(4 * 32, &mut half_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&common::init_f32_image(4 * 32, &mut half_seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&init_f16_image(32, &mut half_seed))?;
    let mut dx_dev = DeviceMemory::from_slice(&init_f16_image(4 * 32 * 16 * 16, &mut half_seed))?;
    let mut dscale_dev = DeviceMemory::from_slice(&init_f16_image(32, &mut half_seed))?;
    let mut bias_gradient_dev = DeviceMemory::from_slice(&init_f16_image(32, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(input, &mut input_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(bias_gradient, &mut bias_gradient_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let dx_host = dx_dev.copy_to_host_vec()?;
    let dscale_host = dscale_dev.copy_to_host_vec()?;
    let bias_gradient_host = bias_gradient_dev.copy_to_host_vec()?;

    let dx_first = dx_host[0].to_f32();
    let dx_last = dx_host[dx_host.len() - 1].to_f32();
    let dx_checksum = dx_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dx_nonfinite = count_nonfinite_f16(&dx_host);
    let dscale_first = dscale_host[0].to_f32();
    let dscale_last = dscale_host[dscale_host.len() - 1].to_f32();
    let dscale_checksum = dscale_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dscale_nonfinite = count_nonfinite_f16(&dscale_host);
    let bias_gradient_first = bias_gradient_host[0].to_f32();
    let bias_gradient_last = bias_gradient_host[bias_gradient_host.len() - 1].to_f32();
    let bias_gradient_checksum = bias_gradient_host
        .iter()
        .map(|value| value.to_f32())
        .sum::<f32>();
    let bias_gradient_nonfinite = count_nonfinite_f16(&bias_gradient_host);

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
    assert_close!(dx_first, -0.072631836, 1.0e-6, "dx.first");
    assert_close!(dx_last, 0.008911133, 1.0e-6, "dx.last");
    assert_close!(dx_checksum, -111.40835, 1.0e-3, "dx.checksum");
    assert_close!(dscale_first, 15.6953125, 1.0e-6, "dscale.first");
    assert_close!(dscale_last, -13.6953125, 1.0e-6, "dscale.last");
    assert_close!(dscale_checksum, 178.11945, 1.0e-3, "dscale.checksum");
    assert_close!(bias_gradient_first, 521.5, 1.0e-6, "bias_gradient.first");
    assert_close!(bias_gradient_last, 514.5, 1.0e-6, "bias_gradient.last");
    assert_close!(
        bias_gradient_checksum,
        16423.5,
        1.0e-2,
        "bias_gradient.checksum"
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_instance_norm_backward", run())
}
