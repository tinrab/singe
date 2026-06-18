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
        operation::{HeuristicMode, InstanceNormalizationConfig},
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
    let n = 16;
    let c = 32;
    let h = 64;
    let w = 64;
    let mut half_seed = 123_456_789_u32;

    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?,
    ));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?);
    let outputs = graph
        .instance_normalization_infer(input, InstanceNormalizationConfig::training(epsilon))?;
    let output = outputs.output;
    let mean = outputs.mean;
    let inv_variance = outputs.inv_variance;
    let scale = outputs.scale;
    let bias = outputs.bias;

    let compiled = match graph.compile(
        &ctx.cudnn,
        &[HeuristicMode::Instant, HeuristicMode::Fallback],
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut input_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut output_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut mean_dev =
        DeviceMemory::from_slice(&common::init_f32_image((n * c) as usize, &mut half_seed))?;
    let mut inv_variance_dev =
        DeviceMemory::from_slice(&common::init_f32_image((n * c) as usize, &mut half_seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&init_f16_image(c as usize, &mut half_seed))?;
    let mut bias_dev = DeviceMemory::from_slice(&init_f16_image(c as usize, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(input, &mut input_dev)?;
    bindings.set(output, &mut output_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let output_host = output_dev.copy_to_host_vec()?;
    let mean_host = mean_dev.copy_to_host_vec()?;
    let inv_variance_host = inv_variance_dev.copy_to_host_vec()?;

    let output_first = output_host[0].to_f32();
    let output_last = output_host[output_host.len() - 1].to_f32();
    let output_checksum = output_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let output_nonfinite = count_nonfinite_f16(&output_host);
    let mean_first = mean_host[0];
    let mean_last = mean_host[mean_host.len() - 1];
    let mean_checksum = mean_host.iter().copied().sum::<f32>();
    let mean_nonfinite = common::count_nonfinite_f32(&mean_host);
    let inv_variance_first = inv_variance_host[0];
    let inv_variance_last = inv_variance_host[inv_variance_host.len() - 1];
    let inv_variance_checksum = inv_variance_host.iter().copied().sum::<f32>();
    let inv_variance_nonfinite = common::count_nonfinite_f32(&inv_variance_host);

    println!(
        "output: first={output_first}, last={output_last}, checksum={output_checksum}, nonfinite={output_nonfinite}, len={}",
        output_host.len()
    );
    println!(
        "mean: first={mean_first}, last={mean_last}, checksum={mean_checksum}, nonfinite={mean_nonfinite}, len={}",
        mean_host.len()
    );
    println!(
        "inv_variance: first={inv_variance_first}, last={inv_variance_last}, checksum={inv_variance_checksum}, nonfinite={inv_variance_nonfinite}, len={}",
        inv_variance_host.len()
    );

    assert_eq!(output_nonfinite, 0, "output.nonfinite mismatch");
    assert_eq!(mean_nonfinite, 0, "mean.nonfinite mismatch");
    assert_eq!(inv_variance_nonfinite, 0, "inv_variance.nonfinite mismatch");
    assert_close!(output_first, -0.008422852, 1.0e-6, "output.first");
    assert_close!(output_last, 1.4912109, 1.0e-5, "output.last");
    assert_close!(output_checksum, 1.0891788e6, 5.0e2, "output.checksum");
    assert_close!(mean_first, 0.5029528, 1.0e-6, "mean.first");
    assert_close!(mean_last, 0.5001669, 1.0e-6, "mean.last");
    assert_close!(mean_checksum, 256.13507, 1.0e-3, "mean.checksum");
    assert_close!(inv_variance_first, 3.4776814, 1.0e-6, "inv_variance.first");
    assert_close!(inv_variance_last, 3.4717991, 1.0e-6, "inv_variance.last");
    assert_close!(
        inv_variance_checksum,
        1774.1238,
        1.0e-2,
        "inv_variance.checksum",
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_instance_norm", run())
}
