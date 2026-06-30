mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    data_type::f16 as half_f16,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, LayerNormalizationConfig},
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

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<half_f16> {
    init_f32_image(size, seed)
        .into_iter()
        .map(half_f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let batch_size = 4;
    let seq_length = 1024;
    let hidden_size = 128;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
                hidden_size,
                1,
                hidden_size,
                hidden_size,
            ])?,
        )
        .with_name("X"),
    );
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_name("epsilon"));
    let scale = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
                hidden_size,
                1,
                hidden_size,
                hidden_size,
            ])?,
        )
        .with_name("scale"),
    );
    let bias = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, hidden_size, 1, 1])?.with_strides(vec![
                hidden_size,
                1,
                hidden_size,
                hidden_size,
            ])?,
        )
        .with_name("bias"),
    );
    let y = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
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
    graph.layer_normalization(
        x,
        scale,
        bias,
        y,
        mean,
        inv_variance,
        LayerNormalizationConfig::training(epsilon),
    )?;

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut float_seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_f16_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut half_seed,
    ))?;
    let _mean_host_init = init_f32_image((batch_size * seq_length) as usize, &mut float_seed);
    let _inv_variance_host_init =
        init_f32_image((batch_size * seq_length) as usize, &mut float_seed);
    let mut scale_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut bias_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut y_dev =
        DeviceMemory::<half_f16>::zeroes((batch_size * seq_length * hidden_size) as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes((batch_size * seq_length) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    let mean_host = mean_dev.copy_to_host_vec()?;
    let inv_variance_host = inv_variance_dev.copy_to_host_vec()?;

    let y_first = y_host[0].to_f32();
    let y_last = y_host[y_host.len() - 1].to_f32();
    let y_checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let mean_first = mean_host[0];
    let mean_last = mean_host[mean_host.len() - 1];
    let mean_checksum = mean_host.iter().copied().sum::<f32>();
    let inv_variance_first = inv_variance_host[0];
    let inv_variance_last = inv_variance_host[inv_variance_host.len() - 1];
    let inv_variance_checksum = inv_variance_host.iter().copied().sum::<f32>();

    println!(
        "y: first={y_first}, last={y_last}, checksum={y_checksum}, nonfinite=0, len={}",
        y_host.len()
    );
    println!(
        "mean: first={mean_first}, last={mean_last}, checksum={mean_checksum}, nonfinite=0, len={}",
        mean_host.len()
    );
    println!(
        "inv_variance: first={inv_variance_first}, last={inv_variance_last}, checksum={inv_variance_checksum}, nonfinite=0, len={}",
        inv_variance_host.len()
    );

    assert_close!(y_first, -0.533203, 1.0e-3, "y.first");
    assert_close!(y_last, 0.421387, 1.0e-3, "y.last");
    assert_close!(y_checksum, 246493.0, 200.0, "y.checksum");
    assert_close!(mean_first, 0.506307, 1.0e-5, "mean.first");
    assert_close!(mean_last, 0.546668, 1.0e-5, "mean.last");
    assert_close!(mean_checksum, 2049.33, 0.05, "mean.checksum");
    assert_close!(inv_variance_first, 3.4069, 1.0e-4, "inv_variance.first");
    assert_close!(inv_variance_last, 3.57143, 1.0e-4, "inv_variance.last");
    assert_close!(inv_variance_checksum, 14270.2, 0.5, "inv_variance.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_layer_norm", run())
}
