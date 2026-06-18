mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, RmsNormalizationConfig},
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

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let batch_size = 4;
    let seq_length = 1024;
    let hidden_size = 128;

    let mut graph = Graph::new()
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
                hidden_size,
                1,
                hidden_size,
                hidden_size,
            ])?,
        )
        .with_name("X"),
    );
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
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5)?.with_name("epsilon"));
    let y = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size * seq_length, hidden_size, 1, 1])?.with_strides([
            hidden_size,
            1,
            hidden_size,
            hidden_size,
        ])?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([batch_size * seq_length, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    graph.rms_normalization(
        x,
        scale,
        y,
        inv_variance,
        RmsNormalizationConfig::training(epsilon),
    )?;

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut float_seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_f32_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut float_seed,
    ))?;
    let _inv_variance_host_init =
        init_f32_image((batch_size * seq_length) as usize, &mut float_seed);
    let mut scale_dev =
        DeviceMemory::from_slice(&init_f32_image(hidden_size as usize, &mut float_seed))?;
    let mut y_dev = DeviceMemory::from_slice(&init_f32_image(
        (batch_size * seq_length * hidden_size) as usize,
        &mut float_seed,
    ))?;
    let mut inv_variance_dev = DeviceMemory::from_slice(&init_f32_image(
        (batch_size * seq_length) as usize,
        &mut float_seed,
    ))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    let inv_variance_host = inv_variance_dev.copy_to_host_vec()?;

    let y_first = y_host[0];
    let y_last = y_host[y_host.len() - 1];
    let y_checksum = y_host.iter().copied().sum::<f32>();
    let inv_variance_first = inv_variance_host[0];
    let inv_variance_last = inv_variance_host[inv_variance_host.len() - 1];
    let inv_variance_checksum = inv_variance_host.iter().copied().sum::<f32>();

    println!(
        "y: first={y_first}, last={y_last}, checksum={y_checksum}, nonfinite=0, len={}",
        y_host.len()
    );
    println!(
        "inv_variance: first={inv_variance_first}, last={inv_variance_last}, checksum={inv_variance_checksum}, nonfinite=0, len={}",
        inv_variance_host.len()
    );

    assert_close!(y_first, 0.0791731, 1.0e-5, "y.first");
    assert_close!(y_last, 0.571149, 1.0e-5, "y.last");
    assert_close!(y_checksum, 227946.0, 0.5, "y.checksum");
    assert_close!(inv_variance_first, 1.70869, 1.0e-5, "inv_variance.first");
    assert_close!(inv_variance_last, 1.62816, 1.0e-5, "inv_variance.last");
    assert_close!(
        inv_variance_checksum,
        7106.72,
        0.05,
        "inv_variance.checksum",
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_rms_norm", run())
}
