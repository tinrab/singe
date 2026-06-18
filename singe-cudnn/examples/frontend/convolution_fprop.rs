mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{CompileConfig, ConvolutionConfig, HeuristicMode},
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

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 16;
    let c = 128;
    let h = 64;
    let w = 64;
    let k = 256;
    let r = 1;
    let s = 1;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_compute_data_type(DataType::F32);

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .with_id(1002)
        .with_name("image"),
    );
    let w_tensor = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
        )
        .with_id(1001)
        .with_name("filter"),
    );
    let y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
        )
        .with_id(1003),
    );
    graph.convolution_forward(
        x,
        w_tensor,
        y,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_pre_paddings(vec![0, 0])
            .with_post_paddings(vec![0, 0])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let compiled = match graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    ) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((n * c * h * w) as usize, &mut half_seed))?;
    let mut w_dev =
        DeviceMemory::from_slice(&init_f16_image((k * c * r * s) as usize, &mut half_seed))?;
    let mut y_dev =
        DeviceMemory::from_slice(&init_f16_image((n * k * h * w) as usize, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(w_tensor, &mut w_dev)?;
    bindings.set(y, &mut y_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    let y_first = y_host[0].to_f32();
    let y_last = y_host[y_host.len() - 1].to_f32();
    let y_checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let y_nonfinite = count_nonfinite_f16(&y_host);

    println!(
        "y: first={y_first}, last={y_last}, checksum={y_checksum}, nonfinite={y_nonfinite}, len={}",
        y_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_fprop", run())
}
