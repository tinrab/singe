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

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let dy = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([4, 64, 16, 16])?.with_strides(vec![64 * 16 * 16, 1, 64 * 16, 64])?,
        )
        .with_name("grad"),
    );
    let w = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([64, 32, 3, 3])?.with_strides([32 * 3 * 3, 1, 32 * 3, 32])?,
        )
        .with_name("weight"),
    );
    let dx = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([4, 32, 16, 16])?.with_strides([32 * 16 * 16, 1, 32 * 16, 32])?,
    ));
    graph.convolution_dgrad(
        w,
        dy,
        dx,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
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
    let mut dy_dev = DeviceMemory::from_slice(&init_f16_image(4 * 64 * 16 * 16, &mut half_seed))?;
    let mut w_dev = DeviceMemory::from_slice(&init_f16_image(64 * 32 * 3 * 3, &mut half_seed))?;
    let mut dx_dev = DeviceMemory::from_slice(&init_f16_image(4 * 32 * 16 * 16, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(w, &mut w_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let dx_host = dx_dev.copy_to_host_vec()?;
    let dx_first = dx_host[0].to_f32();
    let dx_last = dx_host[dx_host.len() - 1].to_f32();
    let dx_checksum = dx_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dx_nonfinite = count_nonfinite_f16(&dx_host);

    println!(
        "dx: first={dx_first}, last={dx_last}, checksum={dx_checksum}, nonfinite={dx_nonfinite}, len={}",
        dx_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_dgrad", run())
}
