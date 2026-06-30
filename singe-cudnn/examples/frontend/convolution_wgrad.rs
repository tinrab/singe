mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
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

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([4, 64, 16, 16])?.with_strides(vec![64 * 16 * 16, 1, 64 * 16, 64])?,
        )
        .with_name("image"),
    );
    let dy = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([4, 64, 16, 16])?.with_strides(vec![64 * 16 * 16, 1, 64 * 16, 64])?,
        )
        .with_name("grad"),
    );
    let dw = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([64, 64, 3, 3])?.with_strides([64 * 3 * 3, 1, 64 * 3, 64])?,
    ));
    graph.convolution_wgrad(
        x,
        dy,
        dw,
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
    let mut x_dev = DeviceMemory::from_slice(&init_f16_image(4 * 64 * 16 * 16, &mut half_seed))?;
    let mut dy_dev = DeviceMemory::from_slice(&init_f16_image(4 * 64 * 16 * 16, &mut half_seed))?;
    let mut dw_dev = DeviceMemory::from_slice(&init_f16_image(64 * 64 * 3 * 3, &mut half_seed))?;
    let mut workspace = common::workspace(compiled.max_workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(dy, &mut dy_dev)?;
    bindings.set(dw, &mut dw_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let dw_host = dw_dev.copy_to_host_vec()?;
    let dw_first = dw_host[0].to_f32();
    let dw_last = dw_host[dw_host.len() - 1].to_f32();
    let dw_checksum = dw_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let dw_nonfinite = count_nonfinite_f16(&dw_host);

    println!(
        "dw: first={dw_first}, last={dw_last}, checksum={dw_checksum}, nonfinite={dw_nonfinite}, len={}",
        dw_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_wgrad", run())
}
