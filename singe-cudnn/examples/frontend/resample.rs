mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::execution::resample::{PaddingMode, ResampleMode},
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, ResampleConfig},
    },
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 8_i64;
    let h = 56_i64;
    let w = 56_i64;
    let c = 8_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([h * w * c, 1, w * c, c])?,
    ));
    let y = graph.resample_infer(
        x,
        None,
        ResampleConfig::new(ResampleMode::MaxPool, DataType::F32)
            .with_padding_mode(PaddingMode::NegInf)
            .with_window_dims([2_i32, 3_i32])
            .with_strides([4_i32, 5_i32])
            .with_pre_paddings([2_i32, 3_i32])
            .with_post_paddings([4_i32, 5_i32]),
    )?;
    let y_elements = graph.tensor_config(y)?.shape.element_count()?;
    graph.mark_as_output(y)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_f16_image((n * h * w * c) as usize, &mut seed))?;
    let mut y_dev = DeviceMemory::<f16>::zeroes(y_elements)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(y, &mut y_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let _ = y_dev.copy_to_host_vec()?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_resample", run())
}
