mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        operation::{ConvolutionConfig, HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let n = 1_i64;
    let c = 64_i64;
    let h = 32_i64;
    let w = 32_i64;
    let k = 4_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::I8)
        .with_intermediate_data_type(DataType::I32)
        .with_compute_data_type(DataType::I32);

    let x = graph.tensor(named_tensor(
        "image",
        DataType::I8,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let w_tensor = graph.tensor(named_tensor(
        "filter",
        DataType::I8,
        Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
    ));
    let conv_output = graph.tensor(
        TensorSpec::new(
            DataType::I8,
            Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
        )
        .virtual_tensor(),
    );
    graph.convolution_forward(
        x,
        w_tensor,
        conv_output,
        ConvolutionConfig::new(DataType::I32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let y = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([n, k, h, w])?.with_strides([k * h * w, 1, k * w, k])?,
    ));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::Identity,
        input: conv_output,
        output: y,
        compute_type: DataType::I32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    });
    graph.mark_as_output(y)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut x_dev = DeviceMemory::<i8>::zeroes((n * c * h * w) as usize)?;
    let mut w_dev = DeviceMemory::<i8>::zeroes((k * c * r * s) as usize)?;
    let mut y_dev = DeviceMemory::<i32>::zeroes((n * k * h * w) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(w_tensor, &mut w_dev)?;
    bindings.set(y, &mut y_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_int8_fprop", run())
}
