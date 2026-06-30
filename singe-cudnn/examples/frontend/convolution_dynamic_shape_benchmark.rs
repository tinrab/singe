mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::execution::KernelCache,
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{ConvolutionConfig, HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
struct ConvShape {
    n: i64,
    c: i64,
    h: i64,
    w: i64,
    k: i64,
    r: i64,
    s: i64,
}

fn nhwc_shape(n: i64, c: i64, h: i64, w: i64) -> Result<Shape> {
    Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])
}

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn volume_nchw(shape: ConvShape) -> usize {
    (shape.n * shape.c * shape.h * shape.w) as usize
}

fn volume_filter(shape: ConvShape) -> usize {
    (shape.k * shape.c * shape.r * shape.s) as usize
}

fn volume_output(shape: ConvShape) -> usize {
    (shape.n * shape.k * shape.h * shape.w) as usize
}

fn create_conv_relu_forward_graph(
    shape: ConvShape,
    kernel_cache: Arc<Mutex<KernelCache>>,
) -> Result<(Graph, TensorId, TensorId, TensorId)> {
    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("conv-dynamic-shape-benchmark")
            .with_dynamic_shape()
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_compute(DataType::F32),
            ),
    );
    graph.attach_shared_kernel_cache(kernel_cache)?;

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            nhwc_shape(shape.n, shape.c, shape.h, shape.w)?,
        )
        .with_name("image"),
    );

    let w = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([shape.k, shape.c, shape.r, shape.s])?.with_strides(vec![
                shape.c * shape.r * shape.s,
                1,
                shape.c * shape.s,
                shape.c,
            ])?,
        )
        .with_name("filter"),
    );

    let conv_output = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            nhwc_shape(shape.n, shape.k, shape.h, shape.w)?,
        )
        .virtual_tensor(),
    );
    graph.convolution_forward(
        x,
        w,
        conv_output,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    let y = graph.tensor(TensorSpec::new(
        DataType::F16,
        nhwc_shape(shape.n, shape.k, shape.h, shape.w)?,
    ));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input: conv_output,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    });
    graph.mark_as_output(y)?;

    Ok((graph, x, w, y))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let shapes = [
        ConvShape {
            n: 16,
            c: 128,
            h: 56,
            w: 56,
            k: 256,
            r: 3,
            s: 3,
        },
        ConvShape {
            n: 16,
            c: 128,
            h: 80,
            w: 80,
            k: 256,
            r: 3,
            s: 3,
        },
    ];

    let kernel_cache = Arc::new(Mutex::new(KernelCache::create()?));
    let max_x = shapes.iter().copied().map(volume_nchw).max().unwrap_or(0);
    let max_w = shapes.iter().copied().map(volume_filter).max().unwrap_or(0);
    let max_y = shapes.iter().copied().map(volume_output).max().unwrap_or(0);

    let mut seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(&init_f16_image(max_x, &mut seed))?;
    let mut w_dev = DeviceMemory::from_slice(&init_f16_image(max_w, &mut seed))?;
    let mut y_dev = DeviceMemory::<f16>::zeroes(max_y)?;

    for shape in shapes {
        let (graph, x, w, y) = create_conv_relu_forward_graph(shape, Arc::clone(&kernel_cache))?;
        let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;
        let mut workspace = common::workspace(compiled.workspace_size()?)?;
        let mut bindings = compiled.bindings();
        bindings.set(x, &mut x_dev)?;
        bindings.set(w, &mut w_dev)?;
        bindings.set(y, &mut y_dev)?;
        compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_dynamic_shape_benchmark", run())
}
