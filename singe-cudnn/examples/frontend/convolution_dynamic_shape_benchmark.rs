mod common;

use std::{
    env::args,
    sync::{Arc, Mutex},
};

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        execution::KernelCache,
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

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

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn create_conv_relu_forward_graph(
    shape: ConvShape,
    groups: i64,
    kernel_cache: Arc<Mutex<KernelCache>>,
) -> Result<(Graph, TensorId, TensorId, TensorId)> {
    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("conv-dynamic-shape")
            .with_dynamic_shape(true)
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
            Shape::channels_last([shape.n, shape.c, shape.h, shape.w])?,
        )
        .with_name("image"),
    );
    let w = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::channels_last([shape.k, shape.c / groups, shape.r, shape.s])?,
        )
        .with_name("filter"),
    );
    let conv_output = graph.convolution_forward(
        x,
        w,
        ConvolutionConfig::new(2)
            .with_name("convolution")
            .with_pre_paddings([1, 1])
            .with_post_paddings([0, 0]),
        TensorOutputConfig::new(),
    )?;
    let y = graph.pointwise(
        PointwiseOperation::unary(
            PointwiseMode::ReluFwd,
            conv_output,
            TensorOutputConfig::new().output_tensor(),
        )
        .with_name("response_activation")
        .with_nan_propagation(NanPropagation::NotPropagate),
    )?;

    Ok((graph, x, w, y))
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let groups = if args().any(|argument| argument == "--grouped") {
        4
    } else {
        1
    };
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
    let graphs = shapes
        .into_iter()
        .map(|shape| create_conv_relu_forward_graph(shape, groups, Arc::clone(&kernel_cache)))
        .collect::<Result<Vec<_>>>()?;
    let mut max_elements = [0_usize; 3];
    for (graph, x, w, y) in &graphs {
        for (elements, tensor) in max_elements.iter_mut().zip([*x, *w, *y]) {
            *elements = (*elements).max(graph.shape(tensor)?.storage_span()?);
        }
    }
    let [max_x, max_w, max_y] = max_elements;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(cuda, &init_f16_image(max_x, &mut seed))?;
    let w_dev = DeviceMemory::from_slice(cuda, &init_f16_image(max_w, &mut seed))?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(cuda, max_y)?;

    for (graph, x, w, y) in graphs {
        let compiled = graph.compile(
            Some(&example.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?;
        let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
        let mut bindings = compiled.bindings()?;
        bindings.set_input(x, &x_dev)?;
        bindings.set_input(w, &w_dev)?;
        bindings.set(y, &mut y_dev)?;
        // SAFETY: initialized inputs, destination and workspace cover the declared layouts.
        // Plans, bindings and allocations remain live through completion, including enqueue failure.
        example.execute_and_wait(|context| unsafe {
            compiled.execute(
                context,
                ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            )
        })?;
        println!(
            "convolution with {groups} groups: {:?} -> {:?}",
            graph.shape(x)?.dimensions(),
            graph.shape(y)?.dimensions(),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_dynamic_shape_benchmark", run())
}
