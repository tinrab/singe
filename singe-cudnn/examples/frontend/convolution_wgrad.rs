mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let accumulate = args().any(|argument| argument == "--accumulate");
    let fused = args().any(|argument| argument == "--fused-preprocess");
    let example = common::ExampleContext::create()?;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([4, 64, 16, 16])?).with_name("image"),
    );
    let dy = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([4, 64, 16, 16])?).with_name("grad"),
    );
    let (wgrad_input, affine) = if fused {
        let channel = TensorSpec::new(DataType::F16, Shape::channels_last([1, 64, 1, 1])?);
        let scale = graph.tensor(channel.clone().with_name("scale"));
        let bias = graph.tensor(channel.with_name("bias"));
        let scaled = graph.pointwise(
            PointwiseOperation::binary(
                PointwiseMode::Mul,
                x,
                scale,
                TensorOutputConfig::new().virtual_tensor(),
            )
            .with_name("scale"),
        )?;
        let shifted = graph.pointwise(
            PointwiseOperation::binary(
                PointwiseMode::Add,
                scaled,
                bias,
                TensorOutputConfig::new().virtual_tensor(),
            )
            .with_name("bias"),
        )?;
        let activated = graph.pointwise(
            PointwiseOperation::unary(
                PointwiseMode::ReluFwd,
                shifted,
                TensorOutputConfig::new().virtual_tensor(),
            )
            .with_name("relu"),
        )?;
        (activated, Some((scale, bias)))
    } else {
        (x, None)
    };
    let dw = graph.convolution_backward_filter(
        wgrad_input,
        dy,
        [64, 64, 3, 3],
        ConvolutionConfig::new(2)
            .with_name("convolution_backward_filter")
            .with_scaling(1.0, if accumulate { 1.0 } else { 0.0 })
            .with_compute_type(DataType::F32)
            .with_padding([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]),
        TensorOutputConfig::new().output_tensor(),
    )?;

    let input_elements = graph.shape(x)?.storage_span()?;
    let gradient_elements = graph.shape(dy)?.storage_span()?;
    let output_elements = graph.shape(dw)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();

    let mut half_seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f16_image(input_elements, &mut half_seed),
    )?;
    let dy_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f16_image(gradient_elements, &mut half_seed),
    )?;
    let affine_buffers = if let Some((scale, bias)) = affine {
        let scale_dev = DeviceMemory::from_slice(
            cuda,
            &vec![f16::from_f32(1.0); graph.shape(scale)?.storage_span()?],
        )?;
        let bias_dev = DeviceMemory::from_slice(
            cuda,
            &vec![f16::from_f32(-0.5); graph.shape(bias)?.storage_span()?],
        )?;
        Some((scale, scale_dev, bias, bias_dev))
    } else {
        None
    };
    let mut dw_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(if accumulate { 0.5 } else { 0.0 }); output_elements],
    )?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(dy, &dy_dev)?;
    if let Some((scale, scale_dev, bias, bias_dev)) = &affine_buffers {
        bindings.set_input(*scale, scale_dev)?;
        bindings.set_input(*bias, bias_dev)?;
    }
    bindings.set(dw, &mut dw_dev)?;
    // SAFETY: initialized inputs, accumulated output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    let output = compiled.tensor_info(dw)?;
    println!(
        "{:?}: {:?}",
        output.spec.name,
        output.spec.shape.dimensions()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_convolution_backward_filter", run())
}
