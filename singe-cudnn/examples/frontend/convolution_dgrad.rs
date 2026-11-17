mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::ConvolutionConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    tensor::{Shape, TensorSpec},
};

fn run() -> Result<()> {
    let accumulate = args().any(|argument| argument == "--accumulate");
    let example = common::ExampleContext::create()?;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let dy = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([4, 64, 16, 16])?).with_name("grad"),
    );
    let w = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::channels_last([64, 32, 3, 3])?).with_name("weight"),
    );
    let dx = graph.convolution_backward_data(
        w,
        dy,
        [4, 32, 16, 16],
        ConvolutionConfig::new(2)
            .with_name("convolution_backward_data")
            .with_scaling(1.0, if accumulate { 1.0 } else { 0.0 })
            .with_compute_type(DataType::F32)
            .with_padding([1, 1])
            .with_strides([1, 1])
            .with_dilations([1, 1]),
        TensorOutputConfig::new().output_tensor(),
    )?;

    let gradient_elements = graph.shape(dy)?.storage_span()?;
    let filter_elements = graph.shape(w)?.storage_span()?;
    let output_elements = graph.shape(dx)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;

    let mut half_seed = 123_456_789_u32;
    let y_grad_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f16_image(gradient_elements, &mut half_seed),
    )?;
    let w_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &common::init_f16_image(filter_elements, &mut half_seed),
    )?;
    let mut x_grad_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &vec![f16::from_f32(if accumulate { 0.5 } else { 0.0 }); output_elements],
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(dy, &y_grad_dev)?;
    bindings.set_input(w, &w_dev)?;
    bindings.set(dx, &mut x_grad_dev)?;
    // SAFETY: initialized inputs, accumulated output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute_async(
            context,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
        )
    })?;

    let output = compiled.tensor_info(dx)?;
    println!(
        "{:?}: {:?}",
        output.spec.name,
        output.spec.shape.dimensions()
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    common::finish("frontend_convolution_backward_data", run())
}
