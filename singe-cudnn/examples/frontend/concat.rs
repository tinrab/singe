mod common;

use std::env::args;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConcatConfig, ConcatInPlaceMode},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let in_place = args().any(|argument| argument == "--in-place");
    let explicit_output = args().any(|argument| argument == "--explicit-output");
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let input_shape = Shape::with_strides([2, 4, 8, 8], [256, 1, 32, 4])?;
    let input_elements = input_shape.storage_span()?;
    let first = graph.tensor(TensorSpec::new(DataType::F32, input_shape.clone()).with_name("X0"));
    // The second input occupies channels 4..8 of the larger NHWC destination.
    // Its batch and spatial strides therefore use eight output channels.
    let second_shape = if in_place {
        Shape::with_strides([2, 4, 8, 8], [512, 1, 64, 8])?
    } else {
        input_shape
    };
    let second = graph.tensor(TensorSpec::new(DataType::F32, second_shape).with_name("X1"));
    let mode = if in_place {
        ConcatInPlaceMode::Input(1)
    } else {
        ConcatInPlaceMode::None
    };
    let config = ConcatConfig::new(1)
        .with_name("channels")
        .with_in_place(mode);
    let output = if explicit_output {
        let shape = config.infer_output_shape([graph.shape(first)?, graph.shape(second)?])?;
        let output =
            graph.tensor(TensorSpec::new(DataType::F32, shape).with_name("explicit_channels"));
        graph.concat_into(&[first, second], output, config)?;
        output
    } else {
        graph.concat(
            &[first, second],
            config,
            TensorOutputConfig::new().output_tensor(),
        )?
    };
    let output_name = graph.tensor_spec(output)?.name.as_deref().unwrap_or("Y");
    let output_elements = graph.shape(output)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let cuda = example.cudnn.cuda_context();
    let first_device = DeviceMemory::from_slice(cuda, &vec![1.0_f32; input_elements])?;
    let second_device = (!in_place)
        .then(|| DeviceMemory::from_slice(cuda, &vec![2.0_f32; input_elements]))
        .transpose()?;
    // In-place mode consumes the second input from these initialized destination bytes.
    let output_device = DeviceMemory::from_slice(cuda, &vec![2.0_f32; output_elements])?;
    let second_region = match second_device.as_ref() {
        Some(memory) => OwnedDeviceRegion::from_memory(memory),
        None => OwnedDeviceRegion::from_memory(&output_device).slice(4 * size_of::<f32>()..)?,
    };
    let mut bindings = compiled.bindings()?;
    bindings.set_owned_region(
        first,
        OwnedDeviceRegion::from_memory(&first_device),
        BindingAccess::ReadOnly,
    )?;
    bindings.set_owned_region(second, second_region, BindingAccess::ReadOnly)?;
    bindings.set_owned_region(
        output,
        OwnedDeviceRegion::from_memory(&output_device),
        BindingAccess::ReadWrite,
    )?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: all inputs are initialized and the output has sufficient writable storage.
    // The selected in-place input aliases its final output region with matching strides.
    // Bindings retain every allocation, and all resources remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("concatenated NHWC channels into {output_name} with mode {mode:?}");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_concat", run())
}
