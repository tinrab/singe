mod common;

use std::env::args;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{Graph, TensorOutputConfig},
        operation::{
            InstanceNormalizationBackwardInputs, InstanceNormalizationBackwardOutputs,
            NormalizationConfig, NormalizationSavedStatistics,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let mut graph = Graph::new();
    let activation = TensorSpec::new(DataType::F16, Shape::channels_last(vec![4, 32, 16, 16])?);
    let input = graph.tensor(activation.clone());
    let dy = graph.tensor(activation);
    let stats = TensorSpec::new(DataType::F32, Shape::channels_last(vec![4, 32, 1, 1])?);
    let mean = graph.tensor(stats.clone());
    let inv_variance = graph.tensor(stats);
    let scale = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([1, 32, 1, 1])?,
    ));
    let result = TensorOutputConfig::new().output_tensor();
    let outputs = graph.instance_normalization_backward(
        InstanceNormalizationBackwardInputs {
            input,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: dy,
            scale,
        },
        InstanceNormalizationBackwardOutputs {
            input_gradient: result,
            scale_gradient: result,
            bias_gradient: result,
        },
        NormalizationConfig::new().with_name("instance_norm_backward"),
    )?;
    if args().any(|argument| argument == "--input-layout") {
        let input_shape = graph.shape(input)?.clone();
        graph.modify_tensor(outputs.input_gradient, |tensor| {
            tensor.with_shape(input_shape)
        })?;
    }
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::Instant, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    let mut seed = 123_456_789_u32;
    for id in [input, dy] {
        let values = common::init_f32_image(graph.shape(id)?.storage_span()?, &mut seed)
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let memory = DeviceMemory::from_slice(cuda, &values)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    let scale_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(1.0); graph.shape(scale)?.storage_span()?],
    )?;
    bindings.set_input(scale, &scale_dev)?;
    for (id, value) in [(mean, 0.0_f32), (inv_variance, 1.0)] {
        let memory =
            DeviceMemory::from_slice(cuda, &vec![value; graph.shape(id)?.storage_span()?])?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    let output_ids = [
        outputs.input_gradient,
        outputs.scale_gradient,
        outputs.bias_gradient,
    ];
    for id in output_ids {
        let memory = DeviceMemory::<f16>::zeroed(cuda, graph.shape(id)?.storage_span()?)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain initialized operands and independent writable results.
    // Scale, plan and workspace remain live through stream completion, including enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for id in output_ids {
        let tensor = compiled.tensor_info(id)?;
        println!(
            "{:?}: strides {:?}",
            tensor.spec.name,
            tensor.spec.shape.strides()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_instance_norm_backward", run())
}
