mod common;

use std::{env::args, iter::once};

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AdaptiveLayerNormalizationBackwardInputs, AdaptiveLayerNormalizationBackwardOutputs,
            NormalizationConfig, NormalizationSavedStatistics,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let with_bias = !args().any(|argument| argument == "--without-bias-gradient");
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let activation = TensorSpec::new(DataType::F16, Shape::contiguous([4, 1024, 128])?);
    let x = graph.tensor(activation.clone().with_name("X"));
    let dy = graph.tensor(activation.with_name("DY"));
    let scale = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4, 1, 128])?).with_name("scale"));
    let stats = TensorSpec::new(DataType::F32, Shape::contiguous([4, 1024, 1])?);
    let mean = graph.tensor(stats.clone().with_name("mean"));
    let inv_variance = graph.tensor(stats.with_name("inv_variance"));
    let result = TensorOutputConfig::new().output_tensor();
    let affine_result = result.with_data_type(DataType::F32);
    let outputs = graph.adaptive_layer_normalization_backward(
        AdaptiveLayerNormalizationBackwardInputs {
            input: x,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: dy,
            scale,
        },
        AdaptiveLayerNormalizationBackwardOutputs {
            input_gradient: result,
            scale_gradient: affine_result,
            bias_gradient: with_bias.then_some(affine_result),
        },
        NormalizationConfig::new().with_name("adaptive_layer_norm_backward"),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    let mut seed = 123_456_789_u32;
    for id in [x, dy] {
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
    for (id, value) in [(mean, 0.0_f32), (inv_variance, 1.0), (scale, 1.0)] {
        let memory =
            DeviceMemory::from_slice(cuda, &vec![value; graph.shape(id)?.storage_span()?])?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    let dx_dev =
        DeviceMemory::<f16>::zeroed(cuda, graph.shape(outputs.input_gradient)?.storage_span()?)?;
    bindings.set_owned_region(
        outputs.input_gradient,
        OwnedDeviceRegion::from_memory(&dx_dev),
        BindingAccess::ReadWrite,
    )?;
    for id in once(outputs.scale_gradient).chain(outputs.bias_gradient) {
        let memory = DeviceMemory::<f32>::zeroed(cuda, graph.shape(id)?.storage_span()?)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain initialized operands and independent writable results.
    // The plan, buffers and workspace remain live through stream completion, including enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for id in once(outputs.input_gradient)
        .chain(once(outputs.scale_gradient))
        .chain(outputs.bias_gradient)
    {
        let tensor = compiled.tensor_info(id)?;
        println!(
            "{:?}: {:?}",
            tensor.spec.name,
            tensor.spec.shape.dimensions()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_adaptive_layer_norm_backward", run())
}
