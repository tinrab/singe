mod common;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            Activation, BatchNormalizationBackwardConfig, BatchNormalizationBackwardInputs,
            BatchNormalizationBackwardOutputs, BatchNormalizationInferenceInputs,
            NormalizationConfig, NormalizationSavedStatistics, PointwiseOperation,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let activation = TensorSpec::new(DataType::F16, Shape::channels_last([4, 32, 16, 16])?);
    let channel = TensorSpec::new(DataType::F32, Shape::channels_last([1, 32, 1, 1])?);
    let x = graph.tensor(activation.clone().with_name("X"));
    let dy = graph.tensor(activation.with_name("DY"));
    let mean = graph.tensor(channel.clone().with_name("mean"));
    let inv_variance = graph.tensor(channel.clone().with_name("inv_variance"));
    let scale = graph.tensor(channel.clone().with_name("scale"));
    let bias = graph.tensor(channel.with_name("bias"));
    let statistics = NormalizationSavedStatistics { mean, inv_variance };
    let intermediate = TensorOutputConfig::new().with_data_type(DataType::F16);
    let bn_y = graph.batch_normalization_inference(
        BatchNormalizationInferenceInputs {
            input: x,
            statistics,
            scale,
            bias,
        },
        intermediate,
        NormalizationConfig::new().with_name("batch_norm_reconstruction"),
    )?;
    let dx_drelu = graph.pointwise(
        PointwiseOperation::activation_backward(Activation::relu(), dy, bn_y, intermediate)
            .with_name("relu_backward"),
    )?;
    let affine_output = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let outputs = graph.batch_normalization_backward(
        BatchNormalizationBackwardInputs {
            input: x,
            statistics,
            output_gradient: dx_drelu,
            scale,
        },
        BatchNormalizationBackwardOutputs {
            input_gradient: TensorOutputConfig::new().output_tensor(),
            scale_gradient: affine_output,
            bias_gradient: affine_output,
        },
        BatchNormalizationBackwardConfig::new().with_name("batch_norm_backward"),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    let mut seed = 123_456_789_u32;
    for id in [x, dy] {
        let values = common::init_f16_image(graph.shape(id)?.storage_span()?, &mut seed);
        let memory = DeviceMemory::from_slice(cuda, &values)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    for (id, value) in [
        (mean, 0.0_f32),
        (inv_variance, 1.0),
        (scale, 1.0),
        (bias, 0.0),
    ] {
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
    for id in [outputs.scale_gradient, outputs.bias_gradient] {
        let memory = DeviceMemory::<f32>::zeroed(cuda, graph.shape(id)?.storage_span()?)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain initialized inputs and independent writable results.
    // Native intermediates need no caller bindings, and the plan, buffers and workspace
    // remain live through stream completion, including an enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for id in [
        outputs.input_gradient,
        outputs.scale_gradient,
        outputs.bias_gradient,
    ] {
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
    common::finish("frontend_batch_norm_backward", run())
}
