mod common;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            Activation, BatchNormalizationBackwardWeightsConfig,
            BatchNormalizationBackwardWeightsInputs, BatchNormalizationBackwardWeightsOutputs,
            ConvolutionConfig, NormalizationSavedStatistics, PointwiseOperation,
        },
        plan::{BuildPlanPolicy, CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let k = 64;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let dy = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::with_strides([n, k, h, w], [k * h * w, 1, k * w, k])?,
    ));
    let weight = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::with_strides([k, c, 3, 3], [c * 9, 1, c * 3, c])?,
    ));
    let half_intermediate = TensorOutputConfig::new().with_data_type(DataType::F16);
    let dgrad = graph.convolution_backward_data(
        weight,
        dy,
        vec![n, c, h, w],
        ConvolutionConfig::new(2)
            .with_compute_type(DataType::F32)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
        half_intermediate,
    )?;
    let x = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::with_strides([n, c, h, w], [c * h * w, 1, c * w, c])?,
    ));
    let channel = TensorSpec::new(
        DataType::F32,
        Shape::with_strides([1, c, 1, 1], [c, 1, c, c])?,
    );
    let mean = graph.tensor(channel.clone().with_name("mean"));
    let inv_variance = graph.tensor(channel.clone().with_name("inv_variance"));
    let scale = graph.tensor(channel.clone().with_name("scale"));
    let bias = graph.tensor(channel.with_name("bias"));
    let centered = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Sub,
        x,
        mean,
        TensorOutputConfig::new(),
    ))?;
    let scaled_variance = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Mul,
        centered,
        inv_variance,
        TensorOutputConfig::new(),
    ))?;
    let scaled = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Mul,
        scaled_variance,
        scale,
        TensorOutputConfig::new(),
    ))?;
    let bn_output = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Add,
        scaled,
        bias,
        TensorOutputConfig::new(),
    ))?;
    let drelu = graph.pointwise(PointwiseOperation::activation_backward(
        Activation::relu(),
        dgrad,
        bn_output,
        half_intermediate.output_tensor(),
    ))?;
    let result = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let outputs = graph.batch_normalization_backward_weights(
        BatchNormalizationBackwardWeightsInputs {
            input: x,
            statistics: NormalizationSavedStatistics { mean, inv_variance },
            output_gradient: drelu,
            scale,
        },
        BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: result,
            bias_gradient: result,
            equivalent_bias: result,
            equivalent_output_gradient_scale: result,
            equivalent_input_scale: result,
        },
        BatchNormalizationBackwardWeightsConfig::new()
            .with_name("batch_normalization_backward_weights")
            .with_compute_type(DataType::F32),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    let mut seed = 123_456_789_u32;
    for id in [dy, weight, x] {
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
    let drelu_dev = DeviceMemory::<f16>::zeroed(cuda, graph.shape(drelu)?.storage_span()?)?;
    bindings.set_owned_region(
        drelu,
        OwnedDeviceRegion::from_memory(&drelu_dev),
        BindingAccess::ReadWrite,
    )?;
    let output_ids = [
        outputs.scale_gradient,
        outputs.bias_gradient,
        outputs.equivalent_bias,
        outputs.equivalent_output_gradient_scale,
        outputs.equivalent_input_scale,
    ];
    for id in output_ids {
        let memory = DeviceMemory::<f32>::zeroed(cuda, graph.shape(id)?.storage_span()?)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain initialized inputs and independent writable results.
    // Convolution and pointwise intermediates need no caller storage, and all resources
    // remain live through stream completion, including an enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for id in output_ids {
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
    common::finish("frontend_batch_normalization_backward_weights", run())
}
