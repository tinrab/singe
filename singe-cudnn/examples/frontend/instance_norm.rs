mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig},
        operation::{
            InstanceNormalizationConfig, InstanceNormalizationInputs, InstanceNormalizationOutputs,
            NormalizationSavedStatistics, NormalizationStatistics,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([16, 32, 64, 64])?,
    ));
    let affine = TensorSpec::new(DataType::F16, Shape::contiguous([1, 32, 1, 1])?);
    let scale = graph.tensor(affine.clone());
    let bias = graph.tensor(affine);
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));
    let result = TensorOutputConfig::new().output_tensor();
    let statistic = result.with_data_type(DataType::F32);
    let outputs = graph.instance_normalization(
        InstanceNormalizationInputs { input, scale, bias },
        InstanceNormalizationOutputs {
            output: result,
            statistics: NormalizationStatistics::Training(NormalizationSavedStatistics {
                mean: statistic,
                inv_variance: statistic,
            }),
        },
        InstanceNormalizationConfig::new(epsilon).with_name("instance_norm"),
    )?;
    let statistics = outputs
        .statistics
        .saved()
        .ok_or(FrontendError::InternalStateMissing {
            name: "instance norm training statistics",
        })?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::Instant, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let input_values = common::init_f32_image(graph.shape(input)?.storage_span()?, &mut seed)
        .into_iter()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    let input_dev = DeviceMemory::from_slice(cuda, &input_values)?;
    let scale_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(1.0); graph.shape(scale)?.storage_span()?],
    )?;
    let bias_dev = DeviceMemory::from_slice(
        cuda,
        &vec![f16::from_f32(0.0); graph.shape(bias)?.storage_span()?],
    )?;
    let mut output_dev =
        DeviceMemory::<f16>::zeroed(cuda, graph.shape(outputs.output)?.storage_span()?)?;
    let mut mean_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.mean)?.storage_span()?)?;
    let mut inv_variance_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.inv_variance)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(outputs.output, &mut output_dev)?;
    bindings.set(statistics.mean, &mut mean_dev)?;
    bindings.set(statistics.inv_variance, &mut inv_variance_dev)?;
    // SAFETY: matching initialized operands, output storage and workspace remain live
    // until the helper completes the stream, including on enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for tensor in [outputs.output, statistics.mean, statistics.inv_variance] {
        let tensor = compiled.tensor_info(tensor)?;
        println!(
            "{:?}: {:?}",
            tensor.spec.name,
            tensor.spec.shape.dimensions()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_instance_norm", run())
}
