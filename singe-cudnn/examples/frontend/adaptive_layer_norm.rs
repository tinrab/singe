mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AdaptiveLayerNormalizationConfig, AdaptiveLayerNormalizationInputs,
            AdaptiveLayerNormalizationOutputs, NormalizationSavedStatistics,
            NormalizationStatistics,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

#[derive(Clone, Copy)]
struct AdaptiveLayerNormShape {
    b: i64,
    s: i64,
    d: i64,
}

const ADAPTIVE_LAYER_NORM_SHAPES: [AdaptiveLayerNormShape; 4] = [
    AdaptiveLayerNormShape {
        b: 4,
        s: 1024,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 8,
        s: 1024,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 4,
        s: 512,
        d: 128,
    },
    AdaptiveLayerNormShape {
        b: 8,
        s: 512,
        d: 128,
    },
];

fn run_shape(example: &common::ExampleContext, shape: AdaptiveLayerNormShape) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let x = graph.tensor(
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([shape.b, shape.s, shape.d])?,
        )
        .with_name("X"),
    );
    let affine = TensorSpec::new(DataType::F32, Shape::contiguous([shape.b, 1, shape.d])?);
    let scale = graph.tensor(affine.clone().with_name("scale"));
    let bias = graph.tensor(affine.with_name("bias"));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5).with_name("epsilon"));
    let result = TensorOutputConfig::new().output_tensor();
    let statistic = result.with_data_type(DataType::F32);
    let outputs = graph.adaptive_layer_normalization(
        AdaptiveLayerNormalizationInputs {
            input: x,
            scale,
            bias: Some(bias),
        },
        AdaptiveLayerNormalizationOutputs {
            output: result,
            statistics: NormalizationStatistics::Training(NormalizationSavedStatistics {
                mean: statistic,
                inv_variance: statistic,
            }),
        },
        AdaptiveLayerNormalizationConfig::new(epsilon).with_name("adaptive_layer_norm"),
    )?;
    let statistics = outputs
        .statistics
        .saved()
        .ok_or(FrontendError::InternalStateMissing {
            name: "adaptive layer norm training statistics",
        })?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut seed = 123_456_789_u32;
    let x_values = common::init_f32_image(graph.shape(x)?.storage_span()?, &mut seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let x_dev = DeviceMemory::from_slice(cuda, &x_values)?;
    let scale_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f32_image(graph.shape(scale)?.storage_span()?, &mut seed),
    )?;
    let bias_dev = DeviceMemory::from_slice(
        cuda,
        &common::init_f32_image(graph.shape(bias)?.storage_span()?, &mut seed),
    )?;
    let mut y_dev =
        DeviceMemory::<bf16>::zeroed(cuda, graph.shape(outputs.output)?.storage_span()?)?;
    let mut mean_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.mean)?.storage_span()?)?;
    let mut inv_variance_dev =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(statistics.inv_variance)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(outputs.output, &mut y_dev)?;
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

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    for shape in ADAPTIVE_LAYER_NORM_SHAPES {
        run_shape(&context, shape)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_adaptive_layer_norm", run())
}
