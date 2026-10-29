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
        operation::{
            BatchNormalizationFinalizeConfig, BatchNormalizationFinalizeInputs,
            BatchNormalizationFinalizeOutputs, BatchNormalizationRunningStatistics,
            BatchNormalizationRunningStats, NormalizationSavedStatistics,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let with_running = !args().any(|argument| argument == "--without-running");
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F16),
        ),
    );
    // Statistics, equivalent coefficients and this operation's math stay FP32 despite FP16 graph defaults.
    let sum = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides([1, 32, 1, 1], [32, 1, 32, 32])?,
        )
        .with_name("sum"),
    );
    let squared_sum = graph.tensor_like(sum, Some("sq_sum"))?;
    let scale = graph.tensor_like(sum, Some("scale"))?;
    let bias = graph.tensor_like(sum, Some("bias"))?;
    let epsilon = graph.tensor(TensorSpec::scalar(0.001_f32));
    let accumulation_count = graph.tensor(TensorSpec::scalar(64_i64));
    let mut running = None;
    let mut input_values = vec![
        (sum, 64.0_f32),
        (squared_sum, 128.0),
        (scale, 1.0),
        (bias, 0.0),
    ];
    if with_running {
        let mean = graph.tensor_like(sum, Some("prev_running_mean"))?;
        let variance = graph.tensor_like(sum, Some("prev_running_var"))?;
        let momentum = graph.tensor(TensorSpec::scalar(0.001_f32));
        running = Some(BatchNormalizationRunningStats {
            momentum,
            previous: BatchNormalizationRunningStatistics { mean, variance },
        });
        input_values.extend([(mean, 0.0), (variance, 1.0)]);
    }
    let output = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let outputs = graph.batch_normalization_finalize(
        BatchNormalizationFinalizeInputs {
            sum,
            squared_sum,
            scale,
            bias,
            accumulation_count,
        },
        BatchNormalizationFinalizeOutputs {
            equivalent_scale: output,
            equivalent_bias: output,
            statistics: NormalizationSavedStatistics {
                mean: output,
                inv_variance: output,
            },
            running: with_running.then_some(BatchNormalizationRunningStatistics {
                mean: output,
                variance: output,
            }),
        },
        BatchNormalizationFinalizeConfig::training(epsilon)
            .with_name("batch_norm_finalize")
            .with_compute_type(DataType::F32)
            .with_running_stats(running),
    )?;
    let mut output_ids = vec![
        outputs.equivalent_scale,
        outputs.equivalent_bias,
        outputs.statistics.mean,
        outputs.statistics.inv_variance,
    ];
    if let Some(running) = outputs.running {
        output_ids.extend([running.mean, running.variance]);
    }
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    for (id, value) in input_values {
        let count = graph.shape(id)?.storage_span()?;
        let memory = DeviceMemory::from_slice(cuda, &vec![value; count])?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    for id in &output_ids {
        let count = graph.shape(*id)?.storage_span()?;
        let memory = DeviceMemory::<f32>::zeroed(cuda, count)?;
        bindings.set_owned_region(
            *id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain initialized FP32 inputs and writable FP32 destinations.
    // Scalars retain their native encodings, all ranges are independent, and buffers
    // and plan remain live through stream completion, including an enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!(
        "BatchNorm finalize produced {} FP32 channel tensors",
        output_ids.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_batch_norm_finalize", run())
}
