mod common;

use std::env::args;

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
            BatchNormalizationConfig, BatchNormalizationInputs, BatchNormalizationOutputs,
            BatchNormalizationRunningStatistics, BatchNormalizationRunningStats,
            NormalizationSavedStatistics, PointwiseOperation,
        },
        plan::{BuildPlanPolicy, CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let with_running = !args().any(|argument| argument == "--without-running");
    let n = 4;
    let c = 32;
    let h = 16;
    let w = 16;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let activation = TensorSpec::new(
        DataType::F16,
        Shape::with_strides([n, c, h, w], [c * h * w, 1, c * w, c])?,
    );
    let channel = TensorSpec::new(
        DataType::F32,
        Shape::with_strides([1, c, 1, 1], [c, 1, c, c])?,
    );
    let x = graph.tensor(activation.clone().with_name("X"));
    let a = graph.tensor(activation.clone().with_name("A"));
    let y = graph.tensor(activation.with_name("Y"));
    let scale = graph.tensor(channel.clone().with_name("scale"));
    let bias = graph.tensor(channel.clone().with_name("bias"));
    let peer = TensorSpec::new(
        DataType::F32,
        Shape::with_strides([2, n * c, 1, 1], [n * c, 1, n * c, n * c])?,
    );
    let peer_stats_0 = graph.tensor(peer.clone().with_name("peer_stats_0"));
    let peer_stats_1 = graph.tensor(peer.with_name("peer_stats_1"));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));
    let mut running = None;
    let mut channel_inputs = vec![(scale, 1.0_f32), (bias, 0.0)];
    if with_running {
        let mean = graph.tensor(channel.clone().with_name("prev_running_mean"));
        let variance = graph.tensor(channel.with_name("prev_running_var"));
        let momentum = graph.tensor(TensorSpec::scalar_f32(0.1));
        running = Some(BatchNormalizationRunningStats {
            momentum,
            previous: BatchNormalizationRunningStatistics { mean, variance },
        });
        channel_inputs.extend([(mean, 0.0), (variance, 1.0)]);
    }
    let channel_output = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let outputs = graph.batch_normalization(
        BatchNormalizationInputs {
            input: x,
            scale,
            bias,
        },
        BatchNormalizationOutputs {
            output: TensorOutputConfig::new(),
            statistics: NormalizationSavedStatistics {
                mean: channel_output,
                inv_variance: channel_output,
            },
            running: with_running.then_some(BatchNormalizationRunningStatistics {
                mean: channel_output,
                variance: channel_output,
            }),
        },
        BatchNormalizationConfig::new(epsilon)
            .with_name("batch_norm")
            .with_peer_stats(vec![peer_stats_0, peer_stats_1])
            .with_running_stats(running),
    )?;
    let add_output = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Add,
        outputs.output,
        a,
        TensorOutputConfig::new(),
    ))?;
    graph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::ReluFwd,
        input: add_output,
        output: y,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    })?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A, HeuristicMode::Fallback])
            .with_build_policy(BuildPlanPolicy::AllSupported),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut bindings = compiled.bindings()?;
    let mut seed = 123_456_789_u32;
    for id in [x, a] {
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
    for (id, value) in channel_inputs {
        let memory =
            DeviceMemory::from_slice(cuda, &vec![value; graph.shape(id)?.storage_span()?])?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadOnly,
        )?;
    }
    for id in [peer_stats_0, peer_stats_1] {
        let mut values = common::init_f32_image(graph.shape(id)?.storage_span()?, &mut seed);
        if id == peer_stats_0 {
            for index in (1..values.len()).step_by(2) {
                values[index] = f32::from_bits(1);
            }
        }
        let memory = DeviceMemory::from_slice(cuda, &values)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let mut channel_outputs = vec![outputs.statistics.mean, outputs.statistics.inv_variance];
    if let Some(running) = outputs.running {
        channel_outputs.extend([running.mean, running.variance]);
    }
    for id in &channel_outputs {
        let memory = DeviceMemory::<f32>::zeroed(cuda, graph.shape(*id)?.storage_span()?)?;
        bindings.set_owned_region(
            *id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
    }
    let y_dev = DeviceMemory::<f16>::zeroed(cuda, graph.shape(y)?.storage_span()?)?;
    bindings.set_owned_region(
        y,
        OwnedDeviceRegion::from_memory(&y_dev),
        BindingAccess::ReadWrite,
    )?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: bindings retain all initialized operands, peer buffers and writable results.
    // The plan and workspace remain live through stream completion, including enqueue failure.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    for id in channel_outputs {
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
    common::finish("frontend_batch_norm", run())
}
