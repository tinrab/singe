mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{ConvolutionConfig, PointwiseOperation},
        plan::{
            BuildPlanPolicy, CompileConfig, CompiledGraph, ExecuteArgs, HeuristicMode,
            NativePlanRestoreTarget, NativePlanSerializationConfig, NativePlanSet,
            NativePlanStructure,
        },
    },
    math::NanPropagation,
};

const X_ID: TensorId = TensorId::new(1100);
const W_ID: TensorId = TensorId::new(1101);
const SCALE_ID: TensorId = TensorId::new(1102);
const BIAS_ID: TensorId = TensorId::new(1103);
const Y_ID: TensorId = TensorId::new(1104);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn build_graph() -> Result<Graph> {
    let n = 4_i64;
    let c = 32_i64;
    let h = 16_i64;
    let w = 16_i64;
    let k = 64_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("compiled-serialize-demo")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let x = graph.insert_tensor(
        X_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([n, c, h, w], [c * h * w, 1, c * w, c])?,
        ),
    )?;
    let w_tensor = graph.insert_tensor(
        W_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([k, c, r, s], [c * r * s, 1, c * s, c])?,
        ),
    )?;
    let conv = graph.convolution_forward(
        x,
        w_tensor,
        ConvolutionConfig::new(2)
            .with_compute_type(DataType::F32)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
        TensorOutputConfig::new().with_data_type(DataType::F16),
    )?;
    let scale = graph.insert_tensor(
        SCALE_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([1, k, 1, 1], [k, 1, k, k])?,
        ),
    )?;
    let scaled = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            conv,
            scale,
            TensorOutputConfig::new()
                .with_data_type(DataType::F16)
                .virtual_tensor(),
        )
        .with_compute_type(Some(DataType::F32)),
    )?;
    let bias = graph.insert_tensor(
        BIAS_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([1, k, 1, 1], [k, 1, k, k])?,
        ),
    )?;
    let biased = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Add,
            scaled,
            bias,
            TensorOutputConfig::new()
                .with_data_type(DataType::F16)
                .virtual_tensor(),
        )
        .with_compute_type(Some(DataType::F32)),
    )?;
    let y = graph.insert_tensor(
        Y_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([n, k, h, w], [k * h * w, 1, k * w, k])?,
        ),
    )?;
    graph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::ReluFwd,
        input: biased,
        output: y,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    })?;

    Ok(graph)
}

fn run() -> Result<()> {
    let mut example = common::ExampleContext::create()?;
    let graph = build_graph()?;
    let all_plans = args().any(|argument| argument == "--all-plans");
    let plan_only = args().any(|argument| argument == "--plan-only");
    let no_warmup = args().any(|argument| argument == "--no-warmup");

    let config = CompileConfig::new()
        .with_heuristic_modes([HeuristicMode::A])
        .with_build_policy(if all_plans {
            BuildPlanPolicy::AllSupported
        } else {
            BuildPlanPolicy::FirstSupported
        });
    let compiled = graph.compile(Some(&example.cudnn), &config)?;
    let serialized = compiled.serialize(NativePlanSerializationConfig {
        plans: if all_plans {
            NativePlanSet::All
        } else {
            NativePlanSet::Selected
        },
        structure: if plan_only {
            NativePlanStructure::Omit
        } else {
            NativePlanStructure::Include
        },
    })?;
    let target = if no_warmup {
        NativePlanRestoreTarget::ContextWithoutWarmup(&example.cudnn)
    } else {
        NativePlanRestoreTarget::Context(&mut example.cudnn)
    };
    // SAFETY: this unmodified artifact contains the native plans and binding metadata just serialized together.
    let restored = unsafe { CompiledGraph::deserialize(target, &serialized) }?;
    drop(compiled);
    drop(graph);

    println!(
        "Restored graph name: {:?}",
        restored.graph().and_then(Graph::name)
    );
    println!(
        "Declared recipe name: {:?}",
        restored.recipe().and_then(Graph::name)
    );
    let workspace_bytes = restored.workspace_size();
    println!(
        "Restored plans: {:?}, workspace: {workspace_bytes} bytes",
        restored.names()
    );

    let mut seed = 123_456_789_u32;
    let cuda = example.cudnn.cuda_context();
    let inputs = [X_ID, W_ID, SCALE_ID, BIAS_ID]
        .into_iter()
        .map(|id| -> Result<_> {
            let elements = restored.tensor_info(id)?.spec.shape.storage_span()?;
            let host = init_f16_image(elements, &mut seed);
            Ok((id, DeviceMemory::from_slice(cuda, &host)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let output_elements = restored.tensor_info(Y_ID)?.spec.shape.storage_span()?;
    let mut y_dev = DeviceMemory::<f16>::zeroed(cuda, output_elements)?;
    let mut workspace = common::workspace(cuda, workspace_bytes)?;

    let mut bindings = restored.bindings()?;
    for (id, memory) in &inputs {
        bindings.set_input(*id, &memory.view())?;
    }
    bindings.set(Y_ID, &mut y_dev)?;
    // SAFETY: restored plan metadata, initialized inputs, writable output and workspace remain live through completion.
    example.execute_and_wait(|context| unsafe {
        restored.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_compiled_graph_serialize", run())
}
