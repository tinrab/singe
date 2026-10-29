mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::Bindings,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{MatmulConfig, PointwiseOperation},
        plan::{CompileConfig, CompiledGraph, EngineConfigFilter, ExecuteArgs, HeuristicMode},
    },
    scalar::ScalarValue,
};

fn build_graph(
    workspace_initializers: bool,
    workspace_fill: bool,
) -> Result<(Graph, TensorId, TensorId, TensorId, TensorId)> {
    let b = 8_i64;
    let m = 32_i64;
    let n = 16_i64;
    let k = 8_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("cudagraph-matmul-add")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let a = graph.insert_tensor(
        TensorId::new(1100),
        TensorSpec::new(DataType::F16, Shape::contiguous([b, m, k])?),
    )?;
    let scale_spec = if workspace_initializers {
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1])?)
            .with_byte_alignment(4096)?
            .with_workspace_data(0.5_f32.to_ne_bytes().to_vec())?
    } else {
        TensorSpec::scalar_f32(0.5)
    };
    let scale = graph.tensor(scale_spec);
    let scaled = graph
        .tensor(TensorSpec::new(DataType::F16, Shape::contiguous([b, m, k])?).virtual_tensor());
    graph.pointwise_into(PointwiseOperation::binary(
        PointwiseMode::Mul,
        a,
        scale,
        scaled,
    ))?;
    let mut b_spec = TensorSpec::new(DataType::F16, Shape::contiguous([b, k, n])?);
    if workspace_fill {
        b_spec = b_spec.with_workspace_fill(0);
    } else if workspace_initializers {
        let bytes = (0..b_spec.shape.storage_span()?)
            .flat_map(|_| f16::from_f32(1.0).to_bits().to_ne_bytes())
            .collect();
        b_spec = b_spec.with_workspace_data(bytes)?;
    }
    let b_tensor = graph.insert_tensor(TensorId::new(1101), b_spec)?;
    let matmul = graph
        .tensor(TensorSpec::new(DataType::F32, Shape::contiguous([b, m, n])?).virtual_tensor());
    graph.matmul_into(
        scaled,
        b_tensor,
        matmul,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;
    let bias = graph.insert_tensor(
        TensorId::new(1102),
        TensorSpec::new(DataType::F16, Shape::with_strides([1, 1, 1], [1, 1, 1])?).pass_by_value(),
    )?;
    let d = graph.insert_tensor(
        TensorId::new(1103),
        TensorSpec::new(DataType::F16, Shape::contiguous([b, m, n])?),
    )?;
    graph.pointwise_into(PointwiseOperation::binary(
        PointwiseMode::Add,
        matmul,
        bias,
        d,
    ))?;

    Ok((graph, a, b_tensor, bias, d))
}

fn bind_graph<'a>(
    compiled: &'a CompiledGraph,
    a: TensorId,
    b_tensor: TensorId,
    bias: TensorId,
    d: TensorId,
    a_dev: &DeviceMemory<f16>,
    b_dev: Option<&DeviceMemory<f16>>,
    d_dev: &mut DeviceMemory<f16>,
    bias_value: f16,
) -> Result<Bindings<'a>> {
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, a_dev)?;
    if let Some(b_dev) = b_dev {
        bindings.set_input(b_tensor, b_dev)?;
    }
    bindings.set_scalar(bias, ScalarValue::f16_bits(bias_value.to_bits()))?;
    bindings.set(d, d_dev)?;
    Ok(bindings)
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let workspace_fill = args().any(|argument| argument == "--workspace-fill");
    let workspace_initializers =
        workspace_fill || args().any(|argument| argument == "--workspace-initializers");
    let (graph, a, b_tensor, bias, d) = build_graph(workspace_initializers, workspace_fill)?;
    let input_a_elements = graph.shape(a)?.storage_span()?;
    let input_b_elements = graph.shape(b_tensor)?.storage_span()?;
    let output_elements = graph.shape(d)?.storage_span()?;

    let compiled = graph
        .plan_candidates(
            Some(&context.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?
        .filter(
            &EngineConfigFilter::new()
                .with_included_behavior_notes(&[BackendBehaviorNote::SupportsCudaGraphNativeApi]),
        )
        .compile(Some(&context.cudnn), &CompileConfig::new())?;

    let a_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![f16::from_f32(1.0); input_a_elements],
    )?;
    let b_dev = (!workspace_initializers)
        .then(|| {
            DeviceMemory::from_slice(
                context.cudnn.cuda_context(),
                &vec![f16::from_f32(1.0); input_b_elements],
            )
        })
        .transpose()?;
    let mut d_dev = DeviceMemory::<f16>::zeroed(context.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;
    let bindings = bind_graph(
        &compiled,
        a,
        b_tensor,
        bias,
        d,
        &a_dev,
        b_dev.as_ref(),
        &mut d_dev,
        f16::from_f32(2.0),
    )?;

    let mut main_graph = context.cudnn.cuda_context().create_graph()?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    let child_node = unsafe {
        compiled.append_to_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            &mut main_graph,
            &[],
        )
    }?;
    let mut main_exec = main_graph.instantiate()?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        main_exec.launch(&context.stream)?;
    }
    context.stream.synchronize()?;

    let a_dev_new = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![f16::from_f32(1.0); input_a_elements],
    )?;
    let b_dev_new = (!workspace_initializers)
        .then(|| {
            DeviceMemory::from_slice(
                context.cudnn.cuda_context(),
                &vec![f16::from_f32(1.0); input_b_elements],
            )
        })
        .transpose()?;
    let mut d_dev_new = DeviceMemory::<f16>::zeroed(context.cudnn.cuda_context(), output_elements)?;
    let mut workspace_new =
        common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;
    let bindings_new = bind_graph(
        &compiled,
        a,
        b_tensor,
        bias,
        d,
        &a_dev_new,
        b_dev_new.as_ref(),
        &mut d_dev_new,
        f16::from_f32(1.0),
    )?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    unsafe {
        compiled.update_cuda_graph_node(
            &context.cudnn,
            ExecuteArgs::new(&bindings_new).with_workspace(&mut workspace_new),
            &child_node,
            Some(&mut main_exec),
        )
    }?;
    // Recorded kernels retain runtime-compiled code through their native graph references.
    // Caller buffers and both workspace allocations stay alive through the synchronized replay below.
    drop(bindings_new);
    drop(bindings);
    drop(compiled);
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        main_exec.launch(&context.stream)?;
    }
    context.stream.synchronize()?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_cudagraphs", run())
}
