mod common;

use singe_cublas::{
    blas::level3, context::Context as CublasContext, error::Result as CublasResult,
    types::Operation,
};
use singe_cuda::{memory::DeviceMemory, stream::StreamCaptureMode};
use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::PointwiseOperation,
        plan::{CompileConfig, EngineConfigFilter, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn map_cublas<T>(result: CublasResult<T>) -> Result<T> {
    result.map_err(|error| FrontendError::Compile(error.to_string()).into())
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let cublas = map_cublas(CublasContext::create(context.cudnn.cuda_context()))?;
    map_cublas(cublas.set_stream(Some(&context.stream)))?;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("cublas-then-cudnn")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F32)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let x = graph.insert_tensor(
        TensorId::new(1000),
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 2])?),
    )?;
    let bias = graph.insert_tensor(TensorId::new(1001), TensorSpec::scalar_f32(1.0))?;
    let y = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 2])?),
    )?;
    graph.pointwise_into(PointwiseOperation::Binary {
        name: None,
        mode: PointwiseMode::Add,
        lhs: x,
        rhs: bias,
        output: y,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    })?;

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

    let a_dev = DeviceMemory::from_slice(context.cudnn.cuda_context(), &[1.0_f32, 3.0, 2.0, 4.0])?;
    let b_dev = DeviceMemory::from_slice(context.cudnn.cuda_context(), &[5.0_f32, 7.0, 6.0, 8.0])?;
    let mut x_dev = DeviceMemory::<f32>::zeroed(context.cudnn.cuda_context(), 4)?;
    let mut y_dev = DeviceMemory::<f32>::zeroed(context.cudnn.cuda_context(), 4)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set(x, &mut x_dev)?;
    bindings.set(y, &mut y_dev)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    context.stream.begin_capture(StreamCaptureMode::Relaxed)?;
    // SAFETY: these 2x2 allocations match the GEMM and remain live through graph launch and synchronization below.
    map_cublas(unsafe {
        level3::sgemm(
            &cublas,
            Operation::NonTranspose,
            Operation::NonTranspose,
            2,
            2,
            2,
            &alpha,
            &a_dev,
            2,
            &b_dev,
            2,
            &beta,
            &mut x_dev,
            2,
        )
    })?;
    let cublas_graph = context.stream.end_capture()?;

    let mut main_graph = context.cudnn.cuda_context().create_graph()?;
    let cublas_node = main_graph.add_child_graph_node(&[], &cublas_graph)?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    match unsafe {
        compiled.append_to_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            &mut main_graph,
            &[cublas_node],
        )
    } {
        Ok(_) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }

    let executable = main_graph.instantiate()?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&context.stream)?;
    }
    context.stream.synchronize()?;

    let output = y_dev.copy_to_host_vec()?;
    // cuBLAS GEMM uses column-major matrix semantics, so the flattened buffer
    // is the column-major result plus the cuDNN pointwise bias.
    assert_eq!(output, vec![20.0_f32, 44.0, 23.0, 51.0]);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_cublas_graph", run())
}
