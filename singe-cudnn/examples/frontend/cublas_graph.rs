mod common;

use singe_cublas::{blas::level3, context::Context as CublasContext, types::Operation};
use singe_cuda::{graph::Graph as CudaGraph, memory::DeviceMemory, stream::StreamCaptureMode};
use singe_cudnn::{
    backend::behavior::BackendBehaviorNote,
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        graph::Graph,
        operation::{HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};

fn map_cublas<T>(result: singe_cublas::error::Result<T>) -> Result<T> {
    result.map_err(|error| Error::FrontendCompile(error.to_string()))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let cublas = map_cublas(CublasContext::create(ctx.cudnn.cuda_context()))?;
    map_cublas(cublas.set_stream(Some(&ctx.stream)))?;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F32)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32)
        .with_name("cublas-then-cudnn");

    let x =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 2])?).with_id(1000));
    let bias = graph.tensor(TensorSpec::scalar_f32(1.0)?.with_id(1001));
    let y =
        graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([1, 2, 2])?).with_id(1002));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: x,
        rhs: bias,
        output: y,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let compiled = graph
        .plan_candidates(&ctx.cudnn, &[HeuristicMode::A])?
        .require_behavior_notes(vec![BackendBehaviorNote::SupportsCudaGraphNativeApi])?
        .compile(&ctx.cudnn)?;

    let a_dev = DeviceMemory::from_slice(&[1.0_f32, 3.0, 2.0, 4.0])?;
    let b_dev = DeviceMemory::from_slice(&[5.0_f32, 7.0, 6.0, 8.0])?;
    let mut x_dev = DeviceMemory::<f32>::zeroes(4)?;
    let mut y_dev = DeviceMemory::<f32>::zeroes(4)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(y, &mut y_dev)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    ctx.stream.begin_capture(StreamCaptureMode::Relaxed)?;
    map_cublas(level3::sgemm(
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
    ))?;
    let cublas_graph = ctx.stream.end_capture()?;

    let mut main_graph = CudaGraph::create()?;
    let cublas_node = main_graph.add_child_graph_node(&[], &cublas_graph)?;
    match compiled.append_to_cuda_graph(
        &ctx.cudnn,
        &bindings,
        Some(&mut workspace),
        &mut main_graph,
        &[cublas_node],
    ) {
        Ok(_) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }

    let executable = main_graph.instantiate()?;
    executable.launch(&ctx.stream)?;
    ctx.stream.synchronize()?;

    let output = y_dev.copy_to_host_vec()?;
    // cuBLAS GEMM uses column-major matrix semantics, so the flattened buffer
    // is the column-major result plus the cuDNN pointwise bias.
    assert_eq!(output, vec![20.0_f32, 44.0, 23.0, 51.0]);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_cublas_graph", run())
}
