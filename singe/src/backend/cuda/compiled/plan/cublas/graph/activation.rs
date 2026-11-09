use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cublas::lt::types::Epilogue;
use singe_cuda::stream::StreamCaptureMode;

use crate::{
    backend::cuda::{
        CudaStream,
        compiled::plan::cublas::{
            algorithm::cublaslt_matmul_algorithm,
            graph::{CublasLtMatmulGraph, validation::validate_cublaslt_matmul_input},
            launch::{CublasLtMatmulLaunch, launch_cublaslt_matmul},
            layout::CublasLtMatrixLayout,
        },
        graph_binding::{validate_graph_matmul, validate_graph_node_value_tensor},
        graph_value::CudaGraphValue,
        runtime::CudaRuntime,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

/// Captures matrix multiplication with a fused activation epilogue into a child CUDA graph.
/// The compiled graph retains operand, output, and workspace allocations through replay.
///
/// # Errors
///
/// Returns an error if matrix or activation metadata is incompatible, or the operands or algorithm cannot be prepared.
pub fn build_cublaslt_matmul_activation_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    activation: ValueId,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    epilogue: Epilogue,
) -> Result<CublasLtMatmulGraph> {
    validate_graph_matmul(ir, id, inputs)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cublaslt matmul activation graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cublaslt matmul activation graph"))?;
    validate_cublaslt_matmul_input(ir, inputs[0], a)?;
    validate_cublaslt_matmul_input(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, activation, output)?;

    let output_value = activation;
    let a_layout = CublasLtMatrixLayout::create(a, "lhs")?;
    let b_layout = CublasLtMatrixLayout::create(b, "rhs")?;
    let algorithm =
        cublaslt_matmul_algorithm(runtime, &a_layout, &b_layout, output, epilogue, None)?;
    let stream = CudaStream::create(ir.required_value(output_value)?.device())?;
    let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
        launch_cublaslt_matmul(CublasLtMatmulLaunch {
            runtime,
            a,
            b,
            output,
            a_layout: &a_layout,
            b_layout: &b_layout,
            algorithm: algorithm.algorithm(),
            compute_type: algorithm.compute_type(),
            epilogue,
            bias: None,
            addend: None,
            workspace: None,
            stream: None,
        })
    })?;
    Ok(CublasLtMatmulGraph::new(
        graph.into_native(),
        algorithm.workspace_bytes(),
    ))
}
