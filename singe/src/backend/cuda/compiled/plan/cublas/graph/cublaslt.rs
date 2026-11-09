use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cublas::lt::types::Epilogue;
use singe_cuda::stream::StreamCaptureMode;

use crate::{
    backend::cuda::{
        CudaStream,
        compiled::plan::cublas::{
            algorithm::{
                configured_cublaslt_matmul_algorithm, cublaslt_matmul_algorithm,
                cublaslt_matmul_configs, validate_cublaslt_bias,
            },
            graph::{CublasLtMatmulGraph, validation::validate_cublaslt_matmul_input},
            launch::{CublasLtMatmulLaunch, launch_cublaslt_matmul},
            layout::CublasLtMatrixLayout,
        },
        gemm::{cublaslt_takes_bias, cublaslt_takes_strides},
        graph_binding::{
            validate_graph_matmul, validate_graph_matmul_add, validate_graph_node_value_tensor,
        },
        graph_value::CudaGraphValue,
        runtime::{CublasLtMatmulAlgorithmFingerprint, CudaRuntime},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

/// Captures cuBLASLt matrix multiplication into a child CUDA graph.
/// The compiled graph retains input, output, and any workspace allocations through replay.
///
/// # Errors
///
/// Returns an error if operands or result metadata are missing or incompatible, tensor layouts or types are unsupported, required algorithm selection fails, or stream creation or captured native execution fails.
pub fn build_cublaslt_matmul_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    config: Option<&CublasLtMatmulAlgorithmFingerprint>,
) -> Result<CublasLtMatmulGraph> {
    build_matmul_graph(runtime, ir, id, inputs, None, output, cache, config)
}

// What cuBLASLt could run this matmul with, the configuration it would choose first.
//
// # Errors
//
// Returns an error if a value is missing or cuBLASLt fails to answer.
pub fn list_cublaslt_matmul_configs(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    inputs: [ValueId; 2],
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<Vec<CublasLtMatmulAlgorithmFingerprint>> {
    let [a, b] = inputs.map(|input| {
        cache
            .get(&input)
            .ok_or_else(|| Error::missing_input("list cublaslt matmul configurations"))
    });
    let (a, b) = (a?, b?);
    validate_cublaslt_matmul_input(ir, inputs[0], a)?;
    validate_cublaslt_matmul_input(ir, inputs[1], b)?;
    cublaslt_matmul_configs(
        runtime,
        &CublasLtMatrixLayout::create(a, "lhs")?,
        &CublasLtMatrixLayout::create(b, "rhs")?,
        output,
    )
}

/// Captures cuBLASLt matrix multiplication followed by element-wise add `add` of `addend`, which has the product's layout, into a child CUDA graph: cuBLASLt reads the addend as its `C` with a `beta` of one, so the sum is written once.
///
/// # Errors
///
/// Returns an error if [`build_cublaslt_matmul_graph`] would, or the addend's value is missing or has another layout than the output.
pub fn build_cublaslt_matmul_residual_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    add: ValueId,
    addend: ValueId,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<CublasLtMatmulGraph> {
    let addend = cache
        .get(&addend)
        .ok_or_else(|| Error::missing_input("build cublaslt matmul residual graph"))?;
    build_matmul_graph(
        runtime,
        ir,
        id,
        inputs,
        Some((add, addend)),
        output,
        cache,
        None,
    )
}

// Singe's own check of what cuBLASLt can run, before asking it, so what it rejects goes to the next provider.
fn check_cublaslt_strides(
    a: &CublasLtMatrixLayout,
    b: &CublasLtMatrixLayout,
    output: &CudaGraphValue,
) -> Result<()> {
    let row = output.layout().dimensions().last().copied().unwrap_or(1);
    let strides = [
        a.leading_dimension,
        b.leading_dimension,
        a.strided_batch_offset,
        b.strided_batch_offset,
        row,
    ];
    if cublaslt_takes_strides(&strides, output.data_type().size_bytes()) {
        Ok(())
    } else {
        Err(Error::unsupported_operation(
            "cuda graph cuBLASLt matmul with rows narrower than 16 bytes",
        ))
    }
}

fn build_matmul_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    addend: Option<(ValueId, &CudaGraphValue)>,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    config: Option<&CublasLtMatmulAlgorithmFingerprint>,
) -> Result<CublasLtMatmulGraph> {
    validate_graph_matmul(ir, id, inputs)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build matmul graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build matmul graph"))?;
    validate_cublaslt_matmul_input(ir, inputs[0], a)?;
    validate_cublaslt_matmul_input(ir, inputs[1], b)?;
    let output_value = addend.map_or(id, |(add, _)| add);
    validate_graph_node_value_tensor(ir, output_value, output)?;

    let a_layout = CublasLtMatrixLayout::create(a, "lhs")?;
    let b_layout = CublasLtMatrixLayout::create(b, "rhs")?;
    check_cublaslt_strides(&a_layout, &b_layout, output)?;
    let algorithm = match config {
        Some(config) => configured_cublaslt_matmul_algorithm(
            runtime,
            &a_layout,
            &b_layout,
            output,
            Epilogue::Default,
            None,
            config,
        )?,
        None => cublaslt_matmul_algorithm(
            runtime,
            &a_layout,
            &b_layout,
            output,
            Epilogue::Default,
            None,
        )?,
    };
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
            epilogue: Epilogue::Default,
            bias: None,
            addend: addend.map(|(_, addend)| addend),
            workspace: None,
            stream: None,
        })
    })?;
    Ok(CublasLtMatmulGraph::new(
        graph.into_native(),
        algorithm.workspace_bytes(),
    ))
}

/// Captures cuBLASLt matrix multiplication with a bias epilogue into a child CUDA graph.
/// The compiled graph retains input, output, and any workspace allocations through replay.
///
/// # Errors
///
/// Returns an error if operands or result metadata are missing or incompatible, tensor layouts or types are unsupported, required algorithm selection fails, or stream creation or captured native execution fails.
pub fn build_cublaslt_matmul_add_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    add: ValueId,
    bias: ValueId,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<CublasLtMatmulGraph> {
    validate_graph_matmul(ir, id, inputs)?;
    validate_graph_matmul_add(ir, id, add, bias)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cublaslt matmul add graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cublaslt matmul add graph"))?;
    let bias_tensor = cache
        .get(&bias)
        .ok_or_else(|| Error::missing_input("build cublaslt matmul add graph"))?;
    validate_cublaslt_matmul_input(ir, inputs[0], a)?;
    validate_cublaslt_matmul_input(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, bias, bias_tensor)?;
    validate_graph_node_value_tensor(ir, add, output)?;

    let output_value = add;
    let a_layout = CublasLtMatrixLayout::create(a, "lhs")?;
    let b_layout = CublasLtMatrixLayout::create(b, "rhs")?;
    check_cublaslt_strides(&a_layout, &b_layout, output)?;
    if !cublaslt_takes_bias(output.data_type()) {
        return Err(Error::unsupported_operation(
            "cuda graph cuBLASLt bias epilogue of this type under this numerical policy",
        ));
    }

    validate_cublaslt_bias(output, bias_tensor)?;
    let algorithm = cublaslt_matmul_algorithm(
        runtime,
        &a_layout,
        &b_layout,
        output,
        Epilogue::Bias,
        Some(bias_tensor),
    )?;
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
            epilogue: Epilogue::Bias,
            bias: Some(bias_tensor),
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
