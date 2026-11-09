use singe_cuda::{
    graph::{Graph as NativeCudaGraph, GraphNode},
    stream::StreamCaptureMode,
};

use crate::{
    backend::{
        cuda::{
            CudaStream,
            compiled::{
                build::{
                    attach::{
                        CudnnGraphAttachment, attach_cublas_graph, attach_cublaslt_graph,
                        attach_cudnn_graph,
                    },
                    provider::{
                        GraphProviderLowering, GraphProviderNode, GraphProviderRequest,
                        cudnn::note_cudnn_engines, kernel::try_add_kernel_provider_node,
                        try_provider,
                    },
                    zero_work::{GraphZeroWorkNode, try_add_zero_work_matmul_node},
                },
                plan::{
                    cublas::{
                        build_cublas_matmul_graph, build_cublaslt_matmul_graph,
                        list_cublaslt_matmul_configs,
                    },
                    cudnn::build_cudnn_matmul_graph,
                    lowering::CudaGraphProvider,
                },
            },
            cudnn_backend::cudnn_takes_matmul,
            gemm::{ScaledMatmulPointers, enqueue_scaled_matmul},
            graph_value::CudaGraphValue,
            provider_selection::{KernelChoice, Operand, matmul_kernel},
            tensor_region::CudaTensorSource,
            tuning::{ForcedLibrary, TunedChoice, TunedWay, build_forced, way_key},
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::ValueId,
    utility::Locked,
};

pub fn try_add_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    if let Some(node) = try_add_zero_work_matmul_node(request, inputs, lowering)? {
        return Ok(Some(match node {
            GraphZeroWorkNode::Native(graph_node) => GraphProviderNode::Native(graph_node),
            GraphZeroWorkNode::OutputOnly => GraphProviderNode::OutputOnly,
        }));
    }
    try_add_nonzero_matmul_provider_node(request, inputs, lowering)
        .map(|node| Some(GraphProviderNode::Native(node)))
}

fn try_add_nonzero_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<GraphNode> {
    // Operands no library reads whole, and a record that chose generated kernels, skip the libraries.
    let ir = request.ir;
    let operand = |value: ValueId| -> Result<Operand<'_>> {
        Ok(Operand {
            data_type: ir.required_value_data_type(value)?,
            layout: ir.required_value_layout(value)?,
        })
    };
    if (matmul_kernel(operand(inputs[0])?, operand(inputs[1])?) == KernelChoice::General
        || request.lowering_plan.way(request.id) == Some(TunedWay::Generated))
        && let Some(graph_node) = try_add_kernel_provider_node(request, lowering)?
    {
        return Ok(graph_node);
    }

    // cuBLAS reads every operand layout the selector accepts in place and measured faster than cuDNN on a decoder block, so it has the lowest cost.
    if let Some(graph_node) = try_add_cublas_matmul_provider_node(request, inputs, lowering)? {
        return Ok(graph_node);
    }

    if let Some(graph_node) = try_add_cudnn_matmul_provider_node(request, inputs, lowering)? {
        return Ok(graph_node);
    }

    if let Some(graph_node) = try_add_cublaslt_matmul_provider_node(request, inputs, lowering)? {
        return Ok(graph_node);
    }

    if let Some(graph_node) = try_add_kernel_provider_node(request, lowering)? {
        return Ok(graph_node);
    }
    Err(request
        .diagnostics
        .lowering_failure(lowering.provider_failures))
}

pub fn try_add_cublas_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::Cublas,
        |_| {
            build_cublas_matmul_graph(
                request.runtime,
                request.ir,
                request.id,
                inputs,
                request.output,
                request.cache,
            )
        },
        |child_graph, lowering| {
            attach_cublas_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.library_resources,
                request.ir,
                request.id,
                child_graph,
            )
        },
    )
}

pub fn try_add_cublaslt_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::CublasLt,
        |_| {
            let graph = build_forced(request.forced, ForcedLibrary::cublaslt, |config| {
                build_cublaslt_matmul_graph(
                    request.runtime,
                    request.ir,
                    request.id,
                    inputs,
                    request.output,
                    request.cache,
                    config,
                )
            })?;
            if let Some(recorder) = request.library_recorder {
                recorder.note(
                    &way_key(request.ir, request.id)?,
                    CudaGraphProvider::CublasLt,
                    list_cublaslt_matmul_configs(
                        request.runtime,
                        request.ir,
                        inputs,
                        request.output,
                        request.cache,
                    )?
                    .into_iter()
                    .map(TunedChoice::CublasLt)
                    .collect(),
                )?;
            }
            Ok(graph)
        },
        |cublaslt_graph, lowering| {
            attach_cublaslt_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.library_resources,
                request.ir,
                request.id,
                cublaslt_graph,
            )
        },
    )
}

pub fn try_add_cudnn_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    let ir = request.ir;
    if !cudnn_takes_matmul(
        ir.required_value_data_type(inputs[0])?,
        ir.required_value_layout(inputs[0])?,
        ir.required_value_layout(inputs[1])?,
    ) {
        return Ok(None);
    }

    try_provider(
        request,
        lowering,
        CudaGraphProvider::CudnnBackend,
        |_| {
            let graph = build_forced(request.forced, ForcedLibrary::cudnn, |engine| {
                build_cudnn_matmul_graph(
                    request.runtime,
                    request.ir,
                    request.id,
                    inputs,
                    request.output,
                    request.cache,
                    engine,
                )
            })?;
            note_cudnn_engines(request, &graph.1, true)?;
            Ok(graph)
        },
        |(child_graph, executable), lowering| {
            attach_cudnn_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.cudnn_steps,
                request.ir,
                request.id,
                request.output,
                CudnnGraphAttachment {
                    child_graph,
                    executable,
                    update_on_replay: true,
                },
            )
        },
    )
}

// Captures `scaled_mm` on FP8 tensor cores into a child graph, for the operands the graph passes left whole, which cuBLASLt takes.
pub fn try_add_scaled_matmul_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 4],
    output_type: DataType,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::CublasLt,
        |_lowering| build_scaled_matmul_graph(request, inputs, output_type),
        |child_graph, lowering| {
            attach_cublas_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.library_resources,
                request.ir,
                request.id,
                child_graph,
            )
        },
    )
}

fn build_scaled_matmul_graph(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 4],
    output_type: DataType,
) -> Result<NativeCudaGraph> {
    let values = inputs
        .iter()
        .map(|input| {
            request
                .cache
                .get(input)
                .ok_or_else(|| Error::missing_input("build scaled matmul graph"))
        })
        .collect::<Result<Vec<_>>>()?;
    let [lhs, rhs, lhs_scale, rhs_scale] = values[..] else {
        return Err(Error::missing_input("build scaled matmul graph"));
    };
    let operand = |value: &CudaGraphValue| (value.shape().clone(), value.data_type());
    let [
        lhs_operand,
        rhs_operand,
        lhs_scale_operand,
        rhs_scale_operand,
    ] = [lhs, rhs, lhs_scale, rhs_scale].map(operand);
    let problem = spec::scaled_matmul_problem(
        (&lhs_operand.0, lhs_operand.1),
        (&rhs_operand.0, rhs_operand.1),
        (&lhs_scale_operand.0, lhs_scale_operand.1),
        (&rhs_scale_operand.0, rhs_scale_operand.1),
        "cuda graph scaled_mm",
    )?;
    if problem.scaling != spec::MatmulScaling::Tensorwise
        || !spec::takes_scaled_matmul(lhs.layout(), rhs.layout(), &problem)
        || !request.output.is_contiguous()
    {
        return Err(Error::unsupported_operation(
            "cuda graph scaled_mm operand layouts",
        ));
    }

    let pointer = |value: &CudaGraphValue| -> Result<_> {
        Ok(value.tensor_region()?.region().as_ptr().cast_const().cast())
    };
    let pointers = ScaledMatmulPointers {
        lhs: pointer(lhs)?,
        rhs: pointer(rhs)?,
        lhs_scale: pointer(lhs_scale)?,
        rhs_scale: pointer(rhs_scale)?,
        output: request.output.tensor_region()?.region().as_ptr().cast(),
    };
    let stream = CudaStream::create(request.output.device())?;
    let cublaslt = request.runtime.cublaslt().locked("cuda runtime cache")?;
    let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
        // SAFETY: the compiled graph keeps the operands, scales, and output alive while it replays.
        unsafe {
            enqueue_scaled_matmul(
                &cublaslt,
                &problem,
                [lhs.data_type(), rhs.data_type(), output_type],
                &pointers,
                Some(stream.raw()),
            )
        }
    })?;
    Ok(graph.into_native())
}
