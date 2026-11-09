use singe_cuda::graph::GraphNode;

use crate::{
    backend::{
        cuda::{
            compiled::build::{
                operation::{binary_inputs, contraction_operation, convolution_operation},
                provider::{
                    GraphProviderLowering, GraphProviderNode, GraphProviderRequest,
                    cudnn::{try_add_attention_provider_node, try_add_convolution_provider_node},
                    cutensor::try_add_cutensor_contraction_provider_node,
                    fft::try_add_fft_provider_node,
                    kernel::try_add_kernel_provider_node,
                    matmul::{try_add_matmul_provider_node, try_add_scaled_matmul_provider_node},
                },
            },
            cudnn_backend::cudnn_reads_key_lengths,
        },
        registry::{Operation, ScaledDotProductAttentionAttributes},
    },
    error::{Error, Result},
    graph::ir::OperationNode,
};

/// Lowers one graph operation to a provider node.
type ProviderLowering = fn(
    &GraphProviderRequest<'_>,
    &OperationNode,
    Operation,
    &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>>;

/// Returns the library lowering registered for `operation`, if any.
///
/// Every operation may also lower to a generated kernel, which is tried first except for matrix multiplication and attention.
fn provider_lowering(operation: Operation) -> Option<ProviderLowering> {
    Some(match operation {
        Operation::MatMul => lower_matmul,
        Operation::ScaledDotProductAttention => lower_attention,
        Operation::Conv2d | Operation::Conv3d => lower_convolution,
        Operation::Contraction => lower_contraction,
        Operation::ScaledMatMul => lower_scaled_matmul,
        Operation::Rfft | Operation::Irfft => lower_fft,
        _ => return None,
    })
}

/// Returns whether `operation` has a library lowering for compiled CUDA graphs.
#[cfg(test)]
pub fn has_provider_lowering(operation: Operation) -> bool {
    // Paged attention reaches cuDNN where its two results lower together.
    provider_lowering(operation).is_some()
        || matches!(
            operation,
            Operation::PagedAttention
                | Operation::PagedAttentionOutput
                | Operation::PagedAttentionLse
        )
}

/// Lowers the operation producing `request.id` to a generated kernel or its registered library lowering.
pub fn try_add_operation_provider_node(
    request: GraphProviderRequest<'_>,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let ir_operation = request.ir.required_operation_for_value(request.id)?;
    let Some(operation) = ir_operation.opcode() else {
        return Ok(None);
    };
    // Allowed libraries take these before generated kernels, and a record has already narrowed what is allowed.
    let library_first = matches!(
        operation,
        Operation::MatMul
            | Operation::ScaledDotProductAttention
            | Operation::Conv2d
            | Operation::Conv3d
    );
    if !library_first && let Some(node) = try_add_kernel_provider_node(&request, lowering)? {
        return Ok(Some(GraphProviderNode::Native(node)));
    }
    provider_lowering(operation).map_or_else(
        || Ok(None),
        |lower| lower(&request, ir_operation, operation, lowering),
    )
}

fn native(node: Result<Option<GraphNode>>) -> Result<Option<GraphProviderNode>> {
    node.map(|node| node.map(GraphProviderNode::Native))
}

fn lower_matmul(
    request: &GraphProviderRequest<'_>,
    _: &OperationNode,
    operation: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let inputs = binary_inputs(request.ir, request.id, operation)?
        .ok_or_else(|| Error::missing_input("lower matmul"))?;
    try_add_matmul_provider_node(request, inputs, lowering)
}

// Attention goes to cuDNN when it is allowed and takes the options, and otherwise to its generated kernel.
fn lower_attention(
    request: &GraphProviderRequest<'_>,
    operation: &OperationNode,
    _: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    // cuDNN takes key lengths as they are, and neither a mask nor sinks.
    let lengths = |attention: &ScaledDotProductAttentionAttributes| match *operation.inputs() {
        [query, key, value] => Some(([query, key, value], None)),
        [query, key, value, lengths]
            if attention.key_lengths
                && cudnn_reads_key_lengths(request.ir, lengths).unwrap_or(false) =>
        {
            Some(([query, key, value], Some(lengths)))
        }
        _ => None,
    };
    if let Some(attention) = operation.attributes().scaled_dot_product_attention()
        && let Some((inputs, key_lengths)) = lengths(attention)
        && let Some(node) =
            try_add_attention_provider_node(request, inputs, key_lengths, attention, lowering)?
    {
        return Ok(Some(GraphProviderNode::Native(node)));
    }
    native(try_add_kernel_provider_node(request, lowering))
}

fn lower_convolution(
    request: &GraphProviderRequest<'_>,
    _: &OperationNode,
    _: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let convolution = convolution_operation(request.ir, request.id)?
        .ok_or_else(|| Error::missing_node("lower convolution"))?;
    if let Some(node) = try_add_convolution_provider_node(request, &convolution, lowering)? {
        return Ok(Some(GraphProviderNode::Native(node)));
    }
    native(try_add_kernel_provider_node(request, lowering))
}

fn lower_fft(
    request: &GraphProviderRequest<'_>,
    operation_node: &OperationNode,
    operation: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let &[input] = operation_node.inputs() else {
        return Err(Error::missing_input("lower fft"));
    };
    native(try_add_fft_provider_node(
        request, operation, input, lowering,
    ))
}

fn lower_scaled_matmul(
    request: &GraphProviderRequest<'_>,
    operation: &OperationNode,
    _: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let &[lhs, rhs, lhs_scale, rhs_scale] = operation.inputs() else {
        return Err(Error::missing_input("lower scaled matmul"));
    };
    let output = operation
        .attributes()
        .cast_data_type()
        .ok_or_else(|| Error::missing_node("lower scaled matmul"))?;
    native(try_add_scaled_matmul_provider_node(
        request,
        [lhs, rhs, lhs_scale, rhs_scale],
        output,
        lowering,
    ))
}

fn lower_contraction(
    request: &GraphProviderRequest<'_>,
    _: &OperationNode,
    _: Operation,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphProviderNode>> {
    let contraction = contraction_operation(request.ir, request.id)?
        .ok_or_else(|| Error::missing_node("lower contraction"))?;
    native(try_add_cutensor_contraction_provider_node(
        request,
        &contraction,
        lowering,
    ))
}

// Whether cuDNN reads `lengths` as its `[B, 1, 1, 1]` lengths: one contiguous `I32` per batch.
