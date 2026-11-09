use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    fusion::{
                        residual_add, try_add_contraction_fusion,
                        try_add_cudnn_conv2d_bias_relu_fusion, try_add_cudnn_matmul_fusion,
                        try_add_elementwise_fusion, try_add_module_fusion,
                        try_cublaslt_matmul_activation_fusion, try_cublaslt_matmul_residual_fusion,
                    },
                    lower::GraphOperationLoweringRequest,
                    operation::{binary_inputs, conv2d_operation},
                    state::GraphLoweringState,
                },
                plan::{fusion::GroupKind, lowering::CudaGraphProvider},
            },
            diagnostics::GraphProviderFailures,
        },
        registry::Operation,
    },
    error::{Error, Result},
};

/// Lowers the fusion group whose first member is the request's node as one kernel, and returns `false` when there is none or its kernel cannot be planned, so its members lower on their own.
///
/// # Errors
///
/// Returns an error if a group's operand is missing or its kernel's node cannot be added.
pub fn try_lower_fusion_group(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    provider_failures: &mut GraphProviderFailures,
) -> Result<bool> {
    let Some(group) = state.fusion_plan.led_by(request.id).cloned() else {
        return Ok(false);
    };
    // Another fusion, such as cuDNN's, took a member already.
    if group
        .members
        .iter()
        .any(|member| state.skipped_nodes.contains(member))
    {
        return Ok(false);
    }

    let mut fusion_lowering = state.fusion_lowering(request.context, provider_failures);
    let mut fusion_request = *request;
    match &group.kind {
        GroupKind::Contraction { matmul, .. } => {
            fusion_request.id = *matmul;
            fusion_request.index = *request
                .positions
                .get(matmul)
                .ok_or_else(|| Error::missing_node("try lower fusion group"))?;
            try_add_contraction_fusion(&fusion_request, &group, &mut fusion_lowering)
        }
        GroupKind::Module(_) => {
            try_add_module_fusion(&fusion_request, &group, &mut fusion_lowering)
        }
        _ => try_add_elementwise_fusion(&fusion_request, &group, &mut fusion_lowering),
    }
}

/// Lowers a library matmul and the residual add after it as one cuBLASLt call, where the fusion plan fuses neither into another kernel.
///
/// # Errors
///
/// Returns an error if the output cannot be allocated or the call's node cannot be added.
pub fn try_lower_cublaslt_residual(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    provider_failures: &mut GraphProviderFailures,
) -> Result<bool> {
    if !request
        .lowering_plan
        .should_try_provider(request.id, CudaGraphProvider::CublasLt)
        || state.fusion_plan.is_planned(request.id)
    {
        return Ok(false);
    }

    let Some(inputs) = binary_inputs(request.ir, request.id, Operation::MatMul)? else {
        return Ok(false);
    };
    let fusion_request = *request;
    {
        let mut fusion_lowering = state.fusion_lowering(request.context, provider_failures);
        if try_cublaslt_matmul_activation_fusion(&fusion_request, inputs, &mut fusion_lowering)? {
            return Ok(true);
        }
    }

    let Some(residual) = residual_add(&fusion_request, &state.cache)? else {
        return Ok(false);
    };
    if state.fusion_plan.is_planned(residual.0) {
        return Ok(false);
    }

    let mut fusion_lowering = state.fusion_lowering(request.context, provider_failures);
    try_cublaslt_matmul_residual_fusion(&fusion_request, inputs, residual, &mut fusion_lowering)
}

pub fn try_lower_cudnn_fusions(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    provider_failures: &mut GraphProviderFailures,
) -> Result<bool> {
    if !request
        .lowering_plan
        .should_try_provider(request.id, CudaGraphProvider::CudnnBackend)
    {
        return Ok(false);
    }

    if try_lower_cudnn_matmul_fusion(request, state, provider_failures)? {
        return Ok(true);
    }
    try_lower_cudnn_conv2d_bias_relu_fusion(request, state, provider_failures)
}

fn try_lower_cudnn_matmul_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    provider_failures: &mut GraphProviderFailures,
) -> Result<bool> {
    let Some(inputs) = binary_inputs(request.ir, request.id, Operation::MatMul)? else {
        return Ok(false);
    };
    let mut fusion_lowering = state.fusion_lowering(request.context, provider_failures);
    let fusion_request = *request;
    try_add_cudnn_matmul_fusion(&fusion_request, inputs, &mut fusion_lowering)
}

fn try_lower_cudnn_conv2d_bias_relu_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    provider_failures: &mut GraphProviderFailures,
) -> Result<bool> {
    let Some(conv2d) = conv2d_operation(request.ir, request.id)? else {
        return Ok(false);
    };
    let Some(bias) = conv2d.bias else {
        return Ok(false);
    };
    let mut fusion_lowering = state.fusion_lowering(request.context, provider_failures);
    let fusion_request = *request;
    try_add_cudnn_conv2d_bias_relu_fusion(
        &fusion_request,
        conv2d.input,
        conv2d.weight,
        bias,
        conv2d.stride,
        conv2d.padding,
        conv2d.dilation,
        conv2d.groups,
        &mut fusion_lowering,
    )
}
