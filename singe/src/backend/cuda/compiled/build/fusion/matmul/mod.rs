mod activation;
mod add;
mod add_relu;
mod residual;

pub use self::{
    activation::try_cublaslt_matmul_activation_fusion,
    residual::{residual_add, try_cublaslt_matmul_residual_fusion},
};
use self::{add::try_matmul_add_fusion, add_relu::try_cudnn_matmul_add_relu_fusion};
use crate::{
    backend::cuda::compiled::build::{
        fusion::GraphFusionLowering, lower::GraphOperationLoweringRequest,
    },
    error::Result,
    graph::ir::ValueId,
};

pub fn try_add_cudnn_matmul_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    if try_cudnn_matmul_add_relu_fusion(request, inputs, lowering)? {
        return Ok(true);
    }

    if try_matmul_add_fusion(request, inputs, lowering)? {
        return Ok(true);
    }

    Ok(false)
}
