mod attention;
mod concat;
mod conv;
mod matmul;
mod normalization;
mod pointwise;
mod reduction;

use singe_cudnn::backend::{
    execution::{
        OperationGraph as CudnnOperationGraph,
        attention::{DiagonalBandMaskOperation, SdpaForwardOperation},
        convolution::{
            ConvolutionDescriptor as CudnnConvolutionDescriptor,
            ConvolutionForwardOperation as CudnnConvolutionForwardOperation,
        },
        matmul::{
            MatmulDescriptor as CudnnMatmulDescriptor, MatmulOperation as CudnnMatmulOperation,
        },
        normalization::NormalizationForwardOperation,
        pointwise::{
            PointwiseDescriptor, PointwiseDescriptor as CudnnPointwiseDescriptor,
            PointwiseOperation, PointwiseOperation as CudnnPointwiseOperation,
        },
        reduction::{ReductionDescriptor, ReductionOperation},
        tensor_ops::{ConcatOperation, ReshapeOperation},
    },
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor, Tensor as CudnnBackendTensor},
};

use crate::backend::cuda::cudnn_backend::types::{
    operation::{
        attention::CudnnBackendAttentionOperation,
        concat::CudnnBackendConcatOperation,
        conv::{CudnnBackendConv2dBiasReluOperation, CudnnBackendConv2dOperation},
        matmul::{
            CudnnBackendMatmulAddOperation, CudnnBackendMatmulAddReluOperation,
            CudnnBackendMatmulOperation,
        },
        normalization::CudnnBackendNormalizationOperation,
        pointwise::CudnnBackendPointwiseOperation,
        reduction::CudnnBackendReductionOperation,
    },
    tensors::{
        CudnnBackendConv2dBiasReluTensors, CudnnBackendMatmulAddReluTensors,
        CudnnBackendMatmulAddTensors,
    },
};

impl CudnnBackendOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        match self {
            Self::Attention { operation } => operation.binding_tensor(id),
            Self::Normalization { operation } => operation.binding_tensor(id),
            Self::Concat { operation } => operation.binding_tensor(id),
            Self::Conv2d { operation } => operation.binding_tensor(id),
            Self::Conv2dBiasRelu { operation } => operation.binding_tensor(id),
            Self::Matmul { operation } => operation.binding_tensor(id),
            Self::MatmulAdd { operation } => operation.binding_tensor(id),
            Self::MatmulAddRelu { operation } => operation.binding_tensor(id),
            Self::Pointwise { operation } => operation.binding_tensor(id),
            Self::Reduction { operation } => operation.binding_tensor(id),
        }
    }
}

impl CudnnBackendOperation {
    pub fn attention(
        tensors: Vec<Tensor>,
        masks: Vec<DiagonalBandMaskOperation>,
        mask_graph: Option<CudnnOperationGraph>,
        attention: SdpaForwardOperation,
    ) -> Self {
        Self::Attention {
            operation: Box::new(CudnnBackendAttentionOperation::new(
                tensors, masks, mask_graph, attention,
            )),
        }
    }
}

impl CudnnBackendOperation {
    pub fn concat(tensors: Vec<Tensor>, operation: ConcatOperation) -> Self {
        Self::Concat {
            operation: Box::new(CudnnBackendConcatOperation::new(tensors, operation)),
        }
    }
}

impl CudnnBackendOperation {
    pub fn conv2d(
        input: CudnnBackendTensor,
        weight: CudnnBackendTensor,
        output: CudnnBackendTensor,
        convolution: CudnnConvolutionDescriptor,
        operation: CudnnConvolutionForwardOperation,
    ) -> Self {
        Self::Conv2d {
            operation: Box::new(CudnnBackendConv2dOperation::new(
                input,
                weight,
                output,
                convolution,
                operation,
            )),
        }
    }

    pub fn conv2d_bias_relu(
        tensors: CudnnBackendConv2dBiasReluTensors,
        convolution: CudnnConvolutionDescriptor,
        convolution_operation: CudnnConvolutionForwardOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
        relu: CudnnPointwiseDescriptor,
        relu_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self::Conv2dBiasRelu {
            operation: Box::new(CudnnBackendConv2dBiasReluOperation::new(
                tensors,
                convolution,
                convolution_operation,
                add,
                add_operation,
                relu,
                relu_operation,
            )),
        }
    }

    pub fn conv2d_bias(
        input: CudnnBackendTensor,
        weight: CudnnBackendTensor,
        bias: CudnnBackendTensor,
        conv_output: CudnnBackendTensor,
        output: CudnnBackendTensor,
        convolution: CudnnConvolutionDescriptor,
        operation: CudnnConvolutionForwardOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self::Conv2d {
            operation: Box::new(
                CudnnBackendConv2dOperation::new(input, weight, output, convolution, operation)
                    .with_bias(bias, conv_output, add, add_operation),
            ),
        }
    }
}

impl CudnnBackendOperation {
    pub fn matmul(
        a: CudnnBackendTensor,
        b: CudnnBackendTensor,
        output: CudnnBackendTensor,
        matmul: CudnnMatmulDescriptor,
        operation: CudnnMatmulOperation,
    ) -> Self {
        Self::Matmul {
            operation: Box::new(CudnnBackendMatmulOperation::new(
                a, b, output, matmul, operation,
            )),
        }
    }

    pub fn matmul_add(
        tensors: CudnnBackendMatmulAddTensors,
        matmul: CudnnMatmulDescriptor,
        matmul_operation: CudnnMatmulOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self::MatmulAdd {
            operation: Box::new(CudnnBackendMatmulAddOperation::new(
                tensors,
                matmul,
                matmul_operation,
                add,
                add_operation,
            )),
        }
    }

    pub fn matmul_add_relu(
        tensors: CudnnBackendMatmulAddReluTensors,
        matmul: CudnnMatmulDescriptor,
        matmul_operation: CudnnMatmulOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
        relu: CudnnPointwiseDescriptor,
        relu_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self::MatmulAddRelu {
            operation: Box::new(CudnnBackendMatmulAddReluOperation::new(
                tensors,
                matmul,
                matmul_operation,
                add,
                add_operation,
                relu,
                relu_operation,
            )),
        }
    }
}

impl CudnnBackendOperation {
    pub fn normalization(
        tensors: Vec<Tensor>,
        input_reshapes: Vec<ReshapeOperation>,
        normalization: NormalizationForwardOperation,
        pointwise_descriptors: Vec<PointwiseDescriptor>,
        pointwise_operations: Vec<PointwiseOperation>,
        output_reshape: ReshapeOperation,
    ) -> Self {
        Self::Normalization {
            operation: Box::new(CudnnBackendNormalizationOperation::new(
                tensors,
                input_reshapes,
                normalization,
                pointwise_descriptors,
                pointwise_operations,
                output_reshape,
            )),
        }
    }
}

impl CudnnBackendOperation {
    pub fn pointwise(
        tensors: Vec<CudnnBackendTensor>,
        descriptors: Vec<CudnnPointwiseDescriptor>,
        operations: Vec<CudnnPointwiseOperation>,
    ) -> Self {
        Self::Pointwise {
            operation: Box::new(CudnnBackendPointwiseOperation::new(
                tensors,
                descriptors,
                operations,
            )),
        }
    }
}

impl CudnnBackendOperation {
    pub fn reduction(
        input: Tensor,
        output: Tensor,
        reduction: ReductionDescriptor,
        operation: ReductionOperation,
    ) -> Self {
        Self::Reduction {
            operation: Box::new(CudnnBackendReductionOperation::new(
                input, output, reduction, operation,
            )),
        }
    }
}

#[derive(Debug)]
pub enum CudnnBackendOperation {
    Normalization {
        operation: Box<CudnnBackendNormalizationOperation>,
    },
    Concat {
        operation: Box<CudnnBackendConcatOperation>,
    },
    Reduction {
        operation: Box<CudnnBackendReductionOperation>,
    },
    Conv2d {
        operation: Box<CudnnBackendConv2dOperation>,
    },
    Conv2dBiasRelu {
        operation: Box<CudnnBackendConv2dBiasReluOperation>,
    },
    Matmul {
        operation: Box<CudnnBackendMatmulOperation>,
    },
    MatmulAdd {
        operation: Box<CudnnBackendMatmulAddOperation>,
    },
    MatmulAddRelu {
        operation: Box<CudnnBackendMatmulAddReluOperation>,
    },
    Attention {
        operation: Box<CudnnBackendAttentionOperation>,
    },
    Pointwise {
        operation: Box<CudnnBackendPointwiseOperation>,
    },
}
