use singe_cudnn::backend::{
    execution::{
        convolution::{
            ConvolutionDescriptor as CudnnConvolutionDescriptor,
            ConvolutionForwardOperation as CudnnConvolutionForwardOperation,
        },
        pointwise::{
            PointwiseDescriptor as CudnnPointwiseDescriptor,
            PointwiseOperation as CudnnPointwiseOperation,
        },
    },
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
};

use crate::backend::cuda::cudnn_backend::types::tensors::CudnnBackendConv2dBiasReluTensors;

#[derive(Debug)]
pub struct CudnnBackendConv2dOperation {
    input: CudnnBackendTensor,
    weight: CudnnBackendTensor,
    output: CudnnBackendTensor,
    _convolution: CudnnConvolutionDescriptor,
    _operation: CudnnConvolutionForwardOperation,
    bias: Option<CudnnBackendConv2dBias>,
}

#[derive(Debug)]
struct CudnnBackendConv2dBias {
    tensor: CudnnBackendTensor,
    _conv_output: CudnnBackendTensor,
    _add: CudnnPointwiseDescriptor,
    _add_operation: CudnnPointwiseOperation,
}

impl CudnnBackendConv2dOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        [&self.input, &self.weight, &self.output]
            .into_iter()
            .chain(self.bias.as_ref().map(|bias| &bias.tensor))
            .find(|tensor| tensor.id() == id)
    }

    pub(crate) fn new(
        input: CudnnBackendTensor,
        weight: CudnnBackendTensor,
        output: CudnnBackendTensor,
        convolution: CudnnConvolutionDescriptor,
        operation: CudnnConvolutionForwardOperation,
    ) -> Self {
        Self {
            input,
            weight,
            output,
            _convolution: convolution,
            _operation: operation,
            bias: None,
        }
    }

    pub(crate) fn with_bias(
        mut self,
        tensor: CudnnBackendTensor,
        conv_output: CudnnBackendTensor,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
    ) -> Self {
        self.bias = Some(CudnnBackendConv2dBias {
            tensor,
            _conv_output: conv_output,
            _add: add,
            _add_operation: add_operation,
        });
        self
    }
}

#[derive(Debug)]
pub struct CudnnBackendConv2dBiasReluOperation {
    input: CudnnBackendTensor,
    weight: CudnnBackendTensor,
    bias: CudnnBackendTensor,
    _conv_output: CudnnBackendTensor,
    _biased: CudnnBackendTensor,
    output: CudnnBackendTensor,
    _convolution: CudnnConvolutionDescriptor,
    _convolution_operation: CudnnConvolutionForwardOperation,
    _add: CudnnPointwiseDescriptor,
    _add_operation: CudnnPointwiseOperation,
    _relu: CudnnPointwiseDescriptor,
    _relu_operation: CudnnPointwiseOperation,
}

impl CudnnBackendConv2dBiasReluOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        [&self.input, &self.weight, &self.bias, &self.output]
            .into_iter()
            .find(|tensor| tensor.id() == id)
    }

    pub(crate) fn new(
        tensors: CudnnBackendConv2dBiasReluTensors,
        convolution: CudnnConvolutionDescriptor,
        convolution_operation: CudnnConvolutionForwardOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
        relu: CudnnPointwiseDescriptor,
        relu_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self {
            input: tensors.input,
            weight: tensors.weight,
            bias: tensors.bias,
            _conv_output: tensors.conv_output,
            _biased: tensors.biased,
            output: tensors.output,
            _convolution: convolution,
            _convolution_operation: convolution_operation,
            _add: add,
            _add_operation: add_operation,
            _relu: relu,
            _relu_operation: relu_operation,
        }
    }
}
