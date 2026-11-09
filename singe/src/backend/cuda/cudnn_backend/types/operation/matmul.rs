use singe_cudnn::backend::{
    execution::{
        matmul::{
            MatmulDescriptor as CudnnMatmulDescriptor, MatmulOperation as CudnnMatmulOperation,
        },
        pointwise::{
            PointwiseDescriptor as CudnnPointwiseDescriptor,
            PointwiseOperation as CudnnPointwiseOperation,
        },
    },
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
};

use crate::backend::cuda::cudnn_backend::types::tensors::{
    CudnnBackendMatmulAddReluTensors, CudnnBackendMatmulAddTensors,
};

#[derive(Debug)]
pub struct CudnnBackendMatmulOperation {
    a: CudnnBackendTensor,
    b: CudnnBackendTensor,
    output: CudnnBackendTensor,
    _matmul: CudnnMatmulDescriptor,
    _operation: CudnnMatmulOperation,
}

impl CudnnBackendMatmulOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        [&self.a, &self.b, &self.output]
            .into_iter()
            .find(|tensor| tensor.id() == id)
    }

    pub(crate) fn new(
        a: CudnnBackendTensor,
        b: CudnnBackendTensor,
        output: CudnnBackendTensor,
        matmul: CudnnMatmulDescriptor,
        operation: CudnnMatmulOperation,
    ) -> Self {
        Self {
            a,
            b,
            output,
            _matmul: matmul,
            _operation: operation,
        }
    }
}

#[derive(Debug)]
pub struct CudnnBackendMatmulAddOperation {
    a: CudnnBackendTensor,
    b: CudnnBackendTensor,
    bias: CudnnBackendTensor,
    _matmul_output: CudnnBackendTensor,
    output: CudnnBackendTensor,
    _matmul: CudnnMatmulDescriptor,
    _matmul_operation: CudnnMatmulOperation,
    _add: CudnnPointwiseDescriptor,
    _add_operation: CudnnPointwiseOperation,
}

impl CudnnBackendMatmulAddOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        [&self.a, &self.b, &self.bias, &self.output]
            .into_iter()
            .find(|tensor| tensor.id() == id)
    }

    pub(crate) fn new(
        tensors: CudnnBackendMatmulAddTensors,
        matmul: CudnnMatmulDescriptor,
        matmul_operation: CudnnMatmulOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self {
            a: tensors.a,
            b: tensors.b,
            bias: tensors.bias,
            _matmul_output: tensors.matmul_output,
            output: tensors.output,
            _matmul: matmul,
            _matmul_operation: matmul_operation,
            _add: add,
            _add_operation: add_operation,
        }
    }
}

#[derive(Debug)]
pub struct CudnnBackendMatmulAddReluOperation {
    a: CudnnBackendTensor,
    b: CudnnBackendTensor,
    bias: CudnnBackendTensor,
    _matmul_output: CudnnBackendTensor,
    _biased: CudnnBackendTensor,
    output: CudnnBackendTensor,
    _matmul: CudnnMatmulDescriptor,
    _matmul_operation: CudnnMatmulOperation,
    _add: CudnnPointwiseDescriptor,
    _add_operation: CudnnPointwiseOperation,
    _relu: CudnnPointwiseDescriptor,
    _relu_operation: CudnnPointwiseOperation,
}

impl CudnnBackendMatmulAddReluOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        [&self.a, &self.b, &self.bias, &self.output]
            .into_iter()
            .find(|tensor| tensor.id() == id)
    }

    pub(crate) fn new(
        tensors: CudnnBackendMatmulAddReluTensors,
        matmul: CudnnMatmulDescriptor,
        matmul_operation: CudnnMatmulOperation,
        add: CudnnPointwiseDescriptor,
        add_operation: CudnnPointwiseOperation,
        relu: CudnnPointwiseDescriptor,
        relu_operation: CudnnPointwiseOperation,
    ) -> Self {
        Self {
            a: tensors.a,
            b: tensors.b,
            bias: tensors.bias,
            _matmul_output: tensors.matmul_output,
            _biased: tensors.biased,
            output: tensors.output,
            _matmul: matmul,
            _matmul_operation: matmul_operation,
            _add: add,
            _add_operation: add_operation,
            _relu: relu,
            _relu_operation: relu_operation,
        }
    }
}
