use singe_cudnn::backend::{
    execution::pointwise::{
        PointwiseDescriptor as CudnnPointwiseDescriptor,
        PointwiseOperation as CudnnPointwiseOperation,
    },
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
};

#[derive(Debug)]
pub struct CudnnBackendPointwiseOperation {
    tensors: Vec<CudnnBackendTensor>,
    _descriptors: Vec<CudnnPointwiseDescriptor>,
    _operations: Vec<CudnnPointwiseOperation>,
}

impl CudnnBackendPointwiseOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.id() == id && !tensor.is_virtual())
    }

    pub(crate) fn new(
        tensors: Vec<CudnnBackendTensor>,
        descriptors: Vec<CudnnPointwiseDescriptor>,
        operations: Vec<CudnnPointwiseOperation>,
    ) -> Self {
        Self {
            tensors,
            _descriptors: descriptors,
            _operations: operations,
        }
    }
}
