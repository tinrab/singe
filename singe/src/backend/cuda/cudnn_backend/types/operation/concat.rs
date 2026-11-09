use singe_cudnn::backend::{
    execution::tensor_ops::ConcatOperation,
    tensor::{BackendTensorUid, Tensor},
};

#[derive(Debug)]
pub struct CudnnBackendConcatOperation {
    tensors: Vec<Tensor>,
    _operation: ConcatOperation,
}

impl CudnnBackendConcatOperation {
    pub(crate) fn new(tensors: Vec<Tensor>, operation: ConcatOperation) -> Self {
        Self {
            tensors,
            _operation: operation,
        }
    }

    pub(crate) fn binding_tensor(&self, id: BackendTensorUid) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.id() == id)
    }
}
