use singe_cudnn::backend::{
    execution::reduction::{ReductionDescriptor, ReductionOperation},
    tensor::{BackendTensorUid, Tensor},
};

#[derive(Debug)]
pub struct CudnnBackendReductionOperation {
    input: Tensor,
    output: Tensor,
    _reduction: ReductionDescriptor,
    _operation: ReductionOperation,
}

impl CudnnBackendReductionOperation {
    pub(crate) fn new(
        input: Tensor,
        output: Tensor,
        reduction: ReductionDescriptor,
        operation: ReductionOperation,
    ) -> Self {
        Self {
            input,
            output,
            _reduction: reduction,
            _operation: operation,
        }
    }

    pub(crate) fn binding_tensor(&self, id: BackendTensorUid) -> Option<&Tensor> {
        [&self.input, &self.output]
            .into_iter()
            .find(|tensor| tensor.id() == id)
    }
}
