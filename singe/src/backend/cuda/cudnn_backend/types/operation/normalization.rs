use singe_cudnn::backend::{
    execution::{
        normalization::NormalizationForwardOperation,
        pointwise::{PointwiseDescriptor, PointwiseOperation},
        tensor_ops::ReshapeOperation,
    },
    tensor::{BackendTensorUid, Tensor},
};

#[derive(Debug)]
pub struct CudnnBackendNormalizationOperation {
    tensors: Vec<Tensor>,
    _input_reshapes: Vec<ReshapeOperation>,
    _normalization: NormalizationForwardOperation,
    _pointwise_descriptors: Vec<PointwiseDescriptor>,
    _pointwise_operations: Vec<PointwiseOperation>,
    _output_reshape: ReshapeOperation,
}

impl CudnnBackendNormalizationOperation {
    pub(crate) fn new(
        tensors: Vec<Tensor>,
        input_reshapes: Vec<ReshapeOperation>,
        normalization: NormalizationForwardOperation,
        pointwise_descriptors: Vec<PointwiseDescriptor>,
        pointwise_operations: Vec<PointwiseOperation>,
        output_reshape: ReshapeOperation,
    ) -> Self {
        Self {
            tensors,
            _input_reshapes: input_reshapes,
            _normalization: normalization,
            _pointwise_descriptors: pointwise_descriptors,
            _pointwise_operations: pointwise_operations,
            _output_reshape: output_reshape,
        }
    }

    pub(crate) fn binding_tensor(&self, id: BackendTensorUid) -> Option<&Tensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.id() == id && !tensor.is_virtual())
    }
}
