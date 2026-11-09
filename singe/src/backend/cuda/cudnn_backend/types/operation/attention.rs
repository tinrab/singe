use singe_cudnn::backend::{
    execution::{
        OperationGraph as CudnnOperationGraph,
        attention::{DiagonalBandMaskOperation, SdpaForwardOperation},
    },
    tensor::{BackendTensorUid as CudnnTensorUid, Tensor as CudnnBackendTensor},
};

// Owns what a planned attention graph keeps referring to: its tensors, its score masks, and the operation.
#[derive(Debug)]
pub struct CudnnBackendAttentionOperation {
    tensors: Vec<CudnnBackendTensor>,
    _masks: Vec<DiagonalBandMaskOperation>,
    _mask_graph: Option<CudnnOperationGraph>,
    _attention: SdpaForwardOperation,
}

impl CudnnBackendAttentionOperation {
    pub(crate) fn binding_tensor(&self, id: CudnnTensorUid) -> Option<&CudnnBackendTensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.id() == id && !tensor.is_virtual())
    }

    pub(crate) fn new(
        tensors: Vec<CudnnBackendTensor>,
        masks: Vec<DiagonalBandMaskOperation>,
        mask_graph: Option<CudnnOperationGraph>,
        attention: SdpaForwardOperation,
    ) -> Self {
        Self {
            tensors,
            _masks: masks,
            _mask_graph: mask_graph,
            _attention: attention,
        }
    }
}
