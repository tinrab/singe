use crate::{
    backend::cuda::{context::CudaExecutionContext, tensor_region::PendingCudaTensor},
    data_type::DataType,
    error::Result,
    shape::Shape,
    storage::cuda::CudaStorage,
    tensor::Tensor,
};

impl CudaExecutionContext {
    /// Allocates a contiguous CUDA destination without producing readable tensor values.
    /// Use its region for raw producers and promote it only after successful completion.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails, the shape byte count overflows, or the returned allocation byte count differs from the shape requirement.
    pub fn allocate_output(&self, data_type: DataType, shape: &Shape) -> Result<PendingCudaTensor> {
        let storage = self
            .allocator()
            .allocate_uninit_with_shape(data_type, shape)?;
        PendingCudaTensor::create(storage, data_type, shape.clone())
    }

    pub(crate) fn zeroed_tensor(&self, data_type: DataType, shape: &Shape) -> Result<Tensor> {
        let storage = self.allocator().zeroed_with_shape(data_type, shape)?;
        Self::tensor_from_storage(storage, data_type, shape)
    }

    pub(crate) fn tensor_from_storage(
        storage: CudaStorage,
        data_type: DataType,
        shape: &Shape,
    ) -> Result<Tensor> {
        Tensor::create(storage, data_type, shape.clone())
    }
}
