use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::cuda::allocator::{
        CudaAllocator, CudaAllocatorStats, CudaCachingAllocator, allocation::Initialization,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    storage::cuda::{CudaMemoryRecycler, CudaStorage},
    tensor::validate_boolean_bytes,
    utility::{checked_mul, to_i64},
};

impl CudaAllocator for CudaCachingAllocator {
    fn context(&self) -> &Arc<CudaContext> {
        Self::context(self)
    }

    fn device(&self) -> Device {
        self.device
    }

    fn allocate_uninit(
        &self,
        data_type: DataType,
        element_count: usize,
    ) -> Result<CudaStorage<MaybeUninit<u8>>> {
        self.context.bind()?;
        let byte_size = checked_mul(
            element_count,
            data_type.size_bytes(),
            "cuda allocation byte size",
        )?;
        let shape = Some(vec![to_i64(
            element_count,
            "cuda allocation element count",
        )?]);
        let memory = self.take_or_create(byte_size, data_type, shape)?;
        CudaStorage::from_uninit_recyclable(
            data_type,
            memory,
            Arc::clone(&self.cache) as Arc<dyn CudaMemoryRecycler>,
            self.stream.clone(),
        )
    }

    fn allocate_uninit_with_shape(
        &self,
        data_type: DataType,
        shape: &Shape,
    ) -> Result<CudaStorage<MaybeUninit<u8>>> {
        self.context.bind()?;
        let byte_size = checked_mul(
            shape.element_count(),
            data_type.size_bytes(),
            "cuda allocation shape byte size",
        )?;
        let memory =
            self.take_or_create(byte_size, data_type, Some(shape.dimensions().to_vec()))?;
        CudaStorage::from_uninit_recyclable(
            data_type,
            memory,
            Arc::clone(&self.cache) as Arc<dyn CudaMemoryRecycler>,
            self.stream.clone(),
        )
    }

    fn zeroed(&self, data_type: DataType, element_count: usize) -> Result<CudaStorage> {
        self.context.bind()?;
        let byte_size = checked_mul(
            element_count,
            data_type.size_bytes(),
            "cuda allocation byte size",
        )?;
        let shape = Some(vec![to_i64(
            element_count,
            "cuda allocation element count",
        )?]);
        let memory = self.take_or_create(byte_size, data_type, shape)?;
        self.initialize_storage(data_type, memory, Initialization::Zero)
    }

    fn zeroed_with_shape(&self, data_type: DataType, shape: &Shape) -> Result<CudaStorage> {
        self.context.bind()?;
        let byte_size = checked_mul(
            shape.element_count(),
            data_type.size_bytes(),
            "cuda allocation shape byte size",
        )?;
        let memory =
            self.take_or_create(byte_size, data_type, Some(shape.dimensions().to_vec()))?;
        self.initialize_storage(data_type, memory, Initialization::Zero)
    }

    fn allocate_from_bytes(&self, data_type: DataType, data: &[u8]) -> Result<CudaStorage> {
        self.context.bind()?;
        let data_type_size = data_type.size_bytes();
        if data_type_size == 0 {
            return Err(Error::invalid_argument("cuda from_bytes data type size"));
        }

        if !data.len().is_multiple_of(data_type_size) {
            return Err(Error::ByteLengthMismatch {
                name: "cuda from_bytes host byte length".into(),
                element_size: data_type_size,
                actual: data.len(),
            });
        }

        let element_count = data.len() / data_type_size;
        let element_count = to_i64(element_count, "cuda from_bytes element count")?;
        validate_boolean_bytes(data_type, data)?;
        let memory = self.take_or_create(data.len(), data_type, Some(vec![element_count]))?;
        self.initialize_storage(data_type, memory, Initialization::Bytes(data))
    }

    fn allocate_from_bytes_with_shape(
        &self,
        data_type: DataType,
        shape: &Shape,
        data: &[u8],
    ) -> Result<CudaStorage> {
        self.context.bind()?;
        let expected = checked_mul(
            shape.element_count(),
            data_type.size_bytes(),
            "cuda cached allocation host byte length",
        )?;
        if data.len() != expected {
            return Err(Error::length_mismatch(
                "cuda cached allocation host byte length",
                expected,
                data.len(),
            ));
        }

        validate_boolean_bytes(data_type, data)?;
        let memory =
            self.take_or_create(data.len(), data_type, Some(shape.dimensions().to_vec()))?;
        self.initialize_storage(data_type, memory, Initialization::Bytes(data))
    }

    fn stats(&self) -> Result<CudaAllocatorStats> {
        Self::stats(self)
    }

    fn clear_cache(&self) -> Result<()> {
        Self::clear_cache(self)
    }

    fn reset_peak(&self) {
        Self::reset_peak(self);
    }

    fn set_memory_limit(&self, limit: Option<usize>) {
        Self::set_memory_limit(self, limit);
    }
}
