use singe_cuda::error::{Error as CudaError, Status as CudaStatus};

use crate::{
    data_type::DataType, device::Device, error::Error, storage::cuda::map_cuda_allocation_error,
};

#[test]
fn test_cuda_allocation_error_maps_out_of_memory_with_context() {
    let error = map_cuda_allocation_error(
        CudaError::Cuda {
            code: CudaStatus::OutOfMemory,
            message: String::from("out of memory"),
        },
        Device::Cuda(0),
        1024,
        DataType::F32,
        Some(vec![2, 128]),
    );

    match error {
        Error::AllocationFailed {
            device: Device::Cuda(0),
            byte_count: 1024,
            data_type: DataType::F32,
            shape: Some(shape),
        } => assert_eq!(shape, vec![2, 128]),
        error => assert!(
            matches!(error, Error::AllocationFailed { .. }),
            "expected contextual allocation failure, got {error:?}"
        ),
    }
}
