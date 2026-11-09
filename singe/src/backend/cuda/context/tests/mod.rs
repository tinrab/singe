mod context;
mod peer;
mod validation;

use singe_cuda::{stream::StreamFlags, testing::is_unavailable as is_cuda_unavailable};

use crate::{
    backend::cuda::{self, context::CudaExecutionContext, test_support::assert_device_mismatch},
    data_type::DataType,
    device::Device,
    error::{Error, Result, UnsupportedTarget},
    shape::Shape,
    tensor::Tensor,
    testing::maybe_lock_device,
};

fn assert_unsupported_device_details<T>(result: Result<T>, expected: &str) {
    assert!(matches!(
        result,
        Err(Error::Unsupported {
            target: UnsupportedTarget::Device,
            details,
            ..
        }) if details == expected
    ));
}

fn available_device_count() -> Option<i32> {
    match cuda::device_count() {
        Ok(device_count) => Some(device_count),
        Err(Error::Cuda(error)) if is_cuda_unavailable(&error) => None,
        Err(error) => {
            assert!(
                matches!(error, Error::Cuda(_)),
                "unexpected cuda device count error: {error:?}"
            );
            None
        }
    }
}
