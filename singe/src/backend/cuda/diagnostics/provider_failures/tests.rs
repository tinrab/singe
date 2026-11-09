use singe_cudnn::error::{Error as CudnnError, Status as CudnnStatus};

use super::*;

#[test]
fn test_resource_and_internal_errors_do_not_become_negative_capabilities() {
    for status in [
        CudnnStatus::InternalErrorDeviceAllocationFailed,
        CudnnStatus::ExecutionFailedCudaDriver,
        CudnnStatus::BadParam,
    ] {
        let error = Error::Cudnn(CudnnError::Cudnn {
            code: status,
            message: "original diagnostic".into(),
        });
        let mut failures = GraphProviderFailures::default();
        let error = record_cuda_graph_provider_failure(
            &mut failures,
            CudaGraphProvider::CudnnBackend,
            error,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Cudnn(_)));
        assert!(error.to_string().contains("original diagnostic"));
        assert!(failures.is_empty());
    }
}

// Singe's own checks let the next provider try, and a library's own rejection is an error naming the library.
#[test]
fn test_a_library_rejection_is_an_error_and_a_checked_one_is_not() {
    let mut failures = GraphProviderFailures::default();
    record_cuda_graph_provider_failure(
        &mut failures,
        CudaGraphProvider::Cublas,
        Error::unsupported_operation("matmul layout"),
    )
    .unwrap();
    assert_eq!(failures.as_slice().len(), 1);
    let error = record_cuda_graph_provider_failure(
        &mut failures,
        CudaGraphProvider::CudnnBackend,
        Error::Cudnn(CudnnError::Cudnn {
            code: CudnnStatus::NotSupported,
            message: "no engine".into(),
        }),
    )
    .unwrap_err();
    assert!(
        matches!(&error, Error::LibraryRefused { library, .. } if library == "CudnnBackend"),
        "{error}"
    );
    assert_eq!(failures.as_slice().len(), 1);
}
