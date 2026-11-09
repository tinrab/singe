use super::*;

use crate::backend::cuda::CudaStream;

#[test]
fn test_cuda_execution_context_rejects_stream_from_different_device() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    if device_count < 2 {
        return;
    }

    let stream = CudaStream::create(Device::Cuda(1)).unwrap();
    assert_device_mismatch(CudaExecutionContext::create_with_stream(
        Device::Cuda(0),
        Some(stream),
    ));
}

#[test]
fn test_cuda_execution_context_uses_current_stream() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    if device_count == 0 {
        return;
    }

    let device = Device::Cuda(0);
    let stream = CudaStream::create(device).unwrap();
    let stream_id = stream.id().unwrap();
    {
        let _guard = cuda::set_current_stream(&stream).unwrap();
        let context = cuda::execution_context(device).unwrap();
        assert_eq!(context.device(), device);
        assert_eq!(context.stream().unwrap().id().unwrap(), stream_id);
    }

    let context = cuda::execution_context(device).unwrap();
    assert!(context.stream().is_none());
}

#[test]
fn test_cuda_execution_context_create_uses_default_stream_policy() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    if device_count == 0 {
        return;
    }

    let context = CudaExecutionContext::create(Device::Cuda(0)).unwrap();
    assert_eq!(context.device(), Device::Cuda(0));
    assert!(context.stream().is_none());
}

#[test]
fn test_cuda_execution_context_with_cutensor_preserves_owned_stream_operation_error() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    if device_count == 0 {
        return;
    }

    let device = Device::Cuda(0);
    let context = CudaExecutionContext::create(device).unwrap();
    assert!(context.stream().is_none());

    let error = context
        .with_cutensor(|_, stream| {
            assert_eq!(stream.device(), device);
            Err::<(), Error>(Error::invalid_argument("cuda cutensor sentinel operation"))
        })
        .unwrap_err();

    assert!(matches!(
        error,
        Error::InvalidArgument { name } if name == "cuda cutensor sentinel operation"
    ));
    assert!(cuda::current_stream(device).unwrap().is_none());
}
