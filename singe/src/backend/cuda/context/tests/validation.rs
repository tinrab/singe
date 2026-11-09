use super::*;

use crate::backend::cuda::CudaStream;

#[test]
fn test_cuda_negative_device_id_returns_unsupported_device() {
    let shape = Shape::create([1]).unwrap();
    let expected = "cuda device id -1 is negative";

    assert_unsupported_device_details(
        Tensor::zeroed_on(Device::Cuda(-1), DataType::F32, &shape),
        expected,
    );
    assert_unsupported_device_details(
        Tensor::from_vec(Device::Cuda(-1), &shape, vec![1.0_f32]),
        expected,
    );
    assert_unsupported_device_details(CudaStream::create(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(
        CudaStream::create_with_flags(Device::Cuda(-1), StreamFlags::DEFAULT),
        expected,
    );
    assert_unsupported_device_details(cuda::current_stream(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(cuda::synchronize(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(cuda::clear_cache(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(cuda::allocator_stats(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(cuda::execution_context(Device::Cuda(-1)), expected);
    assert_unsupported_device_details(cuda::device(-1), expected);
}

#[test]
fn test_cuda_backend_cpu_device_returns_device_mismatch() {
    assert_device_mismatch(CudaStream::create(Device::Cpu));
    assert_device_mismatch(CudaStream::create_with_flags(
        Device::Cpu,
        StreamFlags::DEFAULT,
    ));
    assert_device_mismatch(cuda::current_stream(Device::Cpu));
    assert_device_mismatch(cuda::synchronize(Device::Cpu));
    assert_device_mismatch(cuda::clear_cache(Device::Cpu));
    assert_device_mismatch(cuda::allocator_stats(Device::Cpu));
    assert_device_mismatch(cuda::execution_context(Device::Cpu));
    assert_device_mismatch(CudaExecutionContext::create(Device::Cpu));
    assert_device_mismatch(CudaExecutionContext::create_with_stream(Device::Cpu, None));
}

#[test]
fn test_cuda_out_of_range_device_id_returns_unsupported_device() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    let out_of_range = Device::Cuda(device_count);
    let shape = Shape::create([1]).unwrap();
    let expected = if device_count == 0 {
        "no cuda devices are available".to_string()
    } else {
        format!(
            "cuda device id {device_count} is out of range; visible device count is {device_count}"
        )
    };

    assert_unsupported_device_details(
        Tensor::zeroed_on(out_of_range, DataType::F32, &shape),
        &expected,
    );
    assert_unsupported_device_details(CudaStream::create(out_of_range), &expected);
    assert_unsupported_device_details(
        CudaStream::create_with_flags(out_of_range, StreamFlags::DEFAULT),
        &expected,
    );
    assert_unsupported_device_details(cuda::current_stream(out_of_range), &expected);
    assert_unsupported_device_details(cuda::synchronize(out_of_range), &expected);
    assert_unsupported_device_details(cuda::clear_cache(out_of_range), &expected);
    assert_unsupported_device_details(cuda::allocator_stats(out_of_range), &expected);
    assert_unsupported_device_details(cuda::execution_context(out_of_range), &expected);
    assert_unsupported_device_details(cuda::device(device_count), &expected);
}

#[test]
fn test_cuda_device_count_reports_non_negative_count() {
    let Some(device_count) = available_device_count() else {
        return;
    };

    assert!(device_count >= 0);
}

#[test]
fn test_cuda_backend_device_validates_device_id() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(device_count) = available_device_count() else {
        return;
    };
    if device_count == 0 {
        assert_unsupported_device_details(cuda::device(0), "no cuda devices are available");
        return;
    }

    assert_eq!(cuda::device(0).unwrap(), Device::Cuda(0));
}

#[test]
fn test_cuda_is_available_matches_device_count() {
    let expected = cuda::device_count().is_ok_and(|count| count > 0);

    assert_eq!(cuda::is_available(), expected);
}
