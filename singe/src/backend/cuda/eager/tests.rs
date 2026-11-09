use singe_core::assert_close;
use singe_cuda::stream::StreamFlags;

use crate::{
    backend::{cuda, cuda::CudaStream},
    data_type::DataType,
    device::Device,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

#[test]
fn test_cuda_tensor_full_and_ones_match_expected_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 2]).unwrap();
    let Some(full) = skip_if_stub(Tensor::full(Device::Cuda(0), &shape, 3.5_f32)) else {
        return;
    };
    let ones = Tensor::ones::<f32>(Device::Cuda(0), &shape).unwrap();

    assert_eq!(full.device(), Device::Cuda(0));
    assert_eq!(full.to_vec::<f32>().unwrap(), vec![3.5; 4]);
    assert_eq!(ones.to_vec::<f32>().unwrap(), vec![1.0; 4]);
}

#[test]
fn test_cuda_tensor_scalar_matches_expected_value() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(scalar) = skip_if_stub(Tensor::scalar(Device::Cuda(0), 2.5_f32)) else {
        return;
    };

    assert_eq!(scalar.device(), Device::Cuda(0));
    assert_eq!(scalar.data_type(), DataType::F32);
    assert_eq!(scalar.shape(), &Shape::scalar());
    assert_eq!(scalar.to_vec::<f32>().unwrap(), vec![2.5]);
}

#[test]
fn test_cuda_tensor_arange_matches_expected_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(tensor) = skip_if_stub(Tensor::arange(Device::Cuda(0), 0.0_f32, 5.0, 2.0)) else {
        return;
    };

    assert_eq!(tensor.device(), Device::Cuda(0));
    assert_eq!(tensor.data_type(), DataType::F32);
    assert_eq!(tensor.shape().dimensions(), &[3]);
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[0.0, 2.0, 4.0]);
}

#[test]
fn test_cuda_tensor_linspace_matches_expected_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let Some(tensor) = skip_if_stub(Tensor::linspace(Device::Cuda(0), 0.0_f32, 1.0, 5)) else {
        return;
    };

    assert_eq!(tensor.device(), Device::Cuda(0));
    assert_eq!(tensor.data_type(), DataType::F32);
    assert_eq!(tensor.shape().dimensions(), &[5]);
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[0.0, 0.25, 0.5, 0.75, 1.0],
    );
}

#[test]
fn test_cuda_tensor_like_constructors_match_expected_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 2]).unwrap();
    let Some(source) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };

    let zeros = source.zeros_like().unwrap();
    let ones = source.ones_like().unwrap();
    let full = source.full_like(7.0_f32).unwrap();

    assert_eq!(zeros.device(), Device::Cuda(0));
    assert_eq!(zeros.data_type(), DataType::F32);
    assert_eq!(zeros.to_vec::<f32>().unwrap(), vec![0.0; 4]);
    assert_eq!(ones.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    assert_eq!(full.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

#[test]
fn test_cuda_tensor_roundtrips_to_host() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 2]).unwrap();
    let Some(tensor) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };

    assert_eq!(tensor.device(), Device::Cuda(0));
    assert_eq!(tensor.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cuda_tensor_transfer_aliases_roundtrip() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 2]).unwrap();
    let host = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let Some(device) = skip_if_stub(host.to_device(Device::Cuda(0))) else {
        return;
    };
    let host_again = device.to_device(Device::Cpu).unwrap();

    assert_eq!(device.device(), Device::Cuda(0));
    assert_eq!(host_again.device(), Device::Cpu);
    assert_eq!(
        host_again.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn test_cuda_tensor_same_device_transfer_uses_current_stream() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2, 2]).unwrap();
    let Some(source) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0],
    )) else {
        return;
    };
    let stream = CudaStream::create_with_flags(Device::Cuda(0), StreamFlags::NON_BLOCKING).unwrap();

    let copied = {
        let _guard = cuda::set_current_stream(&stream).unwrap();
        source.to_device(Device::Cuda(0)).unwrap()
    };

    assert!(cuda::current_stream(Device::Cuda(0)).unwrap().is_none());
    stream.synchronize().unwrap();
    assert_eq!(copied.device(), Device::Cuda(0));
    assert_eq!(copied.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
