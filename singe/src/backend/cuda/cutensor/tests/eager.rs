use super::*;

use crate::backend::cuda::CudaStream;

#[test]
fn test_cuda_cutensor_contraction_matches_expected_values() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let Some(a) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &a_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )) else {
        return;
    };
    let b = Tensor::from_vec(
        Device::Cuda(0),
        &b_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let c = Tensor::from_vec(Device::Cuda(0), &c_shape, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let result = contraction(&a, &[0, 2], &b, &[2, 1], &c, &[0, 1], &[0, 1], 0.5, 2.0).unwrap();

    assert_eq!(result.device(), Device::Cuda(0));
    assert_eq!(result.shape().dimensions(), &[2, 2]);
    assert_close!(&result.to_vec::<f32>().unwrap(), &[31.0, 36.0, 75.5, 85.0],);
}

#[test]
fn test_cuda_cutensor_contraction_returns_empty_for_zero_sized_output() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([0, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([0, 2]).unwrap();
    let Some(a) = skip_if_stub(Tensor::from_vec(
        Device::Cuda(0),
        &a_shape,
        Vec::<f32>::new(),
    )) else {
        return;
    };
    let b = Tensor::from_vec(
        Device::Cuda(0),
        &b_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let c = Tensor::from_vec(Device::Cuda(0), &c_shape, Vec::<f32>::new()).unwrap();

    let result = contraction(&a, &[0, 2], &b, &[2, 1], &c, &[0, 1], &[0, 1], 1.0, 0.0).unwrap();

    assert_eq!(result.device(), Device::Cuda(0));
    assert_eq!(result.shape().dimensions(), &[0, 2]);
    assert!(result.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn test_cuda_cutensor_contraction_runs_on_current_stream() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let a_shape = Shape::create([2, 3]).unwrap();
    let b_shape = Shape::create([3, 2]).unwrap();
    let c_shape = Shape::create([2, 2]).unwrap();
    let Some(stream) = skip_if_stub(CudaStream::create(Device::Cuda(0))) else {
        return;
    };
    let a = Tensor::from_vec(
        Device::Cuda(0),
        &a_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b = Tensor::from_vec(
        Device::Cuda(0),
        &b_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let c = Tensor::from_vec(Device::Cuda(0), &c_shape, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let result = {
        let _guard = cuda::set_current_stream(&stream).unwrap();
        contraction(&a, &[0, 2], &b, &[2, 1], &c, &[0, 1], &[0, 1], 0.5, 2.0).unwrap()
    };

    assert!(cuda::current_stream(Device::Cuda(0)).unwrap().is_none());
    stream.synchronize().unwrap();
    assert_close!(&result.to_vec::<f32>().unwrap(), &[31.0, 36.0, 75.5, 85.0],);
}
