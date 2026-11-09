#![cfg(all(feature = "testing", feature = "cuda"))]

use singe::{DataType, Device, Shape, Tensor, backend::cuda::CudaStream};
use singe_cuda::testing::device_lock;

fn values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|value| (value as f32).mul_add(0.5, -3.0))
        .collect()
}

#[test]
fn test_pinned_tensors_behave_like_cpu_tensors() {
    let _lock = device_lock(0).unwrap();
    let shape = Shape::create([4, 6]).unwrap();
    let pageable = Tensor::from_vec(Device::Cpu, &shape, values(24)).unwrap();
    assert!(!pageable.is_pinned());

    let pinned = pageable.pin_memory().unwrap();
    assert!(pinned.is_pinned());
    assert_eq!(pinned.device(), Device::Cpu);
    assert!(pinned.pin_memory().unwrap().is_pinned());
    assert_eq!(
        pinned.to_vec::<f32>().unwrap(),
        pageable.to_vec::<f32>().unwrap()
    );
    let transposed = pinned.transpose(0, 1).unwrap();
    assert_eq!(
        pinned.matmul(&transposed).unwrap().to_vec::<f32>().unwrap(),
        pageable
            .matmul(&pageable.transpose(0, 1).unwrap())
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    );
    assert!(!pinned.exp().unwrap().is_pinned());

    let strided = pageable.transpose(0, 1).unwrap().pin_memory().unwrap();
    assert!(strided.is_contiguous());
    assert_eq!(
        strided.to_vec::<f32>().unwrap(),
        pageable.transpose(0, 1).unwrap().to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_pinned_tensors_round_trip_through_cuda() {
    let _lock = device_lock(0).unwrap();
    let shape = Shape::create([3, 5]).unwrap();
    let pinned = Tensor::from_vec(Device::Cpu, &shape, values(15))
        .unwrap()
        .pin_memory()
        .unwrap();
    let device = pinned.to_device(Device::Cuda(0)).unwrap();
    assert_eq!(device.device(), Device::Cuda(0));
    assert_eq!(
        device
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        pinned.to_vec::<f32>().unwrap()
    );

    let mask = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![true, false, true],
    )
    .unwrap()
    .pin_memory()
    .unwrap();
    assert_eq!(
        mask.to_device(Device::Cuda(0))
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        [true, false, true]
    );

    let zeros = Tensor::zeros_pinned(DataType::F32, Shape::create([2, 2]).unwrap()).unwrap();
    assert!(zeros.is_pinned());
    assert_eq!(zeros.to_vec::<f32>().unwrap(), [0.0; 4]);
    assert!(device.pin_memory().is_err());
}

#[test]
fn test_non_blocking_copies_return_their_tensors_after_waiting() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let stream = CudaStream::create(device).unwrap();
    let shape = Shape::create([64, 64]).unwrap();
    let expected = values(64 * 64);
    let pinned = Tensor::from_vec(Device::Cpu, &shape, expected.clone())
        .unwrap()
        .pin_memory()
        .unwrap();

    let upload = pinned.to_device_non_blocking(device, &stream).unwrap();
    upload.event().synchronize().unwrap();
    assert!(upload.is_complete().unwrap());
    let (on_device, source) = upload.wait_with_source().unwrap();
    assert!(source.is_pinned());
    assert_eq!(on_device.device(), device);
    assert_eq!(on_device.to_vec::<f32>().unwrap(), expected);

    // One pinned destination serves several downloads.
    let mut destination = Tensor::zeros_pinned(DataType::F32, shape).unwrap();
    for scale in [2.0, 3.0] {
        let scaled = on_device.mul_scalar(scale).unwrap();
        destination = scaled
            .copy_to_non_blocking(destination, &stream)
            .unwrap()
            .wait()
            .unwrap();
        assert!(destination.is_pinned());
        assert_eq!(
            destination.to_vec::<f32>().unwrap(),
            expected
                .iter()
                .map(|value| value * scale)
                .collect::<Vec<_>>()
        );
    }

    let downloaded = on_device
        .transpose(0, 1)
        .unwrap()
        .to_device_non_blocking(Device::Cpu, &stream)
        .unwrap()
        .wait()
        .unwrap();
    assert!(downloaded.is_pinned());
    assert_eq!(
        downloaded.to_vec::<f32>().unwrap(),
        on_device.transpose(0, 1).unwrap().to_vec::<f32>().unwrap()
    );

    let copied = on_device
        .transpose(0, 1)
        .unwrap()
        .to_device_non_blocking(device, &stream)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(copied.device(), device);
    assert_eq!(
        copied.to_vec::<f32>().unwrap(),
        on_device.transpose(0, 1).unwrap().to_vec::<f32>().unwrap()
    );

    // Dropping an unfinished transfer waits for it instead of freeing memory under the copy.
    drop(source.to_device_non_blocking(device, &stream).unwrap());
}

#[test]
fn test_non_blocking_copies_reject_unsafe_host_buffers() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let stream = CudaStream::create(device).unwrap();
    let shape = Shape::create([8]).unwrap();
    let pageable = Tensor::from_vec(Device::Cpu, &shape, values(8)).unwrap();
    assert!(
        pageable
            .clone()
            .to_device_non_blocking(device, &stream)
            .is_err()
    );

    let pinned = pageable.pin_memory().unwrap();
    let shared = pinned.clone();
    assert!(pinned.to_device_non_blocking(device, &stream).is_err());
    drop(shared);

    let on_device = pageable.to_device(device).unwrap();
    let wrong_shape = Tensor::zeros_pinned(DataType::F32, Shape::create([4]).unwrap()).unwrap();
    assert!(
        on_device
            .copy_to_non_blocking(wrong_shape, &stream)
            .is_err()
    );
    assert!(on_device.copy_to_non_blocking(pageable, &stream).is_err());
}
