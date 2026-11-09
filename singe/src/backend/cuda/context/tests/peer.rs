use super::*;
use crate::backend::cuda::eager::storage::staged_copy;

fn values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|value| (value as f32).mul_add(0.25, -4.0))
        .collect()
}

#[test]
fn test_cuda_staged_copy_moves_values_through_pinned_host_memory() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if available_device_count().is_none_or(|count| count < 1) {
        return;
    }

    let shape = Shape::create([5, 7]).unwrap();
    let source = Tensor::from_vec(Device::Cuda(0), &shape, values(35)).unwrap();
    let copy = staged_copy(&source, Device::Cuda(0)).unwrap();
    assert_eq!(copy.device(), Device::Cuda(0));
    assert_eq!(copy.to_vec::<f32>().unwrap(), values(35));
}

#[test]
fn test_cuda_to_device_copies_between_gpus() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    if available_device_count().is_none_or(|count| count < 2) {
        return;
    }

    let shape = Shape::create([64, 33]).unwrap();
    let source = Tensor::from_vec(Device::Cuda(0), &shape, values(64 * 33)).unwrap();
    let strided = source.transpose(0, 1).unwrap();
    for (from, to) in [
        (Device::Cuda(0), Device::Cuda(1)),
        (Device::Cuda(1), Device::Cuda(0)),
    ] {
        let input = strided.to_device(from).unwrap();
        let output = input.to_device(to).unwrap();
        assert_eq!(output.device(), to);
        assert_eq!(
            output.to_vec::<f32>().unwrap(),
            strided.to_vec::<f32>().unwrap()
        );
        assert_eq!(output.add(&output).unwrap().device(), to);
    }
}
