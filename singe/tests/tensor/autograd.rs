#![cfg(all(feature = "testing", feature = "cuda"))]

use singe::{Device, Shape, Tensor};
use singe_cuda::testing::device_lock;

fn gradients(device: Device) -> Vec<Vec<f32>> {
    let tensor = |dimensions: &[i64], values: Vec<f32>| {
        Tensor::from_vec(device, Shape::create(dimensions.to_vec()).unwrap(), values).unwrap()
    };
    let x = tensor(
        &[4, 3],
        (0..12).map(|value| (value as f32 * 0.7).cos()).collect(),
    );
    let w = tensor(
        &[3, 5],
        (0..15).map(|value| (value as f32 * 0.3).sin()).collect(),
    );
    let b = tensor(&[5], vec![0.1, -0.2, 0.3, -0.4, 0.5]);
    let targets = tensor(
        &[4, 5],
        (0..20)
            .map(|value| f32::from(u8::from(value % 6 == 0)))
            .collect(),
    );
    for parameter in [&x, &w, &b] {
        parameter.set_requires_grad(true).unwrap();
    }
    let loss = x
        .matmul(&w)
        .unwrap()
        .add(&b)
        .unwrap()
        .relu()
        .unwrap()
        .log_softmax(1)
        .unwrap()
        .mul(&targets)
        .unwrap()
        .sum(&[], false)
        .unwrap()
        .neg()
        .unwrap();
    loss.backward().unwrap();
    [&x, &w, &b]
        .iter()
        .map(|parameter| {
            let grad = parameter.grad().unwrap();
            assert_eq!(grad.device(), device);
            grad.to_vec::<f32>().unwrap()
        })
        .collect()
}

#[test]
fn test_cuda_backward_matches_cpu_gradients() {
    let _lock = device_lock(0).unwrap();
    for (cuda, cpu) in gradients(Device::Cuda(0))
        .iter()
        .zip(gradients(Device::Cpu))
    {
        for (actual, expected) in cuda.iter().zip(&cpu) {
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        }
    }
}
