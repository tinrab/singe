#![cfg(all(feature = "testing", feature = "cuda"))]

use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    backend::cuda::{CudaGraphProvider, LoweringPolicy},
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
};
use singe_cuda::{device::Device as CudaDevice, testing::device_lock};

fn bf16(device: Device, dimensions: &[i64]) -> Tensor {
    let count = dimensions.iter().product::<i64>();
    let values = (0..count)
        .map(|value| (value as f32).mul_add(0.25, -3.0))
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(dimensions).unwrap(), values)
        .unwrap()
        .cast(DataType::Bf16)
        .unwrap()
}

fn host_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
}

#[test]
fn test_copies_of_misaligned_views_fall_back_to_element_copies() {
    let _lock = device_lock(CudaDevice::new(0)).unwrap();
    for start in [0, 1, 3] {
        let input = bf16(Device::Cuda(0), &[4, 16]).narrow(1, start, 8).unwrap();
        let expected = bf16(Device::Cpu, &[4, 16]).narrow(1, start, 8).unwrap();

        let copied = input.contiguous().unwrap();

        assert_eq!(
            host_values(&copied),
            host_values(&expected.contiguous().unwrap())
        );
    }
}

#[test]
fn test_compiled_graphs_copy_strided_views_without_cudnn() {
    let _lock = device_lock(CudaDevice::new(0)).unwrap();
    let device = Device::Cuda(0);
    let x = bf16(device, &[2, 3, 5, 8]);
    let function = |x: &Tensor| x.transpose(1, 2)?.contiguous()?.reshape([2, 5, 24]);
    let expected = function(&bf16(Device::Cpu, &[2, 3, 5, 8])).unwrap();
    let (captured, _) =
        Graph::capture([("x", x.clone())], |inputs| Ok(vec![function(&inputs[0])?])).unwrap();
    let options = GraphPrepareOptions::new(device).with_cuda_options(
        GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::Kernel,
        ])),
    );

    let mut prepared = captured.prepare(options).unwrap();
    let actual = prepared
        .run(&HashMap::from([("x".to_owned(), x)]))
        .unwrap()
        .remove(0);

    assert_eq!(host_values(&actual), host_values(&expected));
}
