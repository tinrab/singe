mod inference;
mod strided;
mod validation;

use super::*;

fn cuda_device() -> Device {
    Device::Cuda(0)
}

fn contiguous_shape<const N: usize>(dimensions: [i64; N]) -> Shape {
    Shape::create(dimensions).unwrap()
}

fn maybe_cuda_f32_tensor(shape: &Shape, values: Vec<f32>) -> Option<Tensor> {
    skip_if_stub(Tensor::from_vec(cuda_device(), shape, values))
}

fn f32_tensor(shape: &Shape, values: Vec<f32>) -> Tensor {
    Tensor::from_vec(cuda_device(), shape, values).unwrap()
}
