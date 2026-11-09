mod tuned;
mod workspace;

use super::*;

fn matmul_autotune_key(device: Device) -> AutotuneCacheKey {
    let lhs_shape = Shape::create([16, 32]).unwrap();
    let lhs = Tensor::from_vec(device, &lhs_shape, vec![0.0_f32; 16 * 32]).unwrap();
    let rhs_shape = Shape::create([32, 8]).unwrap();
    let rhs = Tensor::from_vec(device, &rhs_shape, vec![0.0_f32; 32 * 8]).unwrap();
    let output_shape = Shape::create([16, 8]).unwrap();
    let output = Tensor::from_vec(device, &output_shape, vec![0.0_f32; 16 * 8]).unwrap();

    AutotuneCacheKey::from_tensors(
        Operation::MatMul,
        DataType::F32,
        [&lhs, &rhs, &output],
        CudaGraphProvider::CublasLt,
        "test-cublaslt",
        Some(4096),
    )
}
