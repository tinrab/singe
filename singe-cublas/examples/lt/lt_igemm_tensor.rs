use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::{MatrixLayout, MatrixTransformDescriptor},
        matmul::{MatmulDescriptor, matmul, matrix_transform},
        types::Order,
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

const M: u64 = 32;
const N: u64 = 32;
const K: u64 = 32;

fn round_up(value: u64, multiple: u64) -> u64 {
    value.div_ceil(multiple) * multiple
}

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&vec![1_i8; (M * K) as usize])?;
    let b = DeviceMemory::from_slice(&vec![1_i8; (K * N) as usize])?;
    let mut c = DeviceMemory::<i32>::zeroes((M * N) as usize)?;

    let ldatransform = 32 * M;
    let ldbtransform = 32 * round_up(N, 8);
    let ldctransform = 32 * M;

    let mut a_transform =
        DeviceMemory::<i8>::zeroes((round_up(K, 32) / 32 * ldatransform) as usize)?;
    let mut b_transform =
        DeviceMemory::<i8>::zeroes((round_up(K, 32) / 32 * ldbtransform) as usize)?;
    let mut c_transform =
        DeviceMemory::<i32>::zeroes((round_up(N, 32) / 32 * ldctransform) as usize)?;

    let a_layout = MatrixLayout::create(DataType::I8, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::I8, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::I32, M, N, M as i64)?;

    let mut a_transform_layout = MatrixLayout::create(DataType::I8, M, K, ldatransform as i64)?;
    a_transform_layout.set_order(Order::Column32)?;

    let mut b_transform_layout = MatrixLayout::create(DataType::I8, N, K, ldbtransform as i64)?;
    b_transform_layout.set_order(Order::Column4_4R2_8C)?;

    let mut c_transform_layout = MatrixLayout::create(DataType::I32, M, N, ldctransform as i64)?;
    c_transform_layout.set_order(Order::Column32)?;

    let mut transform_desc = MatrixTransformDescriptor::create(DataType::F32)?;
    let transform_alpha = 1.0_f32;
    let transform_beta = 0.0_f32;

    matrix_transform(
        &lt,
        &transform_desc,
        &transform_alpha,
        &a,
        &a_layout,
        &transform_beta,
        None::<(&DeviceMemory<i8>, &MatrixLayout)>,
        &mut a_transform,
        &a_transform_layout,
        None,
    )?;

    transform_desc.set_transpose_a(Operation::Transpose)?;
    matrix_transform(
        &lt,
        &transform_desc,
        &transform_alpha,
        &b,
        &b_layout,
        &transform_beta,
        None::<(&DeviceMemory<i8>, &MatrixLayout)>,
        &mut b_transform,
        &b_transform_layout,
        None,
    )?;

    let mut desc = MatmulDescriptor::create(ComputeType::I32, DataType::I32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    let alpha = 1_i32;
    let beta = 0_i32;
    let c_accumulator =
        DeviceMemory::<i32>::zeroes((round_up(N, 32) / 32 * ldctransform) as usize)?;
    matmul(
        &lt,
        &desc,
        &alpha,
        &a_transform,
        &a_transform_layout,
        &b_transform,
        &b_transform_layout,
        &beta,
        &c_accumulator,
        &c_transform_layout,
        &mut c_transform,
        &c_transform_layout,
        None,
        None,
        None,
    )?;

    transform_desc.set_transpose_a(Operation::NonTranspose)?;
    matrix_transform(
        &lt,
        &transform_desc,
        &transform_alpha,
        &c_transform,
        &c_transform_layout,
        &transform_beta,
        None::<(&DeviceMemory<i32>, &MatrixLayout)>,
        &mut c,
        &c_layout,
        None,
    )?;

    let result = c.copy_to_host_vec()?;
    assert!(result.iter().all(|&value| value == K as i32));
    println!("lt_igemm_tensor result[0]: {}", result[0]);
    Ok(())
}
