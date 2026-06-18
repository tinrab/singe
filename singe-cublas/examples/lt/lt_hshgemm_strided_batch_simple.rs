use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul, matmul_algorithm_heuristics},
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::f16,
};

const M: u64 = 2;
const N: u64 = 2;
const K: u64 = 2;
const BATCH_COUNT: i32 = 2;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let one = f16::from_f32(1.0);
    let two = f16::from_f32(2.0);
    let three = f16::from_f32(3.0);
    let four = f16::from_f32(4.0);

    let a = DeviceMemory::from_slice(&[one, three, two, four, two, four, one, three])?;
    let b = DeviceMemory::from_slice(&[one, one, one, one, one, two, three, four])?;
    let c = DeviceMemory::<f16>::zeroes((M * N * BATCH_COUNT as u64) as usize)?;
    let mut d = DeviceMemory::<f16>::zeroes((M * N * BATCH_COUNT as u64) as usize)?;

    let mut a_layout = MatrixLayout::create(DataType::F16, M, K, M as i64)?;
    a_layout.set_batch_count(BATCH_COUNT)?;
    a_layout.set_strided_batch_offset((M * K) as i64)?;

    let mut b_layout = MatrixLayout::create(DataType::F16, K, N, K as i64)?;
    b_layout.set_batch_count(BATCH_COUNT)?;
    b_layout.set_strided_batch_offset((K * N) as i64)?;

    let mut c_layout = MatrixLayout::create(DataType::F16, M, N, M as i64)?;
    c_layout.set_batch_count(BATCH_COUNT)?;
    c_layout.set_strided_batch_offset((M * N) as i64)?;

    let mut d_layout = MatrixLayout::create(DataType::F16, M, N, M as i64)?;
    d_layout.set_batch_count(BATCH_COUNT)?;
    d_layout.set_strided_batch_offset((M * N) as i64)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(WORKSPACE_BYTES)?;
    let heuristics = matmul_algorithm_heuristics(
        &lt,
        &desc,
        &a_layout,
        &b_layout,
        &c_layout,
        &d_layout,
        &preference,
        1,
    )?;

    let mut workspace = DeviceMemory::<u8>::create(heuristics[0].workspace_size)?;
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    matmul(
        &lt,
        &desc,
        &alpha,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &beta,
        &c,
        &c_layout,
        &mut d,
        &d_layout,
        Some(&heuristics[0].algorithm),
        Some(&mut workspace),
        None,
    )?;

    let result = d.copy_to_host_vec()?;
    println!("lt_hshgemm_strided_batch_simple result: {result:?}");
    Ok(())
}
