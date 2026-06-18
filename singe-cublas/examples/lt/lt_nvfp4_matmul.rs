use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul, matmul_algorithm_heuristics},
        types::MatrixScale,
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    memory::DeviceMemory,
    types::{bf16, f4e2m1, f8e4m3},
};

const M: u64 = 16;
const N: u64 = 16;
const K: u64 = 16;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (M * K) as usize])?;
    let b = DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (K * N) as usize])?;
    let c = DeviceMemory::<bf16>::zeroes((M * N) as usize)?;
    let mut d = DeviceMemory::<f4e2m1>::zeroes((M * N) as usize)?;

    let a_scale = DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?;
    let b_scale = DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?;
    let d_scale = DeviceMemory::from_slice(&[1.0_f32])?;
    let d_out_scale = DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?;

    let a_layout = MatrixLayout::create(DataType::F4E2M1, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::F4E2M1, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::Bf16, M, N, M as i64)?;
    let d_layout = MatrixLayout::create(DataType::F4E2M1, M, N, M as i64)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_a_scale_mode(MatrixScale::Vector16UE4M3)?;
    desc.set_b_scale_mode(MatrixScale::Vector16UE4M3)?;
    desc.set_d_scale_mode(MatrixScale::Scalar32F)?;
    desc.set_d_out_scale_mode(MatrixScale::Vector16UE4M3)?;
    desc.set_a_scale_pointer(&a_scale)?;
    desc.set_b_scale_pointer(&b_scale)?;
    desc.set_d_scale_pointer(&d_scale)?;
    desc.set_d_out_scale_pointer(&d_out_scale)?;

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

    println!(
        "lt_nvfp4_matmul output elements: {}",
        d.copy_to_host_vec()?.len()
    );
    Ok(())
}
