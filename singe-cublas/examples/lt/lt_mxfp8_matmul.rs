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
    types::{bf16, f8e4m3, f8ue8m0},
};

const M: u64 = 32;
const N: u64 = 32;
const K: u64 = 32;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&vec![f8e4m3::from_bits(1); (M * K) as usize])?;
    let b = DeviceMemory::from_slice(&vec![f8e4m3::from_bits(1); (K * N) as usize])?;
    let c = DeviceMemory::<bf16>::zeroes((M * N) as usize)?;
    let mut d = DeviceMemory::<f8e4m3>::zeroes((M * N) as usize)?;

    let a_scale = DeviceMemory::from_slice(&[f8ue8m0::from_bits(1)])?;
    let b_scale = DeviceMemory::from_slice(&[f8ue8m0::from_bits(1)])?;
    let d_out_scale = DeviceMemory::from_slice(&[f8ue8m0::from_bits(1)])?;

    let a_layout = MatrixLayout::create(DataType::F8E4M3, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::F8E4M3, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::Bf16, M, N, M as i64)?;
    let d_layout = MatrixLayout::create(DataType::F8E4M3, M, N, M as i64)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_a_scale_mode(MatrixScale::Vector32UE8M0)?;
    desc.set_b_scale_mode(MatrixScale::Vector32UE8M0)?;
    desc.set_d_out_scale_mode(MatrixScale::Vector32UE8M0)?;
    desc.set_a_scale_pointer(&a_scale)?;
    desc.set_b_scale_pointer(&b_scale)?;
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
        "lt_mxfp8_matmul output elements: {}",
        d.copy_to_host_vec()?.len()
    );
    Ok(())
}
