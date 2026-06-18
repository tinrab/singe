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

const M: i64 = 16;
const N: i64 = 16;
const K: i64 = 16;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let rows = DeviceMemory::from_slice(&[M, M])?;
    let cols = DeviceMemory::from_slice(&[N, N])?;
    let reduction = DeviceMemory::from_slice(&[K, K])?;
    let lda = DeviceMemory::from_slice(&[M, M])?;
    let ldb = DeviceMemory::from_slice(&[K, K])?;
    let ldc = DeviceMemory::from_slice(&[M, M])?;
    let ldd = DeviceMemory::from_slice(&[M, M])?;

    let a_batches = [
        DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (M * K) as usize])?,
        DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (M * K) as usize])?,
    ];
    let b_batches = [
        DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (K * N) as usize])?,
        DeviceMemory::from_slice(&vec![f4e2m1::from_bits(1); (K * N) as usize])?,
    ];
    let c_batches = [
        DeviceMemory::<bf16>::zeroes((M * N) as usize)?,
        DeviceMemory::<bf16>::zeroes((M * N) as usize)?,
    ];
    let mut d_batches = [
        DeviceMemory::<bf16>::zeroes((M * N) as usize)?,
        DeviceMemory::<bf16>::zeroes((M * N) as usize)?,
    ];
    let scale_a_tensors = [
        DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?,
        DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?,
    ];
    let scale_b_tensors = [
        DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?,
        DeviceMemory::from_slice(&[f8e4m3::from_bits(1)])?,
    ];

    let a_ptrs = DeviceMemory::from_slice(
        &a_batches
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<_>>(),
    )?;
    let b_ptrs = DeviceMemory::from_slice(
        &b_batches
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<_>>(),
    )?;
    let c_ptrs = DeviceMemory::from_slice(
        &c_batches
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<_>>(),
    )?;
    let mut d_ptrs = DeviceMemory::from_slice(
        &d_batches
            .iter_mut()
            .map(|batch| batch.as_mut_ptr())
            .collect::<Vec<_>>(),
    )?;
    let a_scale_ptrs = DeviceMemory::from_slice(
        &scale_a_tensors
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<_>>(),
    )?;
    let b_scale_ptrs = DeviceMemory::from_slice(
        &scale_b_tensors
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<_>>(),
    )?;

    let a_layout = MatrixLayout::create_grouped(DataType::F4E2M1, &rows, &reduction, &lda)?;
    let b_layout = MatrixLayout::create_grouped(DataType::F4E2M1, &reduction, &cols, &ldb)?;
    let c_layout = MatrixLayout::create_grouped(DataType::Bf16, &rows, &cols, &ldc)?;
    let d_layout = MatrixLayout::create_grouped(DataType::Bf16, &rows, &cols, &ldd)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_a_scale_mode(MatrixScale::Vector16UE4M3)?;
    desc.set_b_scale_mode(MatrixScale::Vector16UE4M3)?;
    desc.set_a_scale_pointer(&a_scale_ptrs)?;
    desc.set_b_scale_pointer(&b_scale_ptrs)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_grouped_desc_d_average_rows(M)?;
    preference.set_grouped_desc_d_average_cols(N)?;
    preference.set_grouped_average_reduction_dim(K)?;
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
        &a_ptrs,
        &a_layout,
        &b_ptrs,
        &b_layout,
        &beta,
        &c_ptrs,
        &c_layout,
        &mut d_ptrs,
        &d_layout,
        Some(&heuristics[0].algorithm),
        Some(&mut workspace),
        None,
    )?;

    println!("lt_nvfp4gemm_grouped_simple groups: {}", rows.len());
    Ok(())
}
