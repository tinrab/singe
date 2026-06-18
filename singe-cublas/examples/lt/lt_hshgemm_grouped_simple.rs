use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul, matmul_algorithm_heuristics},
        types::PointerMode,
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::f16,
};

const GROUP_COUNT: usize = 2;
const M: i64 = 2;
const N: i64 = 2;
const K: i64 = 2;
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
        DeviceMemory::from_slice(&[f16::from_f32(1.0); (M * K) as usize])?,
        DeviceMemory::from_slice(&[f16::from_f32(2.0); (M * K) as usize])?,
    ];
    let b_batches = [
        DeviceMemory::from_slice(&[f16::from_f32(1.0); (K * N) as usize])?,
        DeviceMemory::from_slice(&[f16::from_f32(3.0); (K * N) as usize])?,
    ];
    let c_batches = [
        DeviceMemory::<f16>::zeroes((M * N) as usize)?,
        DeviceMemory::<f16>::zeroes((M * N) as usize)?,
    ];
    let mut d_batches = [
        DeviceMemory::<f16>::zeroes((M * N) as usize)?,
        DeviceMemory::<f16>::zeroes((M * N) as usize)?,
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

    let alpha = DeviceMemory::from_slice(&[1.0_f32, 1.0])?;
    let beta = DeviceMemory::from_slice(&[0.0_f32, 0.0])?;

    let a_layout = MatrixLayout::create_grouped(DataType::F16, &rows, &reduction, &lda)?;
    let b_layout = MatrixLayout::create_grouped(DataType::F16, &reduction, &cols, &ldb)?;
    let c_layout = MatrixLayout::create_grouped(DataType::F16, &rows, &cols, &ldc)?;
    let d_layout = MatrixLayout::create_grouped(DataType::F16, &rows, &cols, &ldd)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_pointer_mode(PointerMode::Device)?;
    desc.set_alpha_batch_stride(1)?;
    desc.set_beta_batch_stride(1)?;

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

    println!("lt_hshgemm_grouped_simple groups: {GROUP_COUNT}");
    Ok(())
}
