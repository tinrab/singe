use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, matmul},
        types::BatchMode,
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

    let a_batches = [
        DeviceMemory::from_slice(&[one, three, two, four])?,
        DeviceMemory::from_slice(&[two, four, one, three])?,
    ];
    let b_batches = [
        DeviceMemory::from_slice(&[one, one, one, one])?,
        DeviceMemory::from_slice(&[one, two, three, four])?,
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
            .collect::<Vec<*const f16>>(),
    )?;
    let b_ptrs = DeviceMemory::from_slice(
        &b_batches
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<*const f16>>(),
    )?;
    let c_ptrs = DeviceMemory::from_slice(
        &c_batches
            .iter()
            .map(DeviceMemory::as_ptr)
            .collect::<Vec<*const f16>>(),
    )?;
    let mut d_ptrs = DeviceMemory::from_slice(
        &d_batches
            .iter_mut()
            .map(|batch| batch.as_mut_ptr())
            .collect::<Vec<*mut f16>>(),
    )?;
    let mut workspace = DeviceMemory::<u8>::create(WORKSPACE_BYTES)?;

    let mut a_layout = MatrixLayout::create(DataType::F16, M, K, M as i64)?;
    a_layout.set_batch_count(BATCH_COUNT)?;
    a_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut b_layout = MatrixLayout::create(DataType::F16, K, N, K as i64)?;
    b_layout.set_batch_count(BATCH_COUNT)?;
    b_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut c_layout = MatrixLayout::create(DataType::F16, M, N, M as i64)?;
    c_layout.set_batch_count(BATCH_COUNT)?;
    c_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut d_layout = MatrixLayout::create(DataType::F16, M, N, M as i64)?;
    d_layout.set_batch_count(BATCH_COUNT)?;
    d_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

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
        None,
        Some(&mut workspace),
        None,
    )?;

    for (index, batch) in d_batches.iter().enumerate() {
        println!(
            "lt_hshgemm_pointer_array_batch_simple batch {index}: {:?}",
            batch.copy_to_host_vec()?
        );
    }
    Ok(())
}
