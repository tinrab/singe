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
    context::Context as CudaContext, data_type::DataType, device::Device, event::EventRecordFlags,
    memory::DeviceMemory,
};

const M: u64 = 64;
const N: u64 = 64;
const K: u64 = 64;
const WORKSPACE_BYTES: usize = 1 << 20;
const REPEATS: usize = 3;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let stream = cuda.create_stream()?;
    let start = cuda.create_event()?;
    let stop = cuda.create_event()?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&vec![1.0_f32; (M * K) as usize])?;
    let b = DeviceMemory::from_slice(&vec![1.0_f32; (K * N) as usize])?;
    let c = DeviceMemory::<f32>::zeroes((M * N) as usize)?;
    let mut d = DeviceMemory::<f32>::zeroes((M * N) as usize)?;

    let a_layout = MatrixLayout::create(DataType::F32, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::F32, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::F32, M, N, M as i64)?;
    let d_layout = MatrixLayout::create(DataType::F32, M, N, M as i64)?;

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
        8,
    )?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let mut best_time = f32::INFINITY;
    let mut best_index = None;

    for (index, heuristic) in heuristics.iter().enumerate() {
        let mut workspace = DeviceMemory::<u8>::create(heuristic.workspace_size)?;
        start.record(&stream, EventRecordFlags::DEFAULT)?;
        for _ in 0..REPEATS {
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
                Some(&heuristic.algorithm),
                Some(&mut workspace),
                Some(&stream),
            )?;
        }
        stop.record(&stream, EventRecordFlags::DEFAULT)?;
        stop.synchronize()?;
        let elapsed = stop.elapsed_time_since(&start)? / REPEATS as f32;
        if elapsed < best_time {
            best_time = elapsed;
            best_index = Some(index);
        }
    }

    println!(
        "lt_sgemm_simple_auto_tuning best heuristic: {:?}, average time: {best_time:.4} ms",
        best_index
    );
    Ok(())
}
