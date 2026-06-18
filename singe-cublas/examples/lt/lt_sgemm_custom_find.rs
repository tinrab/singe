use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{
            MatmulDescriptor, MatmulPreference, check_matmul_algorithm, matmul, matmul_algorithm,
            matmul_algorithm_ids,
        },
        types::{MatmulAlgorithmCapAttribute, MatmulAlgorithmConfigAttribute},
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
const REPEATS: usize = 2;

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
    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    let ids = matmul_algorithm_ids(
        &lt,
        ComputeType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        8,
    )?;

    let mut best = None;
    for id in ids {
        let mut algorithm = matmul_algorithm(
            &lt,
            ComputeType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            id,
        )?;

        let splitk_support =
            algorithm.cap_attribute::<i32>(MatmulAlgorithmCapAttribute::SplitKSupport)? != 0;
        for splitk in [1_i32, 2_i32] {
            if splitk > 1 && !splitk_support {
                continue;
            }
            algorithm.set_config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount, &splitk)?;
            let heuristic = match check_matmul_algorithm(
                &lt, &desc, &a_layout, &b_layout, &c_layout, &d_layout, &algorithm,
            ) {
                Ok(heuristic) if heuristic.workspace_size <= WORKSPACE_BYTES => heuristic,
                _ => continue,
            };

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
                    Some(&algorithm),
                    Some(&mut workspace),
                    Some(&stream),
                )?;
            }
            stop.record(&stream, EventRecordFlags::DEFAULT)?;
            stop.synchronize()?;
            let elapsed = stop.elapsed_time_since(&start)? / REPEATS as f32;
            if best.is_none_or(|(_, _, best_time)| elapsed < best_time) {
                best = Some((id, splitk, elapsed));
            }
        }
    }

    println!("lt_sgemm_custom_find best candidate: {best:?}");
    Ok(())
}
